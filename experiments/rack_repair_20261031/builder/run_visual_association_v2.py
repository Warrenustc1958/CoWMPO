# Copyright 2026 The RLinf Authors.
"""Apply frozen visual bank to proposals and export identity-bound masks, without GT."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cowm import roles
from cowm.init.perceive import filter_reason
from cowm.init.visual_identity import reassociate
from visual_bank_analysis import classify


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for key in ['features', 'pool', 'bank', 'freeze', 'prior', 'out', 'data-role']:
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    if args.data_role not in ['dev', 'fit']:
        raise ValueError('visual development/validation supports only dev and fit')
    frozen = json.loads(Path(args.freeze).read_text())
    for path, digest in frozen['files'].items():
        if sha(path) != digest:
            raise ValueError('frozen file changed: ' + path)
    if sha(args.bank) != frozen['bank_sha256']:
        raise ValueError('bank differs from frozen bank')
    bank = np.load(args.bank, allow_pickle=False)
    roles.require(bank['episodes'].tolist(), {'dev'}, 'visual prototype bank')
    prior = json.loads((Path(args.prior)/'perception_prior.json').read_text())
    prior['robot_mask'] = np.load(Path(args.prior)/'perception_prior.npz')['robot_mask']
    paths = sorted(Path(args.features).glob('*.json'))
    metadata = [json.loads(path.read_text()) for path in paths]
    roles.require([meta['episode'] for meta in metadata], {args.data_role}, 'visual association')
    output = Path(args.out); output.mkdir(parents=True, exist_ok=False)
    totals = {'episodes': len(metadata), 'objects': 0, 'matched': 0, 'ambiguous': 0, 'missing': 0,
              'base_unresolved': 0, 'assignment_changes': 0, 'encoder_seconds': 0.0}
    for path, meta in zip(paths, metadata):
        ep = meta['episode']; source = Path(args.pool)/ep
        if (sha(source/'audit.json'), sha(source/'candidate_masks.npz')) != (
                meta['audit_sha256'], meta['candidate_sha256']):
            raise ValueError('candidate source changed: ' + ep)
        audit = json.loads((source/'audit.json').read_text())
        with np.load(source/'candidate_masks.npz', allow_pickle=False) as a:
            masks = {c['cid']: a[f"c{c['cid']}"] for c in audit['candidates']}
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as a:
            features, cids = a['features'], a['cids'].tolist()
        keep = bank['episodes'] != ep
        names = sorted(meta['base'])
        predictions = dict(zip(cids, classify(features, bank['features'][keep],
                                              bank['labels'][keep].tolist(), names)))
        result = reassociate(names, audit['candidates'], masks, predictions,
                             prior_filter=lambda box, name: filter_reason(box, name, prior), share_all=True)
        base = meta['base']; final = result['objects']
        unresolved = [n for n in names if base[n]['status'] != 'matched']
        changed = [n for n in names if (base[n]['status'], base[n]['cid']) !=
                   (final[n]['status'], final[n]['cid'])]
        log = {n: {'status_before': base[n]['status'], 'status_after': final[n]['status'],
                   'cid_before': base[n]['cid'], 'cid_after': final[n]['cid']}
               for n in sorted(set(unresolved + changed))}
        dest = output/ep; dest.mkdir()
        record = {'episode': ep, 'base': base, 'final': final, 'log': log,
                  'removed': result['rejected'], 'added': result['added'],
                  'base_unresolved': unresolved, 'assignment_changes': changed,
                  'visual_audit_objects': names, 'changed_untriggered': [],
                  'crops': 0, 'dino_forward_passes': meta.get('encoder_forward_passes', 1),
                  'seconds': meta.get('seconds'), 'cost_scope': meta.get('timing_scope', 'not measured'),
                  'protocol': 'frozen_visual_v2', 'freeze_sha256': sha(args.freeze),
                  'features_sha256': sha(path.with_suffix('.npz'))}
        (dest/'recheck.json').write_text(json.dumps(record, indent=1))
        np.savez_compressed(dest/'recheck_masks.npz', **{
            f"c{c['cid']}": result['masks'][c['cid']] for c in result['added']})
        np.savez_compressed(dest/'masks.npz', **{
            n: result['masks'][r['cid']] for n, r in final.items() if r['cid'] is not None})
        (dest/'association.json').write_text(json.dumps({
            'schema': 1, 'episode': ep, 'algorithm': 'frozen_visual_v2', 'objects': final,
            'masks_sha256': sha(dest/'masks.npz'), 'source': {
                'audit_sha256': meta['audit_sha256'], 'candidate_masks_sha256': meta['candidate_sha256'],
                'image_sha256': meta['image_sha256'], 'freeze_sha256': sha(args.freeze)}}, indent=1))
        totals['objects'] += len(names)
        for n in names:
            totals[final[n]['status']] += 1
        totals['base_unresolved'] += len(unresolved)
        totals['assignment_changes'] += len(changed)
        totals['encoder_seconds'] += meta.get('seconds', 0.0)
    (output/'summary.json').write_text(json.dumps(totals, indent=2))
    print(json.dumps(totals))


if __name__ == '__main__':
    main()
