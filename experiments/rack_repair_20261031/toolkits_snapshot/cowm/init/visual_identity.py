# Copyright 2026 The RLinf Authors.
"""Visual identity evidence from a frozen, Builder-visible prototype bank; no GT."""
import numpy as np

from cowm.init.associate import associate

VISUAL_MARGIN = 0.05
NEIGHBORS = 3


def support(name, prediction, margin=VISUAL_MARGIN):
    """Binary evidence for one identity; competing identities may remain mutually ambiguous."""
    scores = prediction['scores']
    if name not in scores or len(scores) < 2 or not all(np.isfinite(v) for v in scores.values()):
        raise ValueError('visual evidence requires finite scores for every identity')
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError('visual margin must be finite and positive')
    gap = scores[name] - max(v for n,v in scores.items() if n != name)
    neighbors = prediction['neighbors']
    if len(neighbors) != NEIGHBORS or any(n not in scores for n in neighbors):
        return 'unknown'
    if gap >= margin and all(n == name for n in neighbors):
        return 'positive'
    if gap <= -margin and all(n != name for n in neighbors):
        return 'negative'
    return 'unknown'


def reassociate(names, candidates, masks, predictions, prior_filter=None, share_all=False, trusted_cids=()):
    """Veto contradicted names; share proposals only when no own proposal has positive visual support.

    Shared proposal scores remain detector proposal scores, not calibrated semantic probabilities.
    """
    pool, rejected = [], []
    trusted_cids=set(trusted_cids)
    for c in candidates:
        item = dict(c)
        evidence = 'positive' if c['cid'] in trusted_cids else support(c['obj'], predictions[c['cid']])
        if evidence == 'negative' and item.get('filtered') is None:
            item['filtered'] = 'visual_identity_negative'
            rejected.append(c['cid'])
        pool.append(item)
    have_positive = {c['obj'] for c in pool if c.get('filtered') is None and
                     (c['cid'] in trusted_cids or support(c['obj'], predictions[c['cid']]) == 'positive')}
    added = []
    next_cid = max((c['cid'] for c in candidates),default=-1)+1
    masks = dict(masks)
    for c in candidates:
        if c['cid'] in trusted_cids:
            continue
        p = predictions[c['cid']]
        name = p['label']
        if (name in have_positive and not share_all) or name == c['obj'] or support(name,p) != 'positive':
            continue
        why = prior_filter(c['box'],name) if prior_filter else None
        if why is not None:
            continue
        item = {**c,'cid':next_cid,'obj':name,'filtered':None,'source':'visual_cross_prompt',
                'source_cid':c['cid'],'source_obj':c['obj'],'visual_gap':p['gap']}
        added.append(item)
        masks[next_cid] = masks[c['cid']]
        next_cid += 1
    result = associate(names,pool+added,masks)
    return {'objects':result['objects'],'cost':result['cost'],'rejected':rejected,'added':added,'masks':masks}
