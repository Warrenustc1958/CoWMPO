# Copyright 2026 The RLinf Authors.
"""Estimate explicit dev/fit episodes in independent CPU renderer processes."""
import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cowm import roles


def main():
    parser=argparse.ArgumentParser()
    for key in ['world','masks','evidence','out','data-role']:
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--jobs',type=int,default=4)
    args=parser.parse_args()
    if args.data_role not in ['dev','fit'] or args.jobs<1:
        raise ValueError('expected dev/fit role and positive jobs')
    eps=sorted(p.name for p in Path(args.masks).glob('*') if p.is_dir() and (p/'association.json').is_file())
    if not eps:raise ValueError('no bound episode masks')
    roles.require(eps,{args.data_role},'batch state estimation')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    logs=out/'_logs';logs.mkdir()
    def run(i):
        part=eps[i::args.jobs]
        if not part:return
        target=out/f'_part{i}'
        with (logs/f'part{i}.log').open('w') as stream:
            done=subprocess.run([sys.executable,str(Path(__file__).with_name('estimate_state.py')),
                '--world',args.world,'--masks',args.masks,'--evidence',args.evidence,
                '--out',str(target),'--episodes',','.join(part)],stdout=stream,stderr=subprocess.STDOUT)
        if done.returncode:raise RuntimeError(f'state estimation part {i} failed ({done.returncode})')
        for ep in part:shutil.move(str(target/ep),str(out/ep))
        print('state_chunk_complete',i,len(part),flush=True)
    with ThreadPoolExecutor(args.jobs) as pool:list(pool.map(run,range(args.jobs)))
    print('STATE_COMPLETE',len(eps),flush=True)


if __name__=='__main__':
    main()
