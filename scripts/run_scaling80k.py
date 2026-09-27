"""Post-hoc attack-data scaling: fit the frozen registry on 80,000 attacker-training drops.

Work units are (tag, seed, windows) with windows either [1] or [4, 8], so the two units of a pair use
different encoding caches and no two processes ever write the same cache. Units are claimed through
exclusive file creation, which balances the workers dynamically and lets a worker be restarted safely.

Usage: python scripts/run_scaling80k.py --worker 0
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
from threadpoolctl import threadpool_limits

from pabhbf import revision


def units(c, tags, only=None):
    groups = [[1], [T for T in c['intervals'] if T > 1]]
    if only:
        groups = [[T for T in g if T in only] for g in groups]
    return [(tag, seed, windows) for seed in c['seeds'] for tag in tags for windows in groups if windows]


def claim(claims, name):
    try:
        fd = os.open(claims / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--tags', default=None, help='comma-separated tags; default: the scaling tags of the configuration')
    ap.add_argument('--windows', default=None, help='comma-separated windows; default: every window of the configuration')
    a = ap.parse_args()
    c = revision.config()
    tags = a.tags.split(',') if a.tags else c['scaling']['tags']
    only = [int(T) for T in a.windows.split(',')] if a.windows else None
    claims = revision.OUT / 'attacks' / 'scaling80k' / '.claims'
    claims.mkdir(parents=True, exist_ok=True)
    threads = c['attacks']['threads']
    torch.set_num_threads(threads)
    with threadpool_limits(limits=threads):
        for tag, seed, windows in units(c, tags, only):
            name = f"{tag}_s{seed}_T{'-'.join(map(str, windows))}"
            if not claim(claims, name):
                continue
            print(f'worker {a.worker} claimed {name}', flush=True)
            for T in windows:
                start = time.time()
                revision.fit(tag, seed, T, 'scaling80k')
                print(f'DONE {tag} s{seed} T{T} in {time.time() - start:.0f}s', flush=True)
    print(f'worker {a.worker} complete', flush=True)


if __name__ == '__main__':
    main()
