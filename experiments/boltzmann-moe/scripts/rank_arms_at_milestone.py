#!/usr/bin/env python3
"""Rank arms by training loss at an EARLY token milestone, to decide an ablation without
running it to 32B.

WHY (validated 2026-09-22). Avg11 is not usable for this: for the routing_norm ablation the Avg11
delta CHANGED SIGN between 8B (-1.21pp) and 32B (+0.10pp), while the loss delta reproduced within
10% (+0.0185 -> +0.0203 nats). Loss ranking, by contrast, is near-perfectly preserved:

    134M, 7 arms, 8B vs 32B:  7/7 exact rank matches, Spearman rho = 1.000
    400M, 6 arms, 8B vs 32B:  4/6 exact,              Spearman rho = 0.943
      -- the only inversion is the two Switch arms, whose 32B losses differ by 0.0053 nats

So: an 8B read on loss gives the right ordering except between arms that are effectively tied
(<~0.005 nats). 8B is a quarter of 32B, i.e. ~4x more ablations per GPU-day.

NEEDS NO CHECKPOINT. It reads the training log, so it works even where the tok8B milestone was
never captured (abl_C and abl_D both lack theirs -- the backup script can only hard-link a
checkpoint that still exists, and max_to_keep=2 prunes an 8B checkpoint long before it runs).

Median over a 1000-step window gives ~101 samples at sd ~0.022 nats, so the estimate carries a
standard error near 0.002 nats -- an order of magnitude below the effects we act on.

Usage:
  python scripts/rank_arms_at_milestone.py --tokens 8 --tok-per-step 262144 ARM [ARM ...]
  python scripts/rank_arms_at_milestone.py --step 30517 ARM [ARM ...]
"""
import re, glob, os, statistics, argparse, sys

def series(arm):
    out = []
    for e in sorted(glob.glob(os.path.expanduser(f'~/bsub_logs/{arm}_*.stderr')), key=os.path.getmtime):
        if os.path.basename(e).startswith(('ev_', 'evg_')):
            continue
        for m in re.finditer(r'step = (\d+), train-loss = [0-9.]+, train-lm_loss = ([0-9.]+)',
                             open(e, errors='ignore').read()):
            out.append((int(m.group(1)), float(m.group(2))))
    return out

ap = argparse.ArgumentParser()
ap.add_argument('--tokens', type=float, help='milestone in billions of tokens')
ap.add_argument('--tok-per-step', type=int, default=262144)
ap.add_argument('--step', type=int, help='use this step directly instead of --tokens')
ap.add_argument('--half-window', type=int, default=500)
ap.add_argument('arms', nargs='+')
a = ap.parse_args()
if a.step is None:
    if a.tokens is None: sys.exit('give --tokens or --step')
    a.step = int(a.tokens * 1e9 / a.tok_per_step)
lo, hi = a.step - a.half_window, a.step + a.half_window
print(f"  milestone step {a.step} (window {lo}-{hi}); train-lm_loss in nats, lower better")
rows = []
for arm in a.arms:
    s = series(arm)
    w = [v for st, v in s if lo <= st <= hi]
    fin = s[-1][1] if s else None
    if not w:
        print(f"    {arm:36s} NO log lines in window"); continue
    rows.append((arm, statistics.median(w), statistics.pstdev(w), len(w), fin, s[-1][0]))
rows.sort(key=lambda r: r[1])
print(f"    {'rank':>4}  {'arm':36s} {'loss@milestone':>14s} {'sd':>7s} {'n':>4s} {'final':>8s} {'@step':>8s}")
for i, (arm, med, sd, n, fin, fs) in enumerate(rows, 1):
    se = sd / (n ** 0.5) if n else 0
    print(f"    {i:>4}  {arm:36s} {med:14.4f} {sd:7.4f} {n:4d} "
          f"{(f'{fin:.4f}' if fin else '--'):>8s} {fs:>8d}    (SE {se:.4f})")
if len(rows) > 1:
    print(f"\n  smallest adjacent gap: {min(rows[i+1][1]-rows[i][1] for i in range(len(rows)-1)):.4f} nats"
          f"  -- treat pairs closer than ~0.005 nats as unranked")
