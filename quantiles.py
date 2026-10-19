#!/usr/bin/env python3
"""
The quantiles of the fusion-layer share quoted in Section IV-A and Appendix
S-IV. They depend on the grid, so the script prints the grid with them.

Two things are measured on the region where randomization is worth anything at
all (the denominator otherwise divides by noise):

    share = (S3 - base) / (S4 - base),     base = max(S2, alpha)

using the same corrected baseline as the heat map -- a coin flip attains
(alpha, alpha) with no sensors, so crediting either scheme with the climb up to
the chance line measures nothing.

    python3 quantiles.py             # default grid
    python3 quantiles.py 300 150     # alpha points, Pe points
"""
import sys
import numpy as np

from localroc import D_LOGIT
from fragmentation import (get_rules, deterministic_points, upper_envelope,
                           concave_hull)
from baselines import local_curve, rule_points, hull_at_alpha_rowwise

NA = int(sys.argv[1]) if len(sys.argv) > 1 else 200
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 100
NG = 3000                                   # threshold grid
CUT = 0.01                                  # "worth anything" threshold

al = np.linspace(0, 1, 20001)
alphas = np.linspace(2e-3, 0.998, NA)
pes = np.linspace(0.005, 0.45, NP)

print(f'grid: {NA} alpha x {NP} P_e x {NG} thresholds; '
      f'region S4-base > {CUT}\n')
print(f'{"n":>3} {"samples":>8} {"median":>8} {"Q1":>8} {"<0.9":>7} '
      f'{"max(S4-S3)":>11}')

for n in (2, 3):
    pats, rules = get_rules(n)
    t = np.linspace(1e-12, 1 - 1e-12, NG)
    share, dep_max = [], 0.0
    for pe in pes:
        pts = deterministic_points(n, D_LOGIT, pe, rules, pats, ngrid=NG)
        E = upper_envelope(pts, alphas)
        H, _ = concave_hull(pts, al)
        Hi = np.interp(alphas, al, H)
        u, v = local_curve(t, D_LOGIT, pe)
        X, Y = rule_points(rules, pats, u, v)
        S3 = np.array([np.nanmax(hull_at_alpha_rowwise(X, Y, a))
                       for a in alphas])
        base = np.maximum(E, alphas)
        tot = Hi - base
        dep = np.maximum(Hi - np.maximum(S3, alphas), 0)
        dep_max = max(dep_max, float(dep.max()))
        keep = tot > CUT
        if keep.any():
            share.extend(((tot[keep] - dep[keep]) / tot[keep]).tolist())
    r = np.clip(np.array(share), 0, 1)
    print(f'{n:>3} {len(r):>8} {np.median(r):>8.3f} '
          f'{np.percentile(r, 25):>8.3f} {100*(r < 0.9).mean():>6.1f}% '
          f'{dep_max:>11.4f}')

print('\nThese move with the grid; Appendix S-IV of the supplement states the '
      'grid\nwith the numbers.')
