#!/usr/bin/env python3
"""
The link-budget reading of Section IV-B: one configuration, read in decibels.

    Gaussian location family with shift d = 3.0 (per-sample SNR 9.5 dB),
    n = 4 sensors, alpha = 1e-3 per decision,
    coherent BPSK over AWGN, P_e = Q(sqrt(2 gamma)).

Prints: the feasibility SNR of Corollary 1, the two forms of the link error
budget (17) for P_d = 0.9, the SNR at which the network actually reaches
P_d = 0.9 (search over all monotone rules, common threshold), the band above
the feasibility threshold on which the floor mechanism acts, and the crossing
gain near +1 dB. reproduce.py --slow checks the printed values against the
text.

    python3 linkbudget.py
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import robustness as rb

D, N, ALPHA, PD = 3.0, 4, 1e-3, 0.9
TGRID = 30001            # same dense threshold grid as Tables II and S1
g = lambda t: rb.roc_gauss(t, D)
snr_db = lambda pe: 10 * np.log10(norm.isf(pe) ** 2 / 2)
pe_of_db = lambda db: norm.sf(np.sqrt(2 * 10 ** (db / 10)))


def main(verbose=True):
    out = {}
    pe_f = ALPHA ** (1 / N)
    out['feas_pe'], out['feas_db'] = pe_f, snr_db(pe_f)
    pe_b = 1 / (1 + (PD / ALPHA) ** (1 / N))
    out['budget_pe'], out['budget_db'] = pe_b, snr_db(pe_b)
    pe_t = brentq(lambda p: ALPHA * rb.lam_star(g, p, T=100001) ** N - PD, 0.01, pe_b)
    out['tight_pe'], out['tight_db'] = pe_t, snr_db(pe_t)
    L = rb.lam_star(g, pe_f, T=200001)
    out['floor_rand'] = (pe_f * L) ** N                     # randomized P_d at the floor
    # the actual requirement for P_d = 0.9 (common-threshold search, S2 and S4)
    def s2_at(pe): return rb.Plane(g, N, pe, T=TGRID).S2(ALPHA)
    def s4_at(pe): return rb.Plane(g, N, pe, T=TGRID).S4(ALPHA)
    pe_90 = brentq(lambda p: s2_at(p) - PD, 0.02, 0.06, xtol=2e-4)
    out['pd90_pe'], out['pd90_db'] = pe_90, snr_db(pe_90)
    # the band above the threshold on which the floor mechanism acts
    band = []
    for db in (-3.70, -3.60, -3.40, -3.10):
        pe = pe_of_db(db)
        P = rb.Plane(g, N, pe, T=TGRID)
        band.append((db, pe, P.S2(ALPHA), P.S4(ALPHA)))
    out['band'] = band
    # the crossing near +1 dB: largest S4 - S2 for P_e in [0.04, 0.08], and the fusion-layer share
    best = (0, None, None)
    for pe in np.linspace(0.045, 0.075, 13):
        P = rb.Plane(g, N, pe, T=TGRID)
        gain = P.S4(ALPHA) - max(P.S2(ALPHA), ALPHA)
        if gain > best[0]:
            share = (P.S3(ALPHA) - max(P.S2(ALPHA), ALPHA)) / gain if gain > 1e-9 else float('nan')
            best = (gain, pe, share)
    out['cross_gain'], out['cross_pe'], out['cross_share'] = best
    out['cross_db'] = snr_db(best[1])
    if verbose:
        print(f"d={D}, n={N}, alpha={ALPHA:g}, uncoded BPSK/AWGN")
        print(f"  feasibility (Cor. 1):      P_e <= {pe_f:.4f}  SNR >= {out['feas_db']:6.2f} dB")
        print(f"  budget (17), loose, Pd=0.9: P_e <= {pe_b:.4f}  SNR >= {out['budget_db']:6.2f} dB")
        print(f"  budget (17), tight (lam*):  P_e <= {pe_t:.4f}  SNR >= {out['tight_db']:6.2f} dB")
        print(f"  P_d = 0.9 actually reached: P_e <= {pe_90:.4f}  SNR >= {out['pd90_db']:6.2f} dB")
        print(f"  randomized P_d at the floor (closed form): {out['floor_rand']:.4f}; deterministic: {ALPHA:g}")
        for db, pe, s2, s4 in band:
            print(f"  {db:6.2f} dB  P_e={pe:.4f}  deterministic S2={s2:.4f}  randomized S4={s4:.4f}")
        print(f"  crossing: gain {best[0]:.4f} at P_e={best[1]:.4f} ({out['cross_db']:.2f} dB), "
              f"fusion-layer share {best[2]:.2f}")
    return out


if __name__ == '__main__':
    main()
