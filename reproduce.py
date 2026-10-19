#!/usr/bin/env python3
"""
Check the values the paper and its supplement quote: 133 of them with --slow,
47 without.

Each entry below names the claim, where it appears, and recomputes it. The
script prints the paper's value beside the computed one and exits non-zero if
any pair disagrees, so it can be run as a regression test after an edit.

    python3 reproduce.py            # 47 checks, seconds
    python3 reproduce.py --slow     # adds the sweeps: 133 checks, about 15 minutes

Nothing here reads the manuscript; the paper's values are hard-coded below, on
purpose, so that a disagreement means either the text drifted or the code did.
"""
import sys
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from localroc import lam_star, roc, phi, D_LOGIT

SLOW = '--slow' in sys.argv
rows, failed = [], 0


def check(claim, where, paper, computed, tol=5e-4):
    global failed
    ok = abs(paper - computed) <= tol
    if not ok:
        failed += 1
    rows.append((claim, where, paper, computed, ok))


# Link error budget, the inverted form of Corollary 3, and the small-alpha
# example that shows where the bound actually bites
for n_, a_, pd_, val in ((3, 1e-6, 0.9, 0.010), (3, 1e-2, 0.9, 0.182)):
    check(f'BER budget, alpha={a_:g}, n={n_}', 'eq (17)', val,
          1 / (1 + (pd_ / a_) ** (1 / n_)), 1e-3)
check('P_d cap at alpha=1e-6, n=3, P_e=0.05', 'Sec III-D', 0.007,
      1e-6 * ((1 - .05) / .05) ** 3, 5e-4)

# ---------------------------------------------------------------- Section III
# Onset: the root of (1-e)^n = e^{n-1}
for n, val in ((2, 0.3820), (3, 0.4302), (4, 0.4503), (5, 0.4614), (6, 0.4684)):
    check(f'onset P_e^o({n})', 'Thm 2, Fig S2(a)', val,
          brentq(lambda x: (1 - x) ** n - x ** (n - 1), 1e-12, 0.4999999), 1e-4)

check('onset(2) closed form', 'Thm 2', (3 - np.sqrt(5)) / 2,
      brentq(lambda x: (1 - x) ** 2 - x, 1e-12, 0.4999999), 1e-9)

# Asymptote 1/2 - onset ~ ln2/(4n-2), stated to hold to three decimals from n=4
for n in (4, 5, 6):
    r = brentq(lambda x: (1 - x) ** n - x ** (n - 1), 1e-12, 0.4999999)
    check(f'asymptote at n={n}', 'Thm 2 proof', round(r, 3),
          round(0.5 - np.log(2) / (4 * n - 2), 3), 1e-9)

# Slope cap and the value of randomization at the floor
check('lambda*(0.10)', 'Lem 2', 3.0623, lam_star(0.10), 1e-3)
check('mu^n - P_e^n, n=2, P_e=0.10', 'Thm 3', 0.0838,
      (0.10 * lam_star(0.10)) ** 2 - 0.01, 1e-3)
for pe, val in ((0.05, 0.041), (0.30, 0.135)):
    check(f'gain at floor, P_e={pe}', 'Supp S-II',
          val, (pe * lam_star(pe)) ** 2 - pe ** 2, 1e-3)
check('peak gain, alpha<=0.1', 'Sec IV-A', 0.140,
      max((p * lam_star(p)) ** 2 - p ** 2
          for p in np.linspace(0.005, 0.3162, 400)), 1e-3)

# Sensor-count requirement
for a, pd, pe, val in ((.01, .9, .05, 2), (.01, .9, .10, 3), (.01, .9, .20, 4),
                       (.001, .9, .05, 3), (.001, .9, .10, 4), (.001, .9, .20, 5)):
    check(f'n >= ... (alpha={a}, P_e={pe})', 'Cor 3', val,
          np.ceil(np.log(pd / a) / np.log((1 - pe) / pe)), 1e-9)

# ---------------------------------------------------------------- Section II
# The 2-LD benchmark of Section II-C: both LDs at A under AND, both at B under
# OR, and the interpolation C of the two global points at alpha = 0.2009
A, B, alpha = (0.3976, 0.8871), (0.1304, 0.6328), 0.2009
Af, Bf = A[0] ** 2, 1 - (1 - B[0]) ** 2
p = (Bf - alpha) / (Bf - Af)
check('benchmark AND point, P_f', 'Sec II-C', 0.1581, Af, 1e-4)
check('benchmark AND point, P_d', 'Sec II-C', 0.787, A[1] ** 2, 5e-4)
check('benchmark OR point, P_f', 'Sec II-C', 0.2438, Bf, 1e-4)
check('benchmark OR point, P_d', 'Sec II-C', 0.865, 1 - (1 - B[1]) ** 2, 5e-4)
check('benchmark point C, P_d', 'Sec II-C', 0.826,
      p * A[1] ** 2 + (1 - p) * (1 - (1 - B[1]) ** 2), 5e-4)

# Logistic fit residuals quoted in the caption of Fig. 1
check('fit residual at A', 'Fig 1 caption', 0.003,
      abs(float(roc(A[0])) - A[1]), 5e-4)
check('fit residual at B', 'Fig 1 caption', 0.001,
      abs(float(roc(B[0])) - B[1]), 5e-4)

# Differential-privacy comparison: lambda* against (1-P_e)/P_e
for pe_, val in ((0.05, 4.6), (0.10, 2.9), (0.20, 1.9)):
    check(f'DP tightening, P_e={pe_}', 'Sec I-A', val,
          (1 - pe_) / pe_ / lam_star(pe_), 0.05)
check('DP tightening^n, n=2, P_e=0.05', 'Sec I-A', 21,
      ((1 - .05) / .05 / lam_star(.05)) ** 2, 0.6)

# Ideal-channel scheduling gain, n=2: peak, location, and the width of the
# window in which it exceeds a tenth of the peak (Fig. 1 caption, Sec. II-C)
def _gain_curve_n2(pe, T=40001):
    from fragmentation import get_rules
    pats, rules = get_rules(2)
    t = np.linspace(1e-9, 1 - 1e-9, T)
    u, v = phi(t, pe), phi(roc(t), pe)
    W0 = np.prod(np.where(pats == 1, u[:, None, None], 1 - u[:, None, None]), -1)
    W1 = np.prod(np.where(pats == 1, v[:, None, None], 1 - v[:, None, None]), -1)
    px = np.concatenate([W0 @ f for f in rules]); py = np.concatenate([W1 @ f for f in rules])
    P = np.column_stack([np.append(px, [0, 1]), np.append(py, [0, 1])])
    P = P[np.lexsort((-P[:, 1], P[:, 0]))]; st = []
    for x, y in P:
        while len(st) >= 2 and (st[-1][0] - st[-2][0]) * (y - st[-2][1]) \
                - (st[-1][1] - st[-2][1]) * (x - st[-2][0]) >= 0:
            st.pop()
        st.append((x, y))
    H = np.array(st); o = np.argsort(px)
    aa = np.linspace(0.05, 0.5, 9001)
    g = np.interp(aa, H[:, 0], H[:, 1]) - np.interp(aa, px[o], np.maximum.accumulate(py[o]))
    return aa, g
_aa, _g = _gain_curve_n2(0.0)
check('ideal-channel peak gain, n=2', 'Fig 1(a)', 5.1e-3, float(_g.max()), 1e-4)
check('ideal-channel crossing', 'Fig 1(a)', 0.191, float(_aa[_g.argmax()]), 1e-3)
_m = _g > 0.1 * _g.max()
check('window where gain > peak/10', 'Fig 1(a), Sec II-C', 0.05, float(_aa[_m].max() - _aa[_m].min()), 5e-3)
# Fig. 1(c) sits at P_e = 0.25: the floor, the gain there (closed form, which
# fig_merged.py's search reproduces to 1e-4) and the P_d values quoted in Remark 4
check('Fig 1(c) floor P_e^2', 'Fig 1(c)', 0.0625, 0.25 ** 2, 1e-9)
check('Fig 1(c) gain at the floor', 'Fig 1(c), Rem 4', 0.1404, (0.25 * lam_star(0.25)) ** 2 - 0.25 ** 2, 2e-4)
check('Fig 1(c) BPSK SNR at P_e = 0.25 (dB)', 'Fig 1(c)', -6.4, 10 * np.log10(norm.isf(0.25) ** 2 / 2), 0.05)
check('floor gain at the heatmap grid peak, P_e=0.264', 'Sec IV-A', 0.1400, (0.264 * lam_star(0.264)) ** 2 - 0.264 ** 2, 2e-4)
check('Fig 1(c) randomized P_d at the floor', 'Rem 4', 0.20, (0.25 * lam_star(0.25)) ** 2, 5e-3)
# Fig. 2 caption: the slope of (14) at the floor is lambda*^n - g'(0); for the logistic
# benchmark g'(0) = e^d, so at P_e = 0.05 (lambda*^2 = 17.3 > 11.6) the maximum of the
# value sits just right of the floor, at alpha/P_e^n = 1.13; at 0.10 and 0.20 at the floor
_gp0 = roc(1e-7) / 1e-7
check("Fig 2 caption: g'(0) = e^d", 'Fig 2 caption', 11.6, _gp0, 0.05)
check('Fig 2 caption: lambda*(0.05)^2', 'Fig 2 caption', 17.3, lam_star(0.05) ** 2, 0.1)
for _pe, _ratio in ((0.05, 1.13), (0.10, 1.00), (0.20, 1.00)):
    _L = lam_star(_pe); _rr = np.linspace(1.0, 1.6, 6001); _al = _pe ** 2 * _rr
    _ta = (np.sqrt(_al) - _pe) / (1 - 2 * _pe)
    _g = _al * _L ** 2 - (_pe + (1 - 2 * _pe) * roc(np.clip(_ta, 0, 1))) ** 2
    check(f'Fig 2: maximum of the value at alpha/P_e^n, P_e={_pe}', 'Fig 2 caption', _ratio, float(_rr[int(np.argmax(_g))]), 5e-3)

# ---------------------------------------------------------------- sweeps
if SLOW:
    # ---- robustness.py: Table S1 rows, Theorem 3 on the first segment, mismatch, SNR mapping
    import robustness as rb
    from scipy.stats import norm
    for d, n_, pe_, floor_, cross_, share_ in ((1.5, 2, 0.10, 0.096, 0.032, 0.01),
                                               (3.0, 3, 0.20, 0.282, 0.057, 0.70)):
        rocf = lambda t, d=d: rb.roc_gauss(t, d)
        P = rb.Plane(rocf, n_, pe_, T=30001)      # the dense grid of Tables II and S1
        cf = (pe_ * rb.lam_star(rocf, pe_)) ** n_ - pe_ ** n_
        check(f'Gauss d={d} n={n_} P_e={pe_}: floor gain', 'Tab S1', floor_, P.floor_gain(), 1e-3)
        check(f'Gauss d={d} n={n_} P_e={pe_}: closed form = measured', 'Tab S1', P.floor_gain(), cf, 1e-4)
        cg, ca, sh = P.crossing()
        check(f'Gauss d={d} n={n_} P_e={pe_}: crossing gain', 'Tab S1', cross_, cg, 2e-3)
        check(f'Gauss d={d} n={n_} P_e={pe_}: fusion-layer share', 'Tab S1', share_, sh, 0.05)
    g3 = lambda t: rb.roc_gauss(t, 3.0)
    # the row the S1 caption quotes for grid sensitivity: d=3, n=4, P_e=0.05
    _Pd = rb.Plane(g3, 4, 0.05, T=30001); _cd = _Pd.crossing(na=800)
    check('Gauss d=3.0 P_e=0.05: lambda* (Table S1 prints 12.28)', 'Tab S1', 12.28, rb.lam_star(g3, 0.05), 5e-3)
    check('Gauss d=3.0 n=4 P_e=0.05: crossing gain (dense grid)', 'Tab S1', 0.033, _cd[0], 1e-3)
    check('Gauss d=3.0 n=4 P_e=0.05: share (dense grid)', 'Tab S1', 0.00, max(_cd[2], 0.0), 0.02)
    _Pc = rb.Plane(g3, 4, 0.05, T=1501); _cc = _Pc.crossing(na=800)
    check('same row at 1501 thresholds: inflated gain (caption)', 'Tab S1 caption', 0.035, _cc[0], 1e-3)
    P3 = rb.Plane(g3, 2, 0.10, T=4001)
    for a_, v_ in ((0.0100, 0.424), (0.0101, 0.255), (0.0105, 0.116), (0.0110, 0.053), (0.0128, 0.0)):
        check(f'first-segment gain, alpha={a_}: closed form', 'Sec III-D', v_, rb.gain_closed_form(g3, 0.10, 2, a_), 2e-3)
        check(f'first-segment gain, alpha={a_}: search', 'Sec III-D', v_, P3.gain(a_), 3e-3)
    check('width factor n=2', 'eq (15)', 1.28, rb.first_segment(g3, 0.10, 2)[4], 1e-2)
    check('width factor n=5', 'Sec III-D', 1.87, rb.first_segment(g3, 0.10, 5)[4], 1e-2)
    check('mu^5, d=3.0, P_e=0.10', 'Sec III-D', 0.124, (0.10 * rb.first_segment(g3, 0.10, 5)[2]) ** 5, 1e-3)
    for ph, pf_r, pd_r in ((0.099, 1.017, 0.442), (0.095, 1.092, 0.474), (0.105, 0.920, 0.399), (0.110, 0.851, 0.369)):
        r, _ = rb.mismatch(g3, 0.10, 2, 0.0100, ph)
        check(f'mismatch P_e_hat={ph}: randomized Pf/alpha', 'Sec III-D', pf_r, r[0], 2e-3)
        check(f'mismatch P_e_hat={ph}: randomized Pd', 'Sec III-D', pd_r, r[1], 2e-3)
    _, d_ = rb.mismatch(g3, 0.10, 2, 0.0100, 0.099)
    check('mismatch P_e_hat=0.099: deterministic Pf/alpha', 'Sec III-D', 1.020, d_[0], 2e-3)
    check('mismatch P_e_hat=0.099: deterministic Pd', 'Sec III-D', 0.242, d_[1], 2e-3)
    check('AWGN BPSK SNR (dB) at P_e=0.10', 'Sec IV-A', -0.9, 10 * np.log10(norm.isf(0.10) ** 2 / 2), 0.05)
    check('Rayleigh BPSK mean SNR (dB) at P_e=0.10', 'Sec IV-A', 2.5, 10 * np.log10(((1 - 2 * 0.10) ** 2) / (1 - (1 - 2 * 0.10) ** 2)), 0.05)

    from fragmentation import (get_rules, deterministic_points,
                               upper_envelope, concave_hull)
    from baselines import local_curve, rule_points, hull_at_alpha_rowwise

    # Fusion-layer gain at P_e=0 (Han--Kim: S3 = S2 exactly at n = 2). The paper quotes
    # the residual on the threshold grid of Table S1 (robustness.tgrid at P_e = 0:
    # 30 000 linear points with log tails), so that one grid serves the whole paper.
    alphas = np.linspace(1e-3, .999, 600)
    for n, val in ((2, 6.9e-5), (3, 8.0e-5)):
        _P = rb.Plane(lambda t: roc(t), n, 0.0, T=30001)
        check(f'S3-S2 at P_e=0, n={n} (Table S1 grid)', 'Sec IV-A, S-IV', val,
              float(max(_P.S3(a) - _P.S2(a) for a in alphas)), 0.5e-5)
    # the same quantity on the older 25 000-point linear grid through the second code
    # path (baselines.py); not quoted in the paper any more, kept as a cross-check
    for n, val in ((2, 1.05e-4), (3, 1.04e-4)):
        pats, rules = get_rules(n)
        pts = deterministic_points(n, D_LOGIT, 0.0, rules, pats, ngrid=25000)
        E = upper_envelope(pts, alphas)
        t = np.linspace(1e-12, 1 - 1e-12, 25000)
        u, v = local_curve(t, D_LOGIT, 0.0)
        X, Y = rule_points(rules, pats, u, v)
        S3 = np.array([np.nanmax(hull_at_alpha_rowwise(X, Y, a)) for a in alphas])
        check(f'S3-S2 at P_e=0, n={n} (25 000-point grid, second code path)', 'cross-check', val,
              float(np.max(S3 - E)), 1.5e-5)

    # Fig. 3(a): the peak of S4 - max(S2, alpha) sits on the floor, which is
    # the check that the heat map agrees with Theorem 3 rather than
    # contradicting it. Measuring against S2 alone put this peak in an
    # interior gap at P_e ~ 0.49, where most of the apparent gain was movement
    # along the chance line.
    alphas_h = np.linspace(2e-3, .998, 200)
    pes_h = np.linspace(0.005, .49, 100)
    best, loc = 0.0, None
    pats, rules = get_rules(2)
    for pe_ in pes_h:
        pts = deterministic_points(2, D_LOGIT, pe_, rules, pats, ngrid=3000)
        E = upper_envelope(pts, alphas_h)
        H, _ = concave_hull(pts, np.linspace(0, 1, 20001))
        Hi = np.interp(alphas_h, np.linspace(0, 1, 20001), H)
        g = Hi - np.maximum(E, alphas_h)
        k = int(g.argmax())
        if g[k] > best:
            best, loc = float(g[k]), (alphas_h[k], pe_)
    check('peak of S4-max(S2,alpha)', 'Fig 3(a)', 0.140, best, 3e-3)
    check('peak sits on the floor', 'Fig 3(a)', loc[0], loc[1] ** 2, 2e-3)

    # Table I: the floor peak against the largest crossing beyond 3 P_e^n.
    # This is what shows the gain is not confined to a neighborhood of the
    # floor -- at n=3, P_e=0.1 the crossing exceeds it.
    for n_, pe_, at_floor, at_cross in ((2, .05, 0.041, 0.022),
                                        (2, .10, 0.084, 0.043),
                                        (2, .20, 0.134, 0.046),
                                        (3, .10, 0.028, 0.032),
                                        (3, .20, 0.065, 0.030)):
        pats, rules = get_rules(n_)
        pts = deterministic_points(n_, D_LOGIT, pe_, rules, pats, ngrid=6000)
        F = pe_ ** n_
        check(f'floor peak, n={n_}, P_e={pe_}', 'Tab I', at_floor,
              (pe_ * lam_star(pe_)) ** n_ - F, 1e-3)
        aa = np.linspace(3 * F, 0.995, 3000)
        E = upper_envelope(pts, aa)
        H, _ = concave_hull(pts, np.linspace(0, 1, 40001))
        Hi = np.interp(aa, np.linspace(0, 1, 40001), H)
        g = Hi - np.maximum(E, aa)
        check(f'crossing peak, n={n_}, P_e={pe_}', 'Tab I', at_cross,
              float(g.max()), 2e-3)

    # What the fusion layer alone recovers, at the floor and at a crossing.
    # deterministic_points / rule_points search a COMMON threshold, so the
    # values here are the common-threshold figures the paper quotes (18% and
    # 2%); the heterogeneous 24% at n=2 is checked separately below.
    for tag, n_, a_, pe_, val in (('floor', 2, 0.067, 0.259, 1.00),
                                  ('crossing', 2, 0.289, 0.10, 0.18),
                                  ('crossing', 3, 0.078, 0.10, 0.02)):
        pats, rules = get_rules(n_)
        pts = deterministic_points(n_, D_LOGIT, pe_, rules, pats, ngrid=6000)
        E = upper_envelope(pts, np.array([a_]))
        H, _ = concave_hull(pts, np.linspace(0, 1, 40001))
        Hi = float(np.interp(a_, np.linspace(0, 1, 40001), H))
        tt = np.linspace(1e-12, 1 - 1e-12, 6000)
        u_, v_ = local_curve(tt, D_LOGIT, pe_)
        X, Y = rule_points(rules, pats, u_, v_)
        S3v = float(np.nanmax(hull_at_alpha_rowwise(X, Y, a_)))
        b_ = max(float(E[0]), a_)
        check(f'S3 share at {tag}, n={n_} (common thresholds)', 'Sec IV-A', val,
              max(S3v - b_, 0) / (Hi - b_), 0.015)

    # The heterogeneous-threshold share at the n=2 crossing: S3 is the hull of
    # the rule points at ONE threshold pair, maximized over all pairs; S4 is the
    # hull over everything. The paper quotes 24% (18% with equal thresholds).
    tt = np.linspace(1e-9, 1 - 1e-9, 400); gg = roc(tt)
    pats2, rules2 = get_rules(2)
    a_, pe_ = 0.289, 0.10
    u1, v1 = phi(tt, pe_), phi(gg, pe_)
    U1, U2 = np.meshgrid(u1, u1, indexing='ij'); V1, V2 = np.meshgrid(v1, v1, indexing='ij')
    uu = np.stack([U1.ravel(), U2.ravel()], 1); vv = np.stack([V1.ravel(), V2.ravel()], 1)
    w0 = np.prod(np.where(pats2 == 1, uu[:, None, :], 1 - uu[:, None, :]), -1)
    w1 = np.prod(np.where(pats2 == 1, vv[:, None, :], 1 - vv[:, None, :]), -1)
    Xh = np.column_stack([np.zeros(len(uu)), np.ones(len(uu))] + [(w0 * f).sum(-1) for f in rules2])
    Yh = np.column_stack([np.zeros(len(uu)), np.ones(len(uu))] + [(w1 * f).sum(-1) for f in rules2])
    S3h = float(np.nanmax(hull_at_alpha_rowwise(Xh, Yh, a_)))
    pts = deterministic_points(2, D_LOGIT, pe_, rules2, pats2, ngrid=6000)
    E = upper_envelope(pts, np.array([a_]))
    H, _ = concave_hull(pts, np.linspace(0, 1, 40001))
    Hi = float(np.interp(a_, np.linspace(0, 1, 40001), H))
    b_ = max(float(E[0]), a_)
    check('S3 share at crossing, n=2 (heterogeneous thresholds)', 'Sec IV-A', 0.24,
          max(S3h - b_, 0) / (Hi - b_), 0.02)

    # The share of the (alpha, P_e) plane on which the gain exceeds 0.01, from
    # the saved map make_baseline_figs.py writes (260 log-spaced alpha in
    # [1e-3, 0.999], 200 P_e in [0, 0.49]); restricted to alpha <= 1/2 and
    # P_e <= 0.45 as the text states. It moves by a few points with the grid,
    # which is why the text says "roughly half".
    try:
        import os
        gm = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gain_maps.npz'))
        m_a = gm['alphas'] <= 0.5; m_p = gm['pes'] <= 0.45
        frac = float((gm['Gtot'][m_p][:, m_a] > 0.01).mean())
        check('share of plane with gain > 0.01, n=2', 'Sec IV-A', 0.50, frac, 0.03)
    except FileNotFoundError:
        pass

    # Corollary 3 is loose: the network that actually attains (alpha, P_d)
    # with randomization, against the bound. Uses k-out-of-n rules with a
    # common threshold and the concave hull, so it is an upper bound on the
    # true requirement -- if anything the bound is looser still.
    from scipy.stats import binom
    from localroc import roc as _roc, phi as _phi
    def _need_n(alpha, pd_target, pe_):
        tt = np.linspace(1e-9, 1 - 1e-9, 3000)
        u_, v_ = _phi(tt, pe_), _phi(_roc(tt), pe_)
        for n_ in range(1, 41):
            px, py = [0., 1.], [0., 1.]
            for k in range(1, n_ + 1):
                px += list(binom.sf(k - 1, n_, u_)); py += list(binom.sf(k - 1, n_, v_))
            px, py = np.array(px), np.array(py); o = np.argsort(px)
            px, py = px[o], py[o]; st = []
            for x, y in zip(px, py):
                while len(st) >= 2 and (st[-1][0]-st[-2][0])*(y-st[-2][1]) \
                        - (st[-1][1]-st[-2][1])*(x-st[-2][0]) >= 0:
                    st.pop()
                st.append((x, y))
            hx = np.array([q[0] for q in st]); hy = np.array([q[1] for q in st])
            if np.interp(alpha, hx, hy) >= pd_target:
                return n_
        return 99
    for a_, pd_, pe_, val in ((.01, .9, .05, 12), (.01, .9, .10, 15),
                              (.01, .9, .20, 29), (.001, .9, .10, 23)):
        check(f'actual n at alpha={a_}, P_e={pe_}', 'Sec III-D', val,
              _need_n(a_, pd_, pe_), 1e-9)

    # Heterogeneous thresholds do not create the crossing gains: at each
    # crossing of Table I the heterogeneous S2 equals the common one.
    tt = np.linspace(1e-9, 1 - 1e-9, 400); gg = _roc(tt)
    pats2, rules2 = get_rules(2)
    def _env_at(px, py, a):
        o = np.argsort(px); return float(np.interp(a, px[o], np.maximum.accumulate(py[o])))
    for a_, pe_ in ((0.24, 0.05), (0.289, 0.10), (0.28, 0.20)):
        u1, v1 = _phi(tt, pe_), _phi(gg, pe_)
        U1, U2 = np.meshgrid(u1, u1, indexing='ij'); V1, V2 = np.meshgrid(v1, v1, indexing='ij')
        uu = np.stack([U1.ravel(), U2.ravel()], 1); vv = np.stack([V1.ravel(), V2.ravel()], 1)
        w0 = np.prod(np.where(pats2 == 1, uu[:, None, :], 1 - uu[:, None, :]), -1)
        w1 = np.prod(np.where(pats2 == 1, vv[:, None, :], 1 - vv[:, None, :]), -1)
        px = [0., 1.]; py = [0., 1.]
        for f in rules2: px += list((w0 * f).sum(-1)); py += list((w1 * f).sum(-1))
        Eh = _env_at(np.array(px), np.array(py), a_)
        w0c = np.prod(np.where(pats2 == 1, u1[:, None, None], 1 - u1[:, None, None]), -1)
        w1c = np.prod(np.where(pats2 == 1, v1[:, None, None], 1 - v1[:, None, None]), -1)
        px2 = [0., 1.]; py2 = [0., 1.]
        for f in rules2: px2 += list((w0c * f).sum(-1)); py2 += list((w1c * f).sum(-1))
        Ec = _env_at(np.array(px2), np.array(py2), a_)
        # both envelopes jitter by ~0.004 between grids, so this tests only
        # that heterogeneity does not open a systematic gap
        check(f'hetero - common S2 at crossing ({a_}, {pe_})', 'Sec IV-A', 0.0,
              Eh - Ec, 5e-3)

    # Table S1, P_e = 0 rows: the ideal-link scheduling gain on the dense grid. The
    # logistic control must reproduce Fig. 1(a); the Gaussian family stays below 6e-4
    # on alpha in [1e-4, 1/2] (a linear grid gave up to 0.0085 there: staircase).
    _P0 = rb.Plane(lambda t: roc(t), 2, 0.0, T=3001)
    _c0 = _P0.crossing(na=800)
    check('ideal-link gain, logistic n=2 (Table S1 control = Fig 1(a))', 'Tab S1', 0.0051, _c0[0], 2e-4)
    check('ideal-link crossing alpha, logistic n=2', 'Tab S1', 0.191, _c0[1], 3e-3)
    _ideal_max = max(rb.make_plane(lambda t, d=d: rb.roc_gauss(t, d), n_, 0.0, T=3001).crossing(na=800)[0]
                     for d in (1.5, 3.0) for n_ in (2, 3, 4, 5))
    check('largest ideal-link gain, Gaussian family (text: at most 6e-4)', 'Sec IV-B, Tab S1', 0.0006, _ideal_max, 1e-4)

    # Section IV-B, 'One configuration, read in decibels' (linkbudget.py)
    import linkbudget as lb
    L_ = lb.main(verbose=False)
    check('dB example: feasibility SNR', 'Sec IV-B', -3.7, L_['feas_db'], 0.05)
    check('dB example: budget (17), loose', 'Sec IV-B', -2.9, L_['budget_db'], 0.05)
    check('dB example: budget (17), tight', 'Sec IV-B', -1.7, L_['tight_db'], 0.05)
    check('dB example: SNR for P_d = 0.9', 'Sec IV-B', 2.4, L_['pd90_db'], 0.1)
    check('dB example: randomized P_d at the floor', 'Sec IV-B', 0.20, L_['floor_rand'], 5e-3)
    for (db_, _, s2_, s4_), (v2, v4) in zip(L_['band'], ((None, 0.20), (0.12, 0.21), (0.22, 0.24), (0.29, 0.29))):
        if v2 is not None:
            check(f'dB example: deterministic P_d at {db_} dB', 'Sec IV-B', v2, s2_, 6e-3)
        check(f'dB example: randomized P_d at {db_} dB', 'Sec IV-B', v4, s4_, 6e-3)
    check('dB example: crossing gain near +1 dB', 'Sec IV-B', 0.032, L_['cross_gain'], 2e-3)
    check('dB example: crossing SNR', 'Sec IV-B', 1.1, L_['cross_db'], 0.1)
    check('dB example: fusion-layer share at that crossing', 'Sec IV-B', 0.0, L_['cross_share'], 0.02)

    # Fig. 4: the gap opens well to the left of the threshold
    from baselines import evaluate
    r = evaluate(2, D_LOGIT, 0.01, np.array([0.0, 0.08, 0.099]), ngrid=8001)
    for i, (pe_, val) in enumerate(((0.0, 0.0004), (0.08, 0.046), (0.099, 0.083))):
        check(f'S4-S2 at alpha=0.01, P_e={pe_}', 'Sec IV-A', val,
              float(r['S4'][i] - r['S2'][i]), 2e-3)

# ---------------------------------------------------------------- report
w = max(len(r[0]) for r in rows)
print(f'{"claim":<{w}}  {"where":<16}  {"paper":>10}  {"computed":>10}')
print('-' * (w + 44))
for claim, where, paper, comp, ok in rows:
    mark = '' if ok else '   <-- MISMATCH'
    print(f'{claim:<{w}}  {where:<16}  {paper:>10.5g}  {comp:>10.5g}{mark}')
print('-' * (w + 44))
print(f'{len(rows)} checks, {failed} mismatched'
      + ('' if SLOW else '   (run with --slow for the sweeps)'))
sys.exit(1 if failed else 0)
