"""
Baseline comparison for robust decision fusion under a symmetric DMC.

Five schemes, in increasing order of what the DFC is allowed to do.  All are
evaluated at the same Neyman-Pearson constraint alpha and the same crossover
probability Pe, on the same local ROC.

  S0  AND-only              fusion rule frozen to AND; local threshold retuned
                            for the channel.  The trivial "always safe" scheme.
  S1  Channel-blind         local thresholds and fusion rule designed for
                            Pe = 0 and applied unchanged.  VIOLATES alpha.
  S2  Channel-aware det.    best monotone fusion rule with local thresholds
                            re-optimized for the channel.  Strongest purely
                            deterministic competitor (Chair-Varshney fusion on
                            channel-corrected probabilities, cf. Chen 2004).
  S3  Fusion-layer rand.    local operating point held FIXED at its best value;
                            the DFC randomizes over fusion rules only.  This is
                            the classical randomized NP test at the FC.
  S4  Dependent Rand.       LDs and DFC switch jointly -> concave hull of the
                            union over all local operating points.  PROPOSED.

S2 <= S3 <= S4 by construction.  The gap S4 - S3 isolates the value of
*dependent* randomization over fusion-only randomization; the gap S4 - S2 is
the total value of randomization.
"""
import numpy as np
from localroc import roc as _roc, D_LOGIT
from fragmentation import (get_rules as kofn_rules, phi_of_u, deterministic_points,
                           upper_envelope, concave_hull)

def local_curve(t, d, pe):
    b = 1 - 2 * pe
    return b * t + pe, b * _roc(t, d) + pe


def rule_points(rules, pats, u, v):
    """(T,m) arrays of global (Pf,Pd) for every rule, incl. trivial rules."""
    X = [np.zeros_like(u), np.ones_like(u)]
    Y = [np.zeros_like(v), np.ones_like(v)]
    for f in rules:
        X.append(phi_of_u(f, pats, u)); Y.append(phi_of_u(f, pats, v))
    return np.column_stack(X), np.column_stack(Y)


def hull_at_alpha_rowwise(X, Y, alpha):
    """For each row (= one local operating point), the upper concave envelope
    evaluated at alpha.  Vectorized over rows."""
    T, m = X.shape
    xi, xj = X[:, :, None], X[:, None, :]
    yi, yj = Y[:, :, None], Y[:, None, :]
    span = (xi <= alpha + 1e-15) & (xj >= alpha - 1e-15) & (xj > xi + 1e-15)
    with np.errstate(invalid='ignore', divide='ignore'):
        val = yi + (yj - yi) * (alpha - xi) / (xj - xi)
    val = np.where(span, val, -np.inf)
    exact = np.where(np.abs(X - alpha) < 1e-12, Y, -np.inf).max(1)
    return np.maximum(val.reshape(T, -1).max(1), exact)


def evaluate(n, d, alpha, pes, ngrid=4001):
    pats, rules = kofn_rules(n)
    t = np.linspace(1e-12, 1 - 1e-12, ngrid)
    al = np.linspace(0, 1, 40001)

    # ---- S1 reference design, fixed once at Pe = 0 -----------------------
    u0, v0 = local_curve(t, d, 0.0)
    X0, Y0 = rule_points(rules, pats, u0, v0)
    ok = np.abs(X0 - alpha) < 2e-3
    if not ok.any():
        raise RuntimeError('no nominal deterministic design at alpha')
    best = np.unravel_index(np.where(ok, Y0, -np.inf).argmax(), Y0.shape)
    t_ref, rule_ref = t[best[0]], best[1]          # column 0,1 = trivial rules

    out = {k: [] for k in ('S0', 'S1f', 'S1d', 'S2', 'S3', 'S4')}
    for pe in pes:
        u, v = local_curve(t, d, pe)
        X, Y = rule_points(rules, pats, u, v)

        # S1 : the nominal design applied unchanged
        i = np.argmin(np.abs(t - t_ref))
        out['S1f'].append(X[i, rule_ref]); out['S1d'].append(Y[i, rule_ref])

        # S0 : AND rule, threshold retuned so that (phi(t))^n = alpha
        ustar = alpha ** (1.0 / n)
        if pe <= ustar <= 1 - pe:
            tt = (ustar - pe) / (1 - 2 * pe)
            _, vv = local_curve(np.array([tt]), d, pe)
            out['S0'].append(float(vv[0] ** n))
        else:                                   # infeasible -> mix with rule 0
            out['S0'].append(alpha)             # collapses onto the diagonal

        # S2 / S4 : deterministic envelope and global concave hull
        pts = deterministic_points(n, d, pe, rules, pats, ngrid=ngrid)
        out['S2'].append(float(upper_envelope(pts, np.array([alpha]))[0]))
        H, _ = concave_hull(pts, al)
        out['S4'].append(float(np.interp(alpha, al, H)))

        # S3 : best single local point + randomization over fusion rules
        out['S3'].append(float(np.nanmax(hull_at_alpha_rowwise(X, Y, alpha))))

    return {k: np.array(v) for k, v in out.items()}


if __name__ == '__main__':
    cfgs = [(2, 1.4673, 0.2009), (3, 1.4673, 0.1300)]
    for n, d, alpha in cfgs:
        print(f'\n=== n={n}, alpha={alpha} ===')
        pes = np.array([0.0, .05, .10, .15, .20, .25, .30, .35, .40])
        r = evaluate(n, d, alpha, pes)
        print(f"{'Pe':>5} {'S1 Pf':>7} {'S1 Pd':>7} {'S0 Pd':>7} {'S2 Pd':>7} "
              f"{'S3 Pd':>7} {'S4 Pd':>7} {'S4-S2':>7} {'S4-S3':>7}")
        for i, pe in enumerate(pes):
            viol = '*' if r['S1f'][i] > alpha + 1e-9 else ' '
            print(f"{pe:>5.2f} {r['S1f'][i]:>6.4f}{viol} {r['S1d'][i]:>7.4f} "
                  f"{r['S0'][i]:>7.4f} {r['S2'][i]:>7.4f} {r['S3'][i]:>7.4f} "
                  f"{r['S4'][i]:>7.4f} {r['S4'][i]-r['S2'][i]:>7.4f} "
                  f"{r['S4'][i]-r['S3'][i]:>7.4f}")
        print('  * = false-alarm constraint violated')
