"""
The deterministic reachable set (Theorem 1)
===========================================
As the crossover probability Pe grows, the set of global false-alarm rates
reachable by *deterministic* monotone fusion shrinks (Theorem 1(iii)). This
module enumerates the fusion rules and computes the deterministic operating
points, their upper envelope and their concave hull.

Model
-----
n conditionally independent identical LDs.
Local ROC: the logistic location family of localroc.py (shift D_LOGIT)
Symmetric DMC, crossover eps:                phi(x) = (1-2eps)x + eps
                                             = 1/2 + (1-2eps)(x - 1/2)
   -> the degraded local ROC is the pristine local ROC contracted toward
      (1/2, 1/2) by the factor beta = 1-2eps.  Local reachable Pf = [eps, 1-eps].
Fusion: every non-constant monotone Boolean rule on {0,1}^n (enumerated exactly).
"""
import itertools
import numpy as np
from localroc import roc as _roc, D_LOGIT

# --------------------------------------------------------------------------
# monotone Boolean rules
# --------------------------------------------------------------------------
def monotone_rules(n):
    """All non-constant monotone Boolean functions on n inputs (exact Dedekind
    enumeration).  Only tractable for n <= 4."""
    if n > 4:
        raise ValueError("exhaustive enumeration only for n<=4; use kofn_rules")
    pats = np.array(list(itertools.product([0, 1], repeat=n)), dtype=int)
    N = 2 ** n
    dom = [(i, j) for i in range(N) for j in range(N)
           if np.all(pats[i] <= pats[j]) and i != j]
    out = []
    for m in range(2 ** N):
        f = np.array([(m >> b) & 1 for b in range(N)], dtype=bool)
        if f.all() or not f.any():
            continue                                   # drop the two constants
        if all(not (f[i] and not f[j]) for i, j in dom):
            out.append(f)
    return pats, out


def kofn_rules(n):
    """The k-out-of-n family, k=1..n.  This is the LRT-optimal fusion family for
    conditionally independent identical local detectors."""
    pats = np.array(list(itertools.product([0, 1], repeat=n)), dtype=int)
    s = pats.sum(1)
    return pats, [s >= k for k in range(1, n + 1)]


def get_rules(n, family='monotone'):
    """Default to the full monotone class.  Restricting to k-out-of-n silently
    drops the dictator rules, whose interval [eps, 1-eps] is the widest of all
    and fills the AND/OR gap -- the same omission that once gave the wrong
    onset in Theorem 2."""
    if family == 'monotone' and n <= 4:
        return monotone_rules(n)
    return kofn_rules(n)


def phi_of_u(f, pats, u):
    """Global rate when every degraded local rate equals u (vectorized in u)."""
    u = np.asarray(u, float)[..., None]
    w = np.prod(np.where(pats[None, :, :] == 1, u[..., None], 1 - u[..., None]),
                axis=-1)                                   # (...,2^n)
    return (w * f[None, :]).sum(-1)


def phi_of_vec(f, pats, U):
    """Global rate for a vector U of per-sensor degraded rates (U shape (...,n))."""
    U = np.asarray(U, float)
    w = np.prod(np.where(pats == 1, U[..., None, :], 1 - U[..., None, :]), axis=-1)
    return (w * f).sum(-1)


# --------------------------------------------------------------------------
# achievable sets
# --------------------------------------------------------------------------
def deterministic_points(n, d, eps, rules, pats, ngrid=1201):
    """(Pf,Pd) reachable by a deterministic monotone rule with identical local
    thresholds, under crossover eps."""
    t = np.linspace(1e-9, 1 - 1e-9, ngrid)
    beta = 1 - 2 * eps
    u = beta * t + eps                                   # degraded local Pf
    v = beta * _roc(t, d) + eps                      # degraded local Pd
    P = [np.array([[0.0, 0.0], [1.0, 1.0]])]
    for f in rules:
        P.append(np.column_stack([phi_of_u(f, pats, u), phi_of_u(f, pats, v)]))
    return np.vstack(P)


def reachable_intervals(n, eps, rules, pats):
    """Exact closed intervals of deterministic global Pf, one per rule
    (Theorem: image of the box [eps,1-eps]^n is [Phi_f(eps), Phi_f(1-eps)])."""
    lo = lambda f: np.ravel(phi_of_u(f, pats, eps))[0]
    hi = lambda f: np.ravel(phi_of_u(f, pats, 1 - eps))[0]
    return [(float(lo(f)), float(hi(f))) for f in rules]


def union_measure(intervals):
    """Lebesgue measure of a union of closed intervals."""
    iv = sorted(intervals)
    tot, cur_lo, cur_hi = 0.0, None, None
    for lo, hi in iv:
        if cur_lo is None:
            cur_lo, cur_hi = lo, hi
        elif lo <= cur_hi:
            cur_hi = max(cur_hi, hi)
        else:
            tot += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
    if cur_lo is not None:
        tot += cur_hi - cur_lo
    return tot


# --------------------------------------------------------------------------
# upper envelope (deterministic) and concave hull (randomized)
# --------------------------------------------------------------------------
def upper_envelope(points, alphas):
    """E(a) = max{Pd : (Pf,Pd) deterministic, Pf <= a}  (step, right-continuous)."""
    o = np.argsort(points[:, 0])
    pf, pd = points[o, 0], np.maximum.accumulate(points[o, 1])
    idx = np.searchsorted(pf, alphas, side='right') - 1
    out = np.where(idx >= 0, pd[np.clip(idx, 0, None)], 0.0)
    return out


def concave_hull(points, alphas):
    """H(a) = upper concave envelope of the deterministic set (= randomized
    optimum, since randomization realizes every convex combination)."""
    o = np.argsort(points[:, 0])
    P = points[o]
    stack = []                                     # monotone upper-hull scan
    for x, y in P:
        while len(stack) >= 1 and y >= stack[-1][1] and x >= stack[-1][0]:
            if x == stack[-1][0]:
                stack.pop()
            else:
                break
        while len(stack) >= 2:
            (x1, y1), (x2, y2) = stack[-2], stack[-1]
            if (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) >= 0:   # x2 below chord
                stack.pop()
            else:
                break
        if stack and x == stack[-1][0]:
            if y > stack[-1][1]:
                stack[-1] = (x, y)
            continue
        stack.append((x, y))
    hx = np.array([s[0] for s in stack]); hy = np.array([s[1] for s in stack])
    return np.interp(alphas, hx, hy), (hx, hy)


# --------------------------------------------------------------------------
# summary metrics vs Pe
# --------------------------------------------------------------------------
def sweep(n, d, eps_grid, nalpha=4001, ngrid=1201, family='kofn'):
    pats, rules = get_rules(n, family)
    alphas = np.linspace(0.0, 1.0, nalpha)
    rec = {k: [] for k in ('D', 'Gbar', 'Zsub', 'AUCe', 'AUCh')}
    detail = {}
    for eps in eps_grid:
        pts = deterministic_points(n, d, eps, rules, pats, ngrid)
        E = upper_envelope(pts, alphas)
        H, hull = concave_hull(pts, alphas)
        G = np.maximum(H - E, 0.0)
        # exact fragmentation measure from the closed-form intervals
        D = 1.0 - union_measure(reachable_intervals(n, eps, rules, pats))
        rec['D'].append(D)
        rec['Gbar'].append(np.trapezoid(G, alphas))
        rec['Zsub'].append(np.trapezoid((E < alphas).astype(float), alphas))
        rec['AUCe'].append(np.trapezoid(E, alphas))
        rec['AUCh'].append(np.trapezoid(H, alphas))
        detail[round(float(eps), 6)] = (pts, hull, alphas, E, H, G)
    return {k: np.array(v) for k, v in rec.items()}, detail, alphas


if __name__ == '__main__':
    d = D_LOGIT                      # SNR matched to the manuscript's 2-LD example
    eg = np.linspace(0.0, 0.49, 50)
    for fam in ('kofn', 'monotone'):
        print(f"=== fusion family: {fam} ===")
        print(f"{'n':>3} {'Pe':>6} {'D(Pe)':>8} {'Gbar':>8} {'Zsub':>8} {'AUC_env':>8} {'AUC_hull':>9}")
        ns = (2, 3, 5) if fam == 'kofn' else (2, 3)
        for n in ns:
            rec, _, _ = sweep(n, d, eg, family=fam)
            for i in (0, 10, 20, 30, 40, 49):
                print(f"{n:>3} {eg[i]:>6.3f} {rec['D'][i]:>8.4f} {rec['Gbar'][i]:>8.4f} "
                      f"{rec['Zsub'][i]:>8.4f} {rec['AUCe'][i]:>8.4f} {rec['AUCh'][i]:>9.4f}")
            print(f"    -> D non-decreasing: {np.all(np.diff(rec['D'])>=-1e-12)};"
                  f"  Gbar non-decreasing: {np.all(np.diff(rec['Gbar'])>=-1e-9)}")
            bad = np.where(np.diff(rec['Gbar']) < -1e-9)[0]
            if len(bad): print(f"       Gbar dips at Pe = {eg[bad][:5]}")
            print()
