from localroc import roc as _roc, D_LOGIT
"""
Brute-force check of Lemma 2 (slope cap): the channel caps the slope of the
global ROC at the origin, however good the sensors are.

Lemma 2.  Write phi(x) = (1-2e)x + e.  For any monotone fusion rule f and any
threshold vector, the global operating point (Pf, Pd) satisfies

        Pd / Pf  <=  ( max_t  phi(g(t)) / phi(t) )^n  =:  S_n(e)

and moreover  S_n(e) <= ((1-e)/e)^n,  with equality in the limit of perfect local
detectors.  Since the concave hull passes through the origin and is concave,
H(alpha) <= alpha * S_n(e) for EVERY alpha, so

        Pd  <=  alpha * ((1-Pe)/Pe)^n                                    (*)

is a bound that depends only on the channel and the target operating point --
not on sensor quality at all.  Rearranged it lower-bounds the number of sensors
(Corollary 3):

        n  >=  ln(Pd/alpha) / ln((1-Pe)/Pe).

Proof sketch for the first inequality: for any Uhat with f(Uhat)=1 the pointwise
likelihood ratio is prod_k (v_k/u_k)^{uhat_k} ((1-v_k)/(1-u_k))^{1-uhat_k}, which
is maximized at Uhat = all-ones, giving prod_k v_k/u_k.  A ratio of sums is at
most the largest pointwise ratio, and the AND rule attains it because it selects
only that atom.
"""
import itertools
import numpy as np

phi = lambda x, e: (1 - 2 * e) * x + e


def lam_star(e, d, N=200001):
    """max_t phi(g(t))/phi(t)."""
    t = np.linspace(1e-12, 1 - 1e-12, N)
    g = _roc(t, d)
    return float(np.max(phi(g, e) / phi(t, e)))


def empirical_max_slope(n, e, d, ngrid=400, hetero=False):
    """Largest Pd/Pf over all monotone rules and (optionally heterogeneous)
    thresholds, found by direct enumeration."""
    pats = np.array(list(itertools.product([0, 1], repeat=n)))
    N = 2 ** n
    dom = [(i, j) for i in range(N) for j in range(N)
           if np.all(pats[i] <= pats[j]) and i != j]
    rules = []
    for m in range(2 ** N):
        f = np.array([(m >> b) & 1 for b in range(N)], dtype=bool)
        if f.all() or not f.any():
            continue
        if all(not (f[i] and not f[j]) for i, j in dom):
            rules.append(f)
    t = np.linspace(1e-9, 1 - 1e-9, ngrid)
    g = _roc(t, d)
    u, v = phi(t, e), phi(g, e)
    best = 0.0
    combos = (itertools.product(range(ngrid), repeat=n) if hetero
              else ((i,) * n for i in range(ngrid)))
    for idx in combos:
        uu = np.array([u[i] for i in idx]); vv = np.array([v[i] for i in idx])
        w0 = np.prod(np.where(pats == 1, uu, 1 - uu), axis=1)
        w1 = np.prod(np.where(pats == 1, vv, 1 - vv), axis=1)
        for f in rules:
            a, b = w0[f].sum(), w1[f].sum()
            if a > 1e-14:
                best = max(best, b / a)
    return best


if __name__ == '__main__':
    print('Does S_n(e) = lam*(e)^n, and is it capped by ((1-e)/e)^n?\n')
    print(f"{'n':>3} {'Pe':>6} {'d':>5} {'lam*^n':>12} {'enumerated max':>16} "
          f"{'((1-e)/e)^n':>13}")
    for n in (2, 3):
        for e in (.05, .10, .20, .35):
            for d in (1.0, 1.4673):
                ls = lam_star(e, d) ** n
                em = empirical_max_slope(n, e, d, ngrid=300)
                cap = ((1 - e) / e) ** n
                mark = 'ok' if em <= ls * (1 + 1e-6) else 'VIOLATED'
                print(f'{n:>3} {e:>6.2f} {d:>5.2f} {ls:>12.4f} '
                      f'{em:>16.4f} {cap:>13.4f}   {mark}')

    print('\nHeterogeneous thresholds (n=2) -- does allowing them beat lam*^2?')
    for e in (.10, .25):
        ls = lam_star(e, 1.4673) ** 2
        em = empirical_max_slope(2, e, 1.4673, ngrid=140, hetero=True)
        print(f'  Pe={e:.2f}: lam*^2={ls:.4f}  heterogeneous max={em:.4f}  '
              f'{"ok" if em <= ls*(1+1e-6) else "VIOLATED"}')

    print('\nIs the cap ((1-e)/e)^n approached as the sensors improve?')
    for e in (.10, .25):
        row = '  Pe=%.2f: ' % e
        for d in (1, 2, 4, 8, 16):
            row += f'd={d}: {lam_star(e, d):.3f}   '
        print(row + f'cap={((1-e)/e):.3f}')
