"""
The measure D_n(P_e) of constraints not attainable with equality (Theorem 2,
Corollary 2).

Reachable set for the k-out-of-n family under crossover eps:
    I_k(eps) = [B_k(eps), B_k(1-eps)],   B_k(u) = P(Bin(n,u) >= k)
    D_n(eps) = 1 - |union_k I_k(eps)|

Corollary 2: take k = ceil(n/2).  Since 1-eps > 1/2 > eps,
Hoeffding gives
    B_k(1-eps) >= 1 - exp(-2 n delta^2)      and     B_k(eps) <= exp(-2 n delta^2)
with delta = 1/2 - eps.  This single interval already covers all but
2 exp(-2 n delta^2) of [0,1], so

    D_n(eps)  <=  2 exp(-2 n (1/2 - eps)^2).

The measure of unattainable constraints therefore decays exponentially in n.
"""
import numpy as np
from scipy.stats import binom


def B(k, u, n):
    return float(binom.sf(k - 1, n, u))


def _monotone(n, _c={}):
    """All non-constant monotone Boolean rules.  NOTE: restricting to the
    k-out-of-n family understates the reachable set -- the dictator rule
    f(U)=u_1 alone contributes [eps, 1-eps] and fills the AND/OR gap."""
    import itertools
    if n in _c:
        return _c[n]
    pats = np.array(list(itertools.product([0, 1], repeat=n)))
    N = 2 ** n
    dom = [(i, j) for i in range(N) for j in range(N)
           if np.all(pats[i] <= pats[j]) and i != j]
    out = []
    for m in range(2 ** N):
        f = np.array([(m >> b) & 1 for b in range(N)], dtype=bool)
        if f.all() or not f.any():
            continue
        if all(not (f[i] and not f[j]) for i, j in dom):
            out.append(f)
    _c[n] = (pats, out)
    return _c[n]


def Phi(f, pats, u):
    u = np.full(pats.shape[1], u, float)
    return float((np.prod(np.where(pats == 1, u, 1 - u), axis=1) * f).sum())


def D_exact(n, eps, family='monotone'):
    if family == 'monotone' and n <= 4:
        pats, rules = _monotone(n)
        iv = sorted((Phi(f, pats, eps), Phi(f, pats, 1 - eps)) for f in rules)
    else:
        iv = sorted((B(k, eps, n), B(k, 1 - eps, n)) for k in range(1, n + 1))
    tot, lo, hi = 0.0, None, None
    for a, b in iv:
        if lo is None:
            lo, hi = a, b
        elif a <= hi:
            hi = max(hi, b)
        else:
            tot += hi - lo; lo, hi = a, b
    if lo is not None:
        tot += hi - lo
    return 1.0 - tot


def bound(n, eps):
    return 2 * np.exp(-2 * n * (0.5 - eps) ** 2)




# --- exact enumeration by weight profile ------------------------------------
# Phi_f depends only on how many patterns of each weight f accepts, so the
# monotone class collapses to a small set of profiles: 3, 8, 24, 94 for
# n = 2..5, against 4, 18, 166, 7579 non-constant functions.
def _profiles(n, _c={}):
    import itertools
    if n in _c:
        return _c[n]
    lv = [[p for p in itertools.product([0, 1], repeat=n) if sum(p) == j]
          for j in range(n + 1)]
    out = set()

    def rec(j, upset):
        if j > n:
            out.add(tuple(len(upset & set(lv[i])) for i in range(n + 1)))
            return
        cur = lv[j]
        forced = {x for x in cur
                  if any(all(y[k] <= x[k] for k in range(n)) for y in upset)}
        free = [x for x in cur if x not in forced]
        for r in range(len(free) + 1):
            for sub in itertools.combinations(free, r):
                rec(j + 1, upset | forced | set(sub))
    rec(0, set())
    _c[n] = [p for p in sorted(out) if 0 < sum(p) < 2 ** n]
    return _c[n]


def _PhiP(prof, n, u):
    return sum(prof[j] * u ** j * (1 - u) ** (n - j) for j in range(n + 1))


def D_profile(n, eps):
    """D_n over the full monotone class, via weight profiles.  n <= 5."""
    iv = sorted((_PhiP(p, n, eps), _PhiP(p, n, 1 - eps)) for p in _profiles(n))
    tot, lo, hi = 0.0, None, None
    for a, b in iv:
        if lo is None:
            lo, hi = a, b
        elif a <= hi:
            hi = max(hi, b)
        else:
            tot += hi - lo; lo, hi = a, b
    if lo is not None:
        tot += hi - lo
    return 1.0 - tot


def onset_profile(n):
    """Smallest eps at which the union acquires an interior gap.  n <= 5."""
    pr = _profiles(n)

    def gap(e):
        iv = sorted((_PhiP(p, n, e), _PhiP(p, n, 1 - e)) for p in pr)
        hi = iv[0][1]
        for a, b in iv[1:]:
            if a > hi + 1e-14:
                return True
            hi = max(hi, b)
        return False
    lo, hi = 1e-9, 0.4999999
    if not gap(hi):
        return float('nan')
    for _ in range(70):
        m = (lo + hi) / 2
        if gap(m):
            hi = m
        else:
            lo = m
    return hi


if __name__ == '__main__':
    print('D_n(Pe): measure of false-alarm constraints that force randomization')
    print('  (exact, k-out-of-n family)   |   Hoeffding bound 2exp(-2n(1/2-Pe)^2)\n')
    hdr = 'Pe    ' + ''.join(f'{"n="+str(n):>21}' for n in (2, 3, 5, 10, 20, 50))
    print(hdr)
    for eps in (0.05, 0.10, 0.20, 0.30, 0.40, 0.45):
        row = f'{eps:<6.2f}'
        for n in (2, 3, 5, 10, 20, 50):
            d, bd = D_exact(n, eps), bound(n, eps)
            row += f'{d:>10.2e}/{min(bd,1.0):<10.2e}'
        print(row)

    print('\nIs the bound valid everywhere tested?')
    bad = [(n, e) for n in range(2, 61) for e in np.linspace(0.001, 0.499, 60)
           if D_exact(n, e) > min(bound(n, e), 1.0) + 1e-12]
    print(f'  violations: {len(bad)}' + (f'   e.g. {bad[:3]}' if bad else '   (none)'))

    print('\nHalf-life: smallest n with D_n(Pe) < 0.01')
    for eps in (0.10, 0.20, 0.30, 0.40, 0.45):
        n = next((n for n in range(2, 400) if D_exact(n, eps) < 0.01), None)
        print(f'  Pe={eps:.2f}: n >= {n}   (floor Pe^n at that n = {eps**n:.2e})')
