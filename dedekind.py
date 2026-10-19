"""
Independent check of the n=6 enumeration.

Two things are verified, not one:

  1. the count of monotone Boolean functions, which must equal the Dedekind
     number M(n) -- a published sequence, so a mismatch means the enumeration is
     wrong rather than merely surprising;
  2. the number of distinct weight profiles, since Phi_f depends on f only
     through how many accepted patterns it has at each Hamming weight.

The recursion walks the cube by decreasing popcount and admits a point only when
all of its immediate supersets are already in the up-set, which generates each
monotone function exactly once.

Run it on n = 2..5 first: those counts are known, so they test the script.
"""
import sys
from itertools import combinations

DEDEKIND = {2: 6, 3: 20, 4: 168, 5: 7581, 6: 7828354}


def enumerate_monotone(n):
    N = 1 << n
    pc = [bin(x).count('1') for x in range(N)]
    levels = [[x for x in range(N) if pc[x] == w] for w in range(n + 1)]
    sup = [[x | (1 << b) for b in range(n) if not (x >> b) & 1]
           for x in range(N)]

    profiles = set()
    total = 0

    def rec(w, upset):
        nonlocal total
        if w < 0:
            total += 1
            if 0 < len(upset) < N:                    # drop the two constants
                profiles.add(tuple(sum(1 for x in levels[j] if x in upset)
                                   for j in range(n + 1)))
            return
        free = [x for x in levels[w] if all(y in upset for y in sup[x])]
        for r in range(len(free) + 1):
            for sub in combinations(free, r):
                rec(w - 1, upset | set(sub))

    rec(n, frozenset())
    return total, profiles


def Phi(prof, n, u):
    return sum(prof[j] * u ** j * (1 - u) ** (n - j) for j in range(n + 1))


def gaps(profiles, n, eps):
    iv = sorted((Phi(p, n, eps), Phi(p, n, 1 - eps)) for p in profiles)
    out, hi = [], iv[0][1]
    for a, b in iv[1:]:
        if a > hi + 1e-13:
            out.append((hi, a))
        hi = max(hi, b)
    return out


if __name__ == '__main__':
    from scipy.optimize import brentq
    for n in [int(x) for x in sys.argv[1:]] or [2, 3, 4, 5]:
        total, profs = enumerate_monotone(n)
        ded = DEDEKIND.get(n)
        ok = 'MATCH' if total == ded else f'MISMATCH (expected {ded})'
        print(f'n={n}:  monotone {total:>9}  [{ok}]   profiles {len(profs):>4}')
        root = brentq(lambda x: (1 - x) ** n - x ** (n - 1), 1e-12, 0.4999999)
        below = [e for e in (0.05, 0.15, 0.25, 0.35, 0.42, 0.45, 0.46)
                 if e < root] + [root - 1e-9]
        bad = [e for e in below if gaps(profs, n, e)]
        above = len(gaps(profs, n, root + 1e-6))
        print(f'        onset {root:.6f};  gaps below: '
              f'{"none at any tested eps" if not bad else bad};  '
              f'gaps just above: {above}')
