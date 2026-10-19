"""
Robustness of the Section IV findings to the local observation model, the
closed form of Theorem 3 on the whole first hull segment, the P_e-mismatch
numbers of Section III-D, and the end-to-end Monte Carlo of Section IV-B
(Appendix S-IV of the supplement).

    python3 robustness.py              # Table S1 (about six minutes)
    python3 robustness.py --mc         # ... plus the Monte Carlo (three runs of 10^6 frames)
    python3 robustness.py --segment    # Theorem 3 numbers and the mismatch table only

Local ROCs: the logistic family of localroc.py (the benchmark) and a Gaussian
location family P_d = Q(Q^{-1}(P_f) - d) at d = 1.5 and 3.0. Common thresholds,
all non-constant monotone rules: the full class by cells for n <= 4 (Plane) and
by weight profile for n = 5 (PlaneProfiles, 94 profiles; the two agree exactly at n = 4).
"""
import sys
import numpy as np
from scipy.stats import norm
from localroc import roc as roc_logit, phi, D_LOGIT
from fragmentation import get_rules

RNG = np.random.default_rng(20260925)


# ---------------------------------------------------------------- local ROCs
def roc_gauss(t, d):
    t = np.clip(t, 1e-300, 1 - 1e-16)
    return norm.sf(norm.isf(t) - d)


def lam_star(rocf, pe, T=200001):
    t = np.linspace(1e-9, 1, T)
    return float(np.max(phi(rocf(t), pe) / phi(t, pe)))


# ---------------------------------------------------------------- geometry
def _hull(px, py):
    P = np.column_stack([np.append(px, [0, 1]), np.append(py, [0, 1])])
    P = P[np.lexsort((-P[:, 1], P[:, 0]))]
    st = []
    for x, y in P:
        while len(st) >= 2 and (st[-1][0] - st[-2][0]) * (y - st[-2][1]) \
                - (st[-1][1] - st[-2][1]) * (x - st[-2][0]) >= 0:
            st.pop()
        st.append((x, y))
    return np.array(st)


def _envelope(px, py):
    o = np.argsort(px)
    return px[o], np.maximum.accumulate(py[o])


def cells(U, pats):
    """(T, n) local rates -> (T, 2^n) cell probabilities, one cell at a time."""
    W = np.ones((U.shape[0], len(pats)))
    for c, pat in enumerate(pats):
        for k, b in enumerate(pat):
            W[:, c] *= U[:, k] if b == 1 else 1 - U[:, k]
    return W


def tgrid(pe, T):
    """Threshold grid. For P_e > 0 the degraded local rates live in [P_e, 1-P_e]
    and a linear grid resolves them (relative step 6.7e-4/P_e at T = 1501); at
    P_e = 0 the rates reach 0, the deterministic envelope at small alpha is made
    of points whose thresholds are 1e-4 apart in ratio, and a linear grid turns
    the envelope's staircase into a spurious 'gain' (the artifact Appendix S-IV
    describes). So at P_e = 0 the grid is logarithmic toward both ends."""
    if pe > 0:
        return np.linspace(1e-9, 1 - 1e-9, T)
    # P_e = 0: a dense linear grid (the staircase ripple of the deterministic
    # envelope scales with the step) with logarithmic tails at both ends
    h = np.geomspace(1e-9, 1e-2, 2000)
    return np.unique(np.concatenate([np.linspace(1e-9, 1 - 1e-9, max(T, 30001)), h, 1 - h]))


class Plane:
    """S2, S3, S4 as functions of alpha for one (roc, n, P_e), common thresholds."""

    def __init__(self, rocf, n, pe, T=3001):
        pats, rules = get_rules(n)
        self.n, self.pe = n, pe
        t = tgrid(pe, T); T = len(t)
        u, v = phi(t, pe), phi(rocf(t), pe)
        W0 = cells(np.repeat(u[:, None], n, 1), pats)
        W1 = cells(np.repeat(v[:, None], n, 1), pats)
        px = np.concatenate([W0 @ f for f in rules])
        py = np.concatenate([W1 @ f for f in rules])
        self.env_x, self.env_y = _envelope(px, py)      # S2
        self.H = _hull(px, py)                          # S4
        # S3: at each t the NP-optimal randomized test on the received vector is
        # the LR test; its ROC is the cumulative path of the cells sorted by LR.
        r = W1 / np.maximum(W0, 1e-300)
        o = np.argsort(-r, axis=1)
        c0 = np.cumsum(np.take_along_axis(W0, o, 1), 1)
        c1 = np.cumsum(np.take_along_axis(W1, o, 1), 1)
        self.c0 = np.concatenate([np.zeros((T, 1)), c0], 1)
        self.c1 = np.concatenate([np.zeros((T, 1)), c1], 1)

    def S2(self, a):
        return float(np.interp(a, self.env_x, self.env_y)) if a >= self.env_x[0] else 0.0

    def S4(self, a):
        return float(np.interp(a, self.H[:, 0], self.H[:, 1]))

    def S3(self, a):
        c0, c1 = self.c0, self.c1
        idx = (c0 <= a).sum(1) - 1
        idx = np.clip(idx, 0, c0.shape[1] - 2)
        x0 = np.take_along_axis(c0, idx[:, None], 1)[:, 0]
        x1 = np.take_along_axis(c0, idx[:, None] + 1, 1)[:, 0]
        y0 = np.take_along_axis(c1, idx[:, None], 1)[:, 0]
        y1 = np.take_along_axis(c1, idx[:, None] + 1, 1)[:, 0]
        w = np.where(x1 > x0, (a - x0) / np.maximum(x1 - x0, 1e-300), 0.0)
        return float(np.max(y0 + np.clip(w, 0, 1) * (y1 - y0)))

    def gain(self, a):
        return self.S4(a) - max(self.S2(a), a)

    def floor_gain(self):
        return self.gain(self.pe ** self.n)

    def crossing(self, na=1500, lo=None, floor_edge=1e-4):
        # alpha in [max(3 P_e^n, floor_edge), 1/2], log-spaced so that the small-alpha
        # region, where the crossings of good sensors sit, is resolved; a linear
        # grid on [max(3 P_e^n, 1e-3), 1/2] pins the d=3.0, n=4, P_e=0.05
        # crossing to the edge at alpha = 1e-3.
        lo = 3 * self.pe ** self.n if lo is None else lo
        aa = np.geomspace(max(lo, floor_edge), 0.5, na)
        g = np.array([self.gain(a) for a in aa])
        k = int(g.argmax())
        # refine on a linear grid within +-8% of the coarse maximum. The
        # deterministic envelope is a staircase in alpha whose ripple (~1e-4 in
        # gain at T = 3001) would otherwise decide where a flat ridge top is
        # reported, so the maximum is taken of an 11-point running mean (about
        # 1e-3 in alpha at alpha = 0.3, negligible at the sharp crossings near
        # 1e-3) and the unsmoothed gain and share at that alpha are returned.
        from scipy.ndimage import uniform_filter1d
        a0 = float(aa[k])
        aa2 = np.linspace(max(a0 * 0.92, floor_edge), min(a0 * 1.08, 0.5), 400)
        g2 = np.array([self.gain(a) for a in aa2])
        k2 = int(np.argmax(uniform_filter1d(g2, 11, mode='nearest')))
        a = float(aa2[k2]); b = max(self.S2(a), a)
        share = (self.S3(a) - b) / (self.S4(a) - b) if self.S4(a) - b > 1e-9 else float('nan')
        return float(g2[k2]), a, share


class PlaneProfiles(Plane):
    """Plane for n >= 5 with a common threshold: every monotone rule's global
    point depends only on its weight profile (c_0, ..., c_n), the number of
    accepted patterns of each weight, so the 7579 rules at n = 5 collapse to the
    94 profiles of scaling._profiles. S3 at a fixed threshold is the hull of the
    counting-rule points, because at a common threshold all cells of one weight
    share a likelihood ratio, so the NP-optimal randomized test mixes adjacent
    counting rules."""

    def __init__(self, rocf, n, pe, T=3001):
        from scaling import _profiles
        from scipy.special import comb
        profs = np.array(_profiles(n), float)                     # (P, n+1)
        self.n, self.pe = n, pe
        t = tgrid(pe, T); T = len(t)
        u, v = phi(t, pe), phi(rocf(t), pe)
        j = np.arange(n + 1)
        Bu = u[:, None] ** j * (1 - u[:, None]) ** (n - j)          # (T, n+1) weight-class masses
        Bv = v[:, None] ** j * (1 - v[:, None]) ** (n - j)
        px = (Bu @ profs.T).ravel(); py = (Bv @ profs.T).ravel()
        self.env_x, self.env_y = _envelope(px, py)
        self.H = _hull(px, py)
        # counting rules k-out-of-n: profile c_j = C(n, j) for j >= k
        C = np.array([comb(n, jj) for jj in j], float)
        cnt = np.array([[C[jj] if jj >= k else 0.0 for jj in j] for k in range(1, n + 1)])  # (n, n+1)
        X = Bu @ cnt.T; Y = Bv @ cnt.T                             # (T, n)
        o = np.argsort(-X, axis=1)                                  # OR has the largest P_f ... sort by P_f ascending
        X = np.take_along_axis(X, o[:, ::-1], 1); Y = np.take_along_axis(Y, o[:, ::-1], 1)
        self.c0 = np.concatenate([np.zeros((T, 1)), X, np.ones((T, 1))], 1)
        self.c1 = np.concatenate([np.zeros((T, 1)), Y, np.ones((T, 1))], 1)


def make_plane(rocf, n, pe, T):
    return Plane(rocf, n, pe, T) if n <= 4 else PlaneProfiles(rocf, n, pe, T)


def _share_txt(gain, share):
    """The share as Tables II and S1 print it: only where the crossing gain
    exceeds 0.01 (below that it is a ratio of grid noise), and not below zero
    (S3 short of max(S2, alpha) recovers nothing of the gain)."""
    if gain < 0.01 or share != share:
        return f"{'---':>10}"
    return f"{max(share, 0.0):>10.2f}"


def table(models, ns=(2, 3, 4, 5), pes=(0.0, 0.05, 0.10, 0.20), T=30001):
    print(f"{'model':<14}{'n':>3}{'P_e':>6}{'lam*':>8}{'floor cf':>10}{'floor num':>11}"
          f"{'crossing':>10}{'at alpha':>10}{'S3 share':>10}")
    rows = []
    for name, rocf in models:
        for n in ns:
            for pe in pes:
                # one dense threshold grid for every row: at T = 1501 the
                # deterministic envelope's staircase near a sharp crossing was 0.009 in
                # P_d and moved the d=3, n=4, P_e=0.05 entry from 0.035 to 0.033 and its
                # share from 0.15 to 0.00 as the grid was refined; at 30 001 it is converged
                P = make_plane(rocf, n, pe, T=T)
                if pe == 0:
                    cg, ca, sh = P.crossing(na=800)
                    print(f"{name:<14}{n:>3}{pe:>6.2f}{'---':>8}{'---':>10}{'---':>11}{cg:>10.4f}{ca:>10.3g}{_share_txt(cg, sh)}")
                    rows.append((name, n, pe, None, None, None, cg, ca, sh)); continue
                L = lam_star(rocf, pe)
                cf = (pe * L) ** n - pe ** n
                fnum = P.floor_gain()
                cg, ca, sh = P.crossing(na=800)
                rows.append((name, n, pe, L, cf, fnum, cg, ca, sh))
                print(f"{name:<14}{n:>3}{pe:>6.2f}{L:>8.4f}{cf:>10.4f}{fnum:>11.4f}"
                      f"{cg:>10.4f}{ca:>10.3g}{_share_txt(cg, sh)}")
    return rows


# ---------------------------------------------------------------- Theorem 3 on the first hull segment
def first_segment(rocf, pe, n, T=400001):
    """t*, phi(t*), lam*, alpha_1 = min(phi(t*)^n, pe^(n-1)) and the width factor."""
    t = np.linspace(1e-9, 1, T)
    r = phi(rocf(t), pe) / phi(t, pe)
    k = int(r.argmax()); ts, lam = float(t[k]), float(r[k])
    a1 = min(phi(ts, pe) ** n, pe ** (n - 1))
    return ts, float(phi(ts, pe)), lam, a1, (phi(ts, pe) / pe) ** n


def gain_closed_form(rocf, pe, n, alpha):
    """alpha*lam*^n - phi(g(t_alpha))^n on [pe^n, phi(t*)^n]; 0 outside."""
    ts, phits, lam, a1, _ = first_segment(rocf, pe, n)
    # tolerance: alpha = 0.0100 and 0.10**2 differ by 2e-18 in floating point,
    # which without it makes the floor itself return 0
    if alpha < pe ** n - 1e-12 or alpha > phits ** n + 1e-12:
        return 0.0
    ta = (alpha ** (1 / n) - pe) / (1 - 2 * pe)
    return alpha * lam ** n - phi(rocf(ta), pe) ** n


def mismatch(rocf, pe_true, n, alpha, pe_hat):
    """Realized (Pf/alpha, Pd) of the randomized floor design and of the
    deterministic AND design made for pe_hat, on a link with pe_true. The
    deterministic entry is None when that design is infeasible under pe_hat."""
    ts, phits, lam, a1, _ = first_segment(rocf, pe_hat, n)
    w = alpha / phi(ts, pe_hat) ** n                   # AND(t*) with weight w, else declare H0
    rand = (w * phi(ts, pe_true) ** n / alpha, w * phi(rocf(ts), pe_true) ** n)
    det = None
    if pe_hat ** n <= alpha:
        ta = (alpha ** (1 / n) - pe_hat) / (1 - 2 * pe_hat)
        det = (phi(ta, pe_true) ** n / alpha, phi(rocf(ta), pe_true) ** n)
    return rand, det


# ---------------------------------------------------------------- Monte Carlo
def simulate_logistic(n, pe, states, weights, frames, d=D_LOGIT):
    """states: list of (t, rule_vector); weights: mixing probabilities.
    Logistic observations y ~ Logistic(0,1) | H0, Logistic(d,1) | H1; local test
    u_k = 1{y_k > tau(t)} with P_f = t; BSC(pe) on each bit; fusion by the rule
    of the state drawn for the frame. Returns realized (P_f, P_d)."""
    pats, _ = get_rules(n)
    tau = np.array([np.log((1 - t) / t) for t, _ in states])
    which = RNG.choice(len(states), size=frames, p=weights)
    key = {int(''.join(map(str, p)), 2): i for i, p in enumerate(pats)}
    out = []
    for h in (0, 1):
        y = RNG.logistic(loc=h * d, scale=1.0, size=(frames, n))
        u = (y > tau[which][:, None]).astype(np.int8)
        u = u ^ (RNG.random((frames, n)) < pe)
        idx = (u * (1 << np.arange(n))[::-1]).sum(1)
        pos = np.vectorize(key.get)(idx)
        dec = np.array([states[w][1][p] for w, p in zip(which, pos)])
        out.append(dec.mean())
    return tuple(out)


def mc_check(frames=1_000_000):
    pats, rules = get_rules(2)
    AND = [f for f in rules if f.sum() == 1][0]
    ZERO = np.zeros(len(pats))
    pe, n = 0.30, 2
    t = np.linspace(1e-9, 1, 200001)
    L = phi(roc_logit(t), pe) / phi(t, pe)
    ts = float(t[L.argmax()]); fa = phi(ts, pe) ** 2
    w = pe ** 2 / fa
    pf, pd = simulate_logistic(n, pe, [(ts, AND), (ts, ZERO)], [w, 1 - w], frames)
    mu = pe * float(L.max())
    print(f"floor  n=2 P_e=0.30 alpha=0.0900: analytic P_f=0.0900 P_d={mu**2:.4f};  simulated P_f={pf:.4f} P_d={pd:.4f}  [{frames} frames]")
    pe, a = 0.10, 0.289
    P = Plane(roc_logit, 2, pe, T=3001)
    H = P.H; k = int((H[:, 0] <= a).sum()) - 1
    (xA, yA), (xB, yB) = H[k], H[k + 1]
    tt = np.linspace(1e-9, 1 - 1e-9, 3001)
    u, v = phi(tt, pe), phi(roc_logit(tt), pe)
    W0 = cells(np.repeat(u[:, None], 2, 1), pats); W1 = cells(np.repeat(v[:, None], 2, 1), pats)
    def locate(x, y):
        best = None
        for f in rules:
            px, py = W0 @ f, W1 @ f
            j = int(np.argmin((px - x) ** 2 + (py - y) ** 2))
            dd = (px[j] - x) ** 2 + (py[j] - y) ** 2
            if best is None or dd < best[0]:
                best = (dd, float(tt[j]), f)
        return best[1], best[2]
    tA, fA = locate(xA, yA); tB, fB = locate(xB, yB)
    wA = (xB - a) / (xB - xA)
    pf, pd = simulate_logistic(2, pe, [(tA, fA), (tB, fB)], [wA, 1 - wA], frames)
    print(f"crossing n=2 P_e=0.10 alpha=0.289: analytic S4={P.S4(a):.4f} (S2={P.S2(a):.4f}); simulated DR P_f={pf:.4f} P_d={pd:.4f}")
    ex, ey = P.env_x, P.env_y
    j = int((ex <= a).sum()) - 1
    tD, fD = locate(ex[j], ey[j])
    pf, pd = simulate_logistic(2, pe, [(tD, fD)], [1.0], frames)
    print(f"                                   simulated best deterministic P_f={pf:.4f} P_d={pd:.4f}")


if __name__ == '__main__':
    if '--segment' in sys.argv:
        g3 = lambda t: roc_gauss(t, 3.0)
        ts, phits, lam, a1, wf = first_segment(g3, 0.10, 2)
        print(f"d=3.0 Pe=0.10 n=2: t*={ts:.4f} phi(t*)={phits:.4f} lam*={lam:.3f} alpha_1={a1:.5f} width factor={wf:.3f}")
        for a in (0.0100, 0.0101, 0.0105, 0.0110, 0.0128):
            print(f"  alpha={a:.4f}: closed-form gain {gain_closed_form(g3, 0.10, 2, a):.4f}")
        for n_ in (2, 3, 4, 5, 6):
            print(f"  n={n_}: width factor {first_segment(g3, 0.10, n_)[4]:.2f}, mu^n {(0.10*lam)**n_:.4f}")
        for ph in (0.090, 0.095, 0.099, 0.100, 0.105, 0.110):
            r, d_ = mismatch(g3, 0.10, 2, 0.0100, ph)
            print(f"  Pe_hat={ph:.3f}: randomized Pf/alpha={r[0]:.3f} Pd={r[1]:.3f} | deterministic {None if d_ is None else tuple(round(float(x),3) for x in d_)}")
        sys.exit()
    models = [('logistic', lambda t: roc_logit(t)),
              ('gauss d=1.5', lambda t: roc_gauss(t, 1.5)),
              ('gauss d=3.0', lambda t: roc_gauss(t, 3.0))]
    table(models)
    if '--mc' in sys.argv:
        mc_check()
