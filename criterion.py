"""
A decision criterion for randomized fusion.

Three quantities settle the design question completely, in the regime
Pe <= Pe*(n) that covers any channel worth deploying over:

  FLOOR      F = Pe^n            no deterministic monotone rule reaches Pf < F
  SLOPE CAP  S = lam*(Pe)^n      the global ROC cannot leave the origin faster,
                                 and S <= ((1-Pe)/Pe)^n whatever the sensors are
  INTERIOR   delta_n(Pe)         residual scalloping strictly inside the
                                 reachable interval

The criterion:  randomization is necessary iff alpha < F (or alpha > 1-F);
inside the interval its value is at most delta_n(Pe), which we measure.

The slope cap inverts into a sensor-count requirement that involves no property
of the sensors at all:

        n  >=  ln(Pd_target / alpha) / ln((1-Pe)/Pe).
"""
import itertools
import numpy as np
from localroc import roc as _roc, D_LOGIT

phi = lambda x, e: (1 - 2 * e) * x + e


def kofn(n):
    """Full monotone class where enumerable; the dictator rules matter."""
    from fragmentation import get_rules
    return get_rules(n)


def det_points(n, d, e, ngrid=40001):
    pats, rules = kofn(n)
    t = np.linspace(1e-12, 1 - 1e-12, ngrid)
    u, v = phi(t, e), phi(_roc(t, d), e)
    X = [np.zeros(1), np.ones(1)]
    Y = [np.zeros(1), np.ones(1)]
    for f in rules:
        w0 = np.prod(np.where(pats[None, :, :] == 1, u[:, None, None],
                              1 - u[:, None, None]), axis=-1)
        w1 = np.prod(np.where(pats[None, :, :] == 1, v[:, None, None],
                              1 - v[:, None, None]), axis=-1)
        X.append((w0 * f).sum(-1)); Y.append((w1 * f).sum(-1))
    return np.concatenate(X), np.concatenate(Y)


def env_and_hull(px, py, alphas):
    o = np.argsort(px); sx, sy = px[o], np.maximum.accumulate(py[o])
    E = np.interp(alphas, sx, sy)
    st = []
    for x, y in zip(px[o], py[o]):
        while len(st) >= 2:
            (x1, y1), (x2, y2) = st[-2], st[-1]
            if (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) >= 0:
                st.pop()
            else:
                break
        st.append((x, y))
    hx = np.array([s[0] for s in st]); hy = np.array([s[1] for s in st])
    return E, np.interp(alphas, hx, hy)


def lam_star(e, d, N=200001):
    t = np.linspace(1e-12, 1 - 1e-12, N)
    return float(np.max(phi(_roc(t, d), e) / phi(t, e)))


if __name__ == '__main__':
    d = D_LOGIT
    print('1.  Is H(alpha) <= alpha * S_n  for every alpha?  '
          '(concavity through the origin)\n')
    print(f"{'n':>3} {'Pe':>6} {'S_n':>10} {'max H(a)/a':>12} {'cap ((1-e)/e)^n':>17}")
    for n in (2, 3):
        for e in (.05, .10, .20, .30):
            px, py = det_points(n, d, e)
            al = np.linspace(1e-4, .999, 40001)
            _, H = env_and_hull(px, py, al)
            S = lam_star(e, d) ** n
            print(f'{n:>3} {e:>6.2f} {S:>10.4f} {np.max(H / al):>12.4f} '
                  f'{((1-e)/e)**n:>17.4f}'
                  f"   {'ok' if np.max(H/al) <= S*(1+2e-3) else 'VIOLATED'}")

    print('\n2.  Interior gain: how much is randomization worth INSIDE '
          '[Pe^n, 1-Pe^n]?\n')
    print(f"{'n':>3} {'Pe':>6} {'floor Pe^n':>12} {'delta_n (interior)':>20} "
          f"{'gain just below floor':>23}")
    for n in (2, 3, 5):
        for e in (.05, .10, .20, .30):
            px, py = det_points(n, d, e)
            F = e ** n
            ain = np.linspace(F * 1.02, 1 - F - 1e-4, 20001)
            E, H = env_and_hull(px, py, ain)
            din = np.max(np.maximum(H - E, 0))
            aout = F * .5
            Eo, Ho = env_and_hull(px, py, np.array([aout]))
            print(f'{n:>3} {e:>6.2f} {F:>12.5f} {din:>20.6f} '
                  f'{float(Ho[0] - Eo[0]):>23.5f}')

    print('\n3.  Sensor-count requirement  n >= ln(Pd/alpha)/ln((1-Pe)/Pe)\n')
    print(f"{'alpha':>7} {'Pd':>6} {'Pe':>6} {'formula':>9} {'n needed':>10} "
          f"{'exact smallest n':>18}")
    for alpha, pdt in ((.01, .90), (.01, .70), (.001, .90), (.05, .95)):
        for e in (.05, .10, .20):
            need = np.log(pdt / alpha) / np.log((1 - e) / e)
            exact = None
            for n in range(1, 13):
                px, py = det_points(n, d, e, ngrid=8001)
                _, H = env_and_hull(px, py, np.array([alpha]))
                if float(H[0]) >= pdt:
                    exact = n; break
            print(f'{alpha:>7.3f} {pdt:>6.2f} {e:>6.2f} {need:>9.2f} '
                  f'{int(np.ceil(need)):>10} {(str(exact) if exact else "> 12"):>18}')
    print("\n  '> 12': not reached with n <= 12 (all monotone rules to n = 4, counting\n"
          "  rules above). reproduce.py --slow finds the counts quoted in Section III-D\n"
          "  (12, 15, 29 and 23) with counting rules up to n = 40.")
