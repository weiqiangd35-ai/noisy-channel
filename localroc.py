"""
Shared local ROC model.

The manuscript's two designed operating points, (0.3976, 0.8871) and
(0.1304, 0.6328), are consistent with a logistic location family to within
0.003 in Pd, with shift d = 2.4474 (kappa = e^-d = 0.08652):

        Pd = Pf / (Pf + kappa (1 - Pf)).

We use the logistic family rather than the Gaussian one for the continuous
sweeps.  Two reasons.  First, the Gaussian likelihood ratio is unbounded at the
origin, so the ideal-channel randomization gain degenerates to ~5e-4 and the
illustration is not representative; the logistic gain is about ten times larger
and visible at a sensible zoom.  Second, the logistic family has a BOUNDED
likelihood ratio, lam*(0) = e^d, so the slope cap of Lemma 2 has two competing
sources -- the sensor and the channel -- rather than only one.

The benchmark points A, B and C of Section II-C are unaffected: they are the
two designed operating points themselves, not the family that interpolates
them.
"""
import numpy as np

D_LOGIT = 2.4474
KAPPA = float(np.exp(-D_LOGIT))


def roc(t, d=D_LOGIT):
    """Local ROC of the logistic location family: Pd as a function of Pf."""
    k = np.exp(-d)
    t = np.asarray(t, float)
    return t / (t + k * (1.0 - t))


def phi(x, pe):
    """Symmetric DMC map."""
    return (1.0 - 2.0 * pe) * np.asarray(x, float) + pe


def lam_star(pe, d=D_LOGIT, N=400001):
    """max_t phi(g(t))/phi(t): the largest post-channel single-link LR."""
    t = np.linspace(1e-12, 1 - 1e-12, N)
    return float(np.max(phi(roc(t, d), pe) / phi(t, pe)))


def sensor_cap(d=D_LOGIT):
    """lam*(0) = e^d for the logistic family."""
    return float(np.exp(d))


if __name__ == '__main__':
    for P in ((0.3976, 0.8871), (0.1304, 0.6328)):
        print(f'  Pf={P[0]:.4f}: manuscript Pd={P[1]:.4f}, '
              f'logistic {float(roc(P[0])):.4f}')
    print(f'  lam*(0) = {lam_star(0.0):.4f}   e^d = {sensor_cap():.4f}')
