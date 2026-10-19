"""
Fig. 1 of the paper, three panels: (a) continuous observations over an ideal
link, with an inset that shows how small the gap between the deterministic
envelope and its concave hull is; (b) a discrete observation alphabet over an
ideal link; (c) continuous observations over a link with P_e = 0.25 (set
FIG1C_PE to change it). All three panels are computed; nothing is drawn by hand.

    FIG_OUT=/path/to/paper/ python3 fig_merged.py
"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from localroc import roc as _roc, D_LOGIT
from scipy.optimize import brentq

import os
OUT = os.environ.get('FIG_OUT', './')
D = D_LOGIT                      # SNR of the running 2-LD example

plt.rcParams.update({'font.size': 7.6, 'axes.linewidth': .6, 'lines.linewidth': 1.3,
                     'xtick.labelsize': 6.6, 'ytick.labelsize': 6.6,
                     'legend.fontsize': 6.2, 'axes.labelsize': 7.6,
                     'figure.dpi': 300, 'savefig.dpi': 300,
                     'font.family': 'serif', 'mathtext.fontset': 'dejavuserif'})

phi = lambda x, pe: (1 - 2 * pe) * x + pe


def upper_hull(px, py):
    o = np.argsort(px); px, py = px[o], py[o]
    st = []
    for x, y in zip(px, py):
        while len(st) >= 2:
            (x1, y1), (x2, y2) = st[-2], st[-1]
            if (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) >= 0:
                st.pop()
            else:
                break
        st.append((x, y))
    return np.array([s[0] for s in st]), np.array([s[1] for s in st])


def rules_of(pe, N=400001):
    t = np.linspace(1e-12, 1 - 1e-12, N)
    u, v = phi(t, pe), phi(_roc(t, D), pe)
    return (u ** 2, v ** 2), (1 - (1 - u) ** 2, 1 - (1 - v) ** 2)


def best_chord(pe):
    """Crossing point of the AND/OR curves, the optimal A/B pair, and the gain."""
    def A(x):
        x = np.clip(np.asarray(x, float), 1e-15, 1 - 1e-15)
        t = np.clip((np.sqrt(x) - pe) / (1 - 2 * pe), 1e-12, 1 - 1e-12)
        return phi(_roc(t, D), pe) ** 2

    def O(x):
        x = np.clip(np.asarray(x, float), 1e-15, 1 - 1e-15)
        t = np.clip((1 - np.sqrt(1 - x) - pe) / (1 - 2 * pe), 1e-12, 1 - 1e-12)
        return 1 - (1 - phi(_roc(t, D), pe)) ** 2

    lo, hi = pe ** 2 + 1e-6, 1 - pe ** 2 - 1e-6
    xs = np.linspace(lo, hi, 6000)
    f = A(xs) - O(xs)
    sgn = np.where(np.diff(np.sign(f)))[0]
    xc = brentq(lambda x: float(A(x) - O(x)), xs[sgn[0]], xs[sgn[0] + 1])

    def gain(a, b):
        ya, yb = max(A(a), O(a)), max(A(b), O(b))
        g = np.linspace(a, b, 4001)
        return float(np.max(ya + (yb - ya) * (g - a) / (b - a)
                            - np.maximum(A(g), O(g))))

    best, arg = 0.0, None
    for a in np.linspace(max(xc - .15, lo), xc, 110):
        for b in np.linspace(xc, min(xc + .15, hi), 110):
            if b - a < 1e-4:
                continue
            v = gain(a, b)
            if v > best:
                best, arg = v, (a, b)
    return xc, arg, best, A, O


# =====================================================================
fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.05))

# ---------------- (a) continuous observations, ideal channel ----------
(af, ad), (of, od) = rules_of(0.0)
tt_a = np.linspace(1e-9, 1 - 1e-9, 4001)
ax[0].plot(af, ad, color='#1f77b4', label='AND rule')
ax[0].plot(tt_a, _roc(tt_a, D), color='#2ca02c', lw=.9, label='single-sensor rule')
ax[0].plot(of, od, color='#ff7f0e', label='OR rule')
px = np.concatenate([[0], af, of, [1]]); py = np.concatenate([[0], ad, od, [1]])
hx, hy = upper_hull(px, py)
ax[0].plot(hx, hy, 'r--', lw=1.1, dashes=(4, 2.2), label='upper concave envelope')
xc, (xa, xb), gmax, A, O = best_chord(0.0)
ya, yb = max(A(xa), O(xa)), max(A(xb), O(xb))
ax[0].plot([xa, xb], [ya, yb], 'ko', ms=2.6)

# inset -- with tick labels, which the original lacked
axi = inset_axes(ax[0], width='96%', height='92%',
                 bbox_to_anchor=(.44, .05, .56, .46),
                 bbox_transform=ax[0].transAxes, loc='center')
w = (xb - xa) * .18
s = np.linspace(xa - w, xb + w, 3000)
env = np.maximum(A(s), O(s))
axi.plot(s, A(s), color='#1f77b4', lw=1.0)
axi.plot(s, O(s), color='#ff7f0e', lw=1.0)
axi.plot([xa, xb], [ya, yb], 'r--', lw=1.2, dashes=(3.4, 1.8))
axi.plot([xa, xb], [ya, yb], 'ko', ms=2.6)
ytop = ya + (yb - ya) * (xc - xa) / (xb - xa)
ybot = float(np.interp(xc, s, env))
axi.annotate('', xy=(xc, ytop), xytext=(xc, ybot),
             arrowprops=dict(arrowstyle='<->', color='k', lw=.8,
                             shrinkA=0, shrinkB=0))
axi.axvline(xc, color='.6', lw=.4, ls=':')
axi.annotate(rf'$\Delta P_d={gmax*1e3:.1f}\times10^{{-3}}$', xy=(xc, (ytop + ybot) / 2),
             xytext=(xa + (xb - xa) * .5, max(ya, yb) + 4.0 * gmax),
             fontsize=4.6, ha='center', arrowprops=dict(arrowstyle='-', color='.4', lw=.4))
axi.set_xlim(s[0], s[-1])
axi.set_ylim(env.min() - .8 * gmax, max(ya, yb) + 5.5 * gmax)
axi.ticklabel_format(useOffset=False)
axi.tick_params(labelsize=4.4, length=1.4, pad=.8)
axi.set_yticks(np.round([env.min(), (env.min()+max(ya,yb))/2, max(ya, yb)], 4))
axi.set_xticks(np.round([xa, xc, xb], 3))
for sp in axi.spines.values():
    sp.set_linewidth(.5)
mark_inset(ax[0], axi, loc1=2, loc2=3, fc='none', ec='.55', lw=.5)   # connectors stay left of the legend
ax[0].set_title(rf'(a) continuous, $P_e=0$', fontsize=7.4)

# ---------------- (b) discrete alphabet, ideal channel ----------------
alph = [(0, 0), (.20, .70), (.45, .90), (1, 1)]
bx, by = [], []
for f1, d1 in alph:
    for f2, d2 in alph:
        bx += [f1 * f2, 1 - (1 - f1) * (1 - f2)]
        by += [d1 * d2, 1 - (1 - d1) * (1 - d2)]
bx, by = np.array(bx), np.array(by)
hx2, hy2 = upper_hull(bx, by)
o = np.argsort(bx); envb = np.maximum.accumulate(by[o])
gapb = np.max(np.interp(bx[o], hx2, hy2) - envb)
ax[1].plot(bx, by, 'o', color='#1f77b4', ms=3.2, label='deterministic points')
ax[1].plot(hx2, hy2, 'r--', lw=1.1, dashes=(4, 2.2), label='upper concave envelope')
ax[1].set_title(rf'(b) discrete alphabet, $P_e=0$', fontsize=7.4)

# ---------------- (c) continuous observations, noisy channel ----------
# P_e = 0.25 (BPSK/AWGN about -6.4 dB) is below the onset P_e^o(2) = 0.382,
# so the geometry is that of Theorem 2(ii)
# (arcs on the chance line, the floor, the dictator rule joining AND and OR)
# is the same. The caption numbers are printed below; reproduce.py checks them.
pe = float(os.environ.get('FIG1C_PE', '0.25'))
(af3, ad3), (of3, od3) = rules_of(pe)
# The dictator rule (listen to one sensor) has interval [P_e, 1-P_e], which
# is what joins the AND and OR arcs below the onset. Omitting it made the
# reachable set look disconnected at P_e = 0.35 < P_e^o(2) = 0.382, in
# apparent contradiction of Theorem 2.
tt3 = np.linspace(1e-9, 1 - 1e-9, 4001)
df3 = (1 - 2 * pe) * tt3 + pe
dd3 = (1 - 2 * pe) * _roc(tt3, D) + pe
ax[2].plot(af3, ad3, color='#1f77b4', label='AND rule')
ax[2].plot(df3, dd3, color='#2ca02c', lw=.9, label='single-sensor (dictator) rule')
ax[2].plot(of3, od3, color='#ff7f0e', label='OR rule')
px3 = np.concatenate([[0], af3, df3, of3, [1]]); py3 = np.concatenate([[0], ad3, dd3, od3, [1]])
hx3, hy3 = upper_hull(px3, py3)
ax[2].plot(hx3, hy3, 'r--', lw=1.1, dashes=(4, 2.2), label='upper concave envelope')
o3 = np.argsort(px3); env3 = np.maximum.accumulate(py3[o3])
gap3 = np.max(np.interp(px3[o3], hx3, hy3) - env3)
_k3 = int(np.argmax(np.interp(px3[o3], hx3, hy3) - env3))
print(f'panel (c): P_e={pe}, floor P_e^2={pe**2:.4f}, max gain {gap3:.4f} at P_f={px3[o3][_k3]:.4f}, '
      f'P_d(det)={env3[_k3]:.4f}, P_d(rand)={np.interp(px3[o3][_k3], hx3, hy3):.4f}')
ax[2].axvspan(0, pe ** 2, color='#d62728', alpha=.14, lw=0)
ax[2].text(pe ** 2 + .035, .62, rf'$P_f<P_e^2$' '\nunreachable',
           fontsize=5.4, color='#a01f1f', rotation=90,
           ha='center', va='center')
ax[2].set_title(rf'(c) continuous, $P_e={pe}$', fontsize=7.4)

for i, a in enumerate(ax):
    a.plot([0, 1], [0, 1], color='.72', lw=.7, ls=':')
    a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.set_xlabel(r'global $P_f$'); a.set_ylabel(r'global $P_d$')
    a.tick_params(length=2.2)
    if i == 0:      # keep clear of the inset (lower right) and of the arcs (top)
        a.legend(frameon=False, fontsize=5.2, handlelength=1.6, labelspacing=.3,
                 loc='center right', bbox_to_anchor=(.99, .72))
    else:
        a.legend(frameon=False, fontsize=5.2, handlelength=1.6, labelspacing=.3,
                 loc='lower right')
fig.tight_layout()
for e in ('png', 'pdf'):
    fig.savefig(OUT + f'fig_why_randomize.{e}', bbox_inches='tight')
plt.close(fig)

print(f'(a) crossing at Pf={xc:.4f}, optimal A={xa:.4f} B={xb:.4f}, '
      f'max gain {gmax:.3e}')
print(f'(b) discrete alphabet, max gain {gapb:.4f}')
print(f'(c) Pe={pe}: floor {pe**2:.4f}, max gain {gap3:.4f}  '
      f'({gap3/gmax:.0f}x panel (a))')
