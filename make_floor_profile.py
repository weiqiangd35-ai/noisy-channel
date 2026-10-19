"""
The gain profile along alpha, normalized by the floor P_e^n, for the 2-LD
benchmark at P_e = 0.05, 0.10, 0.20 -- Fig. 2 of the main text
(Section III-D). The markers at
alpha/P_e^n = 1 are the closed form mu(P_e)^n - P_e^n of Theorem 3; the curves
are the search.

    FIG_OUT=/path/to/paper/ python3 make_floor_profile.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from localroc import lam_star
from criterion import det_points, env_and_hull

OUT = os.environ.get('FIG_OUT', './')
plt.rcParams.update({'font.size': 6.4, 'axes.linewidth': .5,
                     'lines.linewidth': 1.0, 'xtick.labelsize': 5.4,
                     'ytick.labelsize': 5.4, 'legend.fontsize': 5.2,
                     'axes.labelsize': 6.4, 'figure.dpi': 300,
                     'savefig.dpi': 300, 'font.family': 'serif',
                     'mathtext.fontset': 'dejavuserif'})
COL = ('#1f77b4', '#d62728', '#2ca02c')
D = 2.4474

fig, ax = plt.subplots(figsize=(3.45, 1.75))
for c, pe in zip(COL, (.05, .10, .20)):
    n = 2
    F = pe ** n
    px, py = det_points(n, D, pe, ngrid=30001)
    rr = np.logspace(-1.2, 2.6, 1200)
    al = np.clip(F * rr, 1e-9, .999)
    E, H = env_and_hull(px, py, al)
    g = np.maximum(H - np.maximum(E, al), 0)
    ax.semilogx(rr, g, color=c, label=rf'$P_e={pe}$')
    mu = pe * lam_star(pe)
    ax.plot([1], [mu ** n - pe ** n], 'o', color=c, ms=3.0)
    print(f'P_e={pe}: closed form {mu**n - pe**n:.4f}, search at the floor {np.interp(1.0, rr, g):.4f}, '
          f'largest secondary peak {g[rr > 3].max():.4f} at alpha/P_e^n = {rr[rr > 3][g[rr > 3].argmax()]:.1f}')
ax.axvline(1, color='.55', lw=.6, ls=':')
ax.set_xlabel(r'$\alpha / P_e^{\,n}$')
ax.set_ylabel(r'gain over $\max(S_2,\alpha)$')
ax.tick_params(length=1.8)
ax.legend(frameon=False, loc='upper right')
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(OUT + f'fig_floor_profile.{ext}', bbox_inches='tight')
plt.close(fig)
print('fig_floor_profile done')
