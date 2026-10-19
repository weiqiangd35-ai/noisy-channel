"""
Fig. S2 of the supplement, both panels in one call so that they share height
and font size: (a) D_n(P_e) divided by 2 P_e^n, the two regimes of Theorem 2;
(b) the sensor-count requirement (16).

    FIG_OUT=/path/to/paper/ python3 make_fig_regimes.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from scaling import D_profile, onset_profile

import os
OUT = os.environ.get('FIG_OUT', './')
plt.rcParams.update({'font.size': 6.4, 'axes.linewidth': .5,
                     'lines.linewidth': 1.0, 'xtick.labelsize': 5.4,
                     'ytick.labelsize': 5.4, 'legend.fontsize': 5.0,
                     'axes.labelsize': 6.4, 'figure.dpi': 300,
                     'savefig.dpi': 300, 'font.family': 'serif',
                     'mathtext.fontset': 'dejavuserif'})

COL = ('#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b')
fig, (ax_a, ax_c) = plt.subplots(1, 2, figsize=(4.9, 1.75))
ax = {0: ax_a, 2: ax_c}

# ---- (a) D_n against the law it should follow ---------------------------
for n, c in zip((2, 3, 4, 5, 6), COL):
    es = np.linspace(0.02, .4995, 700)
    r = np.array([D_profile(n, e) / (2 * e ** n) for e in es])
    ax[0].plot(es, r, color=c, label=f'$n={n}$')
    e0 = onset_profile(n) if n <= 5 else brentq(
        lambda x: (1 - x) ** 6 - x ** 5, 1e-12, .4999999)
    if np.isfinite(e0):
        ax[0].plot([e0], [1.0], 'o', color=c, ms=2.8, zorder=5)
ax[0].axhline(1.0, color='.55', lw=.6, ls='--', zorder=1)
ax[0].set_ylim(0.97, 1.8)
ax[0].set_xlim(0.02, 0.4995)
ax[0].set_xlabel(r'crossover probability $P_e$')
ax[0].set_ylabel(r'$\mathcal{D}_n(P_e)\,/\,2P_e^{\,n}$')
ax[0].set_title(r'(a) the two regimes', fontsize=6.2)
ax[0].legend(frameon=False, ncol=2, loc='upper left')

# ---- (c) the sensor-count requirement ----------------------------------
pes = np.linspace(.01, .45, 400)
for c, (a, pdt) in zip(COL, ((.05, .90), (.01, .90), (.001, .90))):
    ax[2].plot(pes, np.log(pdt / a) / np.log((1 - pes) / pes), color=c,
               label=rf'$\alpha={a}$')
ax[2].set_ylim(0, 14)
ax[2].set_xlabel(r'crossover probability $P_e$')
ax[2].set_ylabel(r'minimum number of detectors')
ax[2].set_title(r'(b) $n\geq\ln(P_d/\alpha)/\ln\frac{1-P_e}{P_e}$, $P_d=0.9$',
                fontsize=6.2)
ax[2].legend(frameon=False, loc='upper left')

for a_ in (ax_a, ax_c):
    a_.tick_params(length=1.8)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(OUT + f'fig_regimes.{ext}', bbox_inches='tight')
plt.close(fig)
print('fig_regimes done (two panels, one file)')
