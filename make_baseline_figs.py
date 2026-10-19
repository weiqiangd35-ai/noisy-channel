from localroc import roc as _roc, D_LOGIT
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from baselines import evaluate, local_curve, rule_points, hull_at_alpha_rowwise
from fragmentation import (get_rules as kofn_rules, deterministic_points, upper_envelope,
                           concave_hull)

plt.rcParams.update({'font.size': 7.5, 'axes.linewidth': .6, 'lines.linewidth': 1.2,
                     'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8,
                     'legend.fontsize': 6.3, 'axes.labelsize': 7.5,
                     'figure.dpi': 300, 'savefig.dpi': 300,
                     'font.family': 'serif', 'mathtext.fontset': 'dejavuserif'})
import os
D, OUT = D_LOGIT, os.environ.get('FIG_OUT', './')


# ---------------------------------------------------------------- fig A
def fig_baselines():
    fig, axes = plt.subplots(1, 1, figsize=(3.45, 2.15))
    for ax, (n, alpha) in zip([axes], [(2, 0.01)]):
        pes = np.linspace(0, 0.45, 100)
        r = evaluate(n, D, alpha, pes, ngrid=4001)
        dag = alpha ** (1.0 / n)
        ax.axvline(dag, color='.45', ls=':', lw=.9)
        ax.text(dag + .006, .93, r'$P_e^{\dagger}=\alpha^{1/n}$',
                transform=ax.get_xaxis_transform(), fontsize=6.2, color='.35')
        ax.plot(pes, r['S4'], color='#d62728', label='S4  Dependent Rand. (proposed)')
        ax.plot(pes, r['S3'], color='#ff7f0e', ls='--', dashes=(4, 2),
                label='S3  fusion-layer randomization')
        ax.plot(pes, r['S2'], color='#1f77b4', label='S2  best deterministic (channel-aware)')
        ax.plot(pes, r['S0'], color='#2ca02c', ls='-.', label='S0  AND rule only')
        bad = r['S1f'] > alpha + 1e-9
        ax.plot(pes, np.where(bad, r['S1d'], np.nan), color='.35', ls=(0, (1, 1.6)),
                label=r'S1  channel-blind ($P_f>\alpha$)')
        ax.set_xlabel(r'channel crossover probability $P_e$')
        ax.set_ylabel(r'global $P_d$ at $P_f=\alpha$')
        ax.set_title(rf'$n={n}$, $\alpha={alpha}$', fontsize=7.2)
        ax.set_xlim(0, .45); ax.set_ylim(0, 1.0); ax.tick_params(length=2.2)
    axes.legend(frameon=False, loc='upper left', fontsize=5.0,
                bbox_to_anchor=(.02, .90), handlelength=1.6, labelspacing=.3)
    # inset: the interval to the left of the threshold, where the text's
    # argument lives and where the two curves are only ~5% of the axis apart
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    axi = inset_axes(axes, width='52%', height='46%',
                     bbox_to_anchor=(.46, .27, .53, .51),
                     bbox_transform=axes.transAxes, loc='center')
    pe2 = np.linspace(0.055, 0.102, 200)
    r2 = evaluate(2, D, 0.01, pe2, ngrid=4001)
    axi.plot(pe2, r2['S4'], color='#d62728')
    axi.plot(pe2, r2['S2'], color='#1f77b4')
    axi.axvline(0.1, color='.45', ls=':', lw=.7)
    axi.set_xlim(0.055, 0.102); axi.set_ylim(0, 0.16)
    axi.tick_params(labelsize=4.6, length=1.6, pad=1)
    axi.set_xticks([0.06, 0.08, 0.10]); axi.set_yticks([0, 0.08, 0.16])
    axi.set_yticklabels(['0.00', '0.08', '0.16'])
    axi.set_xticklabels(['0.06', '0.08', '0.10'])
    for sp in axi.spines.values():
        sp.set_linewidth(.5)
    mark_inset(axes, axi, loc1=2, loc2=4, fc='none', ec='.6', lw=.5)
    fig.tight_layout()
    for e in ('png', 'pdf'):
        fig.savefig(OUT + f'fig_baseline_comparison.{e}', bbox_inches='tight')
    plt.close(fig); print('fig_baseline_comparison done')


# ---------------------------------------------------------------- fig B
def fig_heatmap(n=2, na=260, np_=200):
    pats, rules = kofn_rules(n)
    alphas = np.logspace(-3, np.log10(.999), na)   # log axis: alpha<0.05 was a sliver
    pes = np.linspace(0, .49, np_)
    Gtot = np.zeros((np_, na)); Gdep = np.zeros((np_, na))
    t = np.linspace(1e-12, 1 - 1e-12, 3001)
    al = np.linspace(0, 1, 40001)
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gain_maps.npz')
    # the sweep takes minutes; reuse the saved map if present
    if os.path.exists(cache):
        z = np.load(cache)
        if z['alphas'].shape == alphas.shape and z['pes'].shape == pes.shape:
            alphas, pes, Gtot, Gdep = z['alphas'], z['pes'], z['Gtot'], z['Gdep']
            np_ = 0                        # skip the sweep below
    for i, pe in enumerate(pes[:np_]):
        pts = deterministic_points(n, D, pe, rules, pats, ngrid=3001)
        E = upper_envelope(pts, alphas)
        H, _ = concave_hull(pts, al)
        Hi = np.interp(alphas, al, H)
        u, v = local_curve(t, D, pe)
        X, Y = rule_points(rules, pats, u, v)
        S3 = np.array([np.nanmax(hull_at_alpha_rowwise(X, Y, a)) for a in alphas])
        # The fair deterministic baseline is max(S2, alpha), not S2: a coin flip
        # uses no sensors and already attains (alpha, alpha), so crediting
        # randomization with the move from S2 up to the chance line counts as
        # gain something that needs no randomization at all. Below the floor
        # S2 = 0, and without this correction the whole of alpha is booked as
        # gain -- which put the peak of panel (a) in an interior gap at
        # Pe ~ 0.49, contradicting Theorem 3.
        base = np.maximum(E, alphas)
        Gtot[i] = np.maximum(Hi - base, 0)
        Gdep[i] = np.maximum(Hi - np.maximum(S3, alphas), 0)

    # full text width: placed with width=\textwidth, so a column-width canvas would be scaled 2x
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 1.87))
    for ax, G, ttl in zip(axes, (Gtot, Gdep),
                          (r'(a) value of randomization, $S_4-\max(S_2,\alpha)$',
                           r'(b) dependent only, $S_4-S_3$ (3.5$\times$ finer)')):
        m = ax.pcolormesh(alphas, pes, G, shading='auto', cmap='YlOrRd',
                          vmin=0, vmax=max(G.max(), 1e-9))
        aa = np.logspace(-4, 0, 400)
        ax.plot(aa, aa ** (1.0 / n), color='#1a3a8f', lw=1.2, ls='--',
                label=r'floor $P_e=\alpha^{1/n}$')
        ax.plot(1 - aa, aa ** (1.0 / n), color='#1a3a8f', lw=1.2, ls=':',
                label=r'ceiling $P_e=(1-\alpha)^{1/n}$')
        ax.set_xscale('log'); ax.set_xlim(1e-3, 1); ax.set_ylim(0, .49)
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(r'$P_e$')
        ax.set_title(ttl.replace(r'\emph{dependent}', 'dependent'), fontsize=7.4)
        ax.tick_params(length=2.2)
        fig.colorbar(m, ax=ax, pad=.02).ax.tick_params(labelsize=6.8)
    # upper left is empty; lower left is the floor ridge at alpha < 1e-2, which the legend must not cover
    axes[0].legend(frameon=False, loc='upper left', labelcolor='k', fontsize=6.0)
    fig.tight_layout()
    for e in ('png', 'pdf'):
        fig.savefig(OUT + f'fig_gain_heatmap.{e}', bbox_inches='tight')
    plt.close(fig)
    print('fig_gain_heatmap done')
    i,j = np.unravel_index(Gtot.argmax(), Gtot.shape)
    print(f'  max of S4-max(S2,alpha) = {Gtot.max():.4f} at alpha={alphas[j]:.4f}, Pe={pes[i]:.4f}')
    print(f'    floor there = {pes[i]**n:.4f}')
    print(f'  max over (alpha,Pe) of S4-S3 = {Gdep.max():.4f}'
          f'   ({100*Gdep.max()/Gtot.max():.1f}% of the total)')
    np.savez(cache, alphas=alphas, pes=pes, Gtot=Gtot, Gdep=Gdep)


# ---------------------------------------------------------------- table
def make_table():
    lines = []
    for n, alpha in ((2, 0.01), (3, 0.01)):
        pes = np.array([.05, .08, .10, .15, .20, .25])
        r = evaluate(n, D, alpha, pes, ngrid=6001)
        lines.append(rf'\multicolumn{{8}}{{l}}{{\textit{{$n={n}$ sensors, '
                     rf'$\alpha={alpha}$, $P_e^{{\dagger}}={alpha**(1/n):.3f}$}}}}\\[1pt]')
        for i, pe in enumerate(pes):
            v = r'$\times$' if r['S1f'][i] > alpha + 1e-9 else r'$\checkmark$'
            lines.append(
                f"{pe:.2f} & {r['S1f'][i]:.4f} {v} & {r['S0'][i]:.4f} & "
                f"{r['S2'][i]:.4f} & {r['S3'][i]:.4f} & {r['S4'][i]:.4f} & "
                f"{r['S4'][i]-r['S2'][i]:.4f} & {r['S4'][i]-r['S3'][i]:.4f}"
                r' \\')
        lines.append(r'\addlinespace')
    open(OUT + 'table_baselines.tex', 'w').write('\n'.join(lines))
    print('table_baselines.tex done')


if __name__ == '__main__':
    fig_baselines(); fig_heatmap(); make_table()
