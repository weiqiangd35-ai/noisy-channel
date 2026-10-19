# Code for "Decision Fusion over Hard-Decision Links: The Reachable Set and the Value of Randomization"

W. Dong and J. Zhou. Supplementary material: the code that computes every
number, table and figure in the paper and its supplementary PDF.

Python 3 with NumPy, SciPy and Matplotlib; nothing to install beyond those.
Tested with Python 3.13, NumPy 2.5, SciPy 1.18 and Matplotlib 3.11. Run every
script from this directory.

## Start here

```bash
python3 reproduce.py            # 47 checks, a few seconds
python3 reproduce.py --slow     # 133 checks, about 15 minutes
```

`reproduce.py` recomputes 133 of the values quoted in the paper and the
supplement (47 without `--slow`) and prints each beside the value in the text,
with the place where it appears. The values from the text are typed into the
script by hand, so a mismatch means the text and the code disagree. The script
exits with a non-zero status if any check fails. `--slow` adds the sweeps over
the (alpha, P_e) plane, rows of Table S1, the link-budget example and the
comparison of the five strategies. The values it does not check are printed by
the scripts that compute them: `robustness.py` prints Table S1 in full and,
with `--mc`, the simulation of Section IV-B.

## The local detector model

`localroc.py` defines the benchmark local ROC, a logistic location family

    Pd = Pf / (Pf + kappa (1 - Pf)),   kappa = exp(-d),   d = 2.4474,

fitted to the two designed operating points (0.3976, 0.8871) and
(0.1304, 0.6328) of Section II-C; it also defines the channel map
phi(x) = (1 - 2 P_e) x + P_e and lambda*(P_e). Every script takes the model
from this file. The Gaussian location family of Section IV-B is defined in
`robustness.py`.

## What computes what

| Script | Computes |
|---|---|
| `localroc.py` | the benchmark local ROC, the channel map, lambda*(P_e) |
| `fragmentation.py` | the monotone fusion rules, the deterministic operating points, their upper envelope and concave hull (Theorem 1) |
| `scaling.py` | D_n(P_e) and the onset of Theorem 2, by weight profile; the bound of Corollary 2 |
| `dedekind.py` | an independent enumeration of the monotone rules, checked against the Dedekind numbers (Theorem 2(ii)); n = 2 to 5 by default, `python3 dedekind.py 6` for n = 6 |
| `criterion.py` | the floor, the slope cap and the sensor-count requirement (Corollaries 1 and 3, Lemma 2) |
| `slope_bound.py` | a brute-force check of the slope cap over all monotone rules (Lemma 2) |
| `baselines.py` | the five strategies S0 to S4 of Section IV-A |
| `quantiles.py` | the fusion-layer share over the plane (Section IV-A, Appendix S-IV), with the grid printed |
| `robustness.py` | Table II and Table S1, Theorem 3 on the first hull segment, the P_e-mismatch numbers of Section III-D, the end-to-end simulation of Section IV-B |
| `linkbudget.py` | the decibel example of Section IV-B |
| `reproduce.py` | the checks described above |

Run times on one core: `criterion.py` about half an hour, `quantiles.py` about
17 minutes, `robustness.py` and `linkbudget.py` about six minutes each, the
other scripts under a minute;
`robustness.py --mc` adds three simulations of 10^6 frames per hypothesis,
and `robustness.py --segment` prints only the Theorem 3 numbers and the
mismatch table, in seconds.

## Figures

| Script | Writes | In the paper |
|---|---|---|
| `fig_merged.py` | `fig_why_randomize` | Fig. 1 |
| `make_floor_profile.py` | `fig_floor_profile` | Fig. 2 |
| `make_baseline_figs.py` | `fig_gain_heatmap`, `fig_baseline_comparison` | Figs. 3 and 4 |
| `make_fig_regimes.py` | `fig_regimes` | Fig. S2 |

Each figure is written as `.pdf` and `.png` to the current directory; set
`FIG_OUT` to write elsewhere, e.g. `FIG_OUT=../out/ python3 fig_merged.py`.
`make_baseline_figs.py` sweeps the (alpha, P_e) plane, which takes several
minutes; it saves the result as `gain_maps.npz` in this directory and reuses
the file when it is present. The `gain_maps.npz` supplied here is that saved
sweep; `reproduce.py --slow` reads it for one check. Figs. S1 and S3 are
diagrams, not computed. Figure layout can differ slightly between Matplotlib
versions; the values do not.

## Notes on the searches

- `deterministic_points` searches a common threshold across sensors.
  Theorem 1(ii) says unequal thresholds cannot extend the reachable interval of
  any rule; a search over threshold pairs on the whole plane at n = 2 (in
  `reproduce.py --slow`) moves S2 by at most 0.0087 and S4 by 10^-4.
- A grid of unequal thresholds gives a spurious S3 - S2 of about 8 x 10^-3 at
  n = 3, P_e = 0 that does not shrink as the grid is refined, because S3
  interpolates between rules while S2 only has grid points. Continuous
  optimization gives zero to 10^-10, and the statements about S3 = S2 rest on
  it.
- Tables II and S1 use one threshold grid of 30 001 points for every row.
  Coarser grids leave staircase artifacts near sharp crossings and in the
  ideal-link rows; between coarse grids the envelopes move by about 0.004, so a
  difference below that is grid noise.
