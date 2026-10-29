# Local stars and edge-weighted source estimation

Promoted: **MPUrge stars + support-first peak-consumption finisher** and
**MPUrge stars + robust range-refinement and peak-consumption finisher**.
They improve the existing methods at low and moderate noise in this rectangle,
especially 0.02–0.2 m. They remain exploratory methods: promotion used these same
room/noise results, not a new independent validation set. Large-noise reconstruction
is still poor. Ordinary clustering with stars is substantially worse on the clean room.

The tested weighting rules are mixed. They can improve individual cases but do not
consistently improve both OSPA and F1 relative to unweighted stars or components.
Keep them as ablations rather than naming a new winner from isolated favorable cases.

## Controlled experiment

One empty rectangle, the same 280 survey positions, AP A and B reconstructed separately.
All peak matches use the frozen empty-room OSPA-selected standard MPUrge:
`alpha=0.7, half_window=2, neighbors=5, unification=False`, including neighborhood
boundary ties. No crossing or triangle changes. No recalibration.

Compare original local connected components with local stars. For a peak node $v$,
each star is $T_v=\{v\}\cup\{u:(u,v)\in E\}$ within the original receiver scope.
Only direct neighbors are included. Identical tracks are removed. Stars overlap;
they are not a partition and do not follow transitive graph paths.

The cluster ending retains radius 0.1 m and support multiplier 2. Each finisher
retains its own selected tolerance, support fraction and duplicate threshold.
The common proposer stays at half-window 2 even for the support-first finisher.
The existing separately tuned support-first reference uses half-window 6. Thus:

- The factorial table isolates grouping/weighting against the same-proposer baseline.
- The full-noise table compares the promoted configuration with the already published
  independently selected method; it is not a pure single-factor comparison for support-first.

The screen covers $\sigma\in\{0,0.1,0.5\}$, both stored evaluation noise realizations
(IDs 2 and 3), one at zero. Promoted unweighted stars additionally cover every usual
noise level. Every variant within a realization sees exactly the same noisy peaks.
OSPA uses $p=1,c=1$ m; F1 uses a 0.5 m gate; matched RMSE excludes unmatched sources.
Assignments never cross APs. OSPA and F1 are averaged over realizations; RMSE pools
matched squared distances and matched counts.

## Exact weighting rules

For accepted matching edge $e=(u,v)$, $\delta_e$ is its MPUrge dissimilarity and
$d_e$ is the distance between its two receiver positions. Define observed scales
$s_\delta=\operatorname{median}_{\delta_e>0}\delta_e$ and
$s_d=\operatorname{median}_{d_e>0}d_e$ over that AP's selected pair graph.
If all costs are zero, set $s_\delta=1$; then every similarity quality is exactly one.

$$q^{\rm sim}_e=\frac1{1+\delta_e/s_\delta},\qquad
q^{\rm dist}_e=\frac1{1+(d_e/s_d)^2},\qquad
q^{\rm both}_e=q^{\rm sim}_eq^{\rm dist}_e.$$

Distance weighting favors shorter receiver edges. This tests one specific distance
heuristic, not every possible use of geometry or long-baseline conditioning.

For retained track $T$, average quality over each node's incident edges whose
other endpoint also belongs to $T$:

$$\widetilde w_v=\frac1{|E_T(v)|}\sum_{e\in E_T(v)}q_e,\qquad
w_v=\frac{\widetilde w_v}{\operatorname{mean}_{u\in T_+}\widetilde w_u},\qquad
\widehat z=\arg\min_z\sum_{v=(i,j)\in T}w_v(\|z-x_i\|_2-r_{ij})^2.$$

Here $x_i$ is receiver position and $r_{ij}$ the observed peak range. Set
$\widetilde w_v=w_v=0$ when $E_T(v)$ is empty, and let $T_+$ contain nodes with
positive weights. Only those nodes enter coordinate estimation; require at least
four noncoplanar receivers. Original track membership is retained for the ordinary
clustering support count. These conventions are part of the tested weighted ablation.
Weights enter candidate estimation only. They do not change pair matching,
component conflict resolution, or either finisher's objective. Uniform uses the
original unweighted estimator exactly. Similarity weights are heuristic qualities,
not calibrated probabilities or measured inverse noise variances.

## Full-noise comparison

Each cell is **OSPA / F1 / matched RMSE (m)**. “Existing” uses the original
independently selected configuration. A missing matched RMSE means no source pair
was within the 0.5 m matching gate; it does not mean zero error.


### Support-first peak-consumption finisher

| Noise σ (m) | Existing components | Promoted unweighted stars |
|---:|---|---|
| 0 | 2.51e-05 / 100.00% / 3.33e-05 | 2.5e-05 / 100.00% / 3.32e-05 |
| 0.01 | 0.0514 / 100.00% / 0.0635 | 0.0404 / 100.00% / 0.0531 |
| 0.02 | 0.1179 / 99.50% / 0.1367 | 0.0792 / 100.00% / 0.0973 |
| 0.05 | 0.3790 / 79.30% / 0.2516 | 0.2189 / 90.20% / 0.1885 |
| 0.1 | 0.6562 / 52.59% / 0.2855 | 0.5874 / 59.87% / 0.2473 |
| 0.2 | 0.8337 / 24.08% / 0.3237 | 0.7805 / 34.42% / 0.3166 |
| 0.5 | 0.9531 / 4.40% / 0.3646 | 0.9565 / 5.56% / 0.3975 |
| 1 | 0.9760 / 1.88% / 0.3301 | 0.9746 / 1.85% / 0.4209 |
| 2 | 0.9885 / 1.29% / 0.4245 | 0.9888 / 0.61% / 0.3130 |
| 5 | 0.9937 / 0.76% / 0.3445 | 0.9985 / 0.00% / — |

### Robust range-refinement and peak-consumption finisher

| Noise σ (m) | Existing components | Promoted unweighted stars |
|---:|---|---|
| 0 | 2.48e-05 / 100.00% / 3.31e-05 | 2.48e-05 / 100.00% / 3.31e-05 |
| 0.01 | 0.0095 / 100.00% / 0.0133 | 0.0080 / 100.00% / 0.0114 |
| 0.02 | 0.0362 / 99.50% / 0.0346 | 0.0200 / 100.00% / 0.0289 |
| 0.05 | 0.4810 / 71.79% / 0.1709 | 0.3492 / 81.97% / 0.0925 |
| 0.1 | 0.7833 / 37.56% / 0.2553 | 0.7089 / 51.48% / 0.2202 |
| 0.2 | 0.9046 / 16.34% / 0.3092 | 0.8594 / 25.72% / 0.2885 |
| 0.5 | 0.9774 / 2.47% / 0.4234 | 0.9627 / 6.62% / 0.3903 |
| 1 | 0.9863 / 0.99% / 0.3794 | 0.9863 / 0.68% / 0.4021 |
| 2 | 0.9951 / 0.32% / 0.4860 | 0.9886 / 1.28% / 0.4003 |
| 5 | 0.9961 / 0.00% / — | 0.9983 / 0.00% / — |

## All grouping/weighting ablations

Cells: OSPA / F1 / matched RMSE (m). No unsuccessful arm is omitted.


### Original candidate clustering and support filter

| Grouping / residual weights | σ = 0 | σ = 0.1 m | σ = 0.5 m |
|---|---|---|---|
| components / uniform | 0.3667 / 69.09% / 0.0001 | 0.9248 / 7.66% / 0.1656 | 0.9955 / 0.00% / — |
| components / similarity | 0.3761 / 69.72% / 0.0001 | 0.9139 / 9.95% / 0.2172 | 1.0000 / 0.00% / — |
| components / distance | 0.3509 / 69.81% / 0.0001 | 0.9229 / 9.06% / 0.2133 | 1.0000 / 0.00% / — |
| components / both | 0.3667 / 69.72% / 0.0001 | 0.9112 / 9.96% / 0.2143 | 1.0000 / 0.00% / — |
| stars / uniform | 0.9076 / 16.92% / 3.31e-05 | 0.9637 / 7.31% / 0.3087 | 0.9793 / 3.11% / 0.4029 |
| stars / similarity | 0.9022 / 17.83% / 3.31e-05 | 0.9659 / 6.49% / 0.2967 | 0.9773 / 3.71% / 0.4047 |
| stars / distance | 0.9042 / 17.48% / 3.31e-05 | 0.9628 / 7.65% / 0.3182 | 0.9749 / 3.11% / 0.3339 |
| stars / both | 0.8978 / 18.55% / 3.31e-05 | 0.9643 / 6.91% / 0.2695 | 0.9774 / 3.42% / 0.3602 |

### Support-first peak-consumption finisher

| Grouping / residual weights | σ = 0 | σ = 0.1 m | σ = 0.5 m |
|---|---|---|---|
| components / uniform | 0.0707 / 94.12% / 3.3e-05 | 0.7006 / 43.80% / 0.2960 | 0.9658 / 2.51% / 0.4048 |
| components / similarity | 0.0655 / 96.08% / 0.0581 | 0.7039 / 41.40% / 0.3038 | 0.9668 / 3.77% / 0.4030 |
| components / distance | 0.0764 / 94.12% / 3.3e-05 | 0.6986 / 43.80% / 0.3002 | 0.9642 / 3.70% / 0.3916 |
| components / both | 0.0651 / 96.08% / 0.0572 | 0.7120 / 37.09% / 0.2788 | 0.9653 / 3.79% / 0.3728 |
| stars / uniform | 2.5e-05 / 100.00% / 3.32e-05 | 0.5874 / 59.87% / 0.2473 | 0.9565 / 5.56% / 0.3975 |
| stars / similarity | 2.5e-05 / 100.00% / 3.31e-05 | 0.5946 / 56.94% / 0.2366 | 0.9446 / 5.02% / 0.3896 |
| stars / distance | 2.5e-05 / 100.00% / 3.32e-05 | 0.5916 / 57.89% / 0.2419 | 0.9300 / 9.41% / 0.3954 |
| stars / both | 2.49e-05 / 100.00% / 3.31e-05 | 0.5860 / 59.21% / 0.2400 | 0.9323 / 8.15% / 0.3368 |

### Robust range-refinement and peak-consumption finisher

| Grouping / residual weights | σ = 0 | σ = 0.1 m | σ = 0.5 m |
|---|---|---|---|
| components / uniform | 2.48e-05 / 100.00% / 3.31e-05 | 0.7833 / 37.56% / 0.2553 | 0.9774 / 2.47% / 0.4234 |
| components / similarity | 2.48e-05 / 100.00% / 3.31e-05 | 0.7692 / 40.40% / 0.2634 | 0.9810 / 2.47% / 0.4018 |
| components / distance | 2.48e-05 / 100.00% / 3.31e-05 | 0.7806 / 37.67% / 0.2520 | 0.9819 / 2.44% / 0.3645 |
| components / both | 2.48e-05 / 100.00% / 3.31e-05 | 0.7751 / 37.90% / 0.2509 | 0.9823 / 2.10% / 0.3683 |
| stars / uniform | 2.48e-05 / 100.00% / 3.31e-05 | 0.7089 / 51.48% / 0.2202 | 0.9627 / 6.62% / 0.3903 |
| stars / similarity | 2.48e-05 / 100.00% / 3.31e-05 | 0.7141 / 50.25% / 0.2120 | 0.9748 / 3.25% / 0.3546 |
| stars / distance | 2.48e-05 / 100.00% / 3.31e-05 | 0.7089 / 51.73% / 0.2191 | 0.9609 / 5.83% / 0.3987 |
| stars / both | 2.48e-05 / 100.00% / 3.31e-05 | 0.7056 / 50.67% / 0.2039 | 0.9643 / 4.72% / 0.3686 |

## Reproduce

```
python scripts/experiment_star_weights.py --levels 0 0.1 0.5
python scripts/experiment_star_weights.py --output runs/star-weight-validation-v1 --levels 0.01 0.02 0.05 0.2 1 2 5 --variants stars:uniform --finishers consumption robust
python scripts/report_star_weights.py
python scripts/promote_star_results.py
```

The experiment resumes saved candidate batches and outcomes. Detailed source arrays
are in the local run directories. [Per-realization metrics](metrics.csv) include
all measured screen and follow-up conditions. The viewer's promoted source maps
come directly from those saved arrays, without a numerical rerun.
