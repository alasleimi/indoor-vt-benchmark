"""Write all ablation outcomes, including unsuccessful alternatives, to a reviewable report."""
import csv
import gzip
import json
from pathlib import Path
import numpy as np

from indoor_vt.acquisition.records import read_json


def aggregate(rows):
    ms = [r['metrics_by_ap']['A+B'] for r in rows]
    tp = sum(m['tp'] for m in ms)
    return {'ospa': float(np.mean([m['ospa'] for m in ms])), 'f1': float(np.mean([m['f1'] for m in ms])),
            'matched_rmse_m': float(np.sqrt(sum(m['matched_squared_error_m2'] for m in ms)/tp)) if tp else None}


def cell(m):
    def number(value):
        return f'{value:.3g}' if value < .00005 else f'{value:.4f}'
    return number(m['ospa']) + f" / {100*m['f1']:.2f}% / " + (number(m['matched_rmse_m']) if m['matched_rmse_m'] is not None else '—')


def main():
    screen = read_json('runs/star-weight-ablation-v1/results.json')
    validation = read_json('runs/star-weight-validation-v1/results.json')
    old = json.loads(gzip.decompress(Path('published/redesign/overview.json.gz').read_bytes()))
    report = Path('docs/star-weight-ablation')
    report.mkdir(parents=True, exist_ok=True)
    text = [r'''# Local stars and edge-weighted source estimation

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
''']
    for finishing in ('consumption', 'robust'):
        title = {'consumption': 'Support-first peak-consumption finisher',
                 'robust': 'Robust range-refinement and peak-consumption finisher'}[finishing]
        text += [f'\n### {title}\n', '| Noise σ (m) | Existing components | Promoted unweighted stars |', '|---:|---|---|']
        for level in old['config']['observations']['noise_levels']:
            rows = [r for r in screen+validation if r['level'] == level and r['grouping'] == 'stars'
                    and r['weight'] == 'uniform' and r['finisher'] == finishing]
            expected = 1 if level == 0 else 2
            assert len(rows) == expected, (finishing, level, len(rows))
            baseline = next(r for r in old['conditions'] if r['method_id'] == 'mpurge_'+finishing+'_ununified'
                            and r['split'] == 'evaluation' and r['level'] == level and 'obstacle_count' not in r)
            text.append(f"| {level:g} | {cell(baseline['source_metrics_by_ap']['A+B'])} | {cell(aggregate(rows))} |")
    text.append('\n## All grouping/weighting ablations\n\nCells: OSPA / F1 / matched RMSE (m). No unsuccessful arm is omitted.\n')
    for finishing in ('cluster', 'consumption', 'robust'):
        title = {'cluster': 'Original candidate clustering and support filter',
                 'consumption': 'Support-first peak-consumption finisher',
                 'robust': 'Robust range-refinement and peak-consumption finisher'}[finishing]
        text += [f'\n### {title}\n', '| Grouping / residual weights | σ = 0 | σ = 0.1 m | σ = 0.5 m |', '|---|---|---|---|']
        for grouping in ('components', 'stars'):
            for weight in ('uniform', 'similarity', 'distance', 'both'):
                cells = []
                for level in (0, .1, .5):
                    rows = [r for r in screen if (r['grouping'], r['weight'], r['finisher'], r['level']) == (grouping, weight, finishing, level)]
                    assert len(rows) == (1 if level == 0 else 2), (grouping, weight, finishing, level)
                    cells.append(cell(aggregate(rows)))
                text.append('| ' + grouping+' / '+weight+' | '+' | '.join(cells)+' |')
    text.append('''
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
''')
    (report/'README.md').write_text('\n'.join(text), encoding='utf-8')
    with (report/'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['phase','sigma_m','realization','grouping','weights','finisher','ap','ospa','f1','matched_rmse_m','tp','fp','fn'])
        for phase, rows in (('screen', screen), ('followup', validation)):
            for row in rows:
                for ap, m in row['metrics_by_ap'].items():
                    writer.writerow([phase,row['level'],row['repetition'],row['grouping'],row['weight'],row['finisher'],ap,
                                     *[m[k] for k in ('ospa','f1','matched_rmse_m','tp','fp','fn')]])
    print(report/'README.md')


if __name__ == '__main__':
    main()
