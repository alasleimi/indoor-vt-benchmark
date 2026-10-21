# Source audit and reproducibility

`upstream_sources.json` hashes the local papers, manuscript code and historical
evidence inspected during implementation. It records evidence, not an assertion
that all adapters exactly reproduce author code. Those external sources are not
copied into this repository, apart from the expressly identified NIST helper.

## NIST and the receiver

The tested NIST Q-D checkout is revision
`1ccf0cb61c1741cc3471db8ec0373c5383da58be`, with MATLAB R2026a Update 4.
Simulation receipts hash its MATLAB source tree and the compatibility overlay.
Each scenario preserves AMF geometry, materials, node positions, seed, all
attempts, executed MATLAB script, effective configuration MAT/JSON, log, native
channel JSON and visualizer path geometry. Failed attempts are retained.

`matlab_compat/xmlreader.m` retains the NIST redistribution notice and marks its
changes: scalar loop bounds and handling a volume containing one triangle,
needed by R2026a and triangulated concave floors. `wrapTo2Pi.m` supplies the
angle-wrapping primitive without the optional Mapping Toolbox. The upstream
checkout is unchanged. These compatibility changes do not replace propagation.

The receiver equations follow the local SAPP implementation. Fifteen frozen
reference cases compare its output with this frontend after disabling the old
peak-count cap. The native direct track has neither that cap nor the receiver's
25 m processing window. The receiver retains its 2 GHz bandwidth, eight-element
cube, coherent path combination, noise/peak thresholds and parabolic refinement.
Its SNR reference is the strongest clean sampled average-array power. It is not
the average power over the entire time window.

Native path gains, phases, AoA/AoD and delays are archived unchanged. NIST's JSON
writer casts fields to single precision; preserving its bytes does not restore
precision it did not emit. `configs/conventions.json` records polar elevation,
units, gain conventions, array pose and SAPP's two speed-of-light constants.

## The 2019 comparator: disclosed assumption

The [TUM publication record](https://portal.fis.tum.de/en/publications/3d-reconstruction-of-indoor-geometry-using-electromagnetic-multip/)
identifies Zayets, Bourguiba and Steinbach's
[2019 ICC paper](https://doi.org/10.1109/ICC.2019.8761636).
The [author's 2022 dissertation](https://mediatum.ub.tum.de/doc/1647114/document.pdf),
section 6.2.3, repeats its cosine clustering/X-means/Ward description. Its
bibliography identifies Bourguiba's 2018 bachelor's thesis. Searches did not
locate that thesis's code or a public author implementation. Majdi's local
bachelor's thesis also flags the unclear one-dimensional clustering.

Cosine similarity between positive scalar ranges is identically one. Neither
the inspected paper nor dissertation resolves the non-degenerate feature
representation. The user authorized a reasonable, explicitly recorded
assumption if author code could not be found.

`zayets2019_assumption` therefore embeds range r as the unit vector proportional
to `[r / scale, 1]`, splits spherical k-means clusters while BIC improves, and
uses constrained Ward grouping of range/count/span features. Range-consensus
support, gamma-times-mean filtering and mean-coordinate cluster consolidation
complete the adapter. Scale, grouping threshold and support parameters are
calibratable; no true cluster/source count is supplied. This is an adaptation
with an assumption, **not an exact reproduction or author implementation**.
Its pilot execution establishes runnability only, not competitive performance.

The 2018 SAPP-ablation adapter is separate and does not use this assumption.

## Source-reference audit

Truth is constructed independently from the survey-time room geometry, including
real APs and physically valid first/second-order images from walls, floor and
ceiling. Finite-wall visibility intervals retain sources even when the actual
survey grid never observes them. Duplicate images keep their reflection
sequences. The saved contact sheet supports visual review; reflection witnesses
and exact coordinates are retained in the evaluator-only source catalogue.

Native survey delay/AoA provides an independent geometry check against that
catalogue. This audit may use privileged data; estimator code may not. Its
0.01 m tolerance checks NIST numeric rounding and is unrelated to the 0.5 m F1
gate. Query-only obstacle sources never alter the survey-time reference bag.

## Determinism and artifacts

Root seeds are pilot 20261020, development 20261021 and reserved publication
20261022. Semantic keys are canonical JSON; the first eight SHA256 bytes in
little-endian order initialize NumPy PCG64. MATLAB uses the derived seed modulo
2^32 with `twister`. Scheduling and directory order do not choose realizations.
The same acquired A/B channels form A+B. Noise draws are independent across
levels and shared across every compared method.

Design manifests preserve realized coordinates and all sampler settings.
Completed native receipts validate input/source/output hashes on reuse.
Frontend artifacts have versioned identities; activating a new version retains
the old files and manifest. Fit caches include actual input bytes, parameters,
code hash and semantic method seed. Calibration freezes include code, design,
observation and reference-source hashes. Adding a freeze file does not change
the implementation hash. Dependencies are pinned in `uv.lock`.

Within-platform replay is tested where reported in `execution.md`. Cross-platform
bitwise identity is not promised: MATLAB/BLAS/compiler differences can affect
floating-point calculations. Native settings and preserved outputs make those
differences auditable.
