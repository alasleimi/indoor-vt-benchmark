"""Generate commit-pinned implementation links and the first method walkthrough."""

import ast
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/alasleimi/indoor-vt-benchmark"


def build(root=ROOT):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()

    def code(module, symbol, label):
        path = f"src/indoor_vt/{module}.py"
        tree = ast.parse((root / path).read_text(encoding="utf8"))
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == symbol
        )
        return {
            "label": label,
            "path": path,
            "symbol": symbol,
            "line": node.lineno,
            "url": f"{REPO}/blob/{revision}/{path}#L{node.lineno}",
        }

    methods = {}
    matcher = code("matching", "PatternMatcher", "Peak-pattern matcher")
    recon = code("reconstruction", "MPUrgeReconstructor", "Reconstruction stages")
    routing = code("registry", "reconstruction_default", "Variant configuration")
    papers = {
        "mpurge": {
            "title": "MPUrge (2025)",
            "url": "https://doi.org/10.1109/IWCMC65282.2025.11059597",
        },
        "mpurge_map": {
            "title": "MPUrge-MAP (2026)",
            "url": "https://doi.org/10.1109/IWCMC69287.2026.11580087",
        },
        "mca": {
            "title": "Multipath Component Analysis (2017)",
            "url": "https://doi.org/10.1109/IPIN.2017.8115943",
        },
        "zayets2018": {
            "title": "Interpolation and Extrapolation of Multipath Fingerprints Using Virtual Transmitter Placement (ICC 2018)",
            "url": "https://doi.org/10.1109/ICC.2018.8422206",
        },
        "zayets2019_assumption": {
            "title": "3D Reconstruction of Indoor Geometry Using Electromagnetic Multipath Fingerprints (ICC 2019)",
            "url": "https://doi.org/10.1109/ICC.2019.8761636",
        },
        "lea": {
            "title": "SMART-LEA (2026)",
            "url": "https://doi.org/10.1109/IWCMC69287.2026.11579893",
        },
    }
    for name in ("mpurge_map", "mpurge", "mca", "chamfer"):
        methods[f"coarse/{name}"] = {
            "code": [
                code("localization", "score_pair", "Fingerprint scoring"),
                code(
                    "localization",
                    "predict_from_scores",
                    "Reference selection and position estimate",
                ),
            ],
            "papers": [papers[name]] if name in papers else [],
            "steps": [],
        }
        if name in ("mpurge", "mpurge_map"):
            methods[f"coarse/{name}"]["code"].append(matcher)
    for base in ("mpurge", "mpurge_star", "mpurge_triangle", "mpurge_diffassign", "pba"):
        for scope in ["local"] if base == "pba" else ["local", "full"]:
            links = [recon, routing, matcher]
            if base == "mpurge_diffassign":
                links.append(code("diffassign", "DiffAssign", "Differentiable refinement"))
            methods[f"vt/{base}_{scope}"] = {
                "code": links,
                "papers": [papers["mpurge"]] if base.startswith("mpurge") else [],
                "steps": [],
            }
    for name in ("ransac", "zayets2018"):
        methods[f"vt/{name}"] = {
            "code": [
                code("consensus", "ConsensusReconstructor", "Source estimator"),
                code("consensus", "score", "Survey-wide consensus score"),
            ],
            "papers": [papers[name]] if name in papers else [],
            "steps": [],
        }
    methods["vt/zayets2019_assumption"] = {
        "code": [code("zayets2019", "Zayets2019", "Disclosed 2019 adapter")],
        "papers": [papers["zayets2019_assumption"]],
        "steps": [],
    }
    methods["lea"] = {
        "code": [code("localization", "SmartLEA", "SMART-LEA refinement")],
        "papers": [papers["lea"]],
        "steps": [],
    }
    methods["pipeline"] = {
        "code": [code("pipeline", "Pipeline", "Pipeline composition")],
        "papers": [],
        "steps": [],
    }

    note = methods["vt/zayets2018"]
    note["code"].extend(
        [
            code("consensus", "matched_tracks", "Delay clusters, grouping and peak tracks"),
            code("fitting", "fit_source", "Signed 3D range fitting"),
        ]
    )
    note["title"] = "Zayets–Steinbach 2018: our SAPP reconstruction adapter"
    note["summary"] = (
        "This entry uses the 2018 paper’s delay-cluster matching to propose sources, followed by the source-bank selection from our SAPP ablation. SAPP is our work. This benchmark extends that adapter to signed 3D fitting and removes its 20-source cap. It does not run the paper’s complete fingerprint synthesis/localization pipeline."
    )
    note["inputs"] = (
        "For each physical AP channel separately: N known survey coordinates sᵢ ∈ ℝ³ and one sorted range list rᵢ per coordinate. Ranges are delay × propagation speed. Empty lists are retained. No wall count, floor plan, AP/VT ground-truth coordinates, path identities, angles or query labels are passed to the estimator. A and B are fitted independently; A+B uses their two labeled source banks."
    )
    note["legend"] = {
        "Paper 2018": "Stated in the original paper (Section III, Algorithms 1–4).",
        "Our interpretation": "A concrete implementation choice for a step the paper does not completely specify.",
        "Our SAPP": "Source-selection machinery from our SAPP source-estimator ablation.",
        "Our benchmark": "Changes or explicit choices in this signed-3D benchmark.",
    }
    note["steps"] = [
        {
            "title": "Construct one neighborhood around every surveyed coordinate",
            "tags": ["Our SAPP", "Our benchmark"],
            "text": "For each survey index i = 0,…,N−1, collect Sᵢ = {j : ‖sⱼ−sᵢ‖₂ ≤ R}, using 3D Euclidean distance and R = {{neighborhood_radius_m}} m. Include i itself. Both height layers may contribute. Sort each neighborhood’s indices, remove identical index sets, then process the remaining neighborhoods in lexicographic order. Distinct neighborhoods can overlap. The 2018 interpolation procedure selects references around a target interpolation location; centering these neighborhoods on every measured survey coordinate is our SAPP source-discovery adaptation.",
            "code_symbol": "ConsensusReconstructor",
        },
        {
            "title": "Split every range list in that neighborhood into consecutive clusters",
            "tags": ["Paper 2018", "Our SAPP"],
            "text": "For each j ∈ Sᵢ with a nonempty sorted range list, start a cluster at its first peak. Append the next peak only if its gap from the previous peak is strictly less than g = {{cluster_gap_m}} m; otherwise start a new cluster. Thus a gap equal to g splits the cluster. This is consecutive-gap clustering (Algorithm 3), not a maximum cluster-width constraint. Several different VTs can contribute peaks to one cluster. The 0.6 m preset comes from our SAPP calibrated adapter, not the paper’s 0.10 m interpolation setting.",
            "code_symbol": "matched_tracks",
        },
        {
            "title": "Calculate a cost for every pair of clusters from different references",
            "tags": ["Paper 2018"],
            "text": "Represent cluster C by f(C) = (its first range, its number of peaks, its last minus first range). For each pair C,D belonging to different survey indices in this neighborhood, compute cost(C,D) = 0.5|first(C)−first(D)| + 0.5|count(C)−count(D)| + 0.5|span(C)−span(D)|. This is Algorithm 4 with α = 0.5, fixed in this implementation. Keep pairs with cost ≤ {{group_threshold}}. The cost combines a count and range features; its threshold is not a spatial distance.",
            "code_symbol": "matched_tracks",
        },
        {
            "title": "Greedily group compatible clusters without duplicate survey owners",
            "tags": ["Paper 2018", "Our interpretation"],
            "text": "Initially each cluster is its own group. Sort the retained pairs by (cost, first cluster index, second cluster index), from smallest to largest. For each pair, merge its two current groups only if the groups are distinct and their sets of survey indices are disjoint. Therefore a group contains at most one cluster per survey position. A merge is driven by the processed pair; all cross-pairs between the two groups need not pass the threshold. The paper states low-cost grouping and the owner constraint; this ordered union procedure and deterministic tie handling are our interpretation.",
            "code_symbol": "matched_tracks",
        },
        {
            "title": "Turn each cluster group into individual peak tracks",
            "tags": ["Paper 2018", "Our interpretation"],
            "text": "Let L be the largest cluster size in the group. Compute an L-element target by averaging, element by element, all clusters of length L. A length-L cluster contributes its first peak to slot 0, second peak to slot 1, and so on. For each shorter cluster, solve a minimum-total-absolute-range-difference assignment to distinct target slots with linear_sum_assignment; some slots remain empty. Each slot is a track of (survey index, peak index) pairs. Positional matching and comparison to the mean full-sized cluster follow Section III-D; the exact one-to-one assignment solver is our interpretation.",
            "code_symbol": "matched_tracks",
        },
        {
            "title": "Deduplicate tracks and fit signed 3D candidate positions",
            "tags": ["Paper 2018", "Our benchmark"],
            "text": "Keep tracks containing at least four reference owners. Translate neighborhood-local indices to global survey indices, deduplicate identical ordered membership tuples across neighborhoods, and process them in sorted order. Reject a track unless its centered survey-coordinate matrix has rank 3. Fit a candidate v by minimizing Σⱼ(‖v−sⱼ‖₂−rⱼ)², initialized by linearized multilateration and solved with unweighted Levenberg–Marquardt (at most 100 function evaluations). Reject failed or nonfinite solutions. There are no room bounds or positive-height constraints. The paper uses trilateration/nonlinear least squares; our benchmark specifies this signed-3D solver, the four-reference/rank checks and rejection rules. No mean-range fallback produces a VT coordinate.",
            "code_symbol": "fit_source",
        },
        {
            "title": "Score each candidate against all currently available survey peaks",
            "tags": ["Our SAPP"],
            "text": "Initialize available peaks to all N survey range lists, not only the neighborhood that generated a candidate. For each active candidate v, predict dᵢ(v) = ‖v−sᵢ‖₂. Set eᵢ(v) = min over available peaks k of |dᵢ(v)−rᵢₖ| (infinity for an empty list). With σ = {{inlier_scale_m}} m, support H(v) counts survey indices whose eᵢ ≤ σ. Score(v) = H(v) + Σᵢ exp[−0.5(eᵢ/σ)²]/(N+1). The fractional term is less than one, so an extra supporting reference always outweighs it; it breaks equal-support comparisons using residual agreement.",
            "code_symbol": "score",
        },
        {
            "title": "Select the best supported candidate and reject duplicate range surfaces",
            "tags": ["Our SAPP"],
            "text": "Set minimum support M = max(5, ceil({{support_fraction}} × N)); N includes references with empty range lists. Repeatedly choose the active candidate with the largest (score, support), using the first candidate in deterministic order for exact ties. Stop if its support is below M. Mark that candidate inactive. Reject it if its predicted distance vector differs from an already accepted source by less than δ = {{surface_separation_m}} m RMS across all N references. This comparison is between predicted range surfaces, not Euclidean distance between source coordinates. This adapter does not refit the candidate after consensus scoring; the robust-refit branch in the same file belongs to RANSAC only.",
            "code_symbol": "ConsensusReconstructor",
        },
        {
            "title": "Consume supporting peaks, suppress duplicate proposals, and rescore",
            "tags": ["Our SAPP"],
            "text": "For an accepted candidate, at every supporting reference remove exactly its nearest currently available peak (set that slot to infinity). Other peaks stay available. Mark every candidate whose predicted range surface is within δ RMS of the accepted surface inactive. Recompute scores of the remaining active candidates using the remaining peaks and repeat. Peak removal means evidence already assigned to one accepted source cannot support another later source.",
            "code_symbol": "ConsensusReconstructor",
        },
        {
            "title": "Return the source bank when remaining evidence is insufficient",
            "tags": ["Our SAPP", "Our benchmark"],
            "text": "Stop when no active candidate remains or the best candidate fails minimum support. Return the accepted signed (x,y,z) coordinates for this AP channel. Our SAPP ablation retained at most 20 sources; this benchmark removes that cap. No true source count is supplied. The output count is controlled by candidate availability, consensus support, evidence consumption and duplicate suppression.",
            "code_symbol": "ConsensusReconstructor",
        },
    ]
    note["parameters"] = [
        [
            "neighborhood_radius_m",
            "Neighborhood radius R",
            "m",
            "Our SAPP preset; paper uses a 1 m interpolation neighborhood",
        ],
        ["cluster_gap_m", "Consecutive range gap g", "m", "Our SAPP calibrated gap"],
        ["group_threshold", "Cluster-pair cost threshold", "", "Recorded adapter preset"],
        ["inlier_scale_m", "Consensus tolerance σ", "m", "Our SAPP"],
        [
            "support_fraction",
            "Minimum support fraction",
            "",
            "Our SAPP (plus an absolute floor of five)",
        ],
        ["surface_separation_m", "Range-surface RMS separation δ", "m", "Our SAPP"],
    ]
    note["limitations"] = (
        "Fixed slot ordering can misassociate crossing paths. Gap thresholds can split or join clusters when noise changes. A 3D neighborhood must span enough noncoplanar references. Global support filtering can reject sources visible only in a small area. The current results are initial calibration presets; main hyperparameter search is unfinished. The trials and retained_per_batch fields belong to RANSAC and do not affect this adapter; batch only partitions its scoring work."
    )
    catalog = {
        "schema": "indoor-vt.method-catalog.v1",
        "repository": REPO,
        "code_revision": revision,
        "methods": methods,
    }
    destination = root / "web/results-explorer/method-catalog.json"
    destination.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
    print(f"Built method catalog: {len(methods)} entries, code revision {revision[:8]}")
    return catalog


if __name__ == "__main__":
    build()
