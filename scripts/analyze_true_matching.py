"""Generate the frozen empty-room truth-matching viewer analysis; no reconstruction."""
import argparse
from pathlib import Path

from indoor_vt.execution.true_matching import TrueMatchingAnalysis
from indoor_vt.execution.viewer import MultiStudyView

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, default=Path("data/redesign/rectangle-20261025"))
parser.add_argument("--run", type=Path, default=Path("runs/redesign/rectangle-clean-selection-v4"))
args = parser.parse_args()
analysis = TrueMatchingAnalysis(MultiStudyView(args.data, args.run))
analysis.compute()
print(analysis.path.resolve(), flush=True)
