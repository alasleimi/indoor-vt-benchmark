"""Exercise the viewer's actual strict-win selection code under Node.js."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_browser_pareto_selection():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for browser selection tests")
    script = Path(__file__).with_name("pareto.test.mjs")
    result = subprocess.run([node, "--test", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
