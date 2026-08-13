"""Keep high-value public claims aligned with the reference manifests."""

import subprocess
import sys


def test_claim_consistency():
    result = subprocess.run(
        [sys.executable, "scripts/check_claim_consistency.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
