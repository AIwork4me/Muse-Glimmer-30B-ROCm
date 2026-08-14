"""Keep high-value public claims aligned with the reference manifests."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_claim_consistency import validate_forward_tracks, validate_hardware_matrix


def test_claim_consistency():
    result = subprocess.run(
        [sys.executable, "scripts/check_claim_consistency.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_future_validated_hardware_uses_generic_evidence_rule(tmp_path: Path):
    primary_evidence = "configs/primary.json"
    primary_path = tmp_path / primary_evidence
    primary_path.parent.mkdir(parents=True)
    primary_path.write_text("{}\n", encoding="utf-8")

    future_evidence = "hardware-validation/future/manifest.json"
    future_path = tmp_path / future_evidence
    future_path.parent.mkdir(parents=True)
    future_path.write_text(
        json.dumps(
            {
                "hardware": {"gpu": "Future Radeon", "gfx_target": "gfx9999"},
                "result": {"status": "pass"},
            }
        ),
        encoding="utf-8",
    )
    hardware = [
        {
            "hardware": "Primary APU",
            "gpu_arch": "gfx0000",
            "status": "validated",
            "evidence": primary_evidence,
        },
        {
            "hardware": "Future Radeon",
            "gpu_arch": "gfx9999",
            "status": "validated",
            "evidence": future_evidence,
        },
    ]

    validate_hardware_matrix(
        hardware,
        {"hardware": "AMD Primary APU", "gpu_arch": "gfx0000"},
        primary_evidence,
        tmp_path,
    )


def test_future_hardware_claim_rejects_failed_evidence(tmp_path: Path):
    primary_evidence = "primary.json"
    (tmp_path / primary_evidence).write_text("{}\n", encoding="utf-8")
    future_evidence = "future.json"
    (tmp_path / future_evidence).write_text(
        json.dumps(
            {
                "hardware": {"gpu": "Future Radeon", "gfx_target": "gfx9999"},
                "result": {"status": "fail"},
            }
        ),
        encoding="utf-8",
    )
    hardware = [
        {
            "hardware": "Primary APU",
            "gpu_arch": "gfx0000",
            "status": "validated",
            "evidence": primary_evidence,
        },
        {
            "hardware": "Future Radeon",
            "gpu_arch": "gfx9999",
            "status": "validated",
            "evidence": future_evidence,
        },
    ]

    with pytest.raises(ValueError, match="does not record a passing result"):
        validate_hardware_matrix(
            hardware,
            {"hardware": "Primary APU", "gpu_arch": "gfx0000"},
            primary_evidence,
            tmp_path,
        )


def test_future_forward_track_needs_no_checker_business_logic_change(tmp_path: Path):
    evidence = tmp_path / "future-evidence"
    evidence.mkdir()
    tracks = validate_forward_tracks(
        [
            {
                "name": "Future backend",
                "status": "project-validated",
                "scope": "synthetic accepted scope",
                "evidence": "future-evidence",
            },
            {
                "name": "Future pending backend",
                "status": "pending",
                "scope": "not yet validated",
                "evidence": None,
            },
        ],
        tmp_path,
    )
    assert set(tracks) == {"Future backend", "Future pending backend"}
