"""Syntax checks for committed JSON and YAML control files."""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_all_non_evidence_json_parses():
    paths = [
        *sorted((ROOT / "configs").glob("*.json")),
        *sorted((ROOT / "schemas").glob("*.json")),
        *sorted((ROOT / "scripts/prompt-sets").glob("*.json")),
    ]
    assert paths
    for path in paths:
        json.loads(path.read_text(encoding="utf-8"))


def test_control_yaml_parses():
    paths = [ROOT / "CITATION.cff", *sorted((ROOT / ".github").rglob("*.yml"))]
    for path in paths:
        assert yaml.compose(path.read_text(encoding="utf-8")) is not None, path
