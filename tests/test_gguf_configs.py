import os
import re

ROOT = os.path.join(os.path.dirname(__file__), "..")
CONF = os.path.join(ROOT, "configs", "gguf-bench")


def _load(name):
    txt = open(os.path.join(CONF, name)).read()
    d = {}
    for line in txt.splitlines():
        m = re.match(r'^([A-Z_]+)=(.*)$', line)
        if m:
            value = m.group(2).strip()
            # Strip inline shell comments before quote handling
            value = value.split('#', 1)[0].rstrip()
            # Only strip surrounding quotes if the entire value is wrapped in them
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            d[m.group(1)] = value
    return d


def test_study1_greedy_single_weight_set():
    d = _load("study1.conf")
    assert d["STUDY"] == "study1"
    assert d["TEMP"] == "0"
    assert d["MAX_TOKENS"] == "256"
    assert d["PER_SLOT_CTX"] == "8192"
    assert "17gb" in d["WEIGHTS"] and "dynamic" in d["WEIGHTS"]
    assert d["NPS"] == "1"
    assert d["SEED"] == "0"
    assert d["REPS"] == "3"


def test_study2_load_concurrency():
    d = _load("study2.conf")
    assert d["TEMP"] == "1.0"
    assert d["MAX_TOKENS"] == "512"
    assert d["REPS"] == "5"
    assert d["NPS"] == "1 4 16"
    assert d["SEED"] == "42"


def test_study3_vision_has_mmproj_flag():
    d = _load("study3.conf")
    assert d["VISION"] == "1"
    assert d["NPS"] == "1 4"
    assert d["TEMP"] == "1.0"
    assert d["MAX_TOKENS"] == "512"
    assert d["SEED"] == "42"
    assert d["REPS"] == "3"
