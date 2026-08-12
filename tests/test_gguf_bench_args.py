import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from gguf_bench_args import build_server_args, build_client_args

W = "models/muse-glimmer-30B-kquant-17gb.gguf"
DF = "models/dflash-kquant.gguf"
MMP = "models/mmproj-kquant.gguf"


def _cell(**kw):
    base = dict(weight=W, dflash=False, vision=False, np=1,
                per_slot_ctx=8192, study="study1", seed=0,
                weights_dir="models", dflash_path=DF, mmproj_path=MMP)
    base.update(kw)
    return base


def test_study1_greedy_baseline_flags():
    a = build_server_args(_cell())
    s = " ".join(a)
    assert "--temp 0" in s
    assert "--seed 0" in s
    assert "-ngl 999" in s
    assert "-np 1" in s
    assert "-c 8192" in s            # np(1) * per_slot_ctx(8192)
    assert "--jinja" in s
    assert "-md" not in s            # baseline, no draft


def test_dflash_adds_draft_model():
    a = build_server_args(_cell(dflash=True))
    s = " ".join(a)
    assert "-md models/dflash-kquant.gguf" in s
    assert "-ngld 99" in s
    # Spec-decoding MUST be explicitly enabled: `-md <dflash>` alone loads the
    # draft model but `--spec-type` defaults to `none`, so 0 drafts are proposed
    # (regression observed on gfx1151: 10.5 tok/s, 1.00x speedup).
    assert "--spec-type draft-dflash" in s
    assert "--spec-draft-n-max 16" in s   # measured sweet spot (= block_size)


def test_dflash_spec_flags_absent_on_baseline():
    """Baseline cells must NOT carry any spec-decoding flags."""
    a = build_server_args(_cell(dflash=False))
    s = " ".join(a)
    assert "--spec-type" not in s
    assert "--spec-draft-n-max" not in s
    assert "-md" not in s


def test_study2_load_uses_temp_1_and_scales_ctx():
    a = build_server_args(_cell(study="study2", np=16))
    s = " ".join(a)
    assert "--temp 1.0" in s
    assert "--top-p 0.95" in s
    assert "--top-k 64" in s
    assert "-c 131072" in s          # 16 * 8192
    assert "-np 16" in s


def test_vision_adds_mmproj():
    a = build_server_args(_cell(vision=True))
    assert "--mmproj" in a
    mmproj_idx = a.index("--mmproj")
    assert a[mmproj_idx + 1] == "models/mmproj-kquant.gguf"


def test_client_args_match_study():
    c = build_client_args(_cell(study="study1"))
    assert c["endpoint"] == "chat"
    assert c["temp"] == 0
    assert c["max_tokens"] == 256
    c2 = build_client_args(_cell(study="study2"))
    assert c2["temp"] == 1.0 and c2["max_tokens"] == 512 and c2["reasoning_strength"] == "high"
