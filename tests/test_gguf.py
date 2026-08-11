import pytest

pytestmark = pytest.mark.gpu  # the full script needs gfx1151 to *run*; the
                              # checks below are static, so run with -m gpu to
                              # exercise them without a build.


def test_gguf_script_builds_llamacpp_and_names_target():
    src = open("scripts/gguf-quickstart.sh").read()
    # HIP build for gfx1151
    assert "-DGGML_HIP=ON" in src
    assert "AMDGPU_TARGETS=gfx1151" in src
    # Meta's official GGUF repo (not the plan's guessed Q4_K_M naming)
    assert "meta-models/Muse-Glimmer-30B-GGUF" in src
    assert "Muse-Glimmer-30B" in src
    # the real calibrated quant filenames
    assert "muse-glimmer-30B-kquant" in src
