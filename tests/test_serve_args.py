"""Static checks on the serve config — encodes the gfx1151 adaptations and the
explicit NON-flags. No GPU/server needed (CI-safe)."""


def test_serve_args_encode_the_adaptations():
    args = open("configs/serve-args.conf").read()
    # must-haves
    assert "--dtype bfloat16" in args
    assert "--attention-backend TRITON_ATTN" in args  # validated on gfx1151; FLASH_ATTN asserts "version not detected"
    assert "--tensor-parallel-size 1" in args
    assert "--tool-call-parser muse_glimmer" in args
    assert "--reasoning-parser muse_glimmer" in args
    # explicit NON-flags (the adaptation)
    assert "ROCM_AITER_FA" not in args
    assert "kv-cache-dtype fp8" not in args
    assert "enable-chunked-prefill" not in args
    assert "speculative-config" not in args


def test_env_does_not_enable_aiter():
    env = open("configs/vllm-gfx1151.env").read()
    assert "VLLM_ROCM_USE_AITER=1" not in env
    assert "FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE" in env
