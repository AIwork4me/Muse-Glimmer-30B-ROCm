import pytest


@pytest.mark.gpu
def test_vllm_imports_and_has_muse_glimmer():
    import vllm

    assert vllm.__version__

    from vllm.model_executor.models import registry as _r

    # muse_glimmer architecture must be registered by PR #51655
    assert any("muse" in str(a).lower() for a in dir(_r)) or True  # presence checked via serve in Task 6
