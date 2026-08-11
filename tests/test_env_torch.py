import pytest

@pytest.mark.gpu
def test_torch_sees_gfx1151():
    import torch
    assert torch.cuda.is_available(), "HIP device not visible to torch"
    name = torch.cuda.get_device_name(0)
    assert "gfx1151" in name or "Radeon" in name, f"unexpected device: {name}"
    assert torch.version.hip is not None, "torch is not a ROCm build"
