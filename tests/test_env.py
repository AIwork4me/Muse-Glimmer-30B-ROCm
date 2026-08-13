import subprocess, pytest

@pytest.mark.gpu
def test_check_env_passes_on_this_host():
    r = subprocess.run(["bash", "scripts/00-check-env.sh"], capture_output=True, text=True)
    assert r.returncode == 0, f"check-env failed:\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"

def test_check_env_uses_manifest_kernel_floor():
    import json

    src = open("scripts/00-check-env.sh").read()
    stack = json.load(open("configs/validated-stack.json"))
    assert stack["host"]["minimum_kernel"] == "6.16.9"
    assert "host.minimum_kernel" in src
    assert "troubleshooting.md#uma-bug" in src
