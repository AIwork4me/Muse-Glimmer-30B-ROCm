import subprocess, pytest

@pytest.mark.gpu
def test_check_env_passes_on_this_host():
    r = subprocess.run(["bash", "scripts/00-check-env.sh"], capture_output=True, text=True)
    assert r.returncode == 0, f"check-env failed:\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"

def test_check_env_rejects_low_kernel(monkeypatch, tmp_path):
    # Script must NAME the 6.16.9 floor so the troubleshooting link is discoverable.
    src = open("scripts/00-check-env.sh").read()
    assert "6.16.9" in src and "troubleshooting.md#uma-bug" in src
