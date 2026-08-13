import subprocess

import pytest


@pytest.mark.parametrize(
    ("actual", "expected_ok"),
    [
        ("6.16.8", False),
        ("6.16.9", True),
        ("6.16.9-oem", True),
        ("6.17.0", True),
        ("7.0.0", True),
    ],
)
def test_kernel_version_floor(actual, expected_ok):
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source scripts/lib/version.sh; version_at_least "$1" 6.16.9',
            "version-test",
            actual,
        ],
        check=False,
    )
    assert (result.returncode == 0) is expected_ok


def test_environment_check_reads_manifest_floor():
    source = open("scripts/00-check-env.sh").read()
    assert "host.minimum_kernel" in source
    assert "version_at_least" in source
