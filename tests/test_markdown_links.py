"""Check local links in public release-readiness documentation."""

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "SECURITY.md",
    ROOT / "CHANGELOG.md",
    ROOT / "handoff.md",
    ROOT / "docs/RELEASE_CHECKLIST.md",
    ROOT / "docs/hardware-validation.md",
    ROOT / "docs/results/METHODOLOGY.md",
    ROOT / "docs/results/benchmark.md",
    ROOT / "schemas/README.md",
]
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")


def test_local_markdown_link_targets_exist():
    missing = []
    for document in DOCS:
        for raw_target in LINK.findall(document.read_text(encoding="utf-8")):
            target = raw_target.strip().strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_text = unquote(target.split("#", 1)[0])
            if not path_text:
                continue
            resolved = (document.parent / path_text).resolve()
            if not resolved.exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not missing, "\n".join(missing)
