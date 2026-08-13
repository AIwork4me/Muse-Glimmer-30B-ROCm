#!/usr/bin/env python3
"""Check high-value public claims against the authoritative manifests."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
HARDWARE_DOC = ROOT / "docs/hardware-validation.md"

STATUS = {
    "validated": "✅ **Validated**",
    "community-validated": "🧪 **Community validated**",
    "planned": "🚧 **Planned**",
    "upstream-recipe": "📘 **Upstream recipe**",
}


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def generated_block(text: str, name: str) -> str:
    start = f"<!-- BEGIN GENERATED: {name} -->"
    end = f"<!-- END GENERATED: {name} -->"
    require(text.count(start) == 1, f"{name}: missing or duplicate start marker")
    require(text.count(end) == 1, f"{name}: missing or duplicate end marker")
    return text.split(start, 1)[1].split(end, 1)[0].strip()


def platform_summary(claims: dict, stack: dict) -> str:
    platform = claims["validated_platform"]
    return (
        f"**Actually validated here:** {platform['hardware']},\n"
        f"`{platform['gpu_arch']}` ({stack['platform']['architecture_family']}). "
        "Radeon dGPUs are planned, not claimed as validated."
    )


def evidence_for(item: dict) -> str:
    if item["status"] == "validated":
        return "Full recorded reference in this repository"
    if item["status"] == "community-validated":
        return "Accepted independent evidence bundle"
    if item["status"] == "upstream-recipe":
        return "Upstream evidence; not revalidated here"
    if item["hardware"] == "Radeon W7900":
        return "No project evidence yet"
    return "Requires a comparable community submission"


def readme_hardware_table(claims: dict) -> str:
    rows = ["| Status | Platform | Evidence |", "|---|---|---|"]
    for item in claims["hardware_matrix"]:
        platform = f"{item['hardware']}, `{item['gpu_arch']}`"
        rows.append(
            f"| {STATUS[item['status']]} | {platform} | {evidence_for(item)} |"
        )
    return "\n".join(rows)


def hardware_doc_table(claims: dict) -> str:
    rows = ["| Platform | Status |", "|---|---|"]
    for item in claims["hardware_matrix"]:
        platform = f"{item['hardware']} (`{item['gpu_arch']}`)"
        rows.append(f"| {platform} | {STATUS[item['status']]} |")
    return "\n".join(rows)


def validation_tracks(claims: dict, forward_manifest: dict) -> str:
    reference = claims["reference_evidence"]
    forward = claims["forward_validation"]
    scope = forward_manifest["scope"]
    platform = forward_manifest["platform"]
    return (
        f"- **Validated historical/reference stack:** ROCm {reference['rocm']} "
        "host toolchain plus the\n"
        "  recorded TheRock runtime. Existing benchmark JSON is immutable evidence.\n"
        f"- **ROCm {forward['rocm'][:4]} gfx1151 track:** the reduced "
        "**GGUF/llama.cpp matrix is\n"
        f"  project-validated** on {platform['hardware'].removeprefix('AMD ')},\n"
        f"  {scope['completed_cells']} of {scope['planned_cells']} planned cells; "
        "the four\n"
        "  `np=16` cells were intentionally deferred). The vLLM/BF16 track remains\n"
        "  pending, so ROCm 7.14 is not presented as a globally validated replacement\n"
        f"  for the historical stack. No ROCm {reference['rocm']} result is relabeled "
        "or overwritten."
    )


def verify_checksums(directory: Path, checksum_file: Path) -> None:
    expected = {}
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        require(Path(name).name == name, f"unsafe checksum path: {name}")
        require(name not in expected, f"duplicate checksum entry: {name}")
        expected[name] = digest
    cells = sorted(directory.glob("cell-*.json"))
    require(set(expected) == {cell.name for cell in cells},
            "ROCm 7.14 checksum inventory disagrees with committed cells")
    for cell in cells:
        actual = hashlib.sha256(cell.read_bytes()).hexdigest()
        require(actual == expected[cell.name], f"checksum mismatch: {cell.name}")


def check() -> None:
    stack = load("configs/validated-stack.json")
    artifacts = load("configs/artifact-manifest.json")
    claims = load("configs/public-claims.json")
    forward_manifest = load(claims["forward_validation"]["manifest"])

    require(claims["validated_stack"] == "configs/validated-stack.json",
            "public claims point to the wrong stack manifest")
    require(claims["artifact_manifest"] == stack["model"]["artifact_manifest"],
            "artifact-manifest path disagrees with validated stack")

    platform = claims["validated_platform"]
    require(platform["status"] == "validated", "reference platform lost validated status")
    require(platform["hardware"] == stack["platform"]["hardware"],
            "validated hardware disagrees with stack manifest")
    require(platform["gpu_arch"] == stack["platform"]["gpu_arch"],
            "validated gfx identity disagrees with stack manifest")

    reference = claims["reference_evidence"]
    require(reference["rocm"] == stack["host"]["rocm_toolchain"],
            "historical ROCm claim disagrees with stack manifest")
    require(reference["matrix"] == stack["benchmark_evidence"]["rocm_7_2_1_matrix"],
            "historical evidence path disagrees with stack manifest")
    require(reference["status"] == stack["status"],
            "historical evidence status disagrees with stack manifest")

    forward = claims["forward_validation"]
    require(forward["rocm"] == "7.14.0" and
            forward["status"] == "partially-validated",
            "ROCm 7.14 must remain scoped rather than globally validated")
    require(stack["benchmark_evidence"]["forward_validation_manifest"] ==
            forward["manifest"],
            "validated stack points to the wrong forward-validation manifest")
    tracks = {track["name"]: track for track in forward["tracks"]}
    require(len(tracks) == 2, "forward validation requires exactly two unique tracks")
    require(tracks["GGUF/llama.cpp"]["status"] == "project-validated" and
            tracks["GGUF/llama.cpp"]["evidence"] == "docs/results/matrix-714/",
            "GGUF track must point to accepted scoped evidence")
    require(tracks["BF16/vLLM"]["status"] == "pending" and
            tracks["BF16/vLLM"]["evidence"] is None,
            "ROCm 7.14 BF16/vLLM must remain pending")

    require(forward_manifest["status"] == "validated-scoped",
            "ROCm 7.14 manifest lost scoped-validation status")
    require(forward_manifest["scope"]["vllm_bf16_status"] == "pending",
            "ROCm 7.14 manifest must keep BF16/vLLM pending")
    require(forward_manifest["platform"]["hardware"] == platform["hardware"] and
            forward_manifest["platform"]["gpu_arch"] == platform["gpu_arch"],
            "ROCm 7.14 platform disagrees with validated hardware identity")
    require(forward_manifest["llama_cpp"]["commit"] == stack["llama_cpp"]["commit"],
            "ROCm 7.14 llama.cpp commit disagrees with the reference stack")
    gguf = artifacts["sets"]["gguf"]
    require(forward_manifest["model"]["repository"] == gguf["repository"] and
            forward_manifest["model"]["revision"] == gguf["revision"],
            "ROCm 7.14 model identity disagrees with the artifact manifest")
    matrix_714 = ROOT / forward_manifest["evidence"]["matrix"]
    cells_714 = sorted(matrix_714.glob("cell-*.json"))
    require(len(cells_714) == forward_manifest["scope"]["completed_cells"],
            "ROCm 7.14 cell count disagrees with its validation scope")
    for cell in cells_714:
        require(json.loads(cell.read_text(encoding="utf-8"))["manifest"]
                ["rocm_version"] == "7.14.0",
                f"ROCm 7.14 cell is mislabeled: {cell.name}")
    verify_checksums(matrix_714, ROOT / forward_manifest["evidence"]["checksums"])

    hardware = claims["hardware_matrix"]
    validated = [item for item in hardware if item["status"] == "validated"]
    require(len(validated) == 1, "exactly one project platform may be validated")
    require(validated[0]["gpu_arch"] == stack["platform"]["gpu_arch"],
            "validated hardware matrix row disagrees with stack")
    w7900 = [item for item in hardware if item["hardware"] == "Radeon W7900"]
    require(len(w7900) == 1 and w7900[0]["status"] == "planned",
            "Radeon W7900 must remain planned without accepted evidence")

    for name, model_key, set_key in (
        ("BF16", "bf16", "bf16"),
        ("GGUF", "gguf", "gguf"),
    ):
        artifact_set = artifacts["sets"][set_key]
        require(stack["model"][f"{model_key}_id"] == artifact_set["repository"],
                f"{name} repository disagrees between manifests")
        require(stack["model"][f"{model_key}_revision"] == artifact_set["revision"],
                f"{name} revision disagrees between manifests")

    benchmark_doc = (ROOT / "docs/results/METHODOLOGY.md").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    require(stack["llama_cpp"]["commit"] in benchmark_doc,
            "methodology does not cite the validated llama.cpp commit")
    require(stack["vllm"]["commit"] in pyproject,
            "pyproject provenance does not cite the validated vLLM commit")

    readme = README.read_text(encoding="utf-8")
    require(generated_block(readme, "validated-platform") ==
            platform_summary(claims, stack),
            "README validated-platform block is stale")
    require(generated_block(readme, "hardware-matrix") ==
            readme_hardware_table(claims),
            "README hardware-matrix block is stale")
    require(generated_block(readme, "validation-tracks") ==
            validation_tracks(claims, forward_manifest),
            "README validation-tracks block is stale")

    hardware_doc = HARDWARE_DOC.read_text(encoding="utf-8")
    require(generated_block(hardware_doc, "hardware-matrix") ==
            hardware_doc_table(claims),
            "hardware-validation status table is stale")


def main() -> int:
    try:
        check()
    except (KeyError, TypeError, ValueError) as exc:
        print(f"claim consistency failed: {exc}", file=sys.stderr)
        return 1
    print("claim consistency: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
