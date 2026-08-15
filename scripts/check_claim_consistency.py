#!/usr/bin/env python3
"""Check high-value public claims against the authoritative manifests."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from statistics import mean

from compare_rocm import load_matrix, tpot_deltas_by_concurrency

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
        "Every additional platform remains evidence-gated by the matrix below."
    )


def evidence_for(item: dict) -> str:
    if item["status"] == "validated":
        return f"[Recorded project evidence]({item['evidence']})"
    if item["status"] == "community-validated":
        return f"[Accepted evidence bundle]({item['evidence']})"
    if item["status"] == "upstream-recipe":
        return "Upstream evidence; not revalidated here"
    return "Requires a comparable community submission"


def resolve_evidence_path(value: str, root: Path = ROOT) -> Path:
    require(bool(value), "accepted evidence path must not be empty")
    relative = Path(value)
    require(not relative.is_absolute(), f"evidence path must be repository-relative: {value}")
    candidate = (root / relative).resolve()
    require(candidate.is_relative_to(root.resolve()), f"evidence path escapes repository: {value}")
    require(candidate.exists(), f"evidence path does not exist: {value}")
    return candidate


def validate_hardware_matrix(
    hardware: list[dict],
    primary: dict,
    primary_evidence: str,
    root: Path = ROOT,
) -> None:
    identities = [(item["hardware"], item["gpu_arch"]) for item in hardware]
    require(len(identities) == len(set(identities)),
            "hardware matrix contains duplicate platform identities")

    def normalized_hardware(value: str) -> str:
        return value.removeprefix("AMD ")

    primary_rows = []
    for item in hardware:
        require(item["status"] in STATUS,
                f"unknown hardware status: {item['status']}")
        if item["status"] not in {"validated", "community-validated"}:
            continue
        evidence = item.get("evidence")
        require(isinstance(evidence, str),
                f"{item['hardware']}: validation status requires evidence")
        evidence_path = resolve_evidence_path(evidence, root)
        is_primary = evidence == primary_evidence
        if is_primary:
            require(item["status"] == "validated",
                    "community validation cannot cite the primary stack as independent evidence")
            require(normalized_hardware(item["hardware"]) ==
                    normalized_hardware(primary["hardware"]) and
                    item["gpu_arch"] == primary["gpu_arch"],
                    "primary evidence is attached to the wrong hardware row")
            primary_rows.append(item)
            continue

        require(evidence_path.is_file(),
                f"{item['hardware']}: evidence must name a hardware-validation manifest")
        try:
            bundle = json.loads(evidence_path.read_text(encoding="utf-8"))
            bundle_hardware = bundle["hardware"]
            result = bundle["result"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(
                f"{item['hardware']}: invalid hardware-validation evidence: {evidence}"
            ) from exc
        require(normalized_hardware(bundle_hardware["gpu"]) ==
                normalized_hardware(item["hardware"]) and
                bundle_hardware["gfx_target"] == item["gpu_arch"],
                f"{item['hardware']}: evidence identity disagrees with public claim")
        require(result.get("status") == "pass",
                f"{item['hardware']}: evidence does not record a passing result")

    require(len(primary_rows) == 1,
            "hardware matrix must contain exactly one row for the primary stack evidence")


def validate_forward_tracks(
    track_items: list[dict], root: Path = ROOT
) -> dict[str, dict]:
    tracks = {track["name"]: track for track in track_items}
    require(len(tracks) == len(track_items),
            "forward validation track names must be unique")
    for track in track_items:
        require(track["status"] in {"project-validated", "pending"},
                f"unknown forward-track status: {track['status']}")
        evidence = track.get("evidence")
        if track["status"] == "project-validated":
            require(isinstance(evidence, str),
                    f"{track['name']}: project-validated track requires evidence")
        if evidence is not None:
            resolve_evidence_path(evidence, root)
    gguf_scope = tracks["GGUF/llama.cpp"]["scope"]
    fwd_manifest = load("configs/rocm-7.14-gguf-validation.json")
    scope = fwd_manifest["scope"]
    deferred = len(scope["deferred_cells"])
    require(gguf_scope.startswith(
                f"{scope['completed_cells']} of {scope['planned_cells']} "
                f"planned cells"),
            "GGUF track scope disagrees with the validation manifest cell count")
    require(f"{deferred} np=16 DFlash cells deferred" in gguf_scope,
            "GGUF track scope disagrees with the deferred-cell count")
    return tracks


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
    deferred = len(scope["deferred_cells"])
    return (
        f"- **ROCm {forward['rocm'][:4]} gfx1151 (recommended default):** the reduced "
        "**GGUF/llama.cpp\n"
        f"  matrix is project-validated** on {platform['hardware'].removeprefix('AMD ')},\n"
        f"  {scope['completed_cells']} of {scope['planned_cells']} planned cells; of the "
        f"four `np=16` cells, both\n"
        f"  baselines were measured 2026-08-15 (healthy, fixed SSE-framing "
        f"client) and the\n"
        f"  {deferred} DFlash cells remain deferred (pathological scope). **Optional / not "
        "prioritized for v0.1; ROCm 7.14\n"
        "  Muse-Glimmer vLLM validation pending.** Current rocBLAS BF16-GEMM proxy "
        "results did not\n"
        "  justify prioritizing a 7.14 rebuild; vLLM/BF16 stays validated on the "
        "7.2.1 reference, so\n"
        "  ROCm 7.14 is not presented as a globally validated replacement for the "
        "historical stack.\n"
        f"- **ROCm {reference['rocm']} (historical reference, supplementary):** the "
        "full validated stack —\n"
        "  the complete benchmark matrix, the vLLM-vs-llama.cpp head-to-head, and "
        "llama-bench — is\n"
        "  preserved as immutable evidence. No result is relabeled or overwritten."
    )


def expected_tpot_claim() -> tuple[str, str]:
    """Render README and compact-doc claims directly from committed raw cells."""
    before = load_matrix(str(ROOT / "docs/results/matrix"))
    after = load_matrix(str(ROOT / "docs/results/matrix-714"))
    grouped = tpot_deltas_by_concurrency(before, after)
    # np=1/np=4 groups come from the original 17-cell pass; the np=16 group
    # holds the two baseline pairs (17gb, dynamic) added 2026-08-15.
    require(set(grouped) == {1, 4, 16}, "unexpected TPOT concurrency groups")

    def pct(value: float) -> str:
        return f"{value:+.1f}%"

    np1 = grouped[1]
    np4 = grouped[4]
    np16 = grouped[16]
    full = (
        f"Mean TPOT delta versus 7.2.1 was {pct(mean(np1))} at np=1 and "
        f"{pct(mean(np4))} at np=4; individual cells ranged from "
        f"{pct(min(np1))} to {pct(max(np1))} and "
        f"{pct(min(np4))} to {pct(max(np4))}, respectively. The comparable "
        f"np=16 baseline pairs averaged {pct(mean(np16))}"
    )
    compact = (
        f"mean TPOT delta was {pct(mean(np1))} at np=1 and "
        f"{pct(mean(np4))} at np=4 (np=16 baseline pairs {pct(mean(np16))}), "
        f"while individual cells varied more"
    )
    return full, compact


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
    require(platform.get("evidence") == claims["validated_stack"],
            "validated platform must cite the primary stack evidence")

    reference = claims["reference_evidence"]
    require(reference["rocm"] == stack["host"]["rocm_toolchain"],
            "historical ROCm claim disagrees with stack manifest")
    require(reference["matrix"] == stack["benchmark_evidence"]["rocm_7_2_1_matrix"],
            "historical evidence path disagrees with stack manifest")
    require(reference["status"] == stack["status"],
            "historical evidence status disagrees with stack manifest")

    forward = claims["forward_validation"]
    require(forward["rocm"] == forward_manifest["host"]["rocm_version"],
            "recommended ROCm version disagrees with its validation manifest")
    require(forward["status"] == "partially-validated",
            "recommended ROCm track must remain scoped rather than globally validated")
    require(forward.get("recommended") is True,
            "ROCm 7.14 must be marked as the recommended default track")
    require(stack["benchmark_evidence"]["forward_validation_manifest"] ==
            forward["manifest"],
            "validated stack points to the wrong forward-validation manifest")
    tracks = validate_forward_tracks(forward["tracks"])
    require(tracks["GGUF/llama.cpp"]["status"] == "project-validated" and
            tracks["GGUF/llama.cpp"]["evidence"] == "docs/results/matrix-714/",
            "GGUF track must point to accepted scoped evidence")
    require(tracks["BF16/vLLM"]["status"] == "pending" and
            tracks["BF16/vLLM"]["evidence"] == "scripts/bench_rocblas_gemm.cpp",
            "ROCm 7.14 BF16/vLLM must be pending with the GEMM proxy evidence")
    vllm_scope = tracks["BF16/vLLM"]["scope"].lower()
    require("not prioritized for v0.1" in vllm_scope and
            "validation is pending" in vllm_scope,
            "ROCm 7.14 BF16/vLLM scope must use the normalized pending vocabulary")

    require(forward_manifest["status"] == "validated-scoped",
            "ROCm 7.14 manifest lost scoped-validation status")
    require(forward_manifest["scope"]["vllm_bf16_status"] == "pending",
            "ROCm 7.14 manifest must record BF16/vLLM as pending")
    require(forward_manifest["platform"]["hardware"] == platform["hardware"] and
            forward_manifest["platform"]["gpu_arch"] == platform["gpu_arch"],
            "ROCm 7.14 platform disagrees with validated hardware identity")
    distribution_scope = forward_manifest["host"]["distribution_scope"]
    require(platform["hardware"] in distribution_scope and
            platform["gpu_arch"] in distribution_scope,
            "AMD distribution scope does not name the claimed platform identity")
    require("independent project evidence" in distribution_scope.lower(),
            "AMD platform support must remain distinct from project workload evidence")
    require(forward_manifest["host"]["release_notes"]["url"].startswith(
                "https://rocm.docs.amd.com/"),
            "ROCm platform support must cite authoritative AMD release notes")
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
                ["rocm_version"] == forward["rocm"],
                f"ROCm 7.14 cell is mislabeled: {cell.name}")
    verify_checksums(matrix_714, ROOT / forward_manifest["evidence"]["checksums"])

    hardware = claims["hardware_matrix"]
    validate_hardware_matrix(hardware, stack["platform"], claims["validated_stack"])

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
    benchmark_doc = (ROOT / "docs/results/benchmark.md").read_text(encoding="utf-8")
    full_tpot_claim, compact_tpot_claim = expected_tpot_claim()
    normalized_readme = " ".join(readme.replace("`", "").split())
    normalized_benchmark = " ".join(
        line.removeprefix("> ").replace("`", "").strip()
        for line in benchmark_doc.splitlines()
    )
    require(full_tpot_claim in normalized_readme,
            "README TPOT claim disagrees with committed matrix evidence")
    require(compact_tpot_claim in normalized_benchmark,
            "benchmark TPOT claim disagrees with committed matrix evidence")
    require("TPOT within ±2%" not in readme and
            "within noise" not in readme.lower() and
            "within noise" not in benchmark_doc.lower(),
            "cross-ROCm prose must not assert an unsupported universal noise bound")
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
