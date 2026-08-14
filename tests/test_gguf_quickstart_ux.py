"""UX-audit regression tests for the gguf-quickstart preflight cluster.

Covers fix-task D of the 2026-08-14 new-user one-pass audit:

- F-16: the serving-port gate must run at the very top of the script, before
  the plan header and the entire reuse chain (ROCm resolution, host-tool
  checks, clone/checkout guards, build-fingerprint compare, artifact
  re-hash). A busy port used to cost all of that before the refusal.
- F-11: the Python port probe must catch the bind OSError and exit 1 so the
  actionable ERROR line is not preceded by a raw traceback.
- F-03 (quickstart half): an upfront available-space check on the filesystem
  holding $MODEL_DEST, with floors derived from configs/artifact-manifest.json
  plus a llama.cpp checkout/build allowance, instead of a mid-download
  "No space left on device".
- One cosmetic folded in from Fix-A's review: the checkout-reuse log line must
  not claim "no fetch needed" seconds after the script itself fetched.

Disk tests drive a copied-tree skeleton (scripts/ + configs/) with a stub
df(1) reporting canned per-mount availability, a stub cmake, and a local git
origin - no network, no GPU, no real ROCm (the resolver gets a fake prefix).
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/gguf-quickstart.sh"
SYSTEM_BASH = shutil.which("bash")

GIB = 1024**3
KB_GIB = 1024 * 1024  # KiB per GiB (df -Pk reports 1024-blocks)

# Mirrors DISK_BUILD_ALLOWANCE_BYTES in scripts/gguf-quickstart.sh: the ~206 MiB
# validated llama.cpp worktree plus the ~1022 MiB build-714 tree measured on
# the validated stack, rounded up for ref/build churn.
BUILD_ALLOWANCE_BYTES = 512 * 1024 * 1024 * 3

GGUF_FILE = "muse-glimmer-30B-kquant-17gb.gguf"
DFLASH_FILE = "dflash-kquant.gguf"
MMPROJ_FILE = "mmproj-kquant.gguf"


def gguf_sizes() -> dict[str, int]:
    manifest = json.loads(
        (ROOT / "configs/artifact-manifest.json").read_text(encoding="utf-8")
    )
    return {item["path"]: item["size_bytes"] for item in manifest["sets"]["gguf"]["files"]}


def gib(bytes_: int) -> str:
    """Render bytes the way the script's awk 'printf %.1f' does."""
    return f"{bytes_ / GIB:.1f} GiB"


def gib2(bytes_: int) -> str:
    """Render bytes the way the plan header's 'printf %.2f' does."""
    return f"{bytes_ / GIB:.2f} GiB"


@contextlib.contextmanager
def held_tcp_port():
    """Hold a 127.0.0.1 bind so the script's probe hits EADDRINUSE."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    try:
        yield port
    finally:
        sock.close()


def free_tcp_port() -> int:
    with held_tcp_port() as port:
        return port


def run_script(script: Path, env: dict, *, cwd: Path = ROOT, timeout: int = 120):
    return subprocess.run(
        [SYSTEM_BASH, str(script)],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def make_rocm_prefix(parent: Path) -> Path:
    prefix = parent / "fake-rocm"
    (prefix / "bin").mkdir(parents=True)
    (prefix / ".info").mkdir()
    (prefix / ".info/version").write_text("7.14.0-test\n", encoding="utf-8")
    hipcc = prefix / "bin/hipcc"
    hipcc.write_text(
        "#!/bin/sh\ncat <<'EOF'\nHIP version: stub-7.14.0-test\nEOF\n",
        encoding="utf-8",
    )
    hipcc.chmod(0o755)
    rocminfo = prefix / "bin/rocminfo"
    rocminfo.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    rocminfo.chmod(0o755)
    return prefix


def install_fake_df(
    bindir: Path,
    arms: list[tuple[str, str, int]],
    default: tuple[str, int],
) -> None:
    """df(1) stub reporting canned availability, keyed by path markers.

    The real preflight parses `df -Pk <dir>`; this stub emits the POSIX shape
    with per-path mount and Available columns, and logs each invocation to
    $DF_LOG so tests can prove the preflight actually ran.
    """
    lines = [
        "#!/bin/sh",
        f"mount={shlex.quote(default[0])}",
        f"avail={default[1]}",
        'case "$*" in',
    ]
    for marker, mount, avail_kb in arms:
        lines.append(f"  *{marker}*) mount={shlex.quote(mount)}; avail={avail_kb} ;;")
    lines += [
        "esac",
        '[ -n "${DF_LOG:-}" ] && printf \'%s\\n\' "$*" >> "$DF_LOG"',
        "printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n'",
        'printf \'fakefs 1 1 %s 99%% %s\\n\' "$avail" "$mount"',
    ]
    stub = bindir / "df"
    stub.write_text("\n".join(lines) + "\n", encoding="utf-8")
    stub.chmod(0o755)


def make_origin(tmp_path: Path) -> tuple[Path, str]:
    """A local llama.cpp stand-in: default-branch tip B, older commit A = pin.

    `uploadpack.allowAnySHA1InWant` mirrors GitHub, which lets the quickstart
    shallow-fetch the pinned 40-hex commit by hash.
    """
    origin = tmp_path / "origin"
    origin.mkdir()

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(origin), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    git("init", "-q", "-b", "master")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    (origin / "README.md").write_text("commit A\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "A")
    pin = git("rev-parse", "HEAD")
    (origin / "README.md").write_text("commit B\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "B")
    git("config", "uploadpack.allowAnySHA1InWant", "true")
    return origin, pin


def make_skeleton(tmp_path: Path) -> Path:
    """A disposable copy of scripts/ + configs/ the quickstart can cd into."""
    repo = tmp_path / "repo"
    shutil.copytree(
        ROOT / "scripts", repo / "scripts", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copytree(ROOT / "configs", repo / "configs")
    return repo


def patch_validated_stack(repo: Path, *, repo_url: str, commit: str) -> None:
    stack_path = repo / "configs/validated-stack.json"
    stack = json.loads(stack_path.read_text(encoding="utf-8"))
    stack["llama_cpp"]["source_repo"] = repo_url
    stack["llama_cpp"]["commit"] = commit
    stack_path.write_text(json.dumps(stack, indent=2) + "\n", encoding="utf-8")


def skeleton_env(
    tmp_path: Path,
    repo: Path,
    *,
    df_arms: list[tuple[str, str, int]] | None = None,
    df_default: tuple[str, int] = ("/fakedisk", 2 * KB_GIB),
    with_artifacts: bool = False,
    extra_artifacts: tuple[str, ...] = (),
    origin: Path | None = None,
    pin: str | None = None,
    extra: dict[str, str] | None = None,
) -> tuple[dict, Path, Path]:
    """Env to drive the skeleton script; returns (env, models_dir, build_dir).

    Without an origin the stack is patched to an unreachable local URL so a
    bug that gets past the preflight fails fast in the clone instead of
    touching the network.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    install_fake_df(bindir, df_arms or [], df_default)
    cmake = bindir / "cmake"
    cmake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cmake.chmod(0o755)

    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    models = repo / "models"
    build = repo / "build-custom"
    models.mkdir(exist_ok=True)
    build.mkdir(exist_ok=True)
    if with_artifacts:
        (models / GGUF_FILE).write_bytes(b"0" * 1024)
    for name in extra_artifacts:
        (models / name).write_bytes(b"0" * 1024)

    if origin is not None:
        patch_validated_stack(repo, repo_url=f"file://{origin}", commit=str(pin))
    else:
        patch_validated_stack(
            repo, repo_url="file:///nonexistent-local-origin", commit="0" * 40
        )

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "PATH": f"{bindir}:{env['PATH']}",
            "ROCM_PREFIX": str(make_rocm_prefix(tmp_path)),
            "MODEL_DEST": str(models),
            "LLAMA_CPP_BUILD_DIR": str(build),
            "PORT": str(free_tcp_port()),
            "DF_LOG": str(tmp_path / "df.log"),
            # Skip hash verification of the dummy stand-in artifacts; the
            # disk preflight must not depend on the revision being validated.
            "GGUF_REVISION": "test-override-rev",
        }
    )
    env.pop("ROCM_PATH", None)
    if extra:
        env.update(extra)
    return env, models, build


# ---------------------------------------------------------------------------
# F-16 + F-11: the port gate runs first and fails clean
# ---------------------------------------------------------------------------


def test_busy_port_refuses_before_all_reuse_work_and_without_traceback(
    tmp_path: Path,
) -> None:
    env = os.environ.copy()
    env["HOME"] = str(tmp_path / "home")
    (tmp_path / "home").mkdir(exist_ok=True)
    # A present-but-incomplete prefix would make the resolver refuse later;
    # with the gate hoisted, that error must never be reached.
    incomplete = tmp_path / "prefix-rocm"
    incomplete.mkdir()
    env["ROCM_PREFIX"] = str(incomplete)
    env.pop("ROCM_PATH", None)

    with held_tcp_port() as port:
        env["PORT"] = str(port)
        result = run_script(SCRIPT, env=env)

    assert result.returncode == 1
    assert f"ERROR: port {port} is already in use; choose PORT=<free-port>." in result.stderr
    assert "Traceback" not in result.stderr, "F-11: probe must not leak a traceback"
    assert "OSError" not in result.stderr, "F-11: the bind error must be handled"
    assert "is incomplete" not in result.stderr, "F-16: gate must precede ROCm resolution"
    assert result.stdout == "", "F-16: gate must precede the plan header and all reuse work"


def test_port_gate_precedes_resolver_tools_and_plan_header() -> None:
    src = SCRIPT.read_text(encoding="utf-8")
    gate_at = src.index("PY_PORT")
    resolve_at = src.index("resolve_rocm_prefix || exit 1")
    tools_at = src.index("for cmd in cmake curl git python3")
    plan_at = src.index('echo "llama.cpp source:')
    assert gate_at < resolve_at, "F-16: port gate must precede ROCm resolution"
    assert gate_at < tools_at, "F-16: port gate must precede the host-tool checks"
    assert resolve_at < plan_at


def test_port_probe_catches_bind_oserror_and_exits_cleanly() -> None:
    src = SCRIPT.read_text(encoding="utf-8")
    assert "sock.bind" in src
    assert "except OSError:" in src, "F-11: bind failures must be caught"
    assert "sys.exit(1)" in src, "F-11: caught bind failures must exit 1 cleanly"


def test_port_env_var_still_parameterizes_the_gate(tmp_path: Path) -> None:
    """Pin: PORT keeps steering the gate (task requires the env knob preserved)."""
    env = os.environ.copy()
    env["HOME"] = str(tmp_path / "home")
    (tmp_path / "home").mkdir(exist_ok=True)
    incomplete = tmp_path / "prefix-rocm"
    incomplete.mkdir()
    env["ROCM_PREFIX"] = str(incomplete)
    env.pop("ROCM_PATH", None)
    env["PORT"] = str(free_tcp_port())

    result = run_script(SCRIPT, env=env)

    assert result.returncode == 1
    assert "already in use" not in result.stderr, "a free port must pass the gate"
    assert "is incomplete" in result.stderr, "execution must reach the resolver"


# ---------------------------------------------------------------------------
# F-03: upfront disk-space preflight
# ---------------------------------------------------------------------------


def test_same_mount_below_combined_floor_refuses_with_expected_and_got(
    tmp_path: Path,
) -> None:
    sizes = gguf_sizes()
    repo = make_skeleton(tmp_path)
    # Same fake mount for models/ and the build dir -> combined floor.
    env, models, _build = skeleton_env(
        tmp_path, repo, extra={"WITH_DFLASH": "1"}
    )

    result = run_script(repo / "scripts/gguf-quickstart.sh", env=env, cwd=repo)

    assert result.returncode == 1
    assert "not enough disk space" in result.stderr
    need = sizes[GGUF_FILE] + sizes[DFLASH_FILE] + BUILD_ALLOWANCE_BYTES
    assert gib(need) in result.stderr, "must state the required space"
    assert "2.0 GiB" in result.stderr, "must state the available space"
    assert "/fakedisk" in result.stderr, "must name the filesystem"
    assert str(models) in result.stderr, "must name the directory it holds"
    assert "MODEL_DEST" in result.stderr, "must offer the MODEL_DEST escape hatch"
    assert "llama.cpp source:" not in result.stdout, "refusal must precede the plan header"
    assert not (repo / "third_party").exists(), "refusal must precede the clone"


def test_separate_mounts_model_floor_refuses_on_the_model_filesystem(
    tmp_path: Path,
) -> None:
    sizes = gguf_sizes()
    repo = make_skeleton(tmp_path)
    env, models, _build = skeleton_env(
        tmp_path,
        repo,
        df_arms=[
            ("models", "/fakemodel", 5 * KB_GIB),
            ("build-custom", "/fakebuild", 40 * KB_GIB),
        ],
        df_default=("/fakemodel", 5 * KB_GIB),
    )

    result = run_script(repo / "scripts/gguf-quickstart.sh", env=env, cwd=repo)

    assert result.returncode == 1
    assert "not enough disk space" in result.stderr
    assert gib(sizes[GGUF_FILE]) in result.stderr, "model-only floor on its own mount"
    assert "5.0 GiB" in result.stderr
    assert "/fakemodel" in result.stderr
    assert "MODEL_DEST" in result.stderr
    assert not (repo / "third_party").exists()


def test_separate_mounts_build_allowance_refuses_on_the_build_filesystem(
    tmp_path: Path,
) -> None:
    repo = make_skeleton(tmp_path)
    env, _models, build = skeleton_env(
        tmp_path,
        repo,
        df_arms=[
            ("models", "/fakemodel", 40 * KB_GIB),
            ("build-custom", "/fakebuild", 1 * KB_GIB),
        ],
        df_default=("/fakemodel", 40 * KB_GIB),
    )

    result = run_script(repo / "scripts/gguf-quickstart.sh", env=env, cwd=repo)

    assert result.returncode == 1
    assert "not enough disk space" in result.stderr
    assert "llama.cpp checkout/build" in result.stderr
    assert gib(BUILD_ALLOWANCE_BYTES) in result.stderr, "build allowance floor"
    assert "1.0 GiB" in result.stderr
    assert str(build) in result.stderr


def test_disk_ok_with_artifacts_present_reaches_serving(tmp_path: Path) -> None:
    origin, pin = make_origin(tmp_path)
    repo = make_skeleton(tmp_path)
    env, models, _build = skeleton_env(
        tmp_path,
        repo,
        with_artifacts=True,
        origin=origin,
        pin=pin,
        df_default=("/fakedisk", 200 * KB_GIB),
    )

    result = run_script(repo / "scripts/gguf-quickstart.sh", env=env, cwd=repo)

    assert "not enough disk space" not in result.stderr + result.stdout
    assert f"have {GGUF_FILE}" in result.stdout, "reuse path: no re-download"
    assert f"Serving on http://127.0.0.1:{env['PORT']}" in result.stdout
    # The stub cmake "built" nothing, so exec of the missing llama-server is
    # the expected terminal state: exit 127 proves every gate was passed.
    assert result.returncode == 127, result.stdout + result.stderr
    df_log = Path(env["DF_LOG"])
    df_calls = df_log.read_text(encoding="utf-8") if df_log.exists() else ""
    assert str(models) in df_calls, "the preflight must actually consult df"
    assert "no fetch needed" not in result.stdout, "cosmetic: fresh clone did fetch"


def test_present_artifacts_skip_the_model_floor_on_a_full_model_filesystem(
    tmp_path: Path,
) -> None:
    """Pin the reuse adjudication: an artifact already on disk must not
    re-require its own space - otherwise every warm rerun on a disk that
    filled up after the download would be locked out of serving. (If hash
    verification quarantines it, the rerun sees it absent and the preflight
    re-engages, so a refetch still never starts unchecked.)"""
    origin, pin = make_origin(tmp_path)
    repo = make_skeleton(tmp_path)
    env, _models, _build = skeleton_env(
        tmp_path,
        repo,
        with_artifacts=True,
        origin=origin,
        pin=pin,
        df_arms=[
            ("models", "/fakemodel", 100 * 1024),  # far below the 15.6 GiB floor
            ("build-custom", "/fakebuild", 200 * KB_GIB),
        ],
        df_default=("/fakebuild", 200 * KB_GIB),
    )

    result = run_script(repo / "scripts/gguf-quickstart.sh", env=env, cwd=repo)

    assert "not enough disk space" not in result.stderr + result.stdout
    assert result.returncode == 127, result.stdout + result.stderr
    assert f"Serving on http://127.0.0.1:{env['PORT']}" in result.stdout


def test_disk_preflight_precedes_the_plan_header() -> None:
    src = SCRIPT.read_text(encoding="utf-8")
    preflight_at = src.index("require_disk_bytes")
    plan_at = src.index('echo "llama.cpp source:')
    assert preflight_at < plan_at


# ---------------------------------------------------------------------------
# Cosmetic folded in from Fix-A's review
# ---------------------------------------------------------------------------


def test_reuse_log_never_claims_no_fetch_right_after_a_fetch() -> None:
    src = SCRIPT.read_text(encoding="utf-8")
    assert "no fetch needed" not in src, (
        "this line prints right after the script's own fetch on a fresh clone;"
        " word it to reflect actual state instead"
    )


# ---------------------------------------------------------------------------
# F-10: WITH_DFLASH / WITH_MMPROJ acknowledged in the plan header and spec args
# ---------------------------------------------------------------------------


def test_dflash_and_mmproj_plan_header_acknowledges_present_artifacts(
    tmp_path: Path,
) -> None:
    origin, pin = make_origin(tmp_path)
    repo = make_skeleton(tmp_path)
    env, models, _build = skeleton_env(
        tmp_path,
        repo,
        with_artifacts=True,
        extra_artifacts=(DFLASH_FILE, MMPROJ_FILE),
        origin=origin,
        pin=pin,
        df_default=("/fakedisk", 200 * KB_GIB),
        extra={"WITH_DFLASH": "1", "WITH_MMPROJ": "1"},
    )

    result = run_script(repo / "scripts/gguf-quickstart.sh", env=env, cwd=repo)

    assert result.returncode == 127, result.stdout + result.stderr
    assert f"Serving on http://127.0.0.1:{env['PORT']}" in result.stdout
    assert (
        f"dflash drafter  : {DFLASH_FILE} (already present; will be verified)"
        in result.stdout
    ), "F-10: the flag must surface in the plan header when the drafter is on disk"
    assert "spec decoding   : draft-dflash" in result.stdout
    assert (
        f"mmproj          : {MMPROJ_FILE} (already present; will be verified)"
        in result.stdout
    ), "F-10: WITH_MMPROJ deserves the same one-line acknowledgment"
    # F-10(b): the effective speculative args on one line just before exec -
    # --spec-type otherwise appears nowhere the user can see it.
    assert (
        f"speculative decoding: draft-dflash "
        f"(draft: {models}/{DFLASH_FILE}, n-max 15)" in result.stdout
    ), "the server never echoes its argv; the script must state the spec args"


def test_dflash_and_mmproj_plan_header_sizes_the_pending_download(
    tmp_path: Path,
) -> None:
    """Absent artifacts: the plan header must announce the download cost.

    The skeleton points at an unreachable local origin, so the run dies in
    the clone right after printing the plan header - exactly the window
    under test, with no network."""
    sizes = gguf_sizes()
    repo = make_skeleton(tmp_path)
    env, _models, _build = skeleton_env(
        tmp_path,
        repo,
        df_default=("/fakedisk", 200 * KB_GIB),
        extra={"WITH_DFLASH": "1", "WITH_MMPROJ": "1"},
    )

    result = run_script(repo / "scripts/gguf-quickstart.sh", env=env, cwd=repo)

    assert result.returncode != 0
    assert "llama.cpp build : " in result.stdout, "the plan header must print"
    assert (
        f"dflash drafter  : {DFLASH_FILE} ({gib2(sizes[DFLASH_FILE])} to fetch)"
        in result.stdout
    ), "F-10: the unannounced ~1.5 GiB drafter cost must be sized up front"
    assert (
        f"mmproj          : {MMPROJ_FILE} ({gib2(sizes[MMPROJ_FILE])} to fetch)"
        in result.stdout
    )


def test_optional_features_stay_silent_without_the_flags(tmp_path: Path) -> None:
    origin, pin = make_origin(tmp_path)
    repo = make_skeleton(tmp_path)
    env, _models, _build = skeleton_env(
        tmp_path,
        repo,
        with_artifacts=True,
        origin=origin,
        pin=pin,
        df_default=("/fakedisk", 200 * KB_GIB),
    )

    result = run_script(repo / "scripts/gguf-quickstart.sh", env=env, cwd=repo)

    assert result.returncode == 127, result.stdout + result.stderr
    assert "dflash" not in result.stdout, "no DFlash lines without WITH_DFLASH=1"
    assert "mmproj" not in result.stdout, "no mmproj lines without WITH_MMPROJ=1"
    assert "spec decoding" not in result.stdout
    assert "speculative decoding" not in result.stdout


# ---------------------------------------------------------------------------
# F-09: expected upstream "failed" noise framed at DFlash serve start
# ---------------------------------------------------------------------------

EXPECTED_NOISE_NOTE = (
    'note: upstream "failed to initialize"/"failed to measure" lines '
    "during DFlash memory fitting are expected; the definitive "
    "confirmation is 'adding speculative implementation' below"
)


def test_dflash_serve_start_frames_expected_upstream_failure_lines(
    tmp_path: Path,
) -> None:
    """F-09: upstream prints E/W "failed ..." lines during a fully
    successful DFlash load (memory fitting), exactly in the window where
    the user is checking whether DFlash engaged. One plain-English note
    from the project's own tooling must frame them just before exec,
    naming the definitive confirmation line to watch for."""
    origin, pin = make_origin(tmp_path)
    repo = make_skeleton(tmp_path)
    env, _models, _build = skeleton_env(
        tmp_path,
        repo,
        with_artifacts=True,
        extra_artifacts=(DFLASH_FILE,),
        origin=origin,
        pin=pin,
        df_default=("/fakedisk", 200 * KB_GIB),
        extra={"WITH_DFLASH": "1"},
    )

    result = run_script(repo / "scripts/gguf-quickstart.sh", env=env, cwd=repo)

    assert result.returncode == 127, result.stdout + result.stderr
    assert EXPECTED_NOISE_NOTE in result.stdout, (
        "F-09: the expected-noise note must print around the serve start"
    )
    spec_at = result.stdout.index("speculative decoding: draft-dflash")
    assert spec_at < result.stdout.index(EXPECTED_NOISE_NOTE), (
        "the note frames what the user is about to see; it belongs after the"
        " spec-args line, immediately before exec"
    )


def test_no_expected_noise_note_without_dflash(tmp_path: Path) -> None:
    origin, pin = make_origin(tmp_path)
    repo = make_skeleton(tmp_path)
    env, _models, _build = skeleton_env(
        tmp_path,
        repo,
        with_artifacts=True,
        origin=origin,
        pin=pin,
        df_default=("/fakedisk", 200 * KB_GIB),
    )

    result = run_script(repo / "scripts/gguf-quickstart.sh", env=env, cwd=repo)

    assert result.returncode == 127, result.stdout + result.stderr
    assert "DFlash memory fitting" not in result.stdout
