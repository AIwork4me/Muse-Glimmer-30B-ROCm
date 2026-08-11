#!/usr/bin/env python3
"""Multi-connection resumable downloader for HuggingFace Xet-bridge files via a mirror.

Why this exists
---------------
`hf download` (huggingface_hub) is single-stream-per-file in its classic HTTP
mode, and its Xet fast-path (`HF_XET_HIGH_PERFORMANCE`) needs direct access to
`cas-server.xethub.hf.co` — which a mirror cannot proxy (it 401s). On a CDN link
that sustains only ~0.2 MiB/s *per connection*, one ~50 GB shard via `hf download`
takes ~16 h.

This script instead opens N parallel HTTP range requests against the signed
CloudFront URL that the mirror's `/resolve` redirects to, and writes each fully
verified chunk straight into the final file. It handles the two things a raw
multi-connection tool (e.g. aria2c) cannot, in this environment:

  * the signed URL expires ~hourly -> it is re-resolved from the mirror on a timer
    and on any 403 mid-transfer (curl sends the signed URL verbatim; urllib would
    re-encode its query string and break CloudFront's signature -> 403);
  * per-chunk progress is persisted to `<file>.parts.json`, so re-running resumes.

No extra dependencies; shells out to `curl` (always present) for the wire.

Usage
-----
  python scripts/hf_parallel_get.py meta-models/Muse-Glimmer-30B \
      model-00001-of-00002.safetensors model-00002-of-00002.safetensors \
      --local-dir models/Muse-Glimmer-30B

Environment
-----------
  HF_ENDPOINT  mirror base (default https://hf-mirror.com)
  NCONNS       parallel range connections (default 24)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
REFRESH_SECS = 45 * 60          # re-resolve signed URL well before the ~1h expiry
CHUNK_BYTES = 32 * 1024 * 1024  # 32 MiB per range request


class NeedRefresh(Exception):
    """Raised when the signed URL 403'd and must be re-resolved."""


def resolve_and_size(mirror, repo, filename):
    """Resolve the mirror /resolve redirect to the signed CDN URL (curl captures
    the 302 Location verbatim), then probe its total size via a 1-byte range."""
    resolve_url = f"{mirror}/{repo}/resolve/main/{filename}"
    r = subprocess.run(
        ["curl", "-sS", "-o", "/dev/null", "-w", "%{redirect_url}",
         "--max-time", "40", resolve_url],
        capture_output=True, text=True,
    )
    signed = r.stdout.strip()
    if not signed.startswith("http"):
        raise RuntimeError(f"no redirect for {filename}: {r.stderr!r}")
    hdr = subprocess.run(
        ["curl", "-sS", "-r", "0-0", "-D", "-", "-o", "/dev/null",
         "--max-time", "60", signed],
        capture_output=True, text=True,
    ).stdout
    m = re.search(r"content-range:\s*bytes\s+\d+-\d+/(\d+)", hdr, re.I)
    if not m:
        raise RuntimeError(f"no Content-Range probing {filename}: {hdr[:200]!r}")
    return signed, int(m.group(1))


def write_chunk(signed_url, offset, length, final_path):
    """Download one byte range from the signed URL into a temp file (curl, verbatim
    URL) then pwrite it into final_path at `offset`. Raises NeedRefresh on 403."""
    end = offset + length - 1
    fd_tmp, tmp = tempfile_mkstemp(offset)
    os.close(fd_tmp)
    try:
        r = subprocess.run(
            ["curl", "-sS", "--fail", "--max-time", "600",
             "-r", f"{offset}-{end}", "-o", tmp, signed_url],
            capture_output=True,
        )
        if r.returncode == 22:  # HTTP >= 400 (--fail)
            msg = (r.stderr or b"").decode("utf-8", "replace")
            if re.search(r"\b40[13]\b", msg):
                raise NeedRefresh()
            raise IOError(f"chunk @ {offset} HTTP error: {msg.strip()}")
        if r.returncode != 0:
            raise IOError(f"chunk @ {offset} curl rc={r.returncode}: "
                          f"{(r.stderr or b'').decode('utf-8','replace').strip()}")
        got = os.path.getsize(tmp)
        if got != length:
            raise IOError(f"short chunk @ {offset}: got {got}/{length}")
        with open(final_path, "r+b") as out, open(tmp, "rb") as inp:
            out.seek(offset)
            while True:
                buf = inp.read(1 << 20)
                if not buf:
                    break
                out.write(buf)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def tempfile_mkstemp(offset):
    import tempfile
    return tempfile.mkstemp(prefix=f"hfpg_{offset}_", suffix=".part")


def download_file(mirror, repo, filename, local_dir, nconns, max_bytes=None):
    dest = os.path.join(local_dir, filename)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    meta_path = dest + ".parts.json"

    signed, total = resolve_and_size(mirror, repo, filename)
    if max_bytes is not None:
        total = min(total, max_bytes)
    n_chunks = (total + CHUNK_BYTES - 1) // CHUNK_BYTES

    completed = [False] * n_chunks
    if os.path.exists(meta_path):
        try:
            m = json.load(open(meta_path))
            if m.get("total") == total and m.get("chunk") == CHUNK_BYTES \
                    and len(m.get("completed", [])) == n_chunks \
                    and os.path.exists(dest) and os.path.getsize(dest) >= total:
                completed = m["completed"]
        except Exception:
            pass

    # (Re)allocate the final file (sparse) to the right size; any size change
    # invalidates the completion map.
    if not os.path.exists(dest) or os.path.getsize(dest) != total:
        with open(dest, "ab") as f:
            f.truncate(total)
        completed = [False] * n_chunks

    todo = [i for i, c in enumerate(completed) if not c]
    print(
        f"[{filename}] {total/1024/1024/1024:.2f} GiB in {n_chunks} chunks; "
        f"{n_chunks - len(todo)} done, {len(todo)} todo, {nconns} conns -> {dest}",
        file=sys.stderr,
    )

    lock = threading.Lock()
    state = {"signed": signed, "resolved_at": time.monotonic()}

    def completed_bytes():
        return sum(min(CHUNK_BYTES, total - i * CHUNK_BYTES)
                   for i in range(n_chunks) if completed[i])

    bytes_done = [completed_bytes()]
    t0 = time.monotonic()
    last_report = [t0]

    def refresh_if_needed():
        with lock:
            if time.monotonic() - state["resolved_at"] > REFRESH_SECS:
                new_url, _ = resolve_and_size(mirror, repo, filename)
                state["signed"] = new_url
                state["resolved_at"] = time.monotonic()
            return state["signed"]

    def worker(idx):
        offset = idx * CHUNK_BYTES
        length = min(CHUNK_BYTES, total - offset)
        for attempt in range(6):
            url = refresh_if_needed()
            try:
                write_chunk(url, offset, length, dest)
                return idx, length
            except NeedRefresh:
                with lock:
                    state["resolved_at"] = 0  # force refresh next time
                time.sleep(2)
            except (IOError, OSError, subprocess.SubprocessError) as e:
                if attempt == 5:
                    sys.stderr.write(f"  give up chunk {idx}: {e}\n")
                    return idx, 0
                time.sleep(2 + 2 * attempt)
        return idx, 0

    def save_meta():
        with open(meta_path, "w") as f:
            json.dump({"total": total, "chunk": CHUNK_BYTES,
                       "completed": completed}, f)

    with ThreadPoolExecutor(max_workers=nconns) as ex:
        for fut in as_completed({ex.submit(worker, i): i for i in todo}):
            idx, n = fut.result()
            if n:
                with lock:
                    completed[idx] = True
                    bytes_done[0] += n
                    save_meta()
                    now = time.monotonic()
                    if now - last_report[0] >= 15:
                        last_report[0] = now
                        dt = now - t0
                        rate = (bytes_done[0] / 1048576 / dt) if dt else 0
                        pct = 100 * sum(completed) / n_chunks
                        remaining_mb = (n_chunks - sum(completed)) * CHUNK_BYTES / 1048576
                        eta = (remaining_mb / rate / 60) if rate else 0
                        sys.stderr.write(
                            f"  [{filename}] {pct:5.1f}%  "
                            f"{bytes_done[0]/1048576:,.0f} MiB  "
                            f"{rate:.2f} MiB/s  ETA {eta:.0f} min\n"
                        )

    if not all(completed):
        save_meta()
        raise SystemExit(f"ERROR: {filename} incomplete "
                         f"({sum(completed)}/{n_chunks})")
    save_meta()
    print(f"[{filename}] OK ({total/1024/1024/1024:.2f} GiB)", file=sys.stderr)
    return total


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo")
    ap.add_argument("filenames", nargs="+")
    ap.add_argument("--local-dir", default=".")
    ap.add_argument("--mirror", default=DEFAULT_MIRROR)
    ap.add_argument("--concurrency", type=int,
                    default=int(os.environ.get("NCONNS", "24")))
    ap.add_argument("--max-bytes", type=int, default=None,
                    help="test: download only the first N bytes then stop")
    args = ap.parse_args()

    grand = 0
    for fn in args.filenames:
        grand += download_file(args.mirror, args.repo, fn, args.local_dir,
                               args.concurrency, args.max_bytes)
    print(f"done: {grand/1024/1024/1024:.2f} GiB", file=sys.stderr)


if __name__ == "__main__":
    main()
