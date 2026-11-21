"""Workspace confinement, atomic publishing, content hashing, path identity, and locking."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def validate_safe_basename(filename: str) -> str:
    """
    Validate that a manifest filename is a strict, canonical single-component basename.

    Rejects empty strings, '.', '..', path separators ('/' or '\\'), drive/root prefixes,
    null bytes, or leading dashes.
    """
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"Invalid manifest filename: {filename!r}")
    if "\x00" in filename:
        raise ValueError(f"Null byte in manifest filename: {filename!r}")
    if "/" in filename or "\\" in filename:
        raise ValueError(f"Path separators are forbidden in manifest filename: {filename!r}")
    if filename in (".", "..") or filename.startswith("-"):
        raise ValueError(f"Unsafe manifest filename: {filename!r}")
    p = Path(filename)
    if p.is_absolute() or p.drive or p.root or p.name != filename:
        raise ValueError(f"Manifest filename must be a plain basename, got {filename!r}")
    return filename


def reject_symlink_path(path: Path, label: str = "path") -> None:
    """Reject a path if it is itself a symlink."""
    if path.is_symlink():
        raise ValueError(f"Symlinked {label} is not allowed inside workspace: {path}")


def ensure_confined_dir(root: Path, sub_path: Path, *, create: bool = True) -> Path:
    """
    Ensure `sub_path` is strictly confined within `root` without traversing any symlinks.
    """
    reject_symlink_path(root, "workspace root")
    if create:
        root.mkdir(parents=True, exist_ok=True)
    root_resolved = root.resolve()

    # Walk relative parts if sub_path is lexically under root, rejecting any symlink component
    try:
        rel = sub_path.relative_to(root)
    except ValueError as exc:
        # Compare resolved only after ensuring neither root nor sub_path is a symlink
        reject_symlink_path(sub_path, "workspace subdirectory")
        sub_res = sub_path.resolve()
        if not sub_res.is_relative_to(root_resolved):
            raise ValueError(
                f"Directory {sub_path} escapes confined workspace root {root}"
            ) from exc
        rel = sub_res.relative_to(root_resolved)

    cur = root
    for part in rel.parts:
        cur = cur / part
        reject_symlink_path(cur, "workspace directory")

    if create:
        cur.mkdir(parents=True, exist_ok=True)
    if not cur.resolve().is_relative_to(root_resolved):
        raise ValueError(f"Directory {sub_path} escapes confined workspace root {root}")
    return cur


def ensure_safe_child_file(
    parent_dir: Path, filename: str, *, allow_missing: bool = False
) -> Path:
    """Validate basename, ensure parent_dir & child are not symlinks, and verify confinement."""
    safe_name = validate_safe_basename(filename)
    reject_symlink_path(parent_dir, "workspace directory")
    target = parent_dir / safe_name
    reject_symlink_path(target, "workspace file")
    if target.parent.resolve() != parent_dir.resolve():
        raise ValueError(f"File {target} escapes expected directory {parent_dir}")
    if not allow_missing and not target.exists():
        raise FileNotFoundError(f"Missing expected workspace file: {target}")
    return target


def ensure_confined_existing_file(root: Path, candidate: Path) -> Path:
    """Verify an existing file path referenced in workspace metadata stays confined inside root."""
    reject_symlink_path(root, "workspace root")
    reject_symlink_path(candidate, "workspace file")
    root_res = root.resolve()
    cand_res = candidate.resolve()
    if not cand_res.is_relative_to(root_res):
        raise ValueError(f"Referenced file {candidate} escapes workspace {root}")
    # Also ensure no intermediate directory component under root is a symlink
    rel = cand_res.relative_to(root_res)
    cur = root
    for part in rel.parts:
        cur = cur / part
        reject_symlink_path(cur, "workspace component")
    return candidate


def paths_alias(a: Path, b: Path) -> bool:
    """
    Determine whether two paths refer to the same file/destination, catching symlinks,
    hardlinks (via os.path.samefile when both exist), and resolved canonical path equality.
    """
    try:
        if a.exists() and b.exists():
            if os.path.samefile(a, b):
                return True
    except OSError:
        pass

    try:
        ra = a.resolve(strict=False)
        rb = b.resolve(strict=False)
        if ra == rb:
            return True
    except OSError:
        pass

    try:
        if (a.parent.resolve(strict=False) / a.name) == (b.parent.resolve(strict=False) / b.name):
            return True
    except OSError:
        pass

    return False


def assert_distinct_output_paths(
    source_path: Path | None, outputs: list[tuple[str, Path | None]]
) -> None:
    """
    Verify no output destination aliases `source_path` or another output destination.
    """
    active: list[tuple[str, Path]] = [(lbl, p) for lbl, p in outputs if p is not None]
    if source_path is not None:
        for lbl, out_p in active:
            if paths_alias(source_path, out_p):
                raise ValueError(
                    f"Output path for {lbl} ({out_p}) aliases input video ({source_path}); "
                    "refusing to overwrite source media."
                )

    for i in range(len(active)):
        lbl_a, p_a = active[i]
        for j in range(i + 1, len(active)):
            lbl_b, p_b = active[j]
            if paths_alias(p_a, p_b):
                raise ValueError(
                    f"Output paths for {lbl_a} ({p_a}) and {lbl_b} ({p_b}) collide."
                )


def hash_file_sha256(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    """Stream SHA-256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_bytes)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def hash_files_manifest(parent_dir: Path, filenames: list[str]) -> str:
    """Compute an order-sensitive SHA-256 digest over validated files inside parent_dir."""
    h = hashlib.sha256()
    h.update(f"count={len(filenames)}\n".encode())
    for fn in filenames:
        fp = ensure_safe_child_file(parent_dir, fn, allow_missing=False)
        h.update(fn.encode("utf-8"))
        h.update(b":")
        h.update(hash_file_sha256(fp).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def verify_image_file(
    path: Path,
    *,
    expected_size: tuple[int, int] | None = None,
    expected_sha256: str | None = None,
) -> bool:
    """Verify an image file exists, is not a symlink, matches optional SHA-256, and decodes cleanly."""
    if path.is_symlink() or not path.exists() or not path.is_file():
        return False
    if path.stat().st_size == 0:
        return False
    if expected_sha256 is not None:
        if hash_file_sha256(path) != expected_sha256:
            return False
    try:
        with Image.open(path) as im:
            im.load()
            if expected_size is not None and im.size != expected_size:
                return False
    except Exception:
        return False
    return True


def atomic_write_text(path: Path, content: str) -> None:
    """Write text to a unique sibling temp file and atomically replace `path`."""
    reject_symlink_path(path.parent, "parent directory")
    reject_symlink_path(path, "target file")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def make_unique_temp_dir(parent: Path, prefix: str = ".stage_") -> Path:
    """Create a unique staging directory directly under `parent` (safe against concurrent collisions)."""
    reject_symlink_path(parent, "parent directory")
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(dir=str(parent), prefix=prefix))


def make_unique_temp_sibling(dest: Path) -> Path:
    """Reserve a unique temporary sibling file path for atomic output encoding (e.g. ffmpeg)."""
    reject_symlink_path(dest.parent, "output directory")
    reject_symlink_path(dest, "output file")
    dest.parent.mkdir(parents=True, exist_ok=True)
    suffix = dest.suffix or ".tmp"
    fd, tmp_name = tempfile.mkstemp(
        dir=str(dest.parent), prefix=f".{dest.stem}.", suffix=f".tmp{suffix}"
    )
    os.close(fd)
    return Path(tmp_name)


def atomic_swap_directory(staging_dir: Path, target_dir: Path) -> None:
    """
    Atomically publish a fully populated `staging_dir` into `target_dir` without in-place mutation.

    Uses a unique backup sibling name if `target_dir` already exists, then cleans up backup.
    """
    reject_symlink_path(target_dir.parent, "parent directory")
    reject_symlink_path(target_dir, "target directory")
    reject_symlink_path(staging_dir, "staging directory")

    backup_dir: Path | None = None
    try:
        if target_dir.exists():
            backup_dir = Path(
                tempfile.mkdtemp(dir=str(target_dir.parent), prefix=f".old_{target_dir.name}_")
            )
            backup_dir.rmdir()
            os.replace(target_dir, backup_dir)
        os.replace(staging_dir, target_dir)
    except Exception:
        # Roll back previous target_dir if swap failed midway
        if backup_dir is not None and backup_dir.exists() and not target_dir.exists():
            try:
                os.replace(backup_dir, target_dir)
            except OSError:
                pass
        raise
    finally:
        if backup_dir is not None and backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)


@contextlib.contextmanager
def workdir_lock(work_dir: Path) -> Iterator[None]:
    """
    Acquire an exclusive cross-process lock for the entire workdir lifecycle so two
    concurrent runs cannot interleave stage outputs or temp files.
    """
    reject_symlink_path(work_dir, "work_dir")
    work_dir.mkdir(parents=True, exist_ok=True)
    lock_path = work_dir / ".workdir.lock"
    reject_symlink_path(lock_path, "workdir lockfile")

    try:
        import fcntl
    except ImportError:
        logger.warning("fcntl unavailable on this platform; skipping workdir lock %s", lock_path)
        yield
        return

    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(str(lock_path), flags, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
