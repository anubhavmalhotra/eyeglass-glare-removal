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


def reject_symlink_components_under(anchor_root: Path, target: Path) -> None:
    """
    Walk every lexical path component from `anchor_root` down to `target` BEFORE resolving,
    rejecting any parent traversal or intermediate/leaf symlink.
    """
    reject_symlink_path(anchor_root, "workspace root")
    if ".." in target.parts:
        raise ValueError(f"Parent traversal '..' forbidden in workspace path: {target}")
    # Compute lexical relative path without resolving symlinks first
    try:
        rel = target.relative_to(anchor_root)
    except ValueError:
        # Compare absolute non-symlink-resolved forms (handles relative vs absolute anchor)
        abs_root = Path(os.path.abspath(str(anchor_root)))
        abs_target = Path(os.path.abspath(str(target)))
        try:
            rel = abs_target.relative_to(abs_root)
        except ValueError as exc:
            raise ValueError(
                f"Path {target} is not lexically confined within workspace root {anchor_root}"
            ) from exc
        anchor_root = abs_root

    cur = anchor_root
    for part in rel.parts:
        if part in ("", "."):
            continue
        if part == "..":
            raise ValueError(f"Parent traversal '..' forbidden in workspace path: {target}")
        cur = cur / part
        reject_symlink_path(cur, "workspace component")


def ensure_confined_dir(root: Path, sub_path: Path, *, create: bool = True) -> Path:
    """
    Ensure `sub_path` is strictly confined within `root` without traversing ANY symlinked
    ancestor or leaf component.
    """
    reject_symlink_components_under(root, sub_path)
    if create:
        root.mkdir(parents=True, exist_ok=True)
        sub_path.mkdir(parents=True, exist_ok=True)
    root_res = root.resolve()
    sub_res = sub_path.resolve(strict=False)
    if not sub_res.is_relative_to(root_res):
        raise ValueError(f"Directory {sub_path} escapes confined workspace root {root}")
    return sub_path


def ensure_safe_child_file(
    parent_dir: Path,
    filename: str,
    *,
    allow_missing: bool = False,
    workspace_root: Path | None = None,
) -> Path:
    """Validate basename, ensure parent_dir (and ancestors under workspace_root) & child are not symlinks."""
    safe_name = validate_safe_basename(filename)
    if workspace_root is not None:
        reject_symlink_components_under(workspace_root, parent_dir)
    else:
        reject_symlink_path(parent_dir, "workspace directory")
    target = parent_dir / safe_name
    reject_symlink_path(target, "workspace file")
    if target.parent.resolve() != parent_dir.resolve():
        raise ValueError(f"File {target} escapes expected directory {parent_dir}")
    if not allow_missing and not target.exists():
        raise FileNotFoundError(f"Missing expected workspace file: {target}")
    return target


def ensure_confined_existing_file(root: Path, candidate: Path) -> Path:
    """
    Verify an existing file path referenced in workspace metadata stays lexically AND canonically
    confined inside `root` with ZERO symlinked ancestor or leaf components.
    """
    reject_symlink_components_under(root, candidate)
    root_res = root.resolve()
    cand_res = candidate.resolve(strict=False)
    if not cand_res.is_relative_to(root_res):
        raise ValueError(f"Referenced file {candidate} escapes workspace {root}")
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


def assert_source_outside_managed_dir(source_path: Path, managed_dir: Path) -> None:
    """
    Reject any source file that aliases or resides inside a managed workspace directory
    (including hardlinks/symlinks to reserved files such as audio.m4a or frames/).
    """
    src_res = source_path.resolve(strict=False)
    dir_res = managed_dir.resolve(strict=False)
    src_lex = Path(os.path.abspath(str(source_path)))
    dir_lex = Path(os.path.abspath(str(managed_dir)))

    if paths_alias(source_path, managed_dir):
        raise ValueError(
            f"Input source ({source_path}) aliases managed directory ({managed_dir})."
        )
    if src_res.is_relative_to(dir_res) or src_lex.is_relative_to(dir_lex):
        raise ValueError(
            f"Input source ({source_path}) resides inside managed workspace directory "
            f"({managed_dir}); refusing to proceed to prevent accidental deletion."
        )

    # Check hardlink/symlink aliases against existing files in managed_dir (e.g. audio.m4a or frames)
    if source_path.exists() and managed_dir.exists() and managed_dir.is_dir():
        for reserved in (
            managed_dir / "audio.m4a",
            managed_dir / "prep" / "audio.m4a",
        ):
            if reserved.exists() and paths_alias(source_path, reserved):
                raise ValueError(
                    f"Input source ({source_path}) aliases managed workspace file ({reserved}); "
                    "refusing to proceed to prevent overwriting source media."
                )
        # Also walk managed frames/roi trees to detect hardlinked source files before swap/delete
        for root_str, _dirs, files in os.walk(managed_dir, followlinks=False):
            root_p = Path(root_str)
            for fn in files:
                fp = root_p / fn
                if paths_alias(source_path, fp):
                    raise ValueError(
                        f"Input source ({source_path}) aliases existing workspace file ({fp}); "
                        "refusing to proceed to prevent source deletion."
                    )


def assert_distinct_output_paths(
    source_path: Path | None, outputs: list[tuple[str, Path | None]]
) -> None:
    """Verify no output destination aliases `source_path` or another output destination."""
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


def hash_files_manifest(
    parent_dir: Path, filenames: list[str], *, workspace_root: Path | None = None
) -> str:
    """Compute an order-sensitive SHA-256 digest over validated files inside parent_dir."""
    if workspace_root is not None:
        reject_symlink_components_under(workspace_root, parent_dir)
    else:
        reject_symlink_path(parent_dir, "workspace directory")
    h = hashlib.sha256()
    h.update(f"count={len(filenames)}\n".encode())
    for fn in filenames:
        fp = ensure_safe_child_file(
            parent_dir, fn, allow_missing=False, workspace_root=workspace_root
        )
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
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def make_unique_temp_dir(parent: Path, prefix: str = ".stage_") -> Path:
    """Create a unique staging directory directly under `parent` (safe against concurrent collisions)."""
    reject_symlink_path(parent, "parent directory")
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(dir=str(parent), prefix=prefix))


def make_unique_temp_sibling(dest: Path) -> Path:
    """Reserve a unique temporary sibling file path for atomic output encoding."""
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

    - Handles `BaseException` (including KeyboardInterrupt) for rollback.
    - Preserves `backup_dir` on disk if rollback itself fails so previous artifacts are never lost.
    """
    reject_symlink_path(target_dir.parent, "parent directory")
    reject_symlink_path(target_dir, "target directory")
    reject_symlink_path(staging_dir, "staging directory")

    backup_dir: Path | None = None
    committed = False
    rollback_succeeded = False
    try:
        if target_dir.exists():
            backup_dir = Path(
                tempfile.mkdtemp(dir=str(target_dir.parent), prefix=f".old_{target_dir.name}_")
            )
            backup_dir.rmdir()
            os.replace(target_dir, backup_dir)
        os.replace(staging_dir, target_dir)
        committed = True
    except BaseException:
        if backup_dir is not None and backup_dir.exists() and not target_dir.exists():
            try:
                os.replace(backup_dir, target_dir)
                rollback_succeeded = True
            except OSError as rb_exc:
                logger.error(
                    "Rollback failed restoring %s from %s (%s); preserving backup copy at %s.",
                    target_dir,
                    backup_dir,
                    rb_exc,
                    backup_dir,
                )
        raise
    finally:
        # Delete backup ONLY when commit succeeded or rollback cleanly restored target_dir
        if backup_dir is not None and backup_dir.exists() and (committed or rollback_succeeded):
            shutil.rmtree(backup_dir, ignore_errors=True)


def atomic_commit_paths(staged_to_dest: list[tuple[Path, Path]]) -> None:
    """
    Commit multiple staged files and/or directories into their final destinations as one
    rollback-safe transaction.

    - Handles `BaseException` (including KeyboardInterrupt) for rollback.
    - Rolls back directories cleanly by moving the newly swapped destination back to its
      staging path before restoring its backup (avoiding POSIX ENOTEMPTY).
    - Preserves backup copies on disk if rollback itself fails.
    """
    if not staged_to_dest:
        return

    for staged, dest in staged_to_dest:
        reject_symlink_path(dest.parent, "destination parent directory")
        reject_symlink_path(dest, "destination path")
        reject_symlink_path(staged, "staged path")

    # Each entry: (staged, dest, backup_path_or_None, dest_was_swapped_in)
    applied: list[tuple[Path, Path, Path | None, bool]] = []
    committed = False
    rollback_ok = True
    try:
        for staged, dest in staged_to_dest:
            backup: Path | None = None
            if dest.exists():
                backup = Path(
                    tempfile.mkdtemp(dir=str(dest.parent), prefix=f".bak_{dest.name}_")
                )
                backup.rmdir()
                os.replace(dest, backup)
            # Record backup before swapping staged -> dest so even if the second replace fails,
            # backup is restored cleanly.
            applied.append((staged, dest, backup, False))
            os.replace(staged, dest)
            applied[-1] = (staged, dest, backup, True)
        committed = True
    except BaseException:
        for staged, dest, backup, swapped in reversed(applied):
            try:
                if swapped and dest.exists():
                    os.replace(dest, staged)
                if backup is not None and backup.exists() and not dest.exists():
                    os.replace(backup, dest)
            except OSError as rb_exc:
                rollback_ok = False
                logger.error(
                    "Rollback failed restoring %s from backup %s (%s); preserving backup.",
                    dest,
                    backup,
                    rb_exc,
                )
        raise
    finally:
        if committed or rollback_ok:
            for _, _, backup, _ in applied:
                if backup is not None and backup.exists():
                    if backup.is_dir() and not backup.is_symlink():
                        shutil.rmtree(backup, ignore_errors=True)
                    else:
                        backup.unlink(missing_ok=True)


@contextlib.contextmanager
def workdir_lock(work_dir: Path) -> Iterator[None]:
    """Acquire an exclusive cross-process lock for the entire workdir lifecycle."""
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
