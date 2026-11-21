"""Workspace confinement, atomic publishing, content hashing, path identity, and locking."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

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


def _dir_is_case_insensitive(path: Path) -> bool:
    """Detect whether the filesystem hosting `path` (or its nearest existing ancestor) is case-insensitive."""
    cur = Path(os.path.abspath(str(path)))
    while not os.path.lexists(cur):
        parent = cur.parent
        if parent == cur:
            break
        cur = parent

    if os.path.lexists(cur):
        try:
            cur_stat = os.lstat(cur)
            probe = cur
            while probe != probe.parent:
                swapped = probe.name.swapcase()
                if swapped != probe.name:
                    alt = probe.with_name(swapped)
                    if os.path.lexists(alt):
                        try:
                            return os.path.samestat(os.lstat(probe), os.lstat(alt))
                        except OSError:
                            pass
                    return False
                probe = probe.parent
            _ = cur_stat
        except OSError:
            pass
    return sys.platform in ("darwin", "win32")


def relative_to_anchor(anchor_root: Path, target: Path) -> tuple[Path, Path] | None:
    """
    Match `target` against `anchor_root` without resolving symlinks beneath `anchor_root`,
    matching existing ancestors by filesystem identity (`samestat`) so case-insensitive
    aliases (e.g. `WORK/link/exports` vs `work` on macOS) cannot bypass anchor checks.

    Returns `(matched_anchor_prefix, lexical_relative_path)` or `None` if `target` is outside
    `anchor_root`.
    """
    abs_root = Path(os.path.abspath(str(anchor_root)))
    abs_target = Path(os.path.abspath(str(target)))

    try:
        return abs_root, abs_target.relative_to(abs_root)
    except ValueError:
        pass

    # Match existing ancestors by (st_dev, st_ino) via lstat (no symlink following)
    if os.path.lexists(abs_root):
        try:
            root_stat = os.lstat(abs_root)
        except OSError:
            root_stat = None
        if root_stat is not None:
            for anc in (abs_target, *abs_target.parents):
                if not os.path.lexists(anc):
                    continue
                try:
                    if os.path.samestat(os.lstat(anc), root_stat):
                        return anc, abs_target.relative_to(anc)
                except OSError:
                    continue

    # Case-insensitive fallback when `anchor_root` itself does not exist yet
    if _dir_is_case_insensitive(abs_root):
        root_parts = abs_root.parts
        target_parts = abs_target.parts
        if len(target_parts) >= len(root_parts) and all(
            a.casefold() == b.casefold() for a, b in zip(root_parts, target_parts, strict=False)
        ):
            matched_prefix = Path(*target_parts[: len(root_parts)])
            rel = Path(*target_parts[len(root_parts) :]) if len(target_parts) > len(root_parts) else Path(".")
            return matched_prefix, rel

    return None


def path_is_under_dir(candidate: Path, target_dir: Path) -> bool:
    """
    Return True if `candidate` equals or resides inside `target_dir`, catching lexical paths,
    resolved symlink targets, filesystem-identity (`samestat`) aliases, and case-insensitive
    aliases even before `target_dir` is created on disk.
    """
    abs_cand = Path(os.path.abspath(str(candidate)))
    abs_dir = Path(os.path.abspath(str(target_dir)))
    if abs_cand.is_relative_to(abs_dir):
        return True
    try:
        if candidate.resolve(strict=False).is_relative_to(target_dir.resolve(strict=False)):
            return True
    except OSError:
        pass
    if relative_to_anchor(target_dir, candidate) is not None:
        return True

    # Walk upward to nearest existing ancestor of `target_dir` and compare remaining parts
    # case-insensitively when hosted on a case-insensitive filesystem (e.g. macOS APFS).
    existing_anc = abs_dir
    tail_parts: list[str] = []
    while not os.path.lexists(existing_anc):
        parent = existing_anc.parent
        if parent == existing_anc:
            break
        tail_parts.append(existing_anc.name)
        existing_anc = parent
    tail_parts.reverse()

    if tail_parts and os.path.lexists(existing_anc) and _dir_is_case_insensitive(existing_anc):
        match = relative_to_anchor(existing_anc, candidate)
        if match is not None:
            rel_parts = match[1].parts
            if len(rel_parts) >= len(tail_parts) and all(
                a.casefold() == b.casefold()
                for a, b in zip(tail_parts, rel_parts, strict=False)
            ):
                return True

    return False


def reject_symlink_components_under(anchor_root: Path, target: Path) -> None:
    """
    Walk every lexical path component from `anchor_root` down to `target` BEFORE resolving,
    rejecting any parent traversal or intermediate/leaf symlink (including case aliases).
    """
    reject_symlink_path(anchor_root, "workspace root")
    if ".." in target.parts:
        raise ValueError(f"Parent traversal '..' forbidden in workspace path: {target}")

    matched = relative_to_anchor(anchor_root, target)
    if matched is None:
        raise ValueError(
            f"Path {target} is not lexically confined within workspace root {anchor_root}"
        )
    cur, rel = matched
    reject_symlink_path(cur, "workspace root")
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
    (including hardlinks/symlinks/case-aliases to reserved files such as audio.m4a or frames/).
    """
    if paths_alias(source_path, managed_dir):
        raise ValueError(
            f"Input source ({source_path}) aliases managed directory ({managed_dir})."
        )
    if path_is_under_dir(source_path, managed_dir):
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
    if dest.exists() and dest.is_dir():
        raise ValueError(f"Output destination is an existing directory, not a file: {dest}")
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
    atomic_commit_paths([(staging_dir, target_dir)])


_PLATFORM_SUPPORTS_DIR_FD = (
    hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
    and os.open in getattr(os, "supports_dir_fd", set())
    and os.stat in getattr(os, "supports_dir_fd", set())
    and os.rename in getattr(os, "supports_dir_fd", set())
    and os.unlink in getattr(os, "supports_dir_fd", set())
    and os.mkdir in getattr(os, "supports_dir_fd", set())
)


def _supports_dir_fd() -> bool:
    """Return True when the platform supports POSIX `dir_fd` + `O_DIRECTORY`/`O_NOFOLLOW`."""
    return _PLATFORM_SUPPORTS_DIR_FD


@dataclass
class ExportDirectoryHandle:
    """Verified cross-platform handle for atomic private (`0600`) PNG frame exports."""

    path: Path
    fd: int | None
    _initial_stat: os.stat_result | None = None

    def write_private_file(self, filename: str, writer: Callable[[BinaryIO], None]) -> None:
        """Create a new mode-0600 file inside the export directory without following symlinks."""
        safe_name = validate_safe_basename(filename)
        open_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        if self.fd is not None:
            file_fd = os.open(safe_name, open_flags, 0o600, dir_fd=self.fd)
        else:
            reject_symlink_path(self.path, "save_frames_dir")
            target = self.path / safe_name
            reject_symlink_path(target, "staged export file")
            file_fd = os.open(str(target), open_flags, 0o600)
        with os.fdopen(file_fd, "wb") as handle:
            writer(handle)

    def unlink_if_exists(self, filename: str) -> None:
        """Best-effort removal of a staged temporary file inside the export directory."""
        safe_name = validate_safe_basename(filename)
        try:
            if self.fd is not None:
                os.unlink(safe_name, dir_fd=self.fd)
            else:
                (self.path / safe_name).unlink(missing_ok=True)
        except FileNotFoundError:
            pass

    def list_regular_filenames(self) -> list[str]:
        """List basenames of existing regular (non-symlink) files inside the export directory."""
        names: list[str] = []
        if self.fd is not None:
            for name in os.listdir(self.fd):
                try:
                    st = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if stat.S_ISREG(st.st_mode):
                    names.append(name)
        else:
            with os.scandir(self.path) as it:
                for entry in it:
                    try:
                        if entry.is_file(follow_symlinks=False):
                            names.append(entry.name)
                    except OSError:
                        continue
        return names

    def verify_unchanged(self, workspace_root: Path | None) -> None:
        """Verify the export directory still refers to the same verified inode with no symlinks."""
        with open_export_directory(self.path, workspace_root) as current:
            if (
                self._initial_stat is not None
                and current._initial_stat is not None
                and not os.path.samestat(self._initial_stat, current._initial_stat)
            ):
                raise ValueError("PNG export directory changed during rendering.")


@contextlib.contextmanager
def open_export_directory(
    path: Path, workspace_root: Path | None
) -> Iterator[ExportDirectoryHandle]:
    """
    Open/verify a PNG export directory without following symlinks under the workspace anchor.

    Uses POSIX `O_DIRECTORY | O_NOFOLLOW` + `dir_fd` walking where supported and falls back
    cleanly to lexical component verification on platforms without `dir_fd` (e.g. Windows).
    """
    absolute = Path(os.path.abspath(str(path)))
    if ".." in path.parts or ".." in absolute.parts:
        raise ValueError(f"Parent traversal '..' forbidden in PNG export path: {path}")

    matched_anchor = (
        relative_to_anchor(workspace_root, path) if workspace_root is not None else None
    )
    if matched_anchor is not None:
        anchor, rel = matched_anchor
        assert workspace_root is not None
        reject_symlink_components_under(workspace_root, path)
    else:
        reject_symlink_path(path, "save_frames_dir")
        anchor = absolute.parent.resolve()
        rel = Path(path.name)

    anchor.mkdir(parents=True, exist_ok=True)
    if _supports_dir_fd():
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        fd = os.open(anchor, flags)
        try:
            for part in rel.parts:
                if part in ("", "."):
                    continue
                try:
                    os.mkdir(part, dir_fd=fd)
                except FileExistsError:
                    pass
                child_fd = os.open(part, flags, dir_fd=fd)
                os.close(fd)
                fd = child_fd
            yield ExportDirectoryHandle(path=path, fd=fd, _initial_stat=os.fstat(fd))
        finally:
            os.close(fd)
        return

    # Cross-platform fallback (e.g., Windows without O_DIRECTORY/O_NOFOLLOW/dir_fd)
    cur = anchor
    reject_symlink_path(cur, "export anchor")
    for part in rel.parts:
        if part in ("", "."):
            continue
        cur = cur / part
        reject_symlink_path(cur, "export directory component")
        if cur.exists():
            if not cur.is_dir():
                raise ValueError(f"Export directory path is not a directory: {cur}")
        else:
            cur.mkdir(exist_ok=True)
        reject_symlink_path(cur, "export directory component")
    st = os.lstat(cur)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise ValueError(f"Export directory is not a regular directory: {cur}")
    yield ExportDirectoryHandle(path=cur, fd=None, _initial_stat=st)


def atomic_commit_paths(
    staged_to_dest: list[tuple[Path, Path]],
    *,
    directory_fds: dict[Path, int] | None = None,
    retire_paths: list[Path] | None = None,
) -> None:
    """
    Publish files and directories (and optionally retire obsolete regular files) in one
    rollback-safe transaction.

    - Strictly prevents a regular-file commit from replacing/deleting an existing directory
      (and prevents a directory commit from replacing a regular file).
    - Allows recursive backup cleanup ONLY for explicit directory swaps.
    - Preserves backup copies if rollback fails.
    """
    fds = directory_fds or {}

    def inspect(path: Path) -> os.stat_result | None:
        try:
            if path.parent in fds:
                return os.stat(path.name, dir_fd=fds[path.parent], follow_symlinks=False)
            return path.lstat()
        except FileNotFoundError:
            return None

    def replace(src: Path, dst: Path) -> None:
        if src.parent in fds or dst.parent in fds:
            os.replace(
                src.name if src.parent in fds else src,
                dst.name if dst.parent in fds else dst,
                src_dir_fd=fds.get(src.parent),
                dst_dir_fd=fds.get(dst.parent),
            )
        else:
            os.replace(src, dst)

    validated_swaps: list[tuple[Path, Path, bool]] = []
    for staged, dest in staged_to_dest:
        if dest.parent not in fds:
            reject_symlink_path(dest.parent, "destination parent directory")

        staged_info = inspect(staged)
        if staged_info is None:
            raise FileNotFoundError(f"Missing staged transaction path: {staged}")
        if stat.S_ISLNK(staged_info.st_mode):
            raise ValueError(f"Symlinked transaction path is not allowed: {staged}")
        staged_is_dir = stat.S_ISDIR(staged_info.st_mode)
        staged_is_reg = stat.S_ISREG(staged_info.st_mode)
        if not staged_is_dir and not staged_is_reg:
            raise ValueError(f"Unsupported staged transaction entry type: {staged}")
        if (staged.parent in fds or dest.parent in fds) and not staged_is_reg:
            raise ValueError(f"Expected a regular transaction file: {staged}")

        dest_info = inspect(dest)
        if dest_info is not None:
            if stat.S_ISLNK(dest_info.st_mode):
                raise ValueError(f"Symlinked transaction path is not allowed: {dest}")
            if staged_is_dir and not stat.S_ISDIR(dest_info.st_mode):
                raise ValueError(
                    f"Refusing to replace non-directory destination ({dest}) with directory ({staged})."
                )
            if staged_is_reg and not stat.S_ISREG(dest_info.st_mode):
                raise ValueError(
                    f"Refusing to replace non-regular destination ({dest}) with regular file ({staged})."
                )

        validated_swaps.append((staged, dest, staged_is_dir))

    validated_retires: list[Path] = []
    for retire_p in retire_paths or []:
        if retire_p.parent not in fds:
            reject_symlink_path(retire_p.parent, "retire parent directory")
        info = inspect(retire_p)
        if info is None:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"Symlinked retire path is not allowed: {retire_p}")
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Refusing to retire non-regular file: {retire_p}")
        validated_retires.append(retire_p)

    # Each applied record: (staged_or_None, dest, backup_or_None, is_dir_swap)
    applied: list[tuple[Path | None, Path, Path | None, bool]] = []
    try:
        for staged, dest, is_dir in validated_swaps:
            backup = None
            if inspect(dest) is not None:
                backup = dest.with_name(f".bak_{dest.name}_{uuid4().hex}")
            # Register intent before either rename so rollback reconciles actual paths safely.
            applied.append((staged, dest, backup, is_dir))
            if backup is not None:
                replace(dest, backup)
            replace(staged, dest)

        for retire_p in validated_retires:
            backup = retire_p.with_name(f".bak_{retire_p.name}_{uuid4().hex}")
            applied.append((None, retire_p, backup, False))
            replace(retire_p, backup)
    except BaseException:
        for staged, dest, backup, _is_dir in reversed(applied):
            try:
                if staged is not None and inspect(staged) is None and inspect(dest) is not None:
                    replace(dest, staged)
                if backup is not None and inspect(backup) is not None:
                    if inspect(dest) is not None:
                        raise OSError("destination still exists")
                    replace(backup, dest)
            except OSError as exc:
                logger.error(
                    "Rollback failed restoring %s from backup %s (%s); preserving backup.",
                    dest,
                    backup,
                    exc,
                )
        raise

    for _, _, backup, is_dir in applied:
        if backup is None:
            continue
        if backup.parent in fds:
            os.unlink(backup.name, dir_fd=fds[backup.parent])
        elif is_dir:
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
