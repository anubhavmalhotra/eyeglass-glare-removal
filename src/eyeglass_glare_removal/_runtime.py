"""Shared PyTorch runtime helpers: device resolution, Blackwell SDPA guard, and GPU file lock."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from eyeglass_glare_removal._workspace import hash_file_sha256

logger = logging.getLogger(__name__)


def configure_sdpa_guard(torch_module: Any) -> bool:
    """
    Apply NVIDIA Blackwell (sm_121 / GB10) SDPA safety guard whenever CUDA is available.

    Disables mem_efficient_sdp (which can silently corrupt attention outputs on Blackwell
    sm_121 GPUs) while keeping flash_sdp and math_sdp enabled. Safe no-op on CPU/MPS.

    Returns True if CUDA SDPA flags were configured, False otherwise.
    """
    cuda = getattr(torch_module, "cuda", None)
    backends_cuda = getattr(getattr(torch_module, "backends", None), "cuda", None)
    if cuda is None or backends_cuda is None:
        return False
    if not cuda.is_available():
        return False

    if hasattr(backends_cuda, "enable_flash_sdp"):
        backends_cuda.enable_flash_sdp(True)
    if hasattr(backends_cuda, "enable_math_sdp"):
        backends_cuda.enable_math_sdp(True)
    if hasattr(backends_cuda, "enable_mem_efficient_sdp"):
        backends_cuda.enable_mem_efficient_sdp(False)
    return True


def resolve_torch_device(torch_module: Any, device_str: str = "auto") -> Any:
    """Resolve 'auto', 'cuda', 'cuda:N', 'mps', or 'cpu' to a torch.device."""
    if device_str == "auto":
        if torch_module.cuda.is_available():
            resolved = "cuda"
        elif (
            hasattr(torch_module.backends, "mps")
            and torch_module.backends.mps.is_available()
        ):
            resolved = "mps"
        else:
            resolved = "cpu"
        device_str = resolved

    if device_str.startswith("cuda") and not torch_module.cuda.is_available():
        raise RuntimeError(
            f"Requested device {device_str!r}, but CUDA is not available in this PyTorch build."
        )
    if device_str == "mps" and not (
        hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available()
    ):
        raise RuntimeError("Requested device 'mps', but MPS is not available.")
    device = torch_module.device(device_str)
    if device.type == "cuda" and device.index is None:
        device = torch_module.device(f"cuda:{torch_module.cuda.current_device()}")
    return device


def preferred_inference_dtype(torch_module: Any, device: Any) -> Any:
    """Pick bfloat16 on CUDA (when supported, else float16), and float32 on CPU/MPS."""
    if getattr(device, "type", "") == "cuda":
        is_bf16 = getattr(torch_module.cuda, "is_bf16_supported", None)
        if callable(is_bf16):
            with torch_module.cuda.device(device):
                if is_bf16():
                    return torch_module.bfloat16
        return torch_module.float16
    return torch_module.float32


def resolve_model_identity(model_id: str, config_files: tuple[str, ...]) -> str:
    """Identify local model assets or resolve the exact hub commit to load."""
    root = Path(model_id)
    if root.is_dir():
        h = hashlib.sha256()
        for directory, subdirs, files in os.walk(root, followlinks=True):
            path = Path(directory)
            ancestors = {ancestor.resolve() for ancestor in path.parents if ancestor.is_relative_to(root)}
            subdirs[:] = sorted(
                name for name in subdirs
                if not name.startswith(".") and (path / name).resolve() not in ancestors | {path.resolve()}
            )
            for name in sorted(files):
                asset = path / name
                if asset.is_file() and not name.startswith("."):
                    h.update(str(asset.relative_to(root)).encode())
                    h.update(b"\0")
                    h.update(hash_file_sha256(asset).encode("ascii"))
        return f"local:{h.hexdigest()}"

    from huggingface_hub import HfApi, try_to_load_from_cache

    def commit_identity(revision: str | None) -> str | None:
        if revision and len(revision) == 40 and all(c in "0123456789abcdef" for c in revision):
            return f"hf:{revision}"
        return None

    lookup_error = None
    try:
        revision = HfApi().model_info(model_id, timeout=5).sha
    except Exception as exc:
        # Hub versions use different HTTP clients and offline exception types.
        lookup_error = exc
    else:
        identity = commit_identity(revision)
        if identity is not None:
            return identity

    for filename in config_files:
        cached = try_to_load_from_cache(model_id, filename)
        if isinstance(cached, str) and "/snapshots/" in cached:
            identity = commit_identity(cached.split("/snapshots/", 1)[1].split("/", 1)[0])
            if identity is not None:
                return identity
    raise RuntimeError(f"Could not resolve an immutable model revision for {model_id!r}.") from lookup_error


def model_revision_kwargs(identity: str | None) -> dict[str, str]:
    """Pass the exact resolved hub commit to every loader of this model."""
    return {"revision": identity[3:]} if identity and identity.startswith("hf:") else {}


def cuda_empty_cache_if_needed(torch_module: Any, device: Any | None = None) -> None:
    """Safely clear CUDA cache only when CUDA is active."""
    if device is not None and getattr(device, "type", "") != "cuda":
        return
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.empty_cache()


def cuda_peak_vram_gib(torch_module: Any, device: Any | None = None) -> float:
    """Return peak CUDA memory allocated in GiB, or 0.0 on CPU/MPS."""
    if device is not None and getattr(device, "type", "") != "cuda":
        return 0.0
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return 0.0
    try:
        return float(cuda.max_memory_allocated(device)) / float(1024**3)
    except Exception:
        return 0.0


@contextlib.contextmanager
def gpu_file_lock(lock_path: Path | str | None) -> Iterator[None]:
    """
    Acquire an exclusive cross-process file lock (flock) when lock_path is specified.

    Serializes GPU-heavy stages on shared machines (e.g., DGX Spark) without modifying
    any host services. Safe no-op when lock_path is None.
    """
    if lock_path is None:
        yield
        return

    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import fcntl
    except ImportError:
        logger.warning("fcntl unavailable on this platform; skipping gpu_lock %s", path)
        yield
        return

    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o666)
    try:
        logger.info("Acquiring GPU lock: %s", path)
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
