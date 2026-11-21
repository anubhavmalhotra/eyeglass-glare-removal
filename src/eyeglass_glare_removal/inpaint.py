"""Stage 4: Resumable chunked Wan2.1-VACE video inpainting with exact 4k+1 handling."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from eyeglass_glare_removal._runtime import (
    configure_sdpa_guard,
    cuda_empty_cache_if_needed,
    cuda_peak_vram_gib,
    gpu_file_lock,
    preferred_inference_dtype,
    resolve_torch_device,
)
from eyeglass_glare_removal._workspace import (
    atomic_swap_directory,
    atomic_write_text,
    ensure_confined_dir,
    ensure_safe_child_file,
    hash_file_sha256,
    make_unique_temp_dir,
    make_unique_temp_sibling,
    reject_symlink_path,
    validate_safe_basename,
    verify_image_file,
)
from eyeglass_glare_removal.config import InpaintConfig

logger = logging.getLogger(__name__)


def pad_frame_count_4k1(num_frames: int) -> int:
    """Return the smallest valid Wan2.1 frame count >= num_frames satisfying (N - 1) % 4 == 0."""
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if (num_frames - 1) % 4 == 0:
        return num_frames
    return ((num_frames - 1) // 4 + 1) * 4 + 1


def pad_sequence_4k1(seq: list[Any]) -> tuple[list[Any], int]:
    """Pad a sequence by repeating its last element to reach exact 4k+1 length."""
    orig_len = len(seq)
    if orig_len < 1:
        raise ValueError("Cannot pad an empty sequence.")
    valid_n = pad_frame_count_4k1(orig_len)
    if valid_n == orig_len:
        return list(seq), valid_n
    return list(seq) + [seq[-1]] * (valid_n - orig_len), valid_n


def unpad_sequence(seq: Any, orig_len: int) -> Any:
    """Slice a padded output sequence back to its exact original length."""
    if orig_len < 1 or orig_len > len(seq):
        raise ValueError(f"Invalid orig_len={orig_len} for sequence of length {len(seq)}")
    return seq[:orig_len]


def compute_inpaint_chunks(
    n_frames: int, chunk_len: int = 81, min_overlap: int = 12
) -> list[tuple[int, int]]:
    """Compute overlapping [start, end) chunk intervals for ANY video length `n_frames >= 1`."""
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    if chunk_len < 5 or (chunk_len - 1) % 4 != 0:
        raise ValueError(
            f"chunk_len must be >= 5 and satisfy (chunk_len - 1) % 4 == 0, got {chunk_len}"
        )
    if min_overlap < 0 or min_overlap * 2 >= chunk_len:
        raise ValueError(
            f"min_overlap must satisfy 0 <= min_overlap < chunk_len / 2 ({chunk_len / 2:.1f}), "
            f"got {min_overlap}"
        )

    if n_frames <= chunk_len:
        return [(0, n_frames)]

    stride = chunk_len - min_overlap
    chunks: list[tuple[int, int]] = []
    start = 0
    while start < n_frames:
        end = min(n_frames, start + chunk_len)
        chunks.append((start, end))
        if end >= n_frames:
            break
        start += stride
    return chunks


def raised_cosine_ramp(length: int, *, rising: bool) -> np.ndarray:
    """Symmetric raised-cosine crossfade ramp in (0, 1) whose rising + falling pairs sum to 1.0."""
    if length <= 0:
        return np.zeros((0,), dtype=np.float32)
    t = np.linspace(0.0, 1.0, length + 2, dtype=np.float32)[1:-1]
    w = 0.5 * (1.0 - np.cos(np.pi * t))
    return w.astype(np.float32) if rising else (1.0 - w).astype(np.float32)


def compute_overlap_weights(
    n_frames: int, chunks: list[tuple[int, int]]
) -> list[list[tuple[int, int, float]]]:
    """Compute per-frame `(chunk_idx, local_idx, weight)` blending contributors summing to 1.0."""
    contribs: list[list[tuple[int, int, float]]] = [[] for _ in range(n_frames)]
    for ci, (s, e) in enumerate(chunks):
        length = e - s
        w = np.ones(length, dtype=np.float32)
        if ci > 0:
            prev_e = chunks[ci - 1][1]
            ov_l = prev_e - s
            if ov_l > 0:
                w[:ov_l] *= raised_cosine_ramp(ov_l, rising=True)
        if ci < len(chunks) - 1:
            next_s = chunks[ci + 1][0]
            ov_r = e - next_s
            if ov_r > 0:
                w[length - ov_r :] *= raised_cosine_ramp(ov_r, rising=False)
        for local_i in range(length):
            contribs[s + local_i].append((ci, local_i, float(w[local_i])))
    return contribs


def compute_inputs_digest(
    frame_paths: list[Path],
    mask_dir: Path,
    clamp_mask_dir: Path | None = None,
) -> str:
    """Content-derived digest over all input ROI frames, glare masks, and optional clamp masks."""
    h = hashlib.sha256()
    h.update(f"n={len(frame_paths)}\n".encode())
    for fp in frame_paths:
        fn = validate_safe_basename(fp.name)
        reject_symlink_path(fp, "ROI frame")
        h.update(fn.encode("utf-8"))
        h.update(b":f=")
        h.update(hash_file_sha256(fp).encode("ascii"))

        mp = ensure_safe_child_file(mask_dir, fn)
        h.update(b":m=")
        h.update(hash_file_sha256(mp).encode("ascii"))

        if clamp_mask_dir is not None:
            cp = ensure_safe_child_file(clamp_mask_dir, fn, allow_missing=True)
            if cp.exists():
                h.update(b":c=")
                h.update(hash_file_sha256(cp).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def compute_run_fingerprint(
    config: InpaintConfig,
    ref_paths: list[Path] | None,
    *,
    inputs_digest: str = "",
) -> str:
    """Stable hash of upstream input bytes + reference image bytes + effective inpaint config."""
    ref_part = "no_ref"
    if ref_paths:
        h_ref = hashlib.sha256()
        for rp in ref_paths:
            h_ref.update(str(rp.name).encode("utf-8"))
            if rp.exists():
                h_ref.update(hash_file_sha256(rp).encode("ascii"))
        ref_part = h_ref.hexdigest()[:24]
    payload = json.dumps(
        {
            "inputs_digest": inputs_digest,
            "model_id": config.model_id,
            "text_encoder_model_id": config.text_encoder_model_id,
            "quantize_4bit": config.quantize_4bit,
            "size": config.size,
            "chunk_len": config.chunk_len,
            "min_overlap": config.min_overlap,
            "mask_dilate": config.mask_dilate,
            "clamp_dilate": config.clamp_dilate,
            "steps": config.steps,
            "guidance_scale": config.guidance_scale,
            "conditioning_scale": config.conditioning_scale,
            "flow_shift": config.flow_shift,
            "prompt": config.prompt,
            "neg_prompt": config.neg_prompt,
            "seed": config.seed,
            "ref_hash": ref_part,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def is_chunk_complete(
    chunk_dir: Path,
    expected_len: int,
    *,
    expected_start: int | None = None,
    expected_end: int | None = None,
    expected_fingerprint: str | None = None,
    expected_size: tuple[int, int] | None = None,
) -> bool:
    """
    Strictly validate an atomic chunk checkpoint:
    - `done.json` schema, range, length, and config_fingerprint match.
    - Every chunk PNG exists, is not a symlink, decodes cleanly as a valid image of expected
      dimensions, and matches its recorded SHA-256 hash in `done.json`.
    """
    if chunk_dir.is_symlink() or not chunk_dir.is_dir():
        return False
    done_file = chunk_dir / "done.json"
    if done_file.is_symlink() or not done_file.exists():
        return False
    try:
        meta = json.loads(done_file.read_text(encoding="utf-8"))
    except Exception:
        return False

    if meta.get("num_frames") != expected_len:
        return False
    if expected_start is not None and meta.get("start") != expected_start:
        return False
    if expected_end is not None and meta.get("end") != expected_end:
        return False
    if expected_fingerprint is not None and meta.get("config_fingerprint") != expected_fingerprint:
        return False

    frame_hashes = meta.get("frame_hashes")
    if not isinstance(frame_hashes, list) or len(frame_hashes) != expected_len:
        return False
    frame_has_edit = meta.get("frame_has_edit")
    if not isinstance(frame_has_edit, list) or len(frame_has_edit) != expected_len:
        return False

    for li in range(expected_len):
        fp = chunk_dir / f"{li:04d}.png"
        if not verify_image_file(
            fp,
            expected_size=expected_size,
            expected_sha256=str(frame_hashes[li]),
        ):
            return False
    return True


def prep_frame_and_mask(
    frame_path: Path,
    mask_path: Path,
    size: int,
    mask_dilate: int = 5,
    clamp_mask_path: Path | None = None,
    clamp_dilate: int = 0,
) -> tuple[Image.Image, Image.Image, int]:
    """Prepare a single (video_frame, binary_inpaint_mask) pair for Wan2.1-VACE."""
    img = Image.open(frame_path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.LANCZOS)

    mask_im = Image.open(mask_path).convert("L")
    if mask_im.size != (size, size):
        mask_im = mask_im.resize((size, size), Image.Resampling.NEAREST)
    m_arr = (np.asarray(mask_im) > 127).astype(np.uint8) * 255

    if mask_dilate > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * mask_dilate + 1, 2 * mask_dilate + 1)
        )
        m_arr = cv2.dilate(m_arr, k, iterations=1)

    if clamp_mask_path is not None and clamp_mask_path.exists():
        cm = Image.open(clamp_mask_path).convert("L")
        if cm.size != (size, size):
            cm = cm.resize((size, size), Image.Resampling.NEAREST)
        cm_arr = (np.asarray(cm) > 127).astype(np.uint8) * 255
        if clamp_dilate > 0:
            kc = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * clamp_dilate + 1, 2 * clamp_dilate + 1)
            )
            cm_arr = cv2.dilate(cm_arr, kc, iterations=1)
        m_arr = np.where(cm_arr > 127, m_arr, 0).astype(np.uint8)

    glare_px = int((m_arr > 127).sum())
    rgb = np.asarray(img, dtype=np.uint8).copy()
    rgb[m_arr > 127] = 128
    return Image.fromarray(rgb), Image.fromarray(m_arr, mode="L"), glare_px


def stream_blend_chunks(
    frame_filenames: list[str],
    chunks: list[tuple[int, int]],
    chunks_root: Path,
    out_dir: Path,
    *,
    orig_frame_paths: list[Path] | None = None,
    frame_has_edit: list[bool] | None = None,
) -> None:
    """
    Stream raised-cosine overlap blending frame-by-frame into a unique staging directory
    and swap atomically into `out_dir`.

    When `frame_has_edit[fi]` is False (zero effective glare mask on that frame), copies
    the original ROI frame directly so mixed-chunk overlap weights never bleed VACE output
    onto clean frames.
    """
    n = len(frame_filenames)
    safe_names = [validate_safe_basename(fn) for fn in frame_filenames]
    contribs = compute_overlap_weights(n, chunks)

    reject_symlink_path(out_dir.parent, "inpaint directory")
    stage_dir = make_unique_temp_dir(out_dir.parent, prefix=".stage_roi_cleaned_")

    try:
        for fi in range(n):
            dst = stage_dir / safe_names[fi]
            if (
                frame_has_edit is not None
                and not frame_has_edit[fi]
                and orig_frame_paths is not None
            ):
                shutil.copyfile(orig_frame_paths[fi], dst)
                continue

            items = contribs[fi]
            if len(items) == 1:
                ci, li, _ = items[0]
                src = chunks_root / f"chunk_{ci:04d}" / f"{li:04d}.png"
                shutil.copyfile(src, dst)
            else:
                acc: np.ndarray | None = None
                wsum = 0.0
                for ci, li, wt in items:
                    src = chunks_root / f"chunk_{ci:04d}" / f"{li:04d}.png"
                    arr = np.asarray(Image.open(src).convert("RGB"), dtype=np.float32)
                    acc = arr * wt if acc is None else (acc + arr * wt)
                    wsum += wt
                assert acc is not None
                blended = np.clip(acc / max(wsum, 1e-6), 0.0, 255.0).round().astype(np.uint8)
                Image.fromarray(blended).save(dst)

        atomic_swap_directory(stage_dir, out_dir)
    except Exception:
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)
        raise


def save_prompt_embeds_safetensors(
    cache_path: Path, prompt_embeds: Any, negative_prompt_embeds: Any
) -> None:
    """Save prompt embeddings atomically in tensor-only Safetensors format (never pickle)."""
    from safetensors.torch import save_file

    reject_symlink_path(cache_path.parent, "embed cache directory")
    reject_symlink_path(cache_path, "embed cache file")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = make_unique_temp_sibling(cache_path)
    try:
        save_file(
            {
                "prompt_embeds": prompt_embeds.contiguous().cpu(),
                "negative_prompt_embeds": negative_prompt_embeds.contiguous().cpu(),
            },
            str(tmp_path),
        )
        os.replace(tmp_path, cache_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def load_prompt_embeds_safetensors(cache_path: Path) -> tuple[Any, Any] | None:
    """
    Load & strictly validate cached embeddings from Safetensors (returns None if invalid).

    Validates exact expected keys, 3D floating-point tensor rank, finite entries, and
    compatible batch/hidden dimensions.
    """
    import torch
    from safetensors.torch import load_file

    if cache_path.is_symlink() or not cache_path.exists():
        return None
    try:
        tensors = load_file(str(cache_path), device="cpu")
    except Exception as exc:
        logger.warning("Failed to load safetensors embed cache %s (%s); recomputing.", cache_path, exc)
        return None

    if set(tensors.keys()) != {"prompt_embeds", "negative_prompt_embeds"}:
        logger.warning("Unexpected keys in embed cache %s; recomputing.", cache_path)
        return None

    pe = tensors["prompt_embeds"]
    npe = tensors["negative_prompt_embeds"]
    for t in (pe, npe):
        if t.ndim != 3 or t.numel() == 0 or not torch.is_floating_point(t):
            return None
        if not bool(torch.isfinite(t).all().item()):
            return None
    if pe.shape[0] != npe.shape[0] or pe.shape[2] != npe.shape[2]:
        return None
    return pe, npe


def get_or_compute_prompt_embeds(
    text_model_id: str,
    prompt: str,
    neg_prompt: str,
    cache_dir: Path,
    device: Any,
    dtype: Any,
) -> tuple[Any, Any]:
    """
    Stage 1 of 2-stage VRAM-safe VACE loading: compute UMT5 prompt embeddings,
    atomically cache to Safetensors (no pickle/torch.load), and unload text_encoder.
    """
    import torch
    from diffusers import WanVACEPipeline

    cache_dir = Path(cache_dir)
    reject_symlink_path(cache_dir, "embed_cache_dir")
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{text_model_id}\n{prompt}\n{neg_prompt}".encode()).hexdigest()[:16]
    cache_path = cache_dir / f"vace_embeds_{key}.safetensors"

    cached = load_prompt_embeds_safetensors(cache_path)
    if cached is not None:
        logger.info("Loaded validated Safetensors UMT5 prompt embeddings from %s", cache_path)
        return cached

    logger.info("Encoding prompt with UMT5 (%s)...", text_model_id)
    pipe_txt = WanVACEPipeline.from_pretrained(
        text_model_id,
        transformer=None,
        vae=None,
        torch_dtype=dtype,
    )
    pipe_txt.text_encoder.to(device)
    with torch.inference_mode():
        pe, npe = pipe_txt.encode_prompt(
            prompt=prompt,
            negative_prompt=neg_prompt,
            do_classifier_free_guidance=True,
            max_sequence_length=512,
            device=device,
            dtype=dtype,
        )
    pe_cpu, npe_cpu = pe.cpu(), npe.cpu()
    save_prompt_embeds_safetensors(cache_path, pe_cpu, npe_cpu)

    del pipe_txt, pe, npe
    gc.collect()
    cuda_empty_cache_if_needed(torch, device)
    return pe_cpu, npe_cpu


def _validate_quantization_prereqs(quantize_4bit: bool, device: Any) -> None:
    """Validate 4-bit quantization prerequisites BEFORE allocating UMT5 or VAE memory."""
    if not quantize_4bit:
        return
    if getattr(device, "type", "") != "cuda":
        raise RuntimeError("--quantize-4bit requires a CUDA GPU and bitsandbytes.")
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "--quantize-4bit requires bitsandbytes (`pip install 'eyeglass-glare-removal[quant]'`)."
        ) from exc


def load_vace_denoiser(
    model_id: str,
    *,
    quantize_4bit: bool,
    device: Any,
    dtype: Any,
    flow_shift: float = 3.0,
) -> Any:
    """Stage 2 of 2-stage loading: load AutoencoderKLWan (fp32) + WanVACETransformer3DModel."""
    import torch
    from diffusers import (
        AutoencoderKLWan,
        WanVACEPipeline,
        WanVACETransformer3DModel,
    )
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    _validate_quantization_prereqs(quantize_4bit, device)

    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).to(device)

    if quantize_4bit:
        from diffusers import BitsAndBytesConfig

        dev_idx = device.index if device.index is not None else 0
        nf4 = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
        transformer = WanVACETransformer3DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=nf4,
            torch_dtype=dtype,
            device_map={"": dev_idx},
            low_cpu_mem_usage=True,
        )
    else:
        transformer = WanVACETransformer3DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)

    scheduler = UniPCMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    scheduler = UniPCMultistepScheduler.from_config(scheduler.config, flow_shift=flow_shift)

    return WanVACEPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
    )


def run_inpaint_chunks(
    frame_paths: list[Path],
    mask_dir: Path,
    inpaint_dir: Path,
    config: InpaintConfig,
    *,
    clamp_mask_dir: Path | None = None,
    ref_image_paths: list[Path] | None = None,
    resume: bool = True,
    chunk_infer_fn: Any = None,
) -> dict[str, Any]:
    """
    Execute Stage 4 resumable chunked inpainting + raised-cosine overlap blending.

    - Content-bound fingerprinting across ROI frames, glare masks, clamp masks, refs, and config.
    - Lazy model loading: UMT5, VAE, and DiT are ONLY loaded if at least one pending chunk
      has non-empty glare masks (zero-glare clips never load VACE!).
    - Validates quantization prerequisites before allocating UMT5/VAE.
    - Tracks per-frame edit support so clean frames are copied bit-for-bit without resampling
      or overlap blending from VACE.
    - Unique temp dirs/files and strict per-frame SHA-256 + decode checkpoint validation.
    """
    if not frame_paths:
        raise ValueError("frame_paths must be non-empty.")

    mask_dir = Path(mask_dir)
    reject_symlink_path(mask_dir, "mask_dir")
    if clamp_mask_dir is not None:
        clamp_mask_dir = Path(clamp_mask_dir)
        reject_symlink_path(clamp_mask_dir, "clamp_mask_dir")

    inpaint_dir = Path(inpaint_dir)
    reject_symlink_path(inpaint_dir, "inpaint_dir")
    inpaint_dir.mkdir(parents=True, exist_ok=True)
    chunks_root = ensure_confined_dir(inpaint_dir, inpaint_dir / "chunks", create=True)
    out_roi_dir = ensure_confined_dir(inpaint_dir, inpaint_dir / "roi_cleaned", create=False)

    n = len(frame_paths)
    chunks = compute_inpaint_chunks(n, chunk_len=config.chunk_len, min_overlap=config.min_overlap)
    with Image.open(frame_paths[0]) as im0:
        orig_sz = im0.size

    valid_refs = [Path(p) for p in (ref_image_paths or []) if Path(p).exists()]
    ref_pils: list[Image.Image] | None = None
    if valid_refs:
        ref_pils = [
            Image.open(rp).convert("RGB").resize((config.size, config.size), Image.Resampling.LANCZOS)
            for rp in valid_refs
        ]

    inputs_digest = compute_inputs_digest(frame_paths, mask_dir, clamp_mask_dir)
    fingerprint = compute_run_fingerprint(config, valid_refs or None, inputs_digest=inputs_digest)

    pending = [
        ci
        for ci, (s, e) in enumerate(chunks)
        if not (
            resume
            and is_chunk_complete(
                chunks_root / f"chunk_{ci:04d}",
                e - s,
                expected_start=s,
                expected_end=e,
                expected_fingerprint=fingerprint,
                expected_size=orig_sz,
            )
        )
    ]

    t_wall0 = time.time()
    peak_vram_gb = 0.0
    skipped_zero_glare_count = 0
    chunk_times: list[float] = []

    if pending:
        model_state: dict[str, Any] = {
            "loaded": False,
            "torch": None,
            "device": None,
            "dtype": None,
            "pipe": None,
            "pe_dev": None,
            "npe_dev": None,
        }

        def _ensure_vace_loaded() -> None:
            if model_state["loaded"] or chunk_infer_fn is not None:
                return
            import torch

            configure_sdpa_guard(torch)
            device = resolve_torch_device(torch, config.device)
            _validate_quantization_prereqs(config.quantize_4bit, device)
            dtype = preferred_inference_dtype(torch, device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            cache_dir = config.embed_cache_dir or (inpaint_dir / "embed_cache")
            pe_cpu, npe_cpu = get_or_compute_prompt_embeds(
                config.text_encoder_model_id or config.model_id,
                config.prompt,
                config.neg_prompt,
                cache_dir,
                device,
                dtype,
            )
            pipe = load_vace_denoiser(
                config.model_id,
                quantize_4bit=config.quantize_4bit,
                device=device,
                dtype=dtype,
                flow_shift=config.flow_shift,
            )
            model_state.update(
                {
                    "loaded": True,
                    "torch": torch,
                    "device": device,
                    "dtype": dtype,
                    "pipe": pipe,
                    "pe_dev": pe_cpu.to(device=device, dtype=dtype),
                    "npe_dev": npe_cpu.to(device=device, dtype=dtype),
                }
            )

        with gpu_file_lock(config.gpu_lock):
            try:
                for done_i, ci in enumerate(pending, 1):
                    s, e = chunks[ci]
                    L = e - s
                    cdir = chunks_root / f"chunk_{ci:04d}"
                    tmp_dir = make_unique_temp_dir(chunks_root, prefix=f".tmp_chunk_{ci:04d}_")

                    try:
                        tc0 = time.time()
                        frames_pil: list[Image.Image] = []
                        masks_pil: list[Image.Image] = []
                        chunk_frame_has_edit: list[bool] = []
                        total_glare_px = 0
                        for fi in range(s, e):
                            fp = frame_paths[fi]
                            fn = validate_safe_basename(fp.name)
                            mp = ensure_safe_child_file(mask_dir, fn)
                            cp = (
                                ensure_safe_child_file(clamp_mask_dir, fn, allow_missing=True)
                                if clamp_mask_dir is not None
                                else None
                            )
                            img_p, m_p, gpx = prep_frame_and_mask(
                                fp,
                                mp,
                                config.size,
                                mask_dilate=config.mask_dilate,
                                clamp_mask_path=cp,
                                clamp_dilate=config.clamp_dilate,
                            )
                            frames_pil.append(img_p)
                            masks_pil.append(m_p)
                            chunk_frame_has_edit.append(gpx > 0)
                            total_glare_px += gpx

                        frame_hashes: list[str] = []
                        if total_glare_px == 0:
                            skipped_zero_glare_count += 1
                            for li, fi in enumerate(range(s, e)):
                                dst_p = tmp_dir / f"{li:04d}.png"
                                shutil.copyfile(frame_paths[fi], dst_p)
                                frame_hashes.append(hash_file_sha256(dst_p))
                            dt = time.time() - tc0
                            meta: dict[str, Any] = {
                                "chunk": ci,
                                "start": s,
                                "end": e,
                                "num_frames": L,
                                "padded_frames": L,
                                "skipped_zero_glare": True,
                                "total_glare_px": 0,
                                "frame_has_edit": chunk_frame_has_edit,
                                "frame_hashes": frame_hashes,
                                "config_fingerprint": fingerprint,
                                "sec": round(dt, 2),
                            }
                        else:
                            # Lazy model load happens ONLY when a chunk actually requires inpainting!
                            _ensure_vace_loaded()

                            f_in, valid_n = pad_sequence_4k1(frames_pil)
                            m_in, _ = pad_sequence_4k1(masks_pil)

                            if chunk_infer_fn is not None:
                                out_arr = chunk_infer_fn(f_in, m_in, ref_pils, valid_n, config.seed)
                                out_arr = unpad_sequence(out_arr, L)
                            else:
                                torch_mod = model_state["torch"]
                                device = model_state["device"]
                                pipe = model_state["pipe"]
                                gen_dev = (
                                    device
                                    if getattr(device, "type", "") != "mps"
                                    else torch_mod.device("cpu")
                                )
                                gen = torch_mod.Generator(device=gen_dev).manual_seed(config.seed)
                                with torch_mod.inference_mode():
                                    out_full = pipe(
                                        video=f_in,
                                        mask=m_in,
                                        reference_images=ref_pils,
                                        prompt_embeds=model_state["pe_dev"],
                                        negative_prompt_embeds=model_state["npe_dev"],
                                        height=config.size,
                                        width=config.size,
                                        num_frames=valid_n,
                                        num_inference_steps=config.steps,
                                        guidance_scale=config.guidance_scale,
                                        conditioning_scale=config.conditioning_scale,
                                        generator=gen,
                                        output_type="np",
                                    ).frames[0]
                                out_arr = unpad_sequence(out_full, L)
                                if device.type == "cuda":
                                    torch_mod.cuda.synchronize(device)
                                    peak_vram_gb = max(
                                        peak_vram_gb, cuda_peak_vram_gib(torch_mod, device)
                                    )

                            out_arr = np.clip(out_arr, 0.0, 1.0)
                            for li in range(L):
                                dst_p = tmp_dir / f"{li:04d}.png"
                                if not chunk_frame_has_edit[li]:
                                    # Clean frame inside a mixed chunk: preserve original ROI bit-for-bit
                                    shutil.copyfile(frame_paths[s + li], dst_p)
                                else:
                                    im = Image.fromarray((out_arr[li] * 255.0).round().astype(np.uint8))
                                    if im.size != orig_sz:
                                        im = im.resize(orig_sz, Image.Resampling.LANCZOS)
                                    im.save(dst_p)
                                frame_hashes.append(hash_file_sha256(dst_p))

                            dt = time.time() - tc0
                            chunk_times.append(dt)
                            meta = {
                                "chunk": ci,
                                "start": s,
                                "end": e,
                                "num_frames": L,
                                "padded_frames": valid_n,
                                "skipped_zero_glare": False,
                                "total_glare_px": total_glare_px,
                                "frame_has_edit": chunk_frame_has_edit,
                                "frame_hashes": frame_hashes,
                                "config_fingerprint": fingerprint,
                                "sec": round(dt, 2),
                                "peak_vram_gib": round(peak_vram_gb, 2),
                            }

                        atomic_write_text(tmp_dir / "done.json", json.dumps(meta, indent=2))
                        atomic_swap_directory(tmp_dir, cdir)
                    except Exception:
                        if tmp_dir.exists():
                            shutil.rmtree(tmp_dir, ignore_errors=True)
                        raise

                    if model_state["torch"] is not None:
                        cuda_empty_cache_if_needed(model_state["torch"], model_state["device"])

                    rem = len(pending) - done_i
                    avg_s = float(np.mean(chunk_times)) if chunk_times else dt
                    prog = {
                        "model_id": config.model_id,
                        "quantize_4bit": config.quantize_4bit,
                        "steps": config.steps,
                        "total_frames": n,
                        "chunks_done": len(chunks) - rem,
                        "chunks_total": len(chunks),
                        "last_chunk": {k: v for k, v in meta.items() if k != "frame_hashes"},
                        "avg_chunk_sec": round(avg_s, 2),
                        "elapsed_min": round((time.time() - t_wall0) / 60.0, 2),
                        "eta_min": round((rem * avg_s) / 60.0, 2),
                        "peak_vram_gib": round(peak_vram_gb, 2),
                    }
                    atomic_write_text(inpaint_dir / "progress.json", json.dumps(prog, indent=2))
            finally:
                torch_mod = model_state.get("torch")
                device = model_state.get("device")
                model_state.clear()
                gc.collect()
                if torch_mod is not None:
                    cuda_empty_cache_if_needed(torch_mod, device)

    # Collect per-frame edit flags across all completed chunks so clean frames bypass overlap blend
    frame_has_edit_global = [False] * n
    for ci, (s, e) in enumerate(chunks):
        meta = json.loads((chunks_root / f"chunk_{ci:04d}" / "done.json").read_text(encoding="utf-8"))
        chunk_flags = meta.get("frame_has_edit") or [True] * (e - s)
        for li in range(e - s):
            if bool(chunk_flags[li]):
                frame_has_edit_global[s + li] = True

    stream_blend_chunks(
        [p.name for p in frame_paths],
        chunks,
        chunks_root,
        out_roi_dir,
        orig_frame_paths=frame_paths,
        frame_has_edit=frame_has_edit_global,
    )
    atomic_write_text(
        inpaint_dir / "edit_support.json",
        json.dumps(
            {
                "config_fingerprint": fingerprint,
                "inputs_digest": inputs_digest,
                "num_frames": n,
                "frame_has_edit": frame_has_edit_global,
            },
            indent=2,
        ),
    )

    summary = {
        "model_id": config.model_id,
        "quantize_4bit": config.quantize_4bit,
        "config_fingerprint": fingerprint,
        "inputs_digest": inputs_digest,
        "steps": config.steps,
        "guidance_scale": config.guidance_scale,
        "conditioning_scale": config.conditioning_scale,
        "chunk_len": config.chunk_len,
        "min_overlap": config.min_overlap,
        "total_frames": n,
        "edited_frames": sum(1 for x in frame_has_edit_global if x),
        "chunks_total": len(chunks),
        "chunks_resumed": len(chunks) - len(pending),
        "chunks_skipped_zero_glare": skipped_zero_glare_count,
        "used_reference": bool(valid_refs),
        "peak_vram_gib": round(peak_vram_gb, 2),
        "total_wall_min": round((time.time() - t_wall0) / 60.0, 2),
        "out_roi_dir": str(out_roi_dir),
    }
    atomic_write_text(inpaint_dir / "inpaint_summary.json", json.dumps(summary, indent=2))
    return summary
