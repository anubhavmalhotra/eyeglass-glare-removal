"""Unit tests for Wan2.1-VACE 4k+1 padding, lazy loading, corruption recovery, and SDPA guard."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from eyeglass_glare_removal import inpaint
from eyeglass_glare_removal._runtime import configure_sdpa_guard, gpu_file_lock
from eyeglass_glare_removal.config import InpaintConfig
from eyeglass_glare_removal.inpaint import (
    compute_inpaint_chunks,
    compute_overlap_weights,
    is_chunk_complete,
    pad_frame_count_4k1,
    pad_sequence_4k1,
    run_inpaint_chunks,
    unpad_sequence,
)


def test_pad_and_unpad_4k1_all_lengths_1_to_300() -> None:
    for n in range(1, 301):
        valid_n = pad_frame_count_4k1(n)
        assert valid_n >= n
        assert valid_n - n <= 3
        assert (valid_n - 1) % 4 == 0

        seq = list(range(n))
        padded, p_len = pad_sequence_4k1(seq)
        assert p_len == valid_n
        assert len(padded) == valid_n
        assert all(x == seq[-1] for x in padded[n:])
        assert unpad_sequence(padded, n) == seq


def test_chunk_schedule_and_exact_overlap_weights_1_to_300() -> None:
    for chunk_len, min_ov in [(81, 12), (21, 6), (17, 0), (13, 4)]:
        for n in range(1, 301):
            chunks = compute_inpaint_chunks(n, chunk_len=chunk_len, min_overlap=min_ov)
            assert chunks[0][0] == 0
            assert chunks[-1][1] == n

            weights = compute_overlap_weights(n, chunks)
            assert len(weights) == n
            for contribs in weights:
                assert 1 <= len(contribs) <= 2
                w_sum = sum(w for _, _, w in contribs)
                assert w_sum == pytest.approx(1.0, abs=1e-5)


def _write_synthetic_inpaint_inputs(
    tmp_path: Path, n_frames: int, glare_frames: set[int], size: int = 64
) -> tuple[list[Path], Path, Path]:
    roi_dir = tmp_path / "roi"
    mask_dir = tmp_path / "glare"
    clamp_dir = tmp_path / "lens"
    for d in (roi_dir, mask_dir, clamp_dir):
        d.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []
    for i in range(n_frames):
        fn = f"{i:06d}.png"
        fp = roi_dir / fn
        val = (20 + i * 7) % 200
        Image.fromarray(np.full((size, size, 3), val, dtype=np.uint8)).save(fp)

        lens = np.zeros((size, size), dtype=np.uint8)
        lens[10:54, 10:54] = 255
        Image.fromarray(lens).save(clamp_dir / fn)

        glare = np.zeros((size, size), dtype=np.uint8)
        if i in glare_frames:
            glare[20:44, 20:44] = 255
        Image.fromarray(glare).save(mask_dir / fn)
        frame_paths.append(fp)

    return frame_paths, mask_dir, clamp_dir


def test_zero_glare_clip_never_loads_vace_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression P1-3: completely clean zero-glare clip must NOT load UMT5, VAE, or DiT."""
    frame_paths, mask_dir, clamp_dir = _write_synthetic_inpaint_inputs(
        tmp_path, n_frames=8, glare_frames=set(), size=64
    )

    def _forbid(*_args, **_kwargs):
        raise AssertionError("VACE model loader must not be invoked on a zero-glare clip!")

    monkeypatch.setattr(inpaint, "get_or_compute_prompt_embeds", _forbid)
    monkeypatch.setattr(inpaint, "load_vace_denoiser", _forbid)

    cfg = InpaintConfig(size=64, chunk_len=81, min_overlap=12, mask_dilate=1)
    summary = run_inpaint_chunks(
        frame_paths,
        mask_dir,
        tmp_path / "inpaint",
        cfg,
        clamp_mask_dir=clamp_dir,
        ref_image_paths=None,
        chunk_infer_fn=None,  # real code path, must remain lazy and never load models!
    )
    assert summary["chunks_skipped_zero_glare"] == 1
    assert summary["edited_frames"] == 0
    for fp in frame_paths:
        assert np.array_equal(
            np.asarray(Image.open(fp)),
            np.asarray(Image.open(tmp_path / "inpaint" / "roi_cleaned" / fp.name)),
        )


def test_run_inpaint_short_clip_4k1_padding_and_clean_frame_preservation(tmp_path: Path) -> None:
    # 10 frames (< 81): frames 0..4 glared, frames 5..9 clean!
    frame_paths, mask_dir, clamp_dir = _write_synthetic_inpaint_inputs(
        tmp_path, n_frames=10, glare_frames={0, 1, 2, 3, 4}, size=64
    )
    seen_valid_n: list[int] = []
    seen_refs: list[object] = []

    def mock_infer(frames_pil, masks_pil, ref_pils, valid_n, seed):
        seen_valid_n.append(valid_n)
        seen_refs.append(ref_pils)
        return np.full((valid_n, 64, 64, 3), 0.5, dtype=np.float32)

    cfg = InpaintConfig(size=64, chunk_len=81, min_overlap=12, mask_dilate=1)
    summary = run_inpaint_chunks(
        frame_paths,
        mask_dir,
        tmp_path / "inpaint",
        cfg,
        clamp_mask_dir=clamp_dir,
        ref_image_paths=None,
        chunk_infer_fn=mock_infer,
    )
    assert seen_valid_n == [13]
    assert seen_refs == [None]
    assert summary["edited_frames"] == 5

    # Clean frames 5..9 inside the mixed chunk MUST match original ROI bit-for-bit!
    for idx in range(5, 10):
        orig = np.asarray(Image.open(frame_paths[idx]))
        clean = np.asarray(Image.open(tmp_path / "inpaint" / "roi_cleaned" / frame_paths[idx].name))
        assert np.array_equal(orig, clean)


def test_corrupt_checkpoint_recovery_and_input_change_invalidation(tmp_path: Path) -> None:
    """Regression P0-2 & P1-10: corrupt chunk PNG or changed input frames MUST force recompute."""
    frame_paths, mask_dir, clamp_dir = _write_synthetic_inpaint_inputs(
        tmp_path, n_frames=20, glare_frames={0, 1, 2, 3, 4, 5}, size=64
    )
    calls = 0

    def mock_infer(frames_pil, masks_pil, ref_pils, valid_n, seed):
        nonlocal calls
        calls += 1
        return np.full((valid_n, 64, 64, 3), 180.0 / 255.0, dtype=np.float32)

    cfg = InpaintConfig(size=64, chunk_len=13, min_overlap=4, mask_dilate=1)
    inp_dir = tmp_path / "inpaint"
    s1 = run_inpaint_chunks(
        frame_paths, mask_dir, inp_dir, cfg, clamp_mask_dir=clamp_dir, chunk_infer_fn=mock_infer
    )
    assert calls == 1 and s1["chunks_skipped_zero_glare"] == 1

    # Overlap frame 10 is clean in BOTH chunks -> must remain bit-exact original (no VACE blend bleed!)
    assert np.array_equal(
        np.asarray(Image.open(frame_paths[10])),
        np.asarray(Image.open(inp_dir / "roi_cleaned" / frame_paths[10].name)),
    )

    # Clean resume -> 0 calls
    s2 = run_inpaint_chunks(
        frame_paths,
        mask_dir,
        inp_dir,
        cfg,
        clamp_mask_dir=clamp_dir,
        chunk_infer_fn=mock_infer,
        resume=True,
    )
    assert calls == 1 and s2["chunks_resumed"] == 2

    # Corrupt chunk 0 PNG with non-empty garbage bytes -> MUST fail validation & recompute chunk 0!
    bad_png = inp_dir / "chunks" / "chunk_0000" / "0000.png"
    bad_png.write_bytes(b"NOT_A_VALID_PNG_CORRUPTED_BYTES")
    assert not is_chunk_complete(inp_dir / "chunks" / "chunk_0000", 13)

    s3 = run_inpaint_chunks(
        frame_paths,
        mask_dir,
        inp_dir,
        cfg,
        clamp_mask_dir=clamp_dir,
        chunk_infer_fn=mock_infer,
        resume=True,
    )
    assert calls == 2 and s3["chunks_resumed"] == 1

    # Replace frame 0 contents on disk (same filename, --no-ref) -> fingerprint changes & recomputes!
    Image.fromarray(np.full((64, 64, 3), 240, dtype=np.uint8)).save(frame_paths[0])
    s4 = run_inpaint_chunks(
        frame_paths,
        mask_dir,
        inp_dir,
        cfg,
        clamp_mask_dir=clamp_dir,
        chunk_infer_fn=mock_infer,
        resume=True,
    )
    assert calls == 3 and s4["chunks_resumed"] == 0


def test_sdpa_guard_cuda_and_cpu_and_gpu_lock(tmp_path: Path) -> None:
    recorded: dict[str, bool] = {}

    fake_cuda_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        backends=SimpleNamespace(
            cuda=SimpleNamespace(
                enable_flash_sdp=lambda v: recorded.__setitem__("flash", v),
                enable_math_sdp=lambda v: recorded.__setitem__("math", v),
                enable_mem_efficient_sdp=lambda v: recorded.__setitem__("mem_eff", v),
            )
        ),
    )
    assert configure_sdpa_guard(fake_cuda_torch) is True
    assert recorded == {"flash": True, "math": True, "mem_eff": False}

    fake_cpu_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(cuda=SimpleNamespace()),
    )
    assert configure_sdpa_guard(fake_cpu_torch) is False

    with gpu_file_lock(None):
        pass
    lock_file = tmp_path / "locks" / "gpu.lock"
    with gpu_file_lock(lock_file):
        assert lock_file.exists()


def test_safetensors_prompt_embed_cache_roundtrip_and_validation(tmp_path: Path) -> None:
    """Regression P0-6: embedding cache uses tensor-only Safetensors & rejects malformed files."""
    torch = pytest.importorskip("torch")
    cache_p = tmp_path / "embeds.safetensors"
    pe = torch.randn((1, 8, 16), dtype=torch.float32)
    npe = torch.randn((1, 8, 16), dtype=torch.float32)

    inpaint.save_prompt_embeds_safetensors(cache_p, pe, npe)
    loaded = inpaint.load_prompt_embeds_safetensors(cache_p)
    assert loaded is not None
    assert torch.allclose(loaded[0], pe)
    assert torch.allclose(loaded[1], npe)

    # Corrupted / non-safetensors file must be rejected safely (never unpickled)
    bad_p = tmp_path / "bad.safetensors"
    bad_p.write_bytes(b"corrupt_pickle_or_garbage")
    assert inpaint.load_prompt_embeds_safetensors(bad_p) is None

    # Wrong tensor rank / NaN values must be rejected
    from safetensors.torch import save_file

    nan_p = tmp_path / "nan.safetensors"
    bad_t = torch.full((1, 8, 16), float("nan"))
    save_file({"prompt_embeds": bad_t, "negative_prompt_embeds": npe}, str(nan_p))
    assert inpaint.load_prompt_embeds_safetensors(nan_p) is None
