"""Unit tests for general rim-safe compositing, zero-glare preservation, atomic publish & 4K RAM."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from eyeglass_glare_removal import composite
from eyeglass_glare_removal.config import CompositeConfig


def _build_rim_test_roi(
    *,
    rim_rgb: tuple[int, int, int] | None,
    clean_rim_rgb: tuple[int, int, int] | None = None,
    bevel_glare_rgb: tuple[int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 128x128 ROI with a circular lens interior (radius 30), a 3px boundary band
    immediately outside (radius 31..33), and bright glare inside the lens.
    """
    H = W = 128
    skin = (155, 120, 100)
    orig = np.full((H, W, 3), skin, dtype=np.uint8)
    clean = np.full((H, W, 3), skin, dtype=np.uint8)

    lens = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(lens, (64, 64), 30, 255, -1)

    glare = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(glare, (64, 64), 28, 255, -1)

    orig[lens > 127] = (242, 246, 252)
    clean[lens > 127] = (138, 106, 88)

    outer = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(outer, (64, 64), 33, 255, -1)
    rim_ring = (outer > 127) & (lens == 0)

    if rim_rgb is not None:
        orig[rim_ring] = rim_rgb
        clean[rim_ring] = clean_rim_rgb if clean_rim_rgb is not None else rim_rgb
    elif bevel_glare_rgb is not None:
        # Continuous rimless glare spilling across bevel with no rim separation
        orig[lens > 127] = bevel_glare_rgb
        orig[rim_ring] = bevel_glare_rgb
        clean[rim_ring] = skin

    return orig, clean, lens, glare, rim_ring


def test_composite_strict_outside_boundary_invariant() -> None:
    rng = np.random.default_rng(7)
    orig_full = rng.integers(0, 256, size=(320, 480, 3), dtype=np.uint8)
    clean_roi = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    lens_mask = np.zeros((128, 128), dtype=np.uint8)
    cv2.ellipse(lens_mask, (42, 64), (20, 14), 0, 0, 360, 255, -1)
    cv2.ellipse(lens_mask, (86, 64), (20, 14), 0, 0, 360, 255, -1)
    glare_mask = lens_mask.copy()

    box = (140, 80, 300, 240)
    comp_full, full_support = composite.composite_single_frame(
        orig_full, clean_roi, lens_mask, glare_mask, box, feather_px=2.0, bevel_band_px=4
    )

    outside = full_support == 0
    assert outside.sum() > 0
    max_outside_diff = int(
        np.max(np.abs(comp_full[outside].astype(np.int16) - orig_full[outside].astype(np.int16)))
    )
    assert max_outside_diff == 0


def test_zero_glare_checkerboard_and_single_clean_lens_zero_pixels_mutated() -> None:
    """Regression P1-11: empty glare mask mutates 0 pixels; clean opposite lens mutates 0 pixels."""
    H, W = 256, 256
    yy, xx = np.mgrid[0:H, 0:W]
    checker = (((xx // 2 + yy // 2) % 2) * 200 + 25).astype(np.uint8)
    orig_full = np.stack([checker, checker, checker], axis=2)

    lens_mask = np.zeros((128, 128), dtype=np.uint8)
    cv2.ellipse(lens_mask, (36, 64), (20, 14), 0, 0, 360, 255, -1)
    cv2.ellipse(lens_mask, (92, 64), (20, 14), 0, 0, 360, 255, -1)

    clean_roi = cv2.resize(orig_full[64:192, 64:192], (128, 128), interpolation=cv2.INTER_LANCZOS4)

    # 1) Empty glare mask -> ZERO pixels changed anywhere (even on high-frequency checkerboard!)
    empty_glare = np.zeros((128, 128), dtype=np.uint8)
    comp_zero, supp_zero = composite.composite_single_frame(
        orig_full, clean_roi, lens_mask, empty_glare, (64, 64, 192, 192)
    )
    assert int(supp_zero.sum()) == 0
    assert np.array_equal(comp_zero, orig_full)

    # 2) Glare ONLY in left lens -> right lens pixels remain 100% untouched!
    left_only_glare = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(left_only_glare, (36, 64), 10, 255, -1)
    comp_left, _ = composite.composite_single_frame(
        orig_full,
        np.full((128, 128, 3), 128, dtype=np.uint8),
        lens_mask,
        left_only_glare,
        (64, 64, 192, 192),
    )
    # Right lens sits at x in [64+72 .. 64+112] = [136..176], y in [114..142]
    assert np.array_equal(comp_left[114:142, 140:172], orig_full[114:142, 140:172])


def test_rim_protection_dark_metallic_adversarial_straddling_tortoiseshell_and_rimless() -> None:
    """Regression P1-12: protects dark, bright neutral metallic (even dL=105), straddling & tortoise."""
    # 1) Dark/black rim -> heavily protected
    orig_d, clean_d, lens, glare, rim_ring = _build_rim_test_roi(
        rim_rgb=(28, 26, 26), clean_rim_rgb=(155, 120, 100)
    )
    alpha_d, _ = composite.compute_rim_safe_alpha(orig_d, clean_d, lens, glare, bevel_band_px=4)
    assert float(np.median(alpha_d[rim_ring])) < 0.10

    # 2) Adversarial bright neutral metallic rim where model darkened it heavily (235 -> 130, dL=105!)
    orig_m, clean_m, lens, glare, rim_ring = _build_rim_test_roi(
        rim_rgb=(235, 235, 235), clean_rim_rgb=(130, 105, 90)
    )
    alpha_m, _ = composite.compute_rim_safe_alpha(orig_m, clean_m, lens, glare, bevel_band_px=4)
    assert float(np.median(alpha_m[rim_ring])) < 0.15

    # 3) Saturated bright chrome metallic rim (255, 255, 255) next to glare (242, 246, 252)
    orig_c, clean_c, lens, glare, rim_ring = _build_rim_test_roi(
        rim_rgb=(255, 255, 255), clean_rim_rgb=(130, 105, 90)
    )
    alpha_c, _ = composite.compute_rim_safe_alpha(orig_c, clean_c, lens, glare, bevel_band_px=4)
    assert float(np.median(alpha_c[rim_ring])) < 0.25

    # 4) Rim straddling INSIDE base_bin (outermost 2px of lens mask) -> protected too!
    H = W = 128
    orig_s = np.full((H, W, 3), (155, 120, 100), dtype=np.uint8)
    clean_s = orig_s.copy()
    inner_cut = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(inner_cut, (64, 64), 28, 255, -1)
    outer_str = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(outer_str, (64, 64), 31, 255, -1)
    straddle = (outer_str > 127) & (inner_cut == 0)
    orig_s[inner_cut > 127] = (242, 246, 252)
    orig_s[straddle] = (215, 215, 215)
    clean_s[lens > 127] = (138, 106, 88)
    clean_s[straddle] = (130, 105, 90)
    alpha_s, _ = composite.compute_rim_safe_alpha(orig_s, clean_s, lens, lens.copy(), bevel_band_px=4)
    assert float(np.median(alpha_s[straddle])) < 0.15

    # 5) Saturated amber tortoiseshell / colored acetate rim -> protected
    orig_t, clean_t, lens, glare, rim_ring = _build_rim_test_roi(
        rim_rgb=(180, 110, 25), clean_rim_rgb=(140, 105, 85)
    )
    alpha_t, _ = composite.compute_rim_safe_alpha(orig_t, clean_t, lens, glare, bevel_band_px=4)
    assert float(np.median(alpha_t[rim_ring])) < 0.20

    # 6) Continuous rimless bevel with heavy white glare spilling outside lens -> suppressed!
    orig_r, clean_r, lens, glare, rim_ring = _build_rim_test_roi(
        rim_rgb=None, bevel_glare_rgb=(245, 248, 254)
    )
    alpha_r, _ = composite.compute_rim_safe_alpha(orig_r, clean_r, lens, glare, bevel_band_px=4)
    assert float(np.median(alpha_r[rim_ring])) > 0.65


def test_suppress_chroma_glint_neutralizes_purple_and_bright_green_only() -> None:
    clean = np.full((64, 64, 3), (150, 120, 100), dtype=np.float32)
    glare = np.full((64, 64), 255, dtype=np.uint8)
    clean[8:20, 8:20] = (190, 120, 195)
    clean[8:20, 36:48] = (120, 195, 130)
    clean[36:48, 8:20] = (62, 84, 58)

    out = composite.suppress_chroma_glint(clean, glare)
    assert float(min(out[14, 14, 0] - out[14, 14, 1], out[14, 14, 2] - out[14, 14, 1])) < 0.25 * 70
    assert float(min(out[14, 42, 1] - out[14, 42, 0], out[14, 42, 1] - out[14, 42, 2])) < 0.25 * 65
    assert float(np.max(np.abs(out[42, 14] - np.array([62, 84, 58])))) < 1.0


@pytest.mark.parametrize("w,h", [(720, 1280), (1920, 1080), (720, 720), (641, 481)])
def test_compare_frame_even_dimensions_for_any_aspect_ratio(w: int, h: int) -> None:
    cw, ch, zh = composite.compute_compare_dimensions(w, h)
    assert cw % 2 == 0 and ch % 2 == 0 and zh % 2 == 0

    orig = np.zeros((h, w, 3), dtype=np.uint8)
    comp = np.zeros((h, w, 3), dtype=np.uint8)
    frame = composite.build_compare_frame(
        orig, comp, (w // 4, h // 4, 3 * w // 4, 3 * h // 4), 0, 30.0
    )
    assert frame.shape == (ch, cw, 3)


def test_4k_compare_adaptive_batch_size_bounds_ram() -> None:
    """Regression P1-17: 4K UHD compare render caps batch size to stay inside RAM budget."""
    bs_4k_cmp = composite.compute_effective_batch_size(
        3840, 2160, compare=True, requested_batch_size=32, max_batch_mem_mb=256
    )
    assert 1 <= bs_4k_cmp <= 3

    bs_720p = composite.compute_effective_batch_size(
        1280, 720, compare=False, requested_batch_size=32, max_batch_mem_mb=256
    )
    assert bs_720p == 32


def test_run_composite_stage_atomic_publish_and_failure_preserves_previous_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    W, H = 160, 120
    n_frames = 4
    orig_dir = tmp_path / "frames"
    clean_dir = tmp_path / "roi_cleaned"
    lens_dir = tmp_path / "roi_lens_exact"
    glare_dir = tmp_path / "roi_glare"
    for d in (orig_dir, clean_dir, lens_dir, glare_dir):
        d.mkdir(parents=True)

    records = []
    for i in range(n_frames):
        fn = f"{i:06d}.png"
        Image.fromarray(np.full((H, W, 3), 120, dtype=np.uint8)).save(orig_dir / fn)
        Image.fromarray(np.full((64, 64, 3), 110, dtype=np.uint8)).save(clean_dir / fn)
        mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.circle(mask, (32, 32), 18, 255, -1)
        Image.fromarray(mask).save(lens_dir / fn)
        Image.fromarray(mask).save(glare_dir / fn)
        records.append({"frame_idx": i, "filename": fn, "x1": 40, "y1": 20, "x2": 104, "y2": 84})

    roi_meta = {"fps": 30.0, "full_width": W, "full_height": H, "frames": records}

    class FakePipe:
        def __init__(self, out_path: Path, audio_path: Path | None) -> None:
            self.out_path = out_path
            self.audio_path = audio_path
            self.stdin = io.BytesIO()
            self.stderr = io.BytesIO(b"")
            self.closed = False

        def poll(self) -> int | None:
            return 0 if self.closed else None

        def wait(self) -> int:
            self.closed = True
            return 0

        def kill(self) -> None:
            self.closed = True

    opened_pipes: list[FakePipe] = []
    fail_on_close = False

    def fake_open_pipe(width, height, fps, out_mp4, **kwargs):
        p = FakePipe(out_mp4, kwargs.get("audio_path"))
        opened_pipes.append(p)
        return p

    def fake_close_pipe(proc, label="video"):
        proc.wait()
        if fail_on_close:
            raise RuntimeError("simulated ffmpeg encode failure")
        proc.out_path.write_bytes(b"ENCODED_MP4_OK")

    monkeypatch.setattr(composite, "open_video_writer_pipe", fake_open_pipe)
    monkeypatch.setattr(composite, "close_video_writer_pipe", fake_close_pipe)

    out_mp4 = tmp_path / "out.mp4"
    out_mp4.write_bytes(b"PREVIOUS_VALID_OUTPUT")

    # 1) Simulate encode failure -> previous out.mp4 MUST remain intact & no temp sibling left!
    fail_on_close = True
    cfg = CompositeConfig(compare=True, qa_grid=True, qa_grid_samples=2, workers=2, batch_size=2)
    with pytest.raises(RuntimeError, match="simulated ffmpeg encode failure"):
        composite.run_composite_stage(
            orig_frames_dir=orig_dir,
            cleaned_roi_dir=clean_dir,
            lens_masks_dir=lens_dir,
            roi_meta=roi_meta,
            output_mp4=out_mp4,
            config=cfg,
            glare_masks_dir=glare_dir,
            audio_path=None,
        )
    assert out_mp4.read_bytes() == b"PREVIOUS_VALID_OUTPUT"
    assert not list(tmp_path.glob(".out*.tmp*"))

    # 2) Successful run writes to temp siblings and atomically publishes out.mp4 & out_compare.mp4
    fail_on_close = False
    opened_pipes.clear()
    res = composite.run_composite_stage(
        orig_frames_dir=orig_dir,
        cleaned_roi_dir=clean_dir,
        lens_masks_dir=lens_dir,
        roi_meta=roi_meta,
        output_mp4=out_mp4,
        config=cfg,
        glare_masks_dir=glare_dir,
        audio_path=None,
    )
    assert res["has_audio"] is False
    assert out_mp4.read_bytes() == b"ENCODED_MP4_OK"
    assert opened_pipes[0].out_path != out_mp4  # encoded via unique temp sibling!
    assert len(opened_pipes[0].stdin.getvalue()) == n_frames * W * H * 3
    cmp_w, cmp_h, _ = composite.compute_compare_dimensions(W, H)
    assert len(opened_pipes[1].stdin.getvalue()) == n_frames * cmp_w * cmp_h * 3
    assert res["qa_grid"] is not None and Path(res["qa_grid"]).exists()


def test_composite_round3_p0_p1_regressions_bridge_wide_rim_relative_src_save_frames_and_rollback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regressions P0-2, P0-4, P1-11, P1-13, P1-14."""
    import os

    # --- P1-13: narrow 6px bridge between two lenses, glare right up to left edge -> 0 px mutated in clean right lens! ---
    H = W = 128
    orig_b = np.full((H, W, 3), (150, 120, 100), dtype=np.uint8)
    lens_b = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(lens_b, (20, 44), (60, 84), 255, -1)  # left lens x=20..60
    cv2.rectangle(lens_b, (67, 44), (107, 84), 255, -1)  # right lens x=67..107 (6px bridge!)
    # Glare fills entire left lens right up to x=60!
    glare_b = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(glare_b, (20, 44), (60, 84), 255, -1)
    orig_b[glare_b > 127] = (248, 250, 255)
    # Give right lens distinctive eye texture so ANY pixel mutation shows up immediately
    orig_b[44:85, 67:108] = (70, 95, 130)
    clean_b = np.full((H, W, 3), (200, 50, 50), dtype=np.uint8)  # stark contrast if leaked!

    comp_b, _ = composite.composite_single_frame(
        orig_b, clean_b, lens_b, glare_b, (0, 0, W, H), feather_px=2.0, bevel_band_px=6
    )
    right_region = lens_b.copy()
    right_region[:, :64] = 0
    changed_in_clean_right = int(np.any(comp_b != orig_b, axis=2)[right_region > 127].sum())
    assert changed_in_clean_right == 0

    # --- P1-14: 6px-wide bright neutral metallic rim straddling BOTH sides of lens boundary ---
    orig_w = np.full((H, W, 3), (150, 118, 96), dtype=np.uint8)
    clean_w = orig_w.copy()
    lens_w = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(lens_w, (64, 64), 30, 255, -1)
    inner_core = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(inner_core, (64, 64), 27, 255, -1)  # 3px inside lens boundary!
    outer_rim = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(outer_rim, (64, 64), 33, 255, -1)  # 3px outside lens boundary! (total 6px wide!)
    rim_inner_rows = (lens_w > 127) & (inner_core == 0)
    rim_outer_rows = (outer_rim > 127) & (lens_w == 0)

    orig_w[inner_core > 127] = (242, 246, 252)
    orig_w[rim_inner_rows | rim_outer_rows] = (232, 232, 232)  # bright neutral metallic 6px rim
    clean_w[lens_w > 127] = (135, 105, 88)
    clean_w[rim_inner_rows | rim_outer_rows] = (135, 105, 88)  # model erased the metallic rim!

    alpha_w, _ = composite.compute_rim_safe_alpha(
        orig_w, clean_w, lens_w, lens_w.copy(), bevel_band_px=4
    )
    assert float(np.median(alpha_w[rim_inner_rows])) < 0.25
    assert float(np.median(alpha_w[rim_outer_rows])) < 0.25

    # --- P0-2, P0-4, P1-11: relative source_video rejection, atomic save-frames-dir, multi-file rollback ---
    orig_dir = tmp_path / "frames"
    clean_dir = tmp_path / "roi_cleaned"
    lens_dir = tmp_path / "roi_lens_exact"
    glare_dir = tmp_path / "roi_glare"
    for d in (orig_dir, clean_dir, lens_dir, glare_dir):
        d.mkdir(parents=True)
    Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8)).save(orig_dir / "000000.png")
    Image.fromarray(np.full((64, 64, 3), 110, dtype=np.uint8)).save(clean_dir / "000000.png")
    Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(lens_dir / "000000.png")
    Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(glare_dir / "000000.png")

    rel_meta = {
        "source_video": "relative_clip.mp4",
        "fps": 30.0,
        "full_width": 64,
        "full_height": 64,
        "frames": [{"frame_idx": 0, "filename": "000000.png", "x1": 0, "y1": 0, "x2": 64, "y2": 64}],
    }
    # P0-2: relative source_video in manifest MUST be rejected when --source is omitted!
    with pytest.raises(ValueError, match="relative source_video"):
        composite.run_composite_stage(
            orig_frames_dir=orig_dir,
            cleaned_roi_dir=clean_dir,
            lens_masks_dir=lens_dir,
            roi_meta=rel_meta,
            output_mp4=tmp_path / "out.mp4",
            config=CompositeConfig(),
        )

    # P0-4: --save-frames-dir must reject symlinked child target without touching outside file!
    save_dir = tmp_path / "saved_frames"
    save_dir.mkdir()
    victim = tmp_path / "victim.png"
    victim.write_bytes(b"DO_NOT_OVERWRITE")
    (save_dir / "000000.png").symlink_to(victim)

    class _DummyPipe:
        def __init__(self, out_p: Path) -> None:
            self.out_path = out_p
            self.stdin = io.BytesIO()
            self.stderr = io.BytesIO(b"")

        def poll(self):
            return None

        def wait(self):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(composite, "open_video_writer_pipe", lambda *a, **kw: _DummyPipe(a[3]))
    monkeypatch.setattr(
        composite, "close_video_writer_pipe", lambda p, label="video": p.out_path.write_bytes(b"NEW")
    )

    real_src = tmp_path / "real_source.mp4"
    real_src.write_bytes(b"SRC")
    with pytest.raises(ValueError, match="Symlinked"):
        composite.run_composite_stage(
            orig_frames_dir=orig_dir,
            cleaned_roi_dir=clean_dir,
            lens_masks_dir=lens_dir,
            roi_meta=rel_meta,
            output_mp4=tmp_path / "out.mp4",
            config=CompositeConfig(save_frames_dir=save_dir, workers=1),
            source_video_path=real_src,
        )
    assert victim.read_bytes() == b"DO_NOT_OVERWRITE"

    # P1-11: failure publishing compare_mp4 MUST roll back output_mp4 to its previous content!
    out_mp4 = tmp_path / "rollback_out.mp4"
    cmp_mp4 = tmp_path / "rollback_out_compare.mp4"
    out_mp4.write_bytes(b"OLD_MAIN")
    cmp_mp4.write_bytes(b"OLD_COMPARE")

    real_replace = os.replace
    replace_count = 0

    def fail_on_compare_replace(src, dst):
        nonlocal replace_count
        if Path(dst) == cmp_mp4 and not str(src).startswith(str(tmp_path / ".bak_")):
            replace_count += 1
            if replace_count == 1:
                raise OSError("simulated compare publish failure")
        return real_replace(src, dst)

    monkeypatch.setattr(composite.os, "replace", fail_on_compare_replace)
    with pytest.raises(OSError, match="simulated compare publish failure"):
        composite.run_composite_stage(
            orig_frames_dir=orig_dir,
            cleaned_roi_dir=clean_dir,
            lens_masks_dir=lens_dir,
            roi_meta=rel_meta,
            output_mp4=out_mp4,
            config=CompositeConfig(compare=True, workers=1),
            source_video_path=real_src,
        )
    assert out_mp4.read_bytes() == b"OLD_MAIN"
    assert cmp_mp4.read_bytes() == b"OLD_COMPARE"
