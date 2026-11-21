"""Unit tests for Stage 3 clean-eye reference selection, eye-visibility checks, and glare-free synthesis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from eyeglass_glare_removal.config import ReferenceConfig
from eyeglass_glare_removal.reference import (
    build_clean_reference,
    mirror_clean_eye_across_bridge,
    stitch_left_right_roi,
    verify_eye_visibility,
)
from eyeglass_glare_removal.segment import compute_glare_hotspot_mask


def _render_eye_roi(
    *,
    left_state: str,
    right_state: str,
    skin_rgb: tuple[int, int, int] = (162, 124, 98),
    left_iris_rgb: tuple[int, int, int] = (42, 34, 28),
    right_iris_rgb: tuple[int, int, int] = (46, 36, 30),
    roi_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Render a realistic synthetic 128x128 ROI with two eyes.
    States: 'clean', 'glared', 'washed_175', 'glare_bridge'.
    Returns `(rgb, union_lens_mask, glare_mask)`.
    """
    H = W = roi_size
    rgb = np.full((H, W, 3), skin_rgb, dtype=np.uint8)
    lens = np.zeros((H, W), dtype=np.uint8)
    glare = np.zeros((H, W), dtype=np.uint8)

    left_center = (36, 64)
    right_center = (92, 64)
    axes = (22, 15)

    cv2.ellipse(lens, left_center, axes, 0, 0, 360, 255, -1)
    cv2.ellipse(lens, right_center, axes, 0, 0, 360, 255, -1)

    for cx, state, iris_col in [
        (left_center[0], left_state, left_iris_rgb),
        (right_center[0], right_state, right_iris_rgb),
    ]:
        # Natural sclera + dark iris + eyelid crease
        cv2.ellipse(rgb, (cx, 64), (11, 5), 0, 0, 360, (176, 172, 168), -1)
        cv2.circle(rgb, (cx, 64), 4, iris_col, -1)
        cv2.line(rgb, (cx - 14, 54), (cx + 14, 54), (65, 50, 42), 2)

        side_lens = np.zeros((H, W), dtype=np.uint8)
        cv2.ellipse(side_lens, (cx, 64), axes, 0, 0, 360, 255, -1)
        if state == "glared":
            cv2.rectangle(rgb, (cx - 16, 54), (cx + 16, 74), (248, 250, 255), -1)
            cv2.rectangle(glare, (cx - 16, 54), (cx + 16, 74), 255, -1)
        elif state == "washed_175":
            rgb[side_lens > 127] = (175, 175, 175)
            glare[side_lens > 127] = 255
        elif state == "glare_seam":
            # Heavy glare on rejected side reaching right up to bridge_x=64 (inside blend band!)
            x1 = 65 if cx == right_center[0] else 14
            x2 = 114 if cx == right_center[0] else 63
            cv2.rectangle(rgb, (x1, 50), (x2, 78), (250, 252, 255), -1)
            cv2.rectangle(glare, (x1, 50), (x2, 78), 255, -1)
        elif state == "glare_invade":
            # Heavy glare invading across the bridge into the clean eye too -> must fall back to None!
            x1 = 54 if cx == right_center[0] else 14
            x2 = 114 if cx == right_center[0] else 74
            cv2.rectangle(rgb, (x1, 50), (x2, 78), (250, 252, 255), -1)
            cv2.rectangle(glare, (x1, 50), (x2, 78), 255, -1)

    return rgb, lens, glare


def _setup_realistic_roi_set(
    tmp_path: Path,
    specs: list[tuple[str, str, tuple[int, int, int], tuple[int, int, int]]],
    glare_pairs: list[tuple[float, float]],
    lens_areas: list[tuple[int, int]] | None = None,
    skin_rgb: tuple[int, int, int] = (162, 124, 98),
) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any]]:
    roi_dir = tmp_path / "roi_frames"
    lens_dir = tmp_path / "lens_masks"
    glare_dir = tmp_path / "glare_masks"
    for d in (roi_dir, lens_dir, glare_dir):
        d.mkdir(parents=True, exist_ok=True)

    frames_meta = []
    stats_frames = []
    for i, ((ls, rs, li_col, ri_col), (gl, gr)) in enumerate(zip(specs, glare_pairs, strict=True)):
        rgb, lens, glare = _render_eye_roi(
            left_state=ls,
            right_state=rs,
            skin_rgb=skin_rgb,
            left_iris_rgb=li_col,
            right_iris_rgb=ri_col,
        )
        fn = f"{i:06d}.png"
        Image.fromarray(rgb).save(roi_dir / fn)
        Image.fromarray(lens).save(lens_dir / fn)
        Image.fromarray(glare).save(glare_dir / fn)
        frames_meta.append({"frame_idx": i, "filename": fn})
        la, ra = lens_areas[i] if lens_areas is not None else (1000, 1000)
        stats_frames.append(
            {
                "frame_idx": i,
                "left_lens_roi_px": la,
                "right_lens_roi_px": ra,
                "glare_left_pct": gl,
                "glare_right_pct": gr,
            }
        )

    roi_meta = {
        "roi_size": 128,
        "init_eye_geometry_roi": {"bridge": [64.0, 64.0]},
        "frames": frames_meta,
    }
    return roi_dir, lens_dir, glare_dir, roi_meta, {"frames": stats_frames}


def test_reference_auto_both_clean_single_frame(tmp_path: Path) -> None:
    roi_dir, lens_dir, glare_dir, roi_meta, mask_stats = _setup_realistic_roi_set(
        tmp_path,
        specs=[
            ("glared", "glared", (40, 30, 25), (40, 30, 25)),
            ("clean", "clean", (35, 25, 20), (35, 25, 20)),  # blink (tiny lens area -> rejected)
            ("clean", "clean", (25, 45, 30), (25, 45, 30)),  # clean & open both eyes!
        ],
        glare_pairs=[(40.0, 35.0), (0.0, 0.0), (0.5, 1.0)],
        lens_areas=[(1000, 1000), (200, 200), (1020, 990)],
    )
    cfg = ReferenceConfig(mode="auto", max_clean_glare_pct=3.0, roi_size=128)
    res = build_clean_reference(
        config=cfg,
        out_dir=tmp_path / "ref",
        roi_dir=roi_dir,
        roi_meta=roi_meta,
        mask_stats=mask_stats,
        lens_masks_dir=lens_dir,
        glare_masks_dir=glare_dir,
    )
    assert res.strategy == "single_clean_frame"
    assert res.ref_path is not None and res.ref_path.exists()
    assert res.left_frame_idx == 2
    assert res.right_frame_idx == 2


def test_reference_rejects_washed_out_rgb175_over_dark_skin_detector_to_reference(
    tmp_path: Path,
) -> None:
    """End-to-end detector -> reference regression: washed-out neutral RGB 175 NEVER passes as clean."""
    roi_dir = tmp_path / "roi_frames"
    lens_dir = tmp_path / "lens_masks"
    glare_dir = tmp_path / "glare_masks"
    for d in (roi_dir, lens_dir, glare_dir):
        d.mkdir(parents=True)

    rgb, lens, _ = _render_eye_roi(
        left_state="washed_175", right_state="washed_175", skin_rgb=(90, 62, 46)
    )
    det_glare = compute_glare_hotspot_mask(rgb, lens, dilate_px=2, feather_sigma=1.0)
    Image.fromarray(rgb).save(roi_dir / "000000.png")
    Image.fromarray(lens).save(lens_dir / "000000.png")
    Image.fromarray(det_glare).save(glare_dir / "000000.png")

    # Even if someone forged 0% glare stats, verify_eye_visibility STILL rejects washed-out frame!
    ok_vis, _ = verify_eye_visibility(rgb, lens)
    assert ok_vis is False

    roi_meta = {"roi_size": 128, "frames": [{"frame_idx": 0, "filename": "000000.png"}]}
    forged_stats = {
        "frames": [
            {
                "frame_idx": 0,
                "left_lens_roi_px": 1000,
                "right_lens_roi_px": 1000,
                "glare_left_pct": 0.0,
                "glare_right_pct": 0.0,
            }
        ]
    }
    cfg = ReferenceConfig(mode="auto", max_clean_glare_pct=3.0, roi_size=128)
    res = build_clean_reference(
        config=cfg,
        out_dir=tmp_path / "ref",
        roi_dir=roi_dir,
        roi_meta=roi_meta,
        mask_stats=forged_stats,
        lens_masks_dir=lens_dir,
        glare_masks_dir=glare_dir,
    )
    assert res.strategy == "reference_free_fallback"
    assert res.ref_path is None


def test_reference_auto_stitched_left_and_right(tmp_path: Path) -> None:
    roi_dir, lens_dir, glare_dir, roi_meta, mask_stats = _setup_realistic_roi_set(
        tmp_path,
        specs=[
            ("clean", "glared", (25, 35, 55), (40, 35, 30)),
            ("glared", "clean", (40, 35, 30), (55, 35, 25)),
        ],
        glare_pairs=[(0.2, 42.0), (38.0, 0.4)],
    )
    cfg = ReferenceConfig(mode="auto", max_clean_glare_pct=3.0, roi_size=128)
    res = build_clean_reference(
        config=cfg,
        out_dir=tmp_path / "ref",
        roi_dir=roi_dir,
        roi_meta=roi_meta,
        mask_stats=mask_stats,
        lens_masks_dir=lens_dir,
        glare_masks_dir=glare_dir,
    )
    assert res.strategy == "stitched_left_right"
    assert res.left_frame_idx == 0
    assert res.right_frame_idx == 1
    assert res.ref_path is not None


def test_reference_auto_mirror_excludes_bridge_bleed_glare_and_respects_no_mirror(
    tmp_path: Path,
) -> None:
    """Regression P1-5: rejected-side glare in seam band must NOT bleed; invading glare falls back."""
    roi_dir, lens_dir, glare_dir, roi_meta, mask_stats = _setup_realistic_roi_set(
        tmp_path / "seam_ok",
        specs=[
            ("clean", "glare_seam", (35, 30, 25), (35, 30, 25)),
            ("glared", "glared", (35, 30, 25), (35, 30, 25)),
        ],
        glare_pairs=[(0.5, 45.0), (35.0, 35.0)],
    )
    cfg_mirror = ReferenceConfig(mode="auto", allow_mirror=True, roi_size=128)
    res_m = build_clean_reference(
        config=cfg_mirror,
        out_dir=tmp_path / "ref_m",
        roi_dir=roi_dir,
        roi_meta=roi_meta,
        mask_stats=mask_stats,
        lens_masks_dir=lens_dir,
        glare_masks_dir=glare_dir,
    )
    assert res_m.strategy == "mirrored_left"
    assert res_m.ref_path is not None
    arr = np.asarray(Image.open(res_m.ref_path))
    # Bright glare (250+) from rejected right half (x=65..114) must NOT bleed onto either side!
    assert float(arr[64, 92].mean()) < 100.0  # mirrored clean iris, not white glare
    assert float(arr[64, 68].mean()) < 170.0  # right side of seam band (was 252 glare -> clean skin!)
    assert float(arr[64, 60].mean()) < 170.0  # left side of seam band (no mirrored glare bleed!)

    # When glare invades into the clean eye across the seam -> post-synthesis check rejects -> None!
    roi_inv, lens_inv, glare_inv, meta_inv, stats_inv = _setup_realistic_roi_set(
        tmp_path / "seam_invade",
        specs=[("clean", "glare_invade", (35, 30, 25), (35, 30, 25))],
        glare_pairs=[(0.5, 50.0)],
    )
    res_inv = build_clean_reference(
        config=cfg_mirror,
        out_dir=tmp_path / "ref_inv",
        roi_dir=roi_inv,
        roi_meta=meta_inv,
        mask_stats=stats_inv,
        lens_masks_dir=lens_inv,
        glare_masks_dir=glare_inv,
    )
    assert res_inv.strategy == "reference_free_fallback"
    assert res_inv.ref_path is None

    cfg_nomirror = ReferenceConfig(mode="auto", allow_mirror=False, roi_size=128)
    res_nm = build_clean_reference(
        config=cfg_nomirror,
        out_dir=tmp_path / "ref_nm",
        roi_dir=roi_dir,
        roi_meta=roi_meta,
        mask_stats=mask_stats,
        lens_masks_dir=lens_dir,
        glare_masks_dir=glare_dir,
    )
    assert res_nm.strategy == "reference_free_fallback"
    assert res_nm.ref_path is None


def test_reference_user_image_and_none_mode(tmp_path: Path) -> None:
    user_img = tmp_path / "user_selfie.png"
    arr = np.full((240, 320, 3), (90, 130, 170), dtype=np.uint8)
    Image.fromarray(arr).save(user_img)

    cfg_img = ReferenceConfig(mode="image", ref_image_path=user_img, roi_size=128)
    res_img = build_clean_reference(config=cfg_img, out_dir=tmp_path / "ref_user")
    assert res_img.ref_path is not None and res_img.ref_path.exists()
    assert Image.open(res_img.ref_path).size == (128, 128)

    cfg_none = ReferenceConfig(mode="none", roi_size=128)
    res_none = build_clean_reference(config=cfg_none, out_dir=tmp_path / "ref_none")
    assert res_none.ref_path is None
    assert res_none.strategy == "disabled"


def test_stitch_and_mirror_helpers_preserve_bridge_symmetry() -> None:
    left = np.full((64, 64, 3), (10, 20, 30), dtype=np.uint8)
    right = np.full((64, 64, 3), (200, 180, 160), dtype=np.uint8)
    stitched = stitch_left_right_roi(left, right, bridge_x=32.0, blend_half_width=4)
    assert tuple(stitched[32, 5]) == (10, 20, 30)
    assert tuple(stitched[32, 58]) == (200, 180, 160)

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[30:35, 16:20] = (120, 220, 80)
    mir = mirror_clean_eye_across_bridge(img, clean_side="left", bridge_x=28.0, blend_half_width=3)
    assert int(mir[32, 38, 1]) > 180
