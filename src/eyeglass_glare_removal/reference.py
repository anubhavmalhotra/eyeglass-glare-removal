"""Automatic clean-eye reference image builder for Wan2.1-VACE conditioning."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from eyeglass_glare_removal._workspace import (
    atomic_write_text,
    ensure_safe_child_file,
    make_unique_temp_sibling,
    reject_symlink_path,
    validate_safe_basename,
)
from eyeglass_glare_removal.config import ReferenceConfig
from eyeglass_glare_removal.detect import (
    compute_adaptive_crop_size,
    detect_eye_geometry,
    make_roi_box,
)
from eyeglass_glare_removal.segment import compute_glare_hotspot_mask, estimate_local_skin_stats

logger = logging.getLogger(__name__)


@dataclass
class ReferenceResult:
    """Outcome of Stage 3 reference construction."""

    mode: str
    strategy: str
    ref_path: Path | None
    left_frame_idx: int | None = None
    right_frame_idx: int | None = None
    left_glare_pct: float | None = None
    right_glare_pct: float | None = None
    bridge_x: float | None = None
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ref_path"] = str(self.ref_path) if self.ref_path is not None else None
        return d


def _prepare_user_reference_image(ref_image_path: Path, roi_size: int) -> tuple[np.ndarray, bool]:
    """Load and crop/resize a user-supplied clean eye reference photo to roi_size x roi_size."""
    if not ref_image_path.exists():
        raise FileNotFoundError(f"Reference image not found: {ref_image_path}")

    bgr = cv2.imread(str(ref_image_path))
    if bgr is None:
        raise RuntimeError(f"Failed to decode reference image: {ref_image_path}")

    H, W = bgr.shape[:2]
    auto_cropped = False
    geom = detect_eye_geometry(bgr, detector="auto")
    if geom is not None:
        crop_sz = compute_adaptive_crop_size(geom.inter_ocular_px, W, H)
        if crop_sz < int(0.75 * min(W, H)):
            cx, cy = geom.roi_center()
            x1, y1, x2, y2 = make_roi_box(cx, cy, crop_sz, W, H)
            bgr = bgr[y1:y2, x1:x2]
            auto_cropped = True

    if not auto_cropped:
        side = min(H, W)
        y1 = (H - side) // 2
        x1 = (W - side) // 2
        bgr = bgr[y1 : y1 + side, x1 : x1 + side]

    roi_bgr = cv2.resize(bgr, (roi_size, roi_size), interpolation=cv2.INTER_LANCZOS4)
    return cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB), auto_cropped


def _split_left_right_lens_masks(
    lens_mask: np.ndarray, bridge_x: float
) -> tuple[np.ndarray, np.ndarray]:
    """Split a union lens mask cleanly into viewer-left and viewer-right binary masks."""
    m = (lens_mask > 127).astype(np.uint8)
    H, W = m.shape[:2]
    left = np.zeros((H, W), dtype=np.uint8)
    right = np.zeros((H, W), dtype=np.uint8)
    n_lbl, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n_lbl >= 3:
        comps = sorted(
            range(1, n_lbl), key=lambda k: int(stats[k, cv2.CC_STAT_AREA]), reverse=True
        )[:2]
        comps.sort(key=lambda k: float(centroids[k][0]))
        left[labels == comps[0]] = 255
        right[labels == comps[1]] = 255
        return left, right

    bx = int(np.clip(round(bridge_x), 1, W - 1))
    left[:, :bx] = m[:, :bx] * 255
    right[:, bx:] = m[:, bx:] * 255
    return left, right


def _estimate_bridge_x(
    roi_size: int,
    roi_meta: dict[str, Any] | None = None,
    lens_mask: np.ndarray | None = None,
) -> float:
    """Estimate horizontal nose-bridge x-coordinate in ROI space."""
    if lens_mask is not None:
        m = (lens_mask > 127).astype(np.uint8)
        n_lbl, _, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n_lbl >= 3:
            comps = sorted(
                range(1, n_lbl), key=lambda k: int(stats[k, cv2.CC_STAT_AREA]), reverse=True
            )[:2]
            comps.sort(key=lambda k: float(centroids[k][0]))
            left_x2 = float(stats[comps[0], cv2.CC_STAT_LEFT] + stats[comps[0], cv2.CC_STAT_WIDTH])
            right_x1 = float(stats[comps[1], cv2.CC_STAT_LEFT])
            if right_x1 >= left_x2 - 10:
                return float(np.clip(0.5 * (left_x2 + right_x1), 0.25 * roi_size, 0.75 * roi_size))

    if roi_meta and isinstance(roi_meta.get("init_eye_geometry_roi"), dict):
        bridge = roi_meta["init_eye_geometry_roi"].get("bridge")
        if bridge and len(bridge) == 2:
            return float(np.clip(float(bridge[0]), 0.25 * roi_size, 0.75 * roi_size))

    return 0.5 * float(roi_size)


def verify_eye_visibility(
    roi_rgb: np.ndarray,
    eye_mask: np.ndarray,
    full_lens_mask: np.ndarray | None = None,
) -> tuple[bool, dict[str, float]]:
    """
    Require positive eye-visibility evidence inside `eye_mask` before accepting a candidate
    as a clean reference eye (prevents washed-out or fogged reflections from passing).

    Checks:
      1) Dynamic range (`p90_L - p10_L >= 18.0`): rejects uniform/washed-out reflections.
      2) Dark iris/pupil/lash presence (`p10_L <= min(125.0, skin_L + 15.0)`): rejects bright
         uniform sheets covering the eye.
      3) Not globally washed out vs surrounding face skin (`med_L <= min(225.0, skin_L + 45.0)`).
      4) Structural gradient/edge energy present (`grad_p75 >= 4.0`).
    """
    region = eye_mask > 127
    if int(region.sum()) < 40:
        return False, {"area": float(region.sum())}

    ref_mask = full_lens_mask if full_lens_mask is not None else eye_mask
    skin_stats = estimate_local_skin_stats(roi_rgb, ref_mask)
    skin_L = skin_stats["L_med"]

    bgr = roi_rgb[:, :, ::-1]
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    L_vals = L[region]

    p10 = float(np.percentile(L_vals, 10))
    p50 = float(np.median(L_vals))
    p90 = float(np.percentile(L_vals, 90))
    dyn_range = p90 - p10

    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy) / 4.0
    grad_p75 = float(np.percentile(grad[region], 75))

    metrics = {
        "skin_L": round(skin_L, 2),
        "p10_L": round(p10, 2),
        "p50_L": round(p50, 2),
        "p90_L": round(p90, 2),
        "dyn_range": round(dyn_range, 2),
        "grad_p75": round(grad_p75, 2),
    }

    has_contrast = dyn_range >= 18.0
    has_dark_iris = p10 <= min(125.0, skin_L + 15.0)
    not_washed_out = p50 <= min(225.0, skin_L + 45.0)
    has_structure = grad_p75 >= 4.0

    return (has_contrast and has_dark_iris and not_washed_out and has_structure), metrics


def _fallback_eye_box_mask(
    roi_size: int, side: str, roi_meta: dict[str, Any] | None
) -> np.ndarray:
    """Fallback eye region mask if lens mask PNGs are unavailable on disk."""
    m = np.zeros((roi_size, roi_size), dtype=np.uint8)
    geom = roi_meta.get("init_eye_geometry_roi") if isinstance(roi_meta, dict) else None
    if isinstance(geom, dict) and "left_eye" in geom and "right_eye" in geom:
        ex, ey = geom["left_eye"] if side == "left" else geom["right_eye"]
        iod = float(geom.get("inter_ocular_px", 0.34 * roi_size))
        rx = max(12, int(round(0.32 * iod)))
        ry = max(8, int(round(0.22 * iod)))
        cv2.ellipse(m, (int(round(ex)), int(round(ey))), (rx, ry), 0, 0, 360, 255, -1)
    else:
        cx = int(round(roi_size * (0.33 if side == "left" else 0.67)))
        cy = int(round(roi_size * 0.48))
        cv2.ellipse(m, (cx, cy), (int(roi_size * 0.12), int(roi_size * 0.08)), 0, 0, 360, 255, -1)
    return m


def stitch_left_right_roi(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    bridge_x: float,
    blend_half_width: int = 12,
    *,
    left_glare_mask: np.ndarray | None = None,
    right_glare_mask: np.ndarray | None = None,
    left_lens_mask: np.ndarray | None = None,
    right_lens_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Seamlessly stitch the clean left half of `left_rgb` with the clean right half of `right_rgb`
    across `bridge_x`, strictly preventing rejected-side glare from bleeding across the seam.
    """
    H, W = left_rgb.shape[:2]
    if right_rgb.shape[:2] != (H, W):
        right_rgb = cv2.resize(right_rgb, (W, H), interpolation=cv2.INTER_LANCZOS4)

    # Narrow blend band so it never crosses into either lens mask when provided
    eff_bw = float(max(1, blend_half_width))
    if left_lens_mask is not None and int((left_lens_mask > 127).sum()) > 0:
        l_xs = np.where(left_lens_mask > 127)[1]
        gap_l = max(1.0, bridge_x - float(l_xs.max()) - 2.0)
        eff_bw = min(eff_bw, gap_l)
    if right_lens_mask is not None and int((right_lens_mask > 127).sum()) > 0:
        r_xs = np.where(right_lens_mask > 127)[1]
        gap_r = max(1.0, float(r_xs.min()) - bridge_x - 2.0)
        eff_bw = min(eff_bw, gap_r)

    xs = np.arange(W, dtype=np.float32)
    t = np.clip((xs - (bridge_x - eff_bw)) / (2.0 * max(1.0, eff_bw)), 0.0, 1.0)
    w_right_2d = np.broadcast_to((0.5 * (1.0 - np.cos(np.pi * t)))[None, :], (H, W)).copy()

    # Constrain using per-eye lens and glare masks so rejected-side glare never bleeds in
    if left_glare_mask is not None:
        g_l = left_glare_mask > 32
        # On right side of bridge, glare in left_rgb must be 100% replaced by right_rgb
        w_right_2d = np.where(g_l & (xs[None, :] >= bridge_x - eff_bw), 1.0, w_right_2d)
    if right_glare_mask is not None:
        g_r = right_glare_mask > 32
        # On left side of bridge, glare in right_rgb must be 100% excluded (keep left_rgb)
        w_right_2d = np.where(g_r & (xs[None, :] <= bridge_x + eff_bw), 0.0, w_right_2d)
    if left_lens_mask is not None:
        w_right_2d = np.where(left_lens_mask > 127, 0.0, w_right_2d)
    if right_lens_mask is not None:
        w_right_2d = np.where(right_lens_mask > 127, 1.0, w_right_2d)

    w_right_2d = cv2.GaussianBlur(w_right_2d.astype(np.float32), (3, 3), 0.7)
    # Re-enforce hard lens and glare exclusion after smoothing so zero glare bleeds across seam
    if left_glare_mask is not None:
        w_right_2d = np.where((left_glare_mask > 32) & (xs[None, :] >= bridge_x - eff_bw), 1.0, w_right_2d)
    if right_glare_mask is not None:
        w_right_2d = np.where((right_glare_mask > 32) & (xs[None, :] <= bridge_x + eff_bw), 0.0, w_right_2d)
    if left_lens_mask is not None:
        w_right_2d = np.where(left_lens_mask > 127, 0.0, w_right_2d)
    if right_lens_mask is not None:
        w_right_2d = np.where(right_lens_mask > 127, 1.0, w_right_2d)

    w_right = np.clip(w_right_2d, 0.0, 1.0)[:, :, None]
    blended = left_rgb.astype(np.float32) * (1.0 - w_right) + right_rgb.astype(np.float32) * w_right
    return np.clip(blended, 0.0, 255.0).round().astype(np.uint8)


def _mirror_array_across_x(arr: np.ndarray, bridge_x: float, interp: int) -> np.ndarray:
    H, W = arr.shape[:2]
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    map_x_1d = np.clip(2.0 * float(bridge_x) - xs, 0.0, float(W - 1))
    map_x = np.broadcast_to(map_x_1d[None, :], (H, W)).astype(np.float32)
    map_y = np.broadcast_to(ys[:, None], (H, W)).astype(np.float32)
    return cv2.remap(arr, map_x, map_y, interpolation=interp, borderMode=cv2.BORDER_REPLICATE)


def mirror_clean_eye_across_bridge(
    rgb: np.ndarray,
    clean_side: str,
    bridge_x: float,
    blend_half_width: int = 12,
    *,
    glare_mask: np.ndarray | None = None,
    clean_lens_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Mirror the clean eye half across the vertical nose-bridge axis x = bridge_x,
    explicitly excluding known glare pixels on the rejected half from bleeding across the seam.
    """
    if clean_side not in ("left", "right"):
        raise ValueError(f"clean_side must be 'left' or 'right', got {clean_side!r}")

    mirrored = _mirror_array_across_x(rgb, bridge_x, cv2.INTER_LINEAR)
    mir_glare = (
        _mirror_array_across_x(glare_mask, bridge_x, cv2.INTER_NEAREST)
        if glare_mask is not None
        else None
    )
    mir_lens = (
        _mirror_array_across_x(clean_lens_mask, bridge_x, cv2.INTER_NEAREST)
        if clean_lens_mask is not None
        else None
    )

    if clean_side == "left":
        return stitch_left_right_roi(
            rgb,
            mirrored,
            bridge_x=bridge_x,
            blend_half_width=blend_half_width,
            left_glare_mask=glare_mask,
            right_glare_mask=mir_glare,
            left_lens_mask=clean_lens_mask,
            right_lens_mask=mir_lens,
        )
    return stitch_left_right_roi(
        mirrored,
        rgb,
        bridge_x=bridge_x,
        blend_half_width=blend_half_width,
        left_glare_mask=mir_glare,
        right_glare_mask=glare_mask,
        left_lens_mask=mir_lens,
        right_lens_mask=clean_lens_mask,
    )


def _frame_sharpness(roi_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_clean_eye_candidates(
    frame_stats: list[dict[str, Any]],
    *,
    max_clean_glare_pct: float = 3.0,
    min_lens_area_ratio: float = 0.7,
    sharpness_fn: Any = None,
    visibility_fn: Any = None,
) -> tuple[int | None, int | None, int | None]:
    """
    Scan per-frame segmentation stats + positive eye-visibility checks to find clean eye frames.

    `visibility_fn(frame_idx, side) -> bool`: verifies positive eye/iris/contrast structure
    inside the candidate eye so undetected uniform washouts can NEVER pass as clean.
    """
    if not frame_stats:
        return None, None, None

    l_areas = [
        int(s.get("left_lens_roi_px", 0))
        for s in frame_stats
        if int(s.get("left_lens_roi_px", 0)) > 0
    ]
    r_areas = [
        int(s.get("right_lens_roi_px", 0))
        for s in frame_stats
        if int(s.get("right_lens_roi_px", 0)) > 0
    ]
    med_l = float(np.median(l_areas)) if l_areas else 0.0
    med_r = float(np.median(r_areas)) if r_areas else 0.0

    min_l = max(1.0, min_lens_area_ratio * med_l)
    min_r = max(1.0, min_lens_area_ratio * med_r)

    both_cands: list[tuple[float, float, int]] = []
    left_cands: list[tuple[float, int]] = []
    right_cands: list[tuple[float, int]] = []

    for s in frame_stats:
        idx = int(s["frame_idx"])
        lp = int(s.get("left_lens_roi_px", 0))
        rp = int(s.get("right_lens_roi_px", 0))
        gl = float(s.get("glare_left_pct", 100.0))
        gr = float(s.get("glare_right_pct", 100.0))

        left_ok = (lp >= min_l) and (gl <= max_clean_glare_pct)
        right_ok = (rp >= min_r) and (gr <= max_clean_glare_pct)

        if left_ok and visibility_fn is not None:
            left_ok = bool(visibility_fn(idx, "left"))
        if right_ok and visibility_fn is not None:
            right_ok = bool(visibility_fn(idx, "right"))

        if left_ok:
            left_cands.append((gl, idx))
        if right_ok:
            right_cands.append((gr, idx))
        if left_ok and right_ok:
            both_cands.append((max(gl, gr), gl + gr, idx))

    def _pick_best_single(cands: list[tuple[float, int]]) -> int | None:
        if not cands:
            return None
        cands.sort(key=lambda x: (x[0], x[1]))
        best_glare = cands[0][0]
        top = [idx for g, idx in cands if g <= best_glare + 0.5][:8]
        if sharpness_fn is not None and len(top) > 1:
            return max(top, key=lambda idx: float(sharpness_fn(idx)))
        return top[0]

    best_both: int | None = None
    if both_cands:
        both_cands.sort(key=lambda x: (x[0], x[1], x[2]))
        best_max_g = both_cands[0][0]
        top_both = [idx for mg, _, idx in both_cands if mg <= best_max_g + 0.5][:8]
        if sharpness_fn is not None and len(top_both) > 1:
            best_both = max(top_both, key=lambda idx: float(sharpness_fn(idx)))
        else:
            best_both = top_both[0]

    return best_both, _pick_best_single(left_cands), _pick_best_single(right_cands)


def _verify_synthesized_reference(
    synth_rgb: np.ndarray,
    left_lens_mask: np.ndarray,
    right_lens_mask: np.ndarray,
    max_clean_glare_pct: float,
) -> tuple[bool, float, float]:
    """
    Validate a stitched/mirrored reference image before saving:
    checks both photometric glare % AND positive eye visibility on both lenses.
    """
    union_lens = np.maximum(left_lens_mask, right_lens_mask)
    if int((union_lens > 127).sum()) == 0:
        return False, 100.0, 100.0

    glare = compute_glare_hotspot_mask(synth_rgb, union_lens, dilate_px=2, feather_sigma=1.0)
    l_px = max(1, int((left_lens_mask > 127).sum()))
    r_px = max(1, int((right_lens_mask > 127).sum()))
    gl_pct = 100.0 * float(((glare > 127) & (left_lens_mask > 127)).sum()) / float(l_px)
    gr_pct = 100.0 * float(((glare > 127) & (right_lens_mask > 127)).sum()) / float(r_px)

    if gl_pct > max_clean_glare_pct or gr_pct > max_clean_glare_pct:
        return False, gl_pct, gr_pct

    vis_l, _ = verify_eye_visibility(synth_rgb, left_lens_mask, full_lens_mask=union_lens)
    vis_r, _ = verify_eye_visibility(synth_rgb, right_lens_mask, full_lens_mask=union_lens)
    return (vis_l and vis_r), gl_pct, gr_pct


def _save_png_atomic(rgb: np.ndarray, dest: Path) -> None:
    tmp = make_unique_temp_sibling(dest)
    try:
        Image.fromarray(rgb).save(tmp, format="PNG")
        os.replace(tmp, dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def build_clean_reference(
    *,
    config: ReferenceConfig,
    out_dir: Path,
    roi_dir: Path | None = None,
    roi_meta: dict[str, Any] | None = None,
    mask_stats: dict[str, Any] | None = None,
    lens_masks_dir: Path | None = None,
    glare_masks_dir: Path | None = None,
) -> ReferenceResult:
    """
    Execute Stage 3: construct a verified clean-eye reference image or gracefully return None.

    Requires positive eye-visibility evidence on candidates AND verifies any stitched/mirrored
    result before saving so glared or washed-out frames are NEVER fed to VACE as a reference.
    """
    out_dir = Path(out_dir)
    reject_symlink_path(out_dir, "reference output directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_out = out_dir / "clean_eye_ref.png"
    meta_out = out_dir / "reference_meta.json"
    reject_symlink_path(ref_out, "clean_eye_ref.png")

    def _save_meta(res: ReferenceResult) -> ReferenceResult:
        atomic_write_text(meta_out, json.dumps(res.to_dict(), indent=2))
        return res

    if config.mode == "none":
        if ref_out.exists():
            ref_out.unlink()
        return _save_meta(
            ReferenceResult(
                mode="none",
                strategy="disabled",
                ref_path=None,
                details="Reference image disabled (--no-ref); running reference-free VACE.",
            )
        )

    if config.mode == "image":
        assert config.ref_image_path is not None
        rgb, auto_cropped = _prepare_user_reference_image(config.ref_image_path, config.roi_size)
        _save_png_atomic(rgb, ref_out)
        return _save_meta(
            ReferenceResult(
                mode="image",
                strategy="user_image_cropped" if auto_cropped else "user_image_direct",
                ref_path=ref_out,
                details=f"Loaded user reference from {config.ref_image_path} (auto_cropped={auto_cropped}).",
            )
        )

    if roi_dir is None or mask_stats is None or roi_meta is None:
        raise ValueError(
            "ReferenceConfig(mode='auto') requires roi_dir, roi_meta, and mask_stats."
        )

    roi_dir = Path(roi_dir)
    reject_symlink_path(roi_dir, "roi_dir")
    frame_stats = mask_stats.get("frames") or []
    frame_records = roi_meta.get("frames") or []
    idx_to_fn = {
        int(r["frame_idx"]): validate_safe_basename(str(r["filename"])) for r in frame_records
    }
    idx_to_stat = {int(s["frame_idx"]): s for s in frame_stats}

    def _load_rgb(idx: int) -> np.ndarray:
        fp = ensure_safe_child_file(roi_dir, idx_to_fn[idx])
        im = Image.open(fp).convert("RGB")
        if im.size != (config.roi_size, config.roi_size):
            im = im.resize((config.roi_size, config.roi_size), Image.Resampling.LANCZOS)
        return np.asarray(im, dtype=np.uint8)

    def _load_gray(d: Path | None, idx: int) -> np.ndarray | None:
        if d is None or not Path(d).is_dir():
            return None
        p = ensure_safe_child_file(Path(d), idx_to_fn[idx], allow_missing=True)
        if not p.exists():
            return None
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        if m.shape[:2] != (config.roi_size, config.roi_size):
            m = cv2.resize(m, (config.roi_size, config.roi_size), interpolation=cv2.INTER_NEAREST)
        return m

    def _get_eye_masks(idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lens_m = _load_gray(lens_masks_dir, idx)
        bx = _estimate_bridge_x(config.roi_size, roi_meta=roi_meta, lens_mask=lens_m)
        if lens_m is not None and int((lens_m > 127).sum()) > 0:
            lm, rm = _split_left_right_lens_masks(lens_m, bx)
            return lm, rm, lens_m
        lm = _fallback_eye_box_mask(config.roi_size, "left", roi_meta)
        rm = _fallback_eye_box_mask(config.roi_size, "right", roi_meta)
        return lm, rm, np.maximum(lm, rm)

    rgb_cache: dict[int, np.ndarray] = {}

    def _cached_rgb(idx: int) -> np.ndarray:
        if idx not in rgb_cache:
            if len(rgb_cache) > 16:
                rgb_cache.clear()
            rgb_cache[idx] = _load_rgb(idx)
        return rgb_cache[idx]

    sharpness_cache: dict[int, float] = {}

    def _sharpness(idx: int) -> float:
        if idx not in sharpness_cache:
            sharpness_cache[idx] = _frame_sharpness(_cached_rgb(idx))
        return sharpness_cache[idx]

    vis_cache: dict[tuple[int, str], bool] = {}

    def _visibility(idx: int, side: str) -> bool:
        key = (idx, side)
        if key not in vis_cache:
            rgb = _cached_rgb(idx)
            lm, rm, full_m = _get_eye_masks(idx)
            eye_m = lm if side == "left" else rm
            ok, _ = verify_eye_visibility(rgb, eye_m, full_lens_mask=full_m)
            vis_cache[key] = ok
        return vis_cache[key]

    best_both, best_l, best_r = select_clean_eye_candidates(
        frame_stats,
        max_clean_glare_pct=config.max_clean_glare_pct,
        min_lens_area_ratio=config.min_lens_area_ratio,
        sharpness_fn=_sharpness,
        visibility_fn=_visibility,
    )

    # Case 1: Single frame clean and eye-visible on both eyes
    if best_both is not None:
        rgb = _load_rgb(best_both)
        _save_png_atomic(rgb, ref_out)
        st = idx_to_stat[best_both]
        return _save_meta(
            ReferenceResult(
                mode="auto",
                strategy="single_clean_frame",
                ref_path=ref_out,
                left_frame_idx=best_both,
                right_frame_idx=best_both,
                left_glare_pct=float(st["glare_left_pct"]),
                right_glare_pct=float(st["glare_right_pct"]),
                details=f"Selected verified clean frame {best_both} for both eyes.",
            )
        )

    # Case 2: Both eyes clean on different frames -> mask-constrained stitch + post-verification
    if best_l is not None and best_r is not None:
        rgb_l = _load_rgb(best_l)
        rgb_r = _load_rgb(best_r)
        lm_l, _, full_l = _get_eye_masks(best_l)
        _, rm_r, _ = _get_eye_masks(best_r)
        glare_l = _load_gray(glare_masks_dir, best_l)
        glare_r = _load_gray(glare_masks_dir, best_r)
        bridge_x = _estimate_bridge_x(config.roi_size, roi_meta=roi_meta, lens_mask=full_l)
        stitched = stitch_left_right_roi(
            rgb_l,
            rgb_r,
            bridge_x=bridge_x,
            left_glare_mask=glare_l,
            right_glare_mask=glare_r,
            left_lens_mask=lm_l,
            right_lens_mask=rm_r,
        )
        ok_syn, gl_syn, gr_syn = _verify_synthesized_reference(
            stitched, lm_l, rm_r, config.max_clean_glare_pct
        )
        if ok_syn:
            _save_png_atomic(stitched, ref_out)
            return _save_meta(
                ReferenceResult(
                    mode="auto",
                    strategy="stitched_left_right",
                    ref_path=ref_out,
                    left_frame_idx=best_l,
                    right_frame_idx=best_r,
                    left_glare_pct=round(gl_syn, 2),
                    right_glare_pct=round(gr_syn, 2),
                    bridge_x=round(bridge_x, 2),
                    details=(
                        f"Stitched verified left eye (frame {best_l}) and right eye (frame {best_r})."
                    ),
                )
            )
        logger.warning(
            "Stitched reference failed clean verification (L=%.1f%%, R=%.1f%%); falling back.",
            gl_syn,
            gr_syn,
        )

    # Case 3: Single clean eye -> glare-masked mirror across bridge + post-verification
    if config.allow_mirror and (best_l is not None or best_r is not None):
        clean_side = "left" if best_l is not None else "right"
        src_idx = best_l if best_l is not None else best_r
        assert src_idx is not None
        rgb = _load_rgb(src_idx)
        lm, rm, full_m = _get_eye_masks(src_idx)
        clean_lens = lm if clean_side == "left" else rm
        glare_m = _load_gray(glare_masks_dir, src_idx)
        bridge_x = _estimate_bridge_x(config.roi_size, roi_meta=roi_meta, lens_mask=full_m)

        mirrored = mirror_clean_eye_across_bridge(
            rgb,
            clean_side=clean_side,
            bridge_x=bridge_x,
            glare_mask=glare_m,
            clean_lens_mask=clean_lens,
        )
        mir_lens = _mirror_array_across_x(clean_lens, bridge_x, cv2.INTER_NEAREST)
        syn_l = clean_lens if clean_side == "left" else mir_lens
        syn_r = mir_lens if clean_side == "left" else clean_lens

        ok_syn, gl_syn, gr_syn = _verify_synthesized_reference(
            mirrored, syn_l, syn_r, config.max_clean_glare_pct
        )
        if ok_syn:
            _save_png_atomic(mirrored, ref_out)
            return _save_meta(
                ReferenceResult(
                    mode="auto",
                    strategy=f"mirrored_{clean_side}",
                    ref_path=ref_out,
                    left_frame_idx=src_idx if clean_side == "left" else None,
                    right_frame_idx=src_idx if clean_side == "right" else None,
                    left_glare_pct=round(gl_syn, 2),
                    right_glare_pct=round(gr_syn, 2),
                    bridge_x=round(bridge_x, 2),
                    details=(
                        f"Mirrored verified clean {clean_side} eye (frame {src_idx}) "
                        f"across bridge x={bridge_x:.1f}."
                    ),
                )
            )
        logger.warning(
            "Mirrored reference failed clean verification (L=%.1f%%, R=%.1f%%); falling back.",
            gl_syn,
            gr_syn,
        )

    # Case 4: Graceful reference-free fallback (never feed glared/washed-out reference)
    if ref_out.exists():
        ref_out.unlink()
    return _save_meta(
        ReferenceResult(
            mode="auto",
            strategy="reference_free_fallback",
            ref_path=None,
            details=(
                f"No clean, visibly sharp eye reference could be verified below "
                f"max_clean_glare_pct={config.max_clean_glare_pct}%; "
                "falling back to reference-free VACE."
            ),
        )
    )
