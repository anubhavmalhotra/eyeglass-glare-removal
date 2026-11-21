"""Automatic clean-eye reference image builder for Wan2.1-VACE conditioning."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from eyeglass_glare_removal._workspace import (
    atomic_swap_directory,
    atomic_write_text,
    ensure_safe_child_file,
    hash_file_sha256,
    make_unique_temp_dir,
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
    prep_generation: str | None = None
    segment_generation: str | None = None
    ref_sha256: str | None = None
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
    Require positive, spatially localized eye-visibility evidence inside the central
    iris/eyelid region of `eye_mask` before accepting a candidate as a clean reference.

    Rejects both washed-out uniform glare sheets AND non-eye periodic/striped reflections
    spanning the lens:
      1) Central eye zone dynamic range (`p90 - p10 >= 18.0`) & non-washout median luminance.
      2) Localized compact dark iris/pupil cluster centered inside the central eye zone
         (bounded width/height & aspect ratio, NOT edge-to-edge stripes).
      3) Non-stripe / non-reflection periphery check: rejects periodic stripe patterns where
         the outer lens periphery mirrors the same high contrast / dark-bright alternation as
         the central eye zone.
    """
    region = eye_mask > 127
    lens_area = int(region.sum())
    if lens_area < 40:
        return False, {"area": float(lens_area)}

    ys, xs = np.where(region)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)

    H, W = eye_mask.shape[:2]
    central_box = np.zeros((H, W), dtype=bool)
    cx1 = x1 + int(round(0.18 * bw))
    cx2 = x1 + int(round(0.82 * bw))
    cy1 = y1 + int(round(0.16 * bh))
    cy2 = y1 + int(round(0.84 * bh))
    central_box[cy1 : max(cy1 + 1, cy2), cx1 : max(cx1 + 1, cx2)] = True
    central_zone = region & central_box
    periph_zone = region & (~central_box)

    if int(central_zone.sum()) < 20:
        return False, {"central_area": float(central_zone.sum())}

    ref_mask = full_lens_mask if full_lens_mask is not None else eye_mask
    skin_stats = estimate_local_skin_stats(roi_rgb, ref_mask)
    skin_L = skin_stats["L_med"]

    bgr = roi_rgb[:, :, ::-1]
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    L_c = L[central_zone]

    p10_c = float(np.percentile(L_c, 10))
    p50_c = float(np.median(L_c))
    p75_c = float(np.percentile(L_c, 75))
    p90_c = float(np.percentile(L_c, 90))
    dyn_range = p90_c - p10_c

    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy) / 4.0
    grad_p75 = float(np.percentile(grad[central_zone], 75))

    has_contrast = dyn_range >= 18.0
    has_dark_iris_level = p10_c <= min(115.0, skin_L + 10.0)
    not_washed_out = p50_c <= min(225.0, skin_L + 40.0)
    has_structure = grad_p75 >= 4.0

    # Locate dark iris/pupil connected components across the entire lens so edge-to-edge stripes
    # cannot impersonate a compact central iris blob!
    dark_thresh = min(115.0, skin_L + 5.0, p75_c - 15.0)
    dark_mask = (L <= dark_thresh) & region
    n_dc, _, d_stats, d_centroids = cv2.connectedComponentsWithStats(
        dark_mask.astype(np.uint8), connectivity=8
    )

    compact_central_iris = False
    spanning_stripes = 0
    for cid in range(1, n_dc):
        area = int(d_stats[cid, cv2.CC_STAT_AREA])
        cw = int(d_stats[cid, cv2.CC_STAT_WIDTH])
        ch = int(d_stats[cid, cv2.CC_STAT_HEIGHT])
        ccx, ccy = float(d_centroids[cid][0]), float(d_centroids[cid][1])

        if cw >= 0.75 * bw or ch >= 0.78 * bh:
            spanning_stripes += 1
            continue

        aspect = float(cw) / float(max(1, ch))
        in_center = (x1 + 0.22 * bw <= ccx <= x1 + 0.78 * bw) and (
            y1 + 0.20 * bh <= ccy <= y1 + 0.80 * bh
        )
        if (
            in_center
            and 10 <= area <= int(0.42 * lens_area)
            and cw <= 0.65 * bw
            and ch <= 0.72 * bh
            and 0.35 <= aspect <= 2.8
        ):
            compact_central_iris = True

    # Reject periodic/striped full-lens reflection patterns:
    # 1) Multiple dark bands spanning across the full lens width/height, OR
    # 2) Outer lens periphery has high glare-like contrast/variance matching the center
    #    (in a real eye, the iris & sclera contrast sit in the center while the periphery is
    #     predominantly smooth skin/eyelid).
    periph_non_eye = False
    if int(periph_zone.sum()) >= 30:
        L_p = L[periph_zone]
        periph_dyn = float(np.percentile(L_p, 90) - np.percentile(L_p, 10))
        periph_grad = float(np.percentile(grad[periph_zone], 75))
        bright_frac_all = float((L[region] > skin_L + 35.0).mean())
        if (periph_dyn >= 0.75 * dyn_range and periph_grad >= 0.75 * grad_p75) or (
            bright_frac_all >= 0.35
        ):
            periph_non_eye = True

    is_localized_eye = compact_central_iris and (spanning_stripes == 0) and (not periph_non_eye)

    metrics = {
        "skin_L": round(skin_L, 2),
        "p10_L": round(p10_c, 2),
        "p50_L": round(p50_c, 2),
        "p90_L": round(p90_c, 2),
        "dyn_range": round(dyn_range, 2),
        "grad_p75": round(grad_p75, 2),
        "compact_iris": float(compact_central_iris),
        "spanning_stripes": float(spanning_stripes),
        "periph_non_eye": float(periph_non_eye),
    }

    ok = (
        has_contrast
        and has_dark_iris_level
        and not_washed_out
        and has_structure
        and is_localized_eye
    )
    return ok, metrics


def _fallback_eye_box_mask(
    roi_size: int, side: str, roi_meta: dict[str, Any] | None
) -> np.ndarray:
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


def _stitch_with_glare_tracking(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    bridge_x: float,
    blend_half_width: int = 12,
    *,
    left_glare_mask: np.ndarray | None = None,
    right_glare_mask: np.ndarray | None = None,
    left_lens_mask: np.ndarray | None = None,
    right_lens_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stitch left half of `left_rgb` and right half of `right_rgb` across `bridge_x`,
    enforcing source-glare exclusion LAST (after smoothing and lens assignment) and
    returning `(stitched_rgb, carried_known_glare_bool)`.
    """
    H, W = left_rgb.shape[:2]
    if right_rgb.shape[:2] != (H, W):
        right_rgb = cv2.resize(right_rgb, (W, H), interpolation=cv2.INTER_LANCZOS4)

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

    if left_lens_mask is not None:
        w_right_2d = np.where(left_lens_mask > 127, 0.0, w_right_2d)
    if right_lens_mask is not None:
        w_right_2d = np.where(right_lens_mask > 127, 1.0, w_right_2d)

    w_right_2d = cv2.GaussianBlur(w_right_2d.astype(np.float32), (3, 3), 0.7)

    # Re-apply lens ownership, THEN enforce source-glare exclusion LAST so lens masks NEVER
    # re-introduce known glare when the alternate source offers a clean pixel!
    if left_lens_mask is not None:
        w_right_2d = np.where(left_lens_mask > 127, 0.0, w_right_2d)
    if right_lens_mask is not None:
        w_right_2d = np.where(right_lens_mask > 127, 1.0, w_right_2d)

    g_l = (left_glare_mask > 32) if left_glare_mask is not None else np.zeros((H, W), dtype=bool)
    g_r = (right_glare_mask > 32) if right_glare_mask is not None else np.zeros((H, W), dtype=bool)

    # Whenever left is glared and right is clean -> strictly take right (1.0);
    # whenever right is glared and left is clean -> strictly take left (0.0).
    w_right_2d = np.where(g_l & (~g_r), 1.0, w_right_2d)
    w_right_2d = np.where(g_r & (~g_l), 0.0, w_right_2d)

    w_right_2d = np.clip(w_right_2d, 0.0, 1.0)
    carried_glare = ((w_right_2d < 0.99) & g_l) | ((w_right_2d > 0.01) & g_r)

    w_right = w_right_2d[:, :, None]
    blended = left_rgb.astype(np.float32) * (1.0 - w_right) + right_rgb.astype(np.float32) * w_right
    return np.clip(blended, 0.0, 255.0).round().astype(np.uint8), carried_glare


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
    """Seamlessly stitch clean left and right halves across `bridge_x`, excluding known glare."""
    stitched, _ = _stitch_with_glare_tracking(
        left_rgb,
        right_rgb,
        bridge_x=bridge_x,
        blend_half_width=blend_half_width,
        left_glare_mask=left_glare_mask,
        right_glare_mask=right_glare_mask,
        left_lens_mask=left_lens_mask,
        right_lens_mask=right_lens_mask,
    )
    return stitched


def _mirror_array_across_x(arr: np.ndarray, bridge_x: float, interp: int) -> np.ndarray:
    H, W = arr.shape[:2]
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    map_x_1d = np.clip(2.0 * float(bridge_x) - xs, 0.0, float(W - 1))
    map_x = np.broadcast_to(map_x_1d[None, :], (H, W)).astype(np.float32)
    map_y = np.broadcast_to(ys[:, None], (H, W)).astype(np.float32)
    return cv2.remap(arr, map_x, map_y, interpolation=interp, borderMode=cv2.BORDER_REPLICATE)


def _mirror_with_glare_tracking(
    rgb: np.ndarray,
    clean_side: str,
    bridge_x: float,
    blend_half_width: int = 12,
    *,
    glare_mask: np.ndarray | None = None,
    clean_lens_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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
        return _stitch_with_glare_tracking(
            rgb,
            mirrored,
            bridge_x=bridge_x,
            blend_half_width=blend_half_width,
            left_glare_mask=glare_mask,
            right_glare_mask=mir_glare,
            left_lens_mask=clean_lens_mask,
            right_lens_mask=mir_lens,
        )
    return _stitch_with_glare_tracking(
        mirrored,
        rgb,
        bridge_x=bridge_x,
        blend_half_width=blend_half_width,
        left_glare_mask=mir_glare,
        right_glare_mask=glare_mask,
        left_lens_mask=mir_lens,
        right_lens_mask=clean_lens_mask,
    )


def mirror_clean_eye_across_bridge(
    rgb: np.ndarray,
    clean_side: str,
    bridge_x: float,
    blend_half_width: int = 12,
    *,
    glare_mask: np.ndarray | None = None,
    clean_lens_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Mirror the clean eye half across `bridge_x`, strictly excluding rejected-side glare."""
    mir, _ = _mirror_with_glare_tracking(
        rgb,
        clean_side=clean_side,
        bridge_x=bridge_x,
        blend_half_width=blend_half_width,
        glare_mask=glare_mask,
        clean_lens_mask=clean_lens_mask,
    )
    return mir


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
    """Scan per-frame segmentation stats + localized eye-visibility checks for clean eye frames."""
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
    carried_glare: np.ndarray | None = None,
) -> tuple[bool, float, float]:
    """
    Validate a stitched/mirrored reference before saving:
    - Reject immediately if ANY known source glare pixel survived inside either required eye lens.
    - Verify re-detected photometric glare % is within `max_clean_glare_pct`.
    - Verify localized eye visibility on both lenses.
    """
    union_lens = np.maximum(left_lens_mask, right_lens_mask)
    if int((union_lens > 127).sum()) == 0:
        return False, 100.0, 100.0

    # Zero tolerance for carried-over known source glare pixels inside either required eye region!
    if carried_glare is not None and int((carried_glare & (union_lens > 127)).sum()) > 0:
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
    """Execute Stage 3 with upstream generation validation, localized eye checks, and atomic publish."""
    out_dir = Path(out_dir)
    reject_symlink_path(out_dir, "reference output directory")
    ref_out = out_dir / "clean_eye_ref.png"
    reject_symlink_path(ref_out, "clean_eye_ref.png")

    prep_gen = roi_meta.get("prep_generation") if isinstance(roi_meta, dict) else None
    seg_gen = mask_stats.get("segment_generation") if isinstance(mask_stats, dict) else None

    # Verify upstream generation chain consistency when both prep and segment manifests are present
    if isinstance(roi_meta, dict) and isinstance(mask_stats, dict):
        ms_prep = mask_stats.get("prep_generation")
        if prep_gen and ms_prep and ms_prep != prep_gen:
            raise ValueError(
                f"Workspace generation mismatch in reference stage: masks prep_generation "
                f"({ms_prep}) != prep roi_boxes.json ({prep_gen}). Rerun segment first."
            )

    def _publish_ref(rgb: np.ndarray | None, res: ReferenceResult) -> ReferenceResult:
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        stage_dir = make_unique_temp_dir(out_dir.parent, prefix=f".stage_{out_dir.name}_")
        try:
            if rgb is not None:
                stage_png = stage_dir / "clean_eye_ref.png"
                Image.fromarray(rgb).save(stage_png, format="PNG")
                res.ref_sha256 = hash_file_sha256(stage_png)
                res.ref_path = ref_out
            else:
                res.ref_path = None
                res.ref_sha256 = None
            res.prep_generation = prep_gen
            res.segment_generation = seg_gen
            atomic_write_text(stage_dir / "reference_meta.json", json.dumps(res.to_dict(), indent=2))
            atomic_swap_directory(stage_dir, out_dir)
            return res
        except BaseException:
            if stage_dir.exists():
                shutil.rmtree(stage_dir, ignore_errors=True)
            raise

    if config.mode == "none":
        return _publish_ref(
            None,
            ReferenceResult(
                mode="none",
                strategy="disabled",
                ref_path=None,
                details="Reference image disabled (--no-ref); running reference-free VACE.",
            ),
        )

    if config.mode == "image":
        assert config.ref_image_path is not None
        rgb, auto_cropped = _prepare_user_reference_image(config.ref_image_path, config.roi_size)
        return _publish_ref(
            rgb,
            ReferenceResult(
                mode="image",
                strategy="user_image_cropped" if auto_cropped else "user_image_direct",
                ref_path=ref_out,
                details=f"Loaded user reference from {config.ref_image_path} (auto_cropped={auto_cropped}).",
            ),
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

    if best_both is not None:
        rgb = _load_rgb(best_both)
        st = idx_to_stat[best_both]
        return _publish_ref(
            rgb,
            ReferenceResult(
                mode="auto",
                strategy="single_clean_frame",
                ref_path=ref_out,
                left_frame_idx=best_both,
                right_frame_idx=best_both,
                left_glare_pct=float(st["glare_left_pct"]),
                right_glare_pct=float(st["glare_right_pct"]),
                details=f"Selected verified clean frame {best_both} for both eyes.",
            ),
        )

    if best_l is not None and best_r is not None:
        rgb_l = _load_rgb(best_l)
        rgb_r = _load_rgb(best_r)
        lm_l, _, full_l = _get_eye_masks(best_l)
        _, rm_r, _ = _get_eye_masks(best_r)
        glare_l = _load_gray(glare_masks_dir, best_l)
        glare_r = _load_gray(glare_masks_dir, best_r)
        bridge_x = _estimate_bridge_x(config.roi_size, roi_meta=roi_meta, lens_mask=full_l)
        stitched, carried = _stitch_with_glare_tracking(
            rgb_l,
            rgb_r,
            bridge_x=bridge_x,
            left_glare_mask=glare_l,
            right_glare_mask=glare_r,
            left_lens_mask=lm_l,
            right_lens_mask=rm_r,
        )
        ok_syn, gl_syn, gr_syn = _verify_synthesized_reference(
            stitched, lm_l, rm_r, config.max_clean_glare_pct, carried_glare=carried
        )
        if ok_syn:
            return _publish_ref(
                stitched,
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
                ),
            )
        logger.warning(
            "Stitched reference failed glare-free verification (L=%.1f%%, R=%.1f%%); falling back.",
            gl_syn,
            gr_syn,
        )

    if config.allow_mirror and (best_l is not None or best_r is not None):
        clean_side = "left" if best_l is not None else "right"
        src_idx = best_l if best_l is not None else best_r
        assert src_idx is not None
        rgb = _load_rgb(src_idx)
        lm, rm, full_m = _get_eye_masks(src_idx)
        clean_lens = lm if clean_side == "left" else rm
        glare_m = _load_gray(glare_masks_dir, src_idx)
        bridge_x = _estimate_bridge_x(config.roi_size, roi_meta=roi_meta, lens_mask=full_m)

        mirrored, carried = _mirror_with_glare_tracking(
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
            mirrored, syn_l, syn_r, config.max_clean_glare_pct, carried_glare=carried
        )
        if ok_syn:
            return _publish_ref(
                mirrored,
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
                ),
            )
        logger.warning(
            "Mirrored reference failed glare-free verification (L=%.1f%%, R=%.1f%%); falling back.",
            gl_syn,
            gr_syn,
        )

    return _publish_ref(
        None,
        ReferenceResult(
            mode="auto",
            strategy="reference_free_fallback",
            ref_path=None,
            details=(
                f"No clean, visibly sharp eye reference could be verified below "
                f"max_clean_glare_pct={config.max_clean_glare_pct}%; "
                "falling back to reference-free VACE."
            ),
        ),
    )
