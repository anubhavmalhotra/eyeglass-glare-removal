"""Stage 2: Chunked pure-PyTorch SAM2.1 lens tracking and general photometric glare detection."""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import logging
import math
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes

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
    hash_files_manifest,
    make_unique_temp_dir,
    make_unique_temp_sibling,
    reject_symlink_path,
    validate_safe_basename,
)
from eyeglass_glare_removal.config import SegmentConfig

logger = logging.getLogger(__name__)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute binary IoU in [0.0, 1.0]; returns 0.0 when both masks are empty."""
    a_b = a > 0
    b_b = b > 0
    union = int((a_b | b_b).sum())
    if union == 0:
        return 0.0
    inter = int((a_b & b_b).sum())
    return float(inter) / float(union)


def _lens_pair_iou(prev_m: np.ndarray, cand_m: np.ndarray) -> tuple[float, bool]:
    """
    Compare a single lens mask across chunks, explicitly distinguishing mutual absence
    (both empty during an occlusion -> `(1.0, True)`) from tracking collapse (`(0.0, False)`).
    """
    a_b = prev_m > 0
    b_b = cand_m > 0
    a_any = bool(a_b.any())
    b_any = bool(b_b.any())
    if not a_any and not b_any:
        return 1.0, True
    if not a_any or not b_any:
        return 0.0, False
    inter = int((a_b & b_b).sum())
    union = int((a_b | b_b).sum())
    return float(inter) / float(max(1, union)), False


def compute_sam2_chunk_ranges(
    n_frames: int, chunk_size: int = 280, overlap: int = 10
) -> list[tuple[int, int]]:
    """Build monotonic overlapping [start, end) chunk ranges covering [0, n_frames)."""
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    if chunk_size < 2:
        raise ValueError(f"chunk_size must be >= 2, got {chunk_size}")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            f"overlap must satisfy 0 <= overlap < chunk_size ({chunk_size}), got {overlap}"
        )

    if n_frames <= chunk_size:
        return [(0, n_frames)]

    ranges: list[tuple[int, int]] = []
    step = chunk_size - overlap
    start = 0
    while start < n_frames:
        end = min(n_frames, start + chunk_size)
        ranges.append((start, end))
        if end >= n_frames:
            break
        start += step
    return ranges


def clean_single_lens_mask(mask_bool: np.ndarray, erode_px: int = 5) -> np.ndarray:
    """Keep largest connected component, fill internal holes, smooth contour, and erode inward."""
    m = (mask_bool > 0).astype(np.uint8)
    if int(m.sum()) == 0:
        return np.zeros_like(m, dtype=np.uint8)

    num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_lbl > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (labels == largest).astype(np.uint8)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
    m = binary_fill_holes(m > 0).astype(np.uint8)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)

    if erode_px > 0:
        k_size = 2 * erode_px + 1
        k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        m = cv2.erode(m, k_erode, iterations=1)

    return (m > 0).astype(np.uint8) * 255


def _prompts_from_box_geometry(
    box: list[float],
    opposite_center: tuple[float, float],
    bridge: tuple[float, float],
    roi_w: int,
    roi_h: int,
    is_left: bool,
) -> dict[str, Any]:
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(8.0, x2 - x1)
    bh = max(8.0, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    def _clamp(px: float, py: float) -> list[float]:
        return [
            round(float(np.clip(px, 2.0, roi_w - 2.0)), 2),
            round(float(np.clip(py, 2.0, roi_h - 2.0)), 2),
        ]

    pos_pts = [
        _clamp(x1 + 0.25 * bw, y1 + 0.45 * bh),
        _clamp(x1 + 0.50 * bw, y1 + 0.35 * bh),
        _clamp(x1 + 0.75 * bw, y1 + 0.45 * bh),
        _clamp(x1 + 0.42 * bw, y1 + 0.68 * bh),
        _clamp(x1 + 0.68 * bw, y1 + 0.66 * bh),
    ]

    pad_y = max(6.0, 0.14 * bh)
    pad_x = max(6.0, 0.12 * bw)
    outer_x = (x1 - pad_x) if is_left else (x2 + pad_x)
    neg_pts = [
        _clamp(cx, y1 - pad_y),
        _clamp(cx, y2 + pad_y),
        _clamp(outer_x, cy),
        _clamp(bridge[0], bridge[1]),
        _clamp(bridge[0], min(y1, bridge[1]) - max(12.0, 0.25 * bh)),
        _clamp(opposite_center[0], opposite_center[1]),
    ]
    return {
        "box": [
            round(float(np.clip(x1, 1.0, roi_w - 2.0)), 2),
            round(float(np.clip(y1, 1.0, roi_h - 2.0)), 2),
            round(float(np.clip(x2, 2.0, roi_w - 1.0)), 2),
            round(float(np.clip(y2, 2.0, roi_h - 1.0)), 2),
        ],
        "points": pos_pts + neg_pts,
        "labels": [1] * len(pos_pts) + [0] * len(neg_pts),
    }


def _normalize_override_prompt(
    raw: dict[str, Any],
    opposite_center: tuple[float, float],
    bridge: tuple[float, float],
    roi_w: int,
    roi_h: int,
    is_left: bool,
) -> dict[str, Any]:
    if not isinstance(raw, dict) or "box" not in raw or len(raw["box"]) != 4:
        raise ValueError("Each side in lens_points override must specify a 4-element 'box'.")
    box = [float(v) for v in raw["box"]]
    if "points" in raw and "labels" in raw:
        pts = [[float(p[0]), float(p[1])] for p in raw["points"]]
        lbls = [int(v) for v in raw["labels"]]
        if len(pts) != len(lbls) or len(pts) == 0:
            raise ValueError(
                "lens_points 'points' and 'labels' must be non-empty and equal length."
            )
        return {"box": box, "points": pts, "labels": lbls}
    return _prompts_from_box_geometry(box, opposite_center, bridge, roi_w, roi_h, is_left=is_left)


def build_initial_lens_prompts(
    eye_geometry_roi: dict[str, Any] | None,
    roi_size: int = 512,
    override_prompts: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Auto-seed left/right lens prompt boxes & points from detected eye geometry (or override)."""
    if override_prompts is not None:
        if "left" not in override_prompts or "right" not in override_prompts:
            raise ValueError("lens_points override must contain both 'left' and 'right' keys.")
        lb = [float(v) for v in override_prompts["left"]["box"]]
        rb = [float(v) for v in override_prompts["right"]["box"]]
        lc = (0.5 * (lb[0] + lb[2]), 0.5 * (lb[1] + lb[3]))
        rc = (0.5 * (rb[0] + rb[2]), 0.5 * (rb[1] + rb[3]))
        bridge = (0.5 * (lb[2] + rb[0]), 0.5 * (lc[1] + rc[1]))
        left_p = _normalize_override_prompt(
            override_prompts["left"], rc, bridge, roi_size, roi_size, is_left=True
        )
        right_p = _normalize_override_prompt(
            override_prompts["right"], lc, bridge, roi_size, roi_size, is_left=False
        )
        return left_p, right_p

    if eye_geometry_roi and "left_eye" in eye_geometry_roi and "right_eye" in eye_geometry_roi:
        lx, ly = float(eye_geometry_roi["left_eye"][0]), float(eye_geometry_roi["left_eye"][1])
        rx, ry = float(eye_geometry_roi["right_eye"][0]), float(eye_geometry_roi["right_eye"][1])
        if lx > rx:
            lx, ly, rx, ry = rx, ry, lx, ly
        iod = max(
            20.0,
            float(eye_geometry_roi.get("inter_ocular_px") or math.hypot(rx - lx, ry - ly)),
        )
    else:
        iod = 0.34 * float(roi_size)
        lx, ly = 0.5 * roi_size - 0.5 * iod, 0.48 * roi_size
        rx, ry = 0.5 * roi_size + 0.5 * iod, 0.48 * roi_size

    bridge_x = 0.5 * (lx + rx)
    bridge_y = 0.5 * (ly + ry)
    bridge_gap = max(6.0, 0.08 * iod)

    half_w = 0.42 * iod
    top_h = 0.24 * iod
    bot_h = 0.30 * iod

    l_box = [lx - half_w, ly - top_h, min(lx + half_w, bridge_x - bridge_gap), ly + bot_h]
    r_box = [max(rx - half_w, bridge_x + bridge_gap), ry - top_h, rx + half_w, ry + bot_h]

    if l_box[2] - l_box[0] < 12.0:
        l_box[0] = max(2.0, l_box[2] - 12.0)
    if r_box[2] - r_box[0] < 12.0:
        r_box[2] = min(roi_size - 2.0, r_box[0] + 12.0)

    left_p = _prompts_from_box_geometry(
        l_box, (rx, ry), (bridge_x, bridge_y), roi_size, roi_size, is_left=True
    )
    right_p = _prompts_from_box_geometry(
        r_box, (lx, ly), (bridge_x, bridge_y), roi_size, roi_size, is_left=False
    )
    return left_p, right_p


def derive_prompts_from_masks(
    left_mask_bool: np.ndarray,
    right_mask_bool: np.ndarray,
    fallback_prompts: tuple[dict[str, Any], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive tight boxes + interior positive + exterior negative points from previous chunk masks."""
    ml = clean_single_lens_mask(left_mask_bool, erode_px=0) > 127
    mr = clean_single_lens_mask(right_mask_bool, erode_px=0) > 127
    H, W = ml.shape[:2]

    ly, lx = np.where(ml)
    ry, rx = np.where(mr)
    min_px = max(64, int(0.001 * H * W))
    if len(lx) < min_px or len(rx) < min_px:
        return fallback_prompts

    lx1, lx2, ly1, ly2 = float(lx.min()), float(lx.max()), float(ly.min()), float(ly.max())
    rx1, rx2, ry1, ry2 = float(rx.min()), float(rx.max()), float(ry.min()), float(ry.max())
    bridge_x = 0.5 * (lx2 + rx1)
    bridge_y = 0.25 * (ly1 + ly2 + ry1 + ry2)
    forehead_y = max(4.0, min(ly1, ry1) - max(12.0, 0.22 * max(ly2 - ly1, ry2 - ry1)))

    left_center = (0.5 * (lx1 + lx2), 0.5 * (ly1 + ly2))
    right_center = (0.5 * (rx1 + rx2), 0.5 * (ry1 + ry2))

    def _build_one(mask_b: np.ndarray, is_left: bool) -> dict[str, Any]:
        ys, xs = np.where(mask_b)
        x1, x2, y1, y2 = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
        dist = cv2.distanceTransform(mask_b.astype(np.uint8), cv2.DIST_L2, 5)

        def _slice_pt(x_frac: float, y_frac: float) -> list[float]:
            tx = int(np.clip(round(x1 + x_frac * (x2 - x1)), 0, W - 1))
            col_ys = np.where(mask_b[:, tx])[0]
            if len(col_ys) == 0:
                return [round(0.5 * (x1 + x2), 2), round(0.5 * (y1 + y2), 2)]
            cy1, cy2 = float(col_ys.min()), float(col_ys.max())
            ty = int(np.clip(round(cy1 + y_frac * (cy2 - cy1)), 0, H - 1))
            if dist[ty, tx] < 3.0:
                ty = int(col_ys[int(np.argmax(dist[col_ys, tx]))])
            return [float(tx), float(ty)]

        pos_pts = [
            _slice_pt(0.22, 0.45),
            _slice_pt(0.50, 0.32),
            _slice_pt(0.78, 0.42),
            _slice_pt(0.48, 0.72),
            _slice_pt(0.76, 0.70),
        ]

        mid_x = 0.5 * (x1 + x2)
        mid_y = 0.5 * (y1 + y2)
        pad = max(8.0, 0.08 * (x2 - x1))
        if is_left:
            outer_x1 = max(2.0, x1 - pad * 1.2)
            outer_x2 = max(2.0, x1 - pad * 0.8)
            opp_c = right_center
        else:
            outer_x1 = min(W - 2.0, x2 + pad * 1.2)
            outer_x2 = min(W - 2.0, x2 + pad * 0.8)
            opp_c = left_center

        neg_pts = [
            [mid_x, max(2.0, y1 - max(7.0, 0.12 * (y2 - y1)))],
            [mid_x, min(H - 2.0, y2 + max(7.0, 0.12 * (y2 - y1)))],
            [outer_x1, max(2.0, mid_y - 8.0)],
            [outer_x2, min(H - 2.0, mid_y + 8.0)],
            [float(np.clip(bridge_x, 2.0, W - 2.0)), float(np.clip(bridge_y, 2.0, H - 2.0))],
            [float(np.clip(bridge_x, 2.0, W - 2.0)), float(np.clip(forehead_y, 2.0, H - 2.0))],
            [float(np.clip(opp_c[0], 2.0, W - 2.0)), float(np.clip(opp_c[1], 2.0, H - 2.0))],
        ]
        return {
            "box": [x1, y1, x2, y2],
            "points": pos_pts + neg_pts,
            "labels": [1] * len(pos_pts) + [0] * len(neg_pts),
        }

    return _build_one(ml, is_left=True), _build_one(mr, is_left=False)


def _capped_sdf_entry(
    mask_u8: np.ndarray, max_sdf: float = 6.0
) -> tuple[np.ndarray | None, np.ndarray, bool]:
    """
    Compute `(capped_sdf_or_None, binary_bool, is_valid)` for consistency-gated temporal smoothing.

    Treats BOTH empty masks (`fg == 0`) AND anomalous full-ROI outlier masks (`fg >= 0.85 * size`)
    as invalid (`is_valid = False`) so neither empty dropouts nor blown-out full masks can erase
    or expand valid neighboring lens masks.
    """
    b_bool = mask_u8 > 127
    fg = int(b_bool.sum())
    if fg == 0 or fg >= int(0.85 * b_bool.size):
        return None, b_bool, False
    b_u8 = b_bool.astype(np.uint8)
    d_in = cv2.distanceTransform(b_u8, cv2.DIST_L2, 5)
    d_out = cv2.distanceTransform(1 - b_u8, cv2.DIST_L2, 5)
    sdf = np.clip(d_in - d_out, -max_sdf, max_sdf).astype(np.float32)
    return sdf, b_bool, True


def _resolve_consistent_window_mask(
    buf: dict[int, tuple[np.ndarray | None, np.ndarray, bool]],
    i: int,
    lo: int,
    hi: int,
    *,
    radius: int,
    min_consistency_iou: float = 0.50,
) -> np.ndarray:
    """
    Resolve frame `i` from a bounded window buffer of `(sdf, mask_bool, is_valid)` entries:
    - Valid current frame `i`: average only neighbors `j` that are valid AND spatially agree
      with frame `i` (`mask_iou >= 0.50`), so motion, missing neighbors, or full-ROI outliers
      can NEVER erase or expand a valid mask.
    - Invalid/empty current frame `i`: bridge/interpolate ONLY when immediately flanking frames
      `i - 1` and `i + 1` BOTH exist, are valid, and agree spatially (`mask_iou >= 0.50`).
    """
    sdf_i, b_i, valid_i = buf[i]
    if radius <= 0:
        return b_i.astype(np.uint8) * 255

    if valid_i and sdf_i is not None:
        agreeing = [sdf_i]
        for j in range(lo, hi):
            if j == i:
                continue
            sdf_j, b_j, valid_j = buf[j]
            if valid_j and sdf_j is not None and mask_iou(b_i, b_j) >= min_consistency_iou:
                agreeing.append(sdf_j)
        if len(agreeing) == 1:
            return b_i.astype(np.uint8) * 255
        avg_sdf = sum(agreeing) / float(len(agreeing))
        return (avg_sdf > 0.0).astype(np.uint8) * 255

    # Invalid/empty/full-outlier frame: bridge isolated anomaly ONLY when flanking valid frames agree
    if (i - 1) in buf and (i + 1) in buf:
        sdf_prev, b_prev, valid_prev = buf[i - 1]
        sdf_next, b_next, valid_next = buf[i + 1]
        if (
            valid_prev
            and valid_next
            and sdf_prev is not None
            and sdf_next is not None
            and mask_iou(b_prev, b_next) >= min_consistency_iou
        ):
            avg_sdf = 0.5 * (sdf_prev + sdf_next)
            return (avg_sdf > 0.0).astype(np.uint8) * 255

    return b_i.astype(np.uint8) * 255


def iter_temporal_smoothed_masks(
    mask_stream: Iterator[np.ndarray],
    n_frames: int,
    radius: int = 1,
    max_sdf: float = 6.0,
) -> Iterator[np.ndarray]:
    """O(1)-RAM streaming consistency-gated temporal binary mask smoother."""
    if n_frames <= 0:
        return
    if radius <= 0 or n_frames < 3:
        for m in mask_stream:
            yield ((m > 127).astype(np.uint8) * 255)
        return

    buf: dict[int, tuple[np.ndarray | None, np.ndarray, bool]] = {}
    next_read = 0

    def _ensure_loaded(up_to_excl: int) -> None:
        nonlocal next_read
        while next_read < up_to_excl and next_read < n_frames:
            m = next(mask_stream)
            buf[next_read] = _capped_sdf_entry(m, max_sdf=max_sdf)
            next_read += 1

    for i in range(n_frames):
        lo = max(0, i - radius)
        hi = min(n_frames, i + radius + 1)
        _ensure_loaded(hi)
        yield _resolve_consistent_window_mask(buf, i, lo, hi, radius=radius)
        for k in list(buf.keys()):
            if k < i - radius:
                del buf[k]


def temporal_smooth_binary_masks(masks: list[np.ndarray], radius: int = 1) -> list[np.ndarray]:
    """Smooth a list of binary masks temporally via consistency-gated capped SDF window."""
    return list(iter_temporal_smoothed_masks(iter(masks), len(masks), radius=radius))


def estimate_local_skin_stats(
    roi_rgb: np.ndarray,
    lens_mask: np.ndarray,
) -> dict[str, float]:
    """Estimate local per-frame facial skin tone in CIE Lab strictly outside the lenses."""
    H, W = lens_mask.shape[:2]
    lens_bin = (lens_mask > 127).astype(np.uint8)
    bgr = roi_rgb[:, :, ::-1]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L_ch, a_ch, b_ch = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    r_inner = max(6, int(round(0.025 * min(H, W))))
    r_outer = max(24, int(round(0.14 * min(H, W))))
    k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_inner + 1, 2 * r_inner + 1))
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_outer + 1, 2 * r_outer + 1))
    ring = (cv2.dilate(lens_bin, k_out) > 0) & (cv2.dilate(lens_bin, k_in) == 0)

    skin_like = ring & (L_ch >= 25.0) & (L_ch <= 240.0) & (a_ch >= 126.0) & (b_ch >= 122.0)
    if int(skin_like.sum()) >= 150:
        ref = skin_like
    elif int(ring.sum()) >= 50:
        ref = ring
    else:
        ref = lens_bin > 0

    L_ref = L_ch[ref]
    a_ref = a_ch[ref]
    b_ref = b_ch[ref]

    if L_ref.size == 0:
        return {
            "L_med": 135.0,
            "L_p85": 150.0,
            "L_mad": 10.0,
            "a_med": 138.0,
            "a_mad": 3.0,
            "b_med": 142.0,
            "b_mad": 4.0,
        }

    L_med = float(np.median(L_ref))
    a_med = float(np.median(a_ref))
    b_med = float(np.median(b_ref))
    return {
        "L_med": L_med,
        "L_p85": float(np.percentile(L_ref, 85)),
        "L_mad": max(6.0, float(1.4826 * np.median(np.abs(L_ref - L_med)))),
        "a_med": a_med,
        "a_mad": max(2.5, float(1.4826 * np.median(np.abs(a_ref - a_med)))),
        "b_med": b_med,
        "b_mad": max(3.0, float(1.4826 * np.median(np.abs(b_ref - b_med)))),
    }


def compute_glare_hotspot_mask(
    roi_rgb: np.ndarray,
    lens_interior: np.ndarray,
    *,
    dilate_px: int = 5,
    feather_sigma: float = 2.0,
    sensitivity: float = 1.0,
) -> np.ndarray:
    """General photometric glare detector strictly inside `lens_interior`."""
    inside = lens_interior > 127
    if int(inside.sum()) == 0:
        return np.zeros(lens_interior.shape[:2], dtype=np.uint8)

    stats = estimate_local_skin_stats(roi_rgb, lens_interior)
    scale = 1.0 / max(0.25, float(sensitivity))

    bgr = roi_rgb[:, :, ::-1]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    chroma = np.hypot(a - 128.0, b - 128.0)

    R = roi_rgb[:, :, 0].astype(np.float32)
    G = roi_rgb[:, :, 1].astype(np.float32)
    B = roi_rgb[:, :, 2].astype(np.float32)

    skin_L = stats["L_med"]
    skin_L85 = stats["L_p85"]
    skin_a = stats["a_med"]
    skin_a_mad = stats["a_mad"]
    skin_b = stats["b_med"]
    skin_b_mad = stats["b_mad"]

    bright_spec_thresh = min(242.0, max(skin_L85 + 26.0 * scale, 202.0 - 10.0 * (1.0 - scale)))
    washout_L_thresh = min(234.0, max(skin_L85 + 34.0 * scale, 194.0 - 8.0 * (1.0 - scale)))
    bright_specular = (L >= bright_spec_thresh) | (
        (L >= washout_L_thresh) & (chroma <= 15.0 / max(0.5, scale))
    )

    broad_neutral = np.zeros(inside.shape, dtype=bool)
    broad_cand = (
        (L >= max(155.0, skin_L85 + 52.0 * scale))
        & (chroma <= 16.0 / max(0.5, scale))
        & inside
    )
    if int(broad_cand.sum()) > 0:
        n_lens, lens_labels, lens_stats, _ = cv2.connectedComponentsWithStats(
            inside.astype(np.uint8), connectivity=8
        )
        for lens_id in range(1, n_lens):
            lens_comp = lens_labels == lens_id
            lens_area = int(lens_stats[lens_id, cv2.CC_STAT_AREA])
            if lens_area < 20:
                continue
            cand_in_lens = (broad_cand & lens_comp).astype(np.uint8)
            n_c, c_labels, c_stats, _ = cv2.connectedComponentsWithStats(
                cand_in_lens, connectivity=8
            )
            for cid in range(1, n_c):
                if int(c_stats[cid, cv2.CC_STAT_AREA]) >= max(24, int(0.12 * lens_area)):
                    broad_neutral |= c_labels == cid

    cool_b_thresh = min(skin_b - 10.0 * scale, 125.0)
    cool_screen = (
        ((b <= cool_b_thresh) & (L >= max(65.0, skin_L + 8.0 * scale)))
        | (((B - R) >= 6.0 * scale) & (L >= max(85.0, skin_L + 8.0 * scale)))
    )

    warm_a_delta = max(6.0, 2.2 * skin_a_mad) * scale
    warm_b_delta = max(10.0, 2.5 * skin_b_mad) * scale
    warm_lamp = (
        ((a >= skin_a + warm_a_delta) & (L >= skin_L + 4.0 * scale))
        | ((b >= skin_b + warm_b_delta) & (a >= skin_a + 3.0 * scale) & (L >= skin_L + 8.0 * scale))
    )

    ar_min_L = max(48.0, skin_L - 25.0)
    green_ar = (
        ((a <= min(skin_a - 10.0 * scale, 123.0)) & (L >= ar_min_L))
        | (((G - R) >= 10.0 * scale) & ((G - B) >= 6.0 * scale) & (L >= 52.0))
    )

    purple_ar = (
        ((b <= min(skin_b - 12.0 * scale, 122.0)) & (a >= 131.0) & (L >= ar_min_L))
        | ((np.minimum(R - G, B - G) >= 12.0 * scale) & (L >= 52.0))
    )

    raw = (bright_specular | broad_neutral | cool_screen | warm_lamp | green_ar | purple_ar) & inside
    m = raw.astype(np.uint8)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
    m = binary_fill_holes(m > 0).astype(np.uint8)

    if dilate_px > 0:
        kd = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        m = cv2.dilate(m, kd, iterations=1)

    m_inside = (m > 0) & inside
    soft = m_inside.astype(np.float32) * 255.0
    if feather_sigma > 0.0:
        soft = cv2.GaussianBlur(soft, (0, 0), sigmaX=feather_sigma, sigmaY=feather_sigma)
        soft = np.maximum(soft, np.where(raw, 255.0, 0.0))

    soft = np.where(inside, np.clip(soft, 0.0, 255.0), 0.0)
    return soft.astype(np.uint8)


def _build_init_session_kwargs(
    processor: Any, frames_pil: list[Image.Image], device: Any, dtype: Any
) -> dict[str, Any]:
    """Explicitly keep video preprocessing & inference state storage on CPU when supported."""
    kwargs: dict[str, Any] = {
        "video": frames_pil,
        "inference_device": device,
        "video_storage_device": "cpu",
        "dtype": dtype,
    }
    try:
        params = inspect.signature(processor.init_video_session).parameters
        if "processing_device" in params:
            kwargs["processing_device"] = "cpu"
        if "inference_state_device" in params:
            kwargs["inference_state_device"] = "cpu"
        if "max_vision_features_cache_size" in params:
            kwargs["max_vision_features_cache_size"] = 1
    except (TypeError, ValueError):
        pass
    return kwargs


def _run_sam2_single_chunk(
    model: Any,
    processor: Any,
    frames_pil: list[Image.Image],
    *,
    device: Any,
    dtype: Any,
    left_prompts: dict[str, Any] | None = None,
    right_prompts: dict[str, Any] | None = None,
    init_masks: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Run pure-PyTorch Sam2VideoModel tracking on a single memory-safe chunk."""
    import torch

    H, W = frames_pil[0].height, frames_pil[0].width
    session_kwargs = _build_init_session_kwargs(processor, frames_pil, device, dtype)
    session = processor.init_video_session(**session_kwargs)

    try:
        if init_masks is not None:
            m_l, m_r = init_masks
            processor.add_inputs_to_inference_session(
                inference_session=session,
                frame_idx=0,
                obj_ids=[1, 2],
                input_masks=[m_l.astype(np.float32), m_r.astype(np.float32)],
            )
        else:
            assert left_prompts is not None and right_prompts is not None
            processor.add_inputs_to_inference_session(
                inference_session=session,
                frame_idx=0,
                obj_ids=[1, 2],
                input_points=[[left_prompts["points"], right_prompts["points"]]],
                input_labels=[[left_prompts["labels"], right_prompts["labels"]]],
                input_boxes=[[left_prompts["box"], right_prompts["box"]]],
            )

        with torch.inference_mode():
            _ = model(inference_session=session, frame_idx=0)

        n = len(frames_pil)
        left_raw: list[np.ndarray | None] = [None] * n
        right_raw: list[np.ndarray | None] = [None] * n

        with torch.inference_mode():
            for out in model.propagate_in_video_iterator(session):
                fidx = int(out.frame_idx)
                pm = out.pred_masks.cpu()
                masks = processor.post_process_masks(
                    [pm],
                    original_sizes=[[H, W]],
                    binarize=True,
                )[0]
                m_np = masks.detach().cpu().numpy()
                if m_np.ndim == 4:
                    m_np = m_np[:, 0]
                id_to_idx = {int(oid): idx for idx, oid in enumerate(session.obj_ids)}
                left_raw[fidx] = m_np[id_to_idx[1]].astype(bool)
                right_raw[fidx] = m_np[id_to_idx[2]].astype(bool)
    finally:
        del session
        gc.collect()
        cuda_empty_cache_if_needed(torch, device)

    empty = np.zeros((H, W), dtype=bool)
    return (
        [m if m is not None else empty.copy() for m in left_raw],
        [m if m is not None else empty.copy() for m in right_raw],
    )


def _hash_overlap_masks(left_ov: list[np.ndarray], right_ov: list[np.ndarray]) -> str:
    """Content digest of the predecessor overlap masks seeding a downstream SAM2 chunk."""
    h = hashlib.sha256()
    for ml, mr in zip(left_ov, right_ov, strict=True):
        h.update(np.ascontiguousarray(ml.astype(np.bool_)).tobytes())
        h.update(np.ascontiguousarray(mr.astype(np.bool_)).tobytes())
    return h.hexdigest()[:16]


def _save_chunk_npz_atomic(
    ckpt_path: Path,
    left_masks: list[np.ndarray],
    right_masks: list[np.ndarray],
    iou_l: float,
    iou_r: float,
    used_fallback: bool,
    fingerprint: str = "",
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_path(ckpt_path.parent, "checkpoint directory")
    fd, tmp_name = tempfile.mkstemp(
        dir=str(ckpt_path.parent), prefix=f".{ckpt_path.stem}.", suffix=".tmp.npz"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(
                f,
                left=np.stack(left_masks, axis=0).astype(np.bool_),
                right=np.stack(right_masks, axis=0).astype(np.bool_),
                iou_l=np.float32(iou_l),
                iou_r=np.float32(iou_r),
                used_fallback=np.bool_(used_fallback),
                fingerprint=np.array(fingerprint),
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, ckpt_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _is_oom_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    if "OutOfMemory" in name:
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def _evaluate_chunk_handoff(
    prev_l_ov: list[np.ndarray],
    prev_r_ov: list[np.ndarray],
    cand_l_ov: list[np.ndarray],
    cand_r_ov: list[np.ndarray],
    splice_i: int,
) -> dict[str, Any]:
    """
    Verify per-lens IoU across seed AND post-seed overlap frames (including splice_i),
    detecting left/right identity swaps while handling mutual occlusion (both empty) cleanly.
    """
    ov_len = min(len(prev_l_ov), len(cand_l_ov))
    if ov_len == 0:
        return {
            "iou_l": 1.0,
            "iou_r": 1.0,
            "splice_iou_l": 1.0,
            "splice_iou_r": 1.0,
            "min_iou": 1.0,
            "swapped": False,
        }

    s_idx = min(max(1 if ov_len >= 2 else 0, splice_i), ov_len - 1)
    l_ious: list[float] = []
    r_ious: list[float] = []
    cross_lr: list[float] = []
    cross_rl: list[float] = []

    for k in range(ov_len):
        li, l_both_empty = _lens_pair_iou(prev_l_ov[k], cand_l_ov[k])
        ri, r_both_empty = _lens_pair_iou(prev_r_ov[k], cand_r_ov[k])
        l_ious.append(li)
        r_ious.append(ri)
        # Only score cross-swap on non-empty lenses (two empty lenses aren't swapped!)
        clr = 0.0 if (l_both_empty or r_both_empty) else mask_iou(prev_l_ov[k], cand_r_ov[k])
        crl = 0.0 if (l_both_empty or r_both_empty) else mask_iou(prev_r_ov[k], cand_l_ov[k])
        cross_lr.append(clr)
        cross_rl.append(crl)

    splice_l = l_ious[s_idx]
    splice_r = r_ious[s_idx]
    post_start = 1 if ov_len >= 2 else 0

    swapped = (
        (cross_lr[s_idx] > splice_l + 0.05)
        or (cross_rl[s_idx] > splice_r + 0.05)
        or any(cross_lr[k] > l_ious[k] + 0.05 for k in range(post_start, ov_len))
        or any(cross_rl[k] > r_ious[k] + 0.05 for k in range(post_start, ov_len))
    )
    min_iou = min(min(l_ious), min(r_ious))
    return {
        "iou_l": splice_l,
        "iou_r": splice_r,
        "splice_iou_l": splice_l,
        "splice_iou_r": splice_r,
        "min_iou": min_iou,
        "swapped": swapped,
    }


def run_chunked_tracking_loop(
    chunk_runner: Any,
    frame_loader: Callable[[int], Any],
    n_frames: int,
    init_left_prompts: dict[str, Any],
    init_right_prompts: dict[str, Any],
    *,
    chunk_size: int = 280,
    overlap: int = 10,
    handoff_iou_threshold: float = 0.85,
    ckpt_dir: Path | None = None,
    resume: bool = True,
    fingerprint: str = "",
    store_in_memory: bool = True,
    ckpt_prefix: str = "chunk",
    oom_cleanup_fn: Callable[[], None] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    """
    Chunk-orchestration engine for SAM2.1 tracking with:
    - Post-seed identity-swap verification (requiring `overlap == 0` or `overlap >= 2`).
    - Mutual-occlusion agreement handling & graceful re-seeding when visibility returns.
    - Dynamic OOM backoff shrinking BOTH chunk size AND effective overlap cleanly.
    - Predecessor-mask-bound chunk fingerprints and single-decompression `.npz` loading.
    """
    if not store_in_memory and ckpt_dir is None:
        raise ValueError("store_in_memory=False requires ckpt_dir.")
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    if chunk_size < 2:
        raise ValueError(f"chunk_size must be >= 2, got {chunk_size}")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            f"overlap must satisfy 0 <= overlap < chunk_size ({chunk_size}), got {overlap}"
        )
    if n_frames > chunk_size and overlap == 1:
        raise ValueError(
            "overlap=1 cannot verify post-seed lens identity; pass overlap=0 (disjoint) or overlap >= 2."
        )

    left_all: list[np.ndarray | None] = [None] * n_frames if store_in_memory else []
    right_all: list[np.ndarray | None] = [None] * n_frames if store_in_memory else []
    chunk_logs: list[dict[str, Any]] = []

    if ckpt_dir is not None:
        reject_symlink_path(ckpt_dir, "SAM2 checkpoint directory")
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    fallback_pair = (init_left_prompts, init_right_prompts)

    prev_end: int = 0
    prev_tail_start: int | None = None
    prev_tail_l: list[np.ndarray] = []
    prev_tail_r: list[np.ndarray] = []

    min_ov = 2 if overlap > 0 else 0
    min_chunk_len = min_ov + 2
    cur_chunk_size = chunk_size
    cur_overlap = overlap
    cs = 0
    c_idx = 0

    while cs < n_frames:
        ce = min(n_frames, cs + cur_chunk_size)
        chunk_len = ce - cs
        tc = time.time()

        actual_ov = 0 if c_idx == 0 else max(0, prev_end - cs)
        prev_l_ov: list[np.ndarray] = []
        prev_r_ov: list[np.ndarray] = []
        handoff_dep_digest = "seed"
        if c_idx > 0 and actual_ov > 0 and prev_tail_start is not None:
            ov_offset = cs - prev_tail_start
            prev_l_ov = prev_tail_l[ov_offset : ov_offset + actual_ov]
            prev_r_ov = prev_tail_r[ov_offset : ov_offset + actual_ov]
            handoff_dep_digest = _hash_overlap_masks(prev_l_ov, prev_r_ov)

        chunk_fp = f"{fingerprint}:{handoff_dep_digest}" if fingerprint else handoff_dep_digest
        ckpt_p = (
            ckpt_dir / f"{ckpt_prefix}_{c_idx:04d}_{cs:06d}_{ce:06d}.npz"
            if ckpt_dir is not None
            else None
        )

        loaded_from_ckpt = False
        c_left: list[np.ndarray] = []
        c_right: list[np.ndarray] = []
        handoff_iou_l = 1.0
        handoff_iou_r = 1.0
        used_fallback = False

        if resume and ckpt_p is not None and ckpt_p.exists() and not ckpt_p.is_symlink():
            try:
                with np.load(ckpt_p, allow_pickle=False) as data:
                    stored_fp = str(data["fingerprint"].item()) if "fingerprint" in data else ""
                    if stored_fp == chunk_fp:
                        # Load compressed arrays ONCE outside the loop (never re-decompress per frame!)
                        left_np = data["left"]
                        right_np = data["right"]
                        if (
                            left_np.ndim == 3
                            and right_np.ndim == 3
                            and left_np.shape == right_np.shape
                            and left_np.shape[0] == chunk_len
                            and left_np.dtype == np.bool_
                            and right_np.dtype == np.bool_
                        ):
                            c_left = [left_np[i] for i in range(chunk_len)]
                            c_right = [right_np[i] for i in range(chunk_len)]
                            handoff_iou_l = float(data["iou_l"]) if "iou_l" in data else 1.0
                            handoff_iou_r = float(data["iou_r"]) if "iou_r" in data else 1.0
                            used_fallback = (
                                bool(data["used_fallback"]) if "used_fallback" in data else False
                            )
                            loaded_from_ckpt = True
            except Exception as exc:
                logger.warning("Corrupt SAM2 chunk checkpoint %s (%s); recomputing.", ckpt_p, exc)

        if not loaded_from_ckpt:
            try:
                chunk_frames = [frame_loader(i) for i in range(cs, ce)]

                if c_idx == 0 or actual_ov == 0:
                    c_left, c_right = chunk_runner(
                        chunk_frames,
                        left_prompts=init_left_prompts,
                        right_prompts=init_right_prompts,
                        init_masks=None,
                    )
                else:
                    prev_l0 = prev_l_ov[0]
                    prev_r0 = prev_r_ov[0]
                    splice_i = max(1 if actual_ov >= 2 else 0, actual_ov // 2)

                    lp, rp = derive_prompts_from_masks(
                        prev_l0, prev_r0, fallback_prompts=fallback_pair
                    )
                    c_left, c_right = chunk_runner(
                        chunk_frames, left_prompts=lp, right_prompts=rp, init_masks=None
                    )
                    ev1 = _evaluate_chunk_handoff(
                        prev_l_ov, prev_r_ov, c_left[:actual_ov], c_right[:actual_ov], splice_i
                    )

                    if ev1["swapped"] or ev1["min_iou"] < handoff_iou_threshold:
                        logger.warning(
                            "Chunk %d box+pts handoff failed (swapped=%s, min_iou=%.3f < %.2f); "
                            "retrying with input_masks.",
                            c_idx,
                            ev1["swapped"],
                            ev1["min_iou"],
                            handoff_iou_threshold,
                        )
                        fb_left, fb_right = chunk_runner(
                            chunk_frames,
                            left_prompts=None,
                            right_prompts=None,
                            init_masks=(
                                clean_single_lens_mask(prev_l0, 0) > 127,
                                clean_single_lens_mask(prev_r0, 0) > 127,
                            ),
                        )
                        ev2 = _evaluate_chunk_handoff(
                            prev_l_ov, prev_r_ov, fb_left[:actual_ov], fb_right[:actual_ov], splice_i
                        )

                        candidates = [
                            (not ev1["swapped"], ev1["min_iou"], False, c_left, c_right, ev1),
                            (not ev2["swapped"], ev2["min_iou"], True, fb_left, fb_right, ev2),
                        ]
                        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
                        best_non_swapped, best_min_iou, best_fb, c_left, c_right, best_ev = (
                            candidates[0]
                        )
                        used_fallback = best_fb
                        handoff_iou_l = best_ev["iou_l"]
                        handoff_iou_r = best_ev["iou_r"]

                        if not best_non_swapped:
                            raise RuntimeError(
                                f"SAM2 chunk {c_idx} [{cs}:{ce}] suffered left/right lens identity swap "
                                "across both prompt and mask handoff attempts."
                            )
                        if best_min_iou < 0.30:
                            raise RuntimeError(
                                f"SAM2 chunk {c_idx} [{cs}:{ce}] handoff collapsed "
                                f"(min_iou={best_min_iou:.3f}); cannot reliably stitch chunk."
                            )
                    else:
                        handoff_iou_l = ev1["iou_l"]
                        handoff_iou_r = ev1["iou_r"]

                if ckpt_p is not None:
                    _save_chunk_npz_atomic(
                        ckpt_p,
                        c_left,
                        c_right,
                        handoff_iou_l,
                        handoff_iou_r,
                        used_fallback,
                        fingerprint=chunk_fp,
                    )
            except Exception as exc:
                if _is_oom_error(exc) and cur_chunk_size > min_chunk_len:
                    new_sz = max(min_chunk_len, cur_chunk_size // 2)
                    if new_sz >= cur_chunk_size:
                        new_sz = cur_chunk_size - 1
                    new_ov = (
                        min(cur_overlap, max(min_ov, new_sz // 4)) if overlap > 0 else 0
                    )
                    if new_ov >= new_sz:
                        new_ov = max(0, new_sz - 2)
                    logger.warning(
                        "CUDA OOM on SAM2 chunk [%d:%d] (size=%d, ov=%d); retrying with size=%d, ov=%d.",
                        cs,
                        ce,
                        cur_chunk_size,
                        cur_overlap,
                        new_sz,
                        new_ov,
                    )
                    cur_chunk_size = new_sz
                    cur_overlap = new_ov
                    # Advance cs cleanly if shrinking chunk_size would stall behind prev_end
                    if c_idx > 0 and prev_end > 0:
                        cs = max(cs, prev_end - cur_overlap)
                    if oom_cleanup_fn is not None:
                        oom_cleanup_fn()
                    gc.collect()
                    continue
                raise

        start_write = 0 if c_idx == 0 else min(max(1 if actual_ov >= 2 else 0, actual_ov // 2), chunk_len - 1)
        if store_in_memory:
            for local_i in range(start_write, chunk_len):
                left_all[cs + local_i] = c_left[local_i]
                right_all[cs + local_i] = c_right[local_i]

        l_areas = [int(m.sum()) for m in c_left]
        r_areas = [int(m.sum()) for m in c_right]
        chunk_logs.append(
            {
                "chunk_idx": c_idx,
                "start": cs,
                "end": ce,
                "ckpt_file": ckpt_p.name if ckpt_p is not None else None,
                "start_write": start_write,
                "from_checkpoint": loaded_from_ckpt,
                "handoff_iou_l": round(handoff_iou_l, 4),
                "handoff_iou_r": round(handoff_iou_r, 4),
                "used_fallback": used_fallback,
                "raw_l_area_minmax": [min(l_areas), max(l_areas)],
                "raw_r_area_minmax": [min(r_areas), max(r_areas)],
                "elapsed_sec": round(time.time() - tc, 2),
            }
        )

        prev_end = ce
        keep_tail = min(chunk_len, max(1, cur_overlap))
        prev_tail_start = ce - keep_tail
        prev_tail_l = c_left[chunk_len - keep_tail :]
        prev_tail_r = c_right[chunk_len - keep_tail :]

        if ce >= n_frames:
            break
        next_cs = ce - cur_overlap
        if next_cs <= cs:
            next_cs = cs + 1
        cs = next_cs
        c_idx += 1

    if store_in_memory:
        assert all(m is not None for m in left_all) and all(m is not None for m in right_all)
        return (
            [m for m in left_all if m is not None],
            [m for m in right_all if m is not None],
            chunk_logs,
        )
    return [], [], chunk_logs


def _iter_stitched_pass_masks(
    ckpt_dir: Path, chunk_logs: list[dict[str, Any]], *, reverse_order: bool = False
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Stream `(left_bool, right_bool)` from chunk `.npz` checkpoints loading each chunk once."""
    if not reverse_order:
        for ci, rec in enumerate(chunk_logs):
            cs = int(rec["start"])
            ce = int(rec["end"])
            start_w = int(rec["start_write"])
            next_write_global = (
                int(chunk_logs[ci + 1]["start"]) + int(chunk_logs[ci + 1]["start_write"])
                if ci + 1 < len(chunk_logs)
                else ce
            )
            end_w = min(ce - cs, max(start_w, next_write_global - cs))
            ckpt_p = ensure_safe_child_file(ckpt_dir, str(rec["ckpt_file"]))
            with np.load(ckpt_p) as data:
                l_arr = data["left"]
                r_arr = data["right"]
                for li in range(start_w, end_w):
                    yield l_arr[li].astype(bool), r_arr[li].astype(bool)
    else:
        for ci in range(len(chunk_logs) - 1, -1, -1):
            rec = chunk_logs[ci]
            cs = int(rec["start"])
            ce = int(rec["end"])
            start_w = int(rec["start_write"])
            next_write_global = (
                int(chunk_logs[ci + 1]["start"]) + int(chunk_logs[ci + 1]["start_write"])
                if ci + 1 < len(chunk_logs)
                else ce
            )
            end_w = min(ce - cs, max(start_w, next_write_global - cs))
            ckpt_p = ensure_safe_child_file(ckpt_dir, str(rec["ckpt_file"]))
            with np.load(ckpt_p) as data:
                l_arr = data["left"]
                r_arr = data["right"]
                for li in range(end_w - 1, start_w - 1, -1):
                    if cs + li == 0:
                        continue
                    yield l_arr[li].astype(bool), r_arr[li].astype(bool)


def track_bidirectional_from_seed(
    chunk_runner: Any,
    frame_loader: Callable[[int], Any],
    n_frames: int,
    seed_idx: int,
    init_left_prompts: dict[str, Any],
    init_right_prompts: dict[str, Any],
    *,
    chunk_size: int = 280,
    overlap: int = 10,
    handoff_iou_threshold: float = 0.85,
    ckpt_dir: Path | None = None,
    resume: bool = True,
    fingerprint: str = "",
    store_in_memory: bool = True,
    oom_cleanup_fn: Callable[[], None] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    """Seed SAM2.1 directly on `seed_idx` and propagate forward and (if seed_idx > 0) backward."""
    if not (0 <= seed_idx < n_frames):
        raise ValueError(f"seed_idx {seed_idx} out of range [0, {n_frames})")

    fwd_n = n_frames - seed_idx
    fwd_l, fwd_r, fwd_logs = run_chunked_tracking_loop(
        chunk_runner,
        lambda k: frame_loader(seed_idx + k),
        fwd_n,
        init_left_prompts,
        init_right_prompts,
        chunk_size=chunk_size,
        overlap=overlap,
        handoff_iou_threshold=handoff_iou_threshold,
        ckpt_dir=ckpt_dir,
        resume=resume,
        fingerprint=f"{fingerprint}_fwd",
        store_in_memory=store_in_memory,
        ckpt_prefix="fwd_chunk",
        oom_cleanup_fn=oom_cleanup_fn,
    )

    if seed_idx == 0:
        return fwd_l, fwd_r, fwd_logs, []

    bwd_n = seed_idx + 1
    bwd_l, bwd_r, bwd_logs = run_chunked_tracking_loop(
        chunk_runner,
        lambda k: frame_loader(seed_idx - k),
        bwd_n,
        init_left_prompts,
        init_right_prompts,
        chunk_size=chunk_size,
        overlap=overlap,
        handoff_iou_threshold=handoff_iou_threshold,
        ckpt_dir=ckpt_dir,
        resume=resume,
        fingerprint=f"{fingerprint}_bwd",
        store_in_memory=store_in_memory,
        ckpt_prefix="bwd_chunk",
        oom_cleanup_fn=oom_cleanup_fn,
    )

    if not store_in_memory:
        return [], [], fwd_logs, bwd_logs

    left_all = list(reversed(bwd_l[1:])) + fwd_l
    right_all = list(reversed(bwd_r[1:])) + fwd_r
    return left_all, right_all, fwd_logs, bwd_logs


def render_mask_overlay_grid(
    roi_dir: Path,
    frame_records: list[dict[str, Any]],
    lens_exact_dir: Path,
    glare_dir: Path,
    sample_idxs: list[int],
    fps: float,
    out_path: Path,
    cols: int = 5,
) -> None:
    """Render zoomed glasses overlay QA sheet (green=lens_exact, red=glare_hotspot)."""
    if not sample_idxs:
        return
    cols = min(cols, len(sample_idxs))
    rows = (len(sample_idxs) + cols - 1) // cols
    tw, th = 320, 200
    sheet = Image.new("RGB", (cols * tw, rows * th), (15, 15, 15))
    dr = ImageDraw.Draw(sheet)

    for idx, fidx in enumerate(sample_idxs):
        fn = validate_safe_basename(str(frame_records[fidx]["filename"]))
        rgb = np.array(Image.open(ensure_safe_child_file(roi_dir, fn)).convert("RGB"))
        lm = np.array(Image.open(ensure_safe_child_file(lens_exact_dir, fn)).convert("L"))
        gm = np.array(Image.open(ensure_safe_child_file(glare_dir, fn)).convert("L"))

        alpha_r = (gm.astype(np.float32) / 255.0) * 0.42
        red_layer = np.zeros_like(rgb, dtype=np.float32)
        red_layer[:, :, 0] = 255.0
        vis = (
            rgb.astype(np.float32) * (1.0 - alpha_r[:, :, None])
            + red_layer * alpha_r[:, :, None]
        ).astype(np.uint8)

        g_cnts, _ = cv2.findContours((gm > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, g_cnts, -1, (255, 60, 60), 1)
        l_cnts, _ = cv2.findContours((lm > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, l_cnts, -1, (0, 255, 80), 2)

        H, W = vis.shape[:2]
        ys, xs = np.where(lm > 127)
        if len(xs) > 0:
            pad_x = max(24, int(0.18 * (xs.max() - xs.min())))
            pad_y = max(24, int(0.45 * (ys.max() - ys.min())))
            x1 = max(0, int(xs.min()) - pad_x)
            x2 = min(W, int(xs.max()) + pad_x)
            y1 = max(0, int(ys.min()) - pad_y)
            y2 = min(H, int(ys.max()) + pad_y)
            crop = vis[y1:y2, x1:x2]
        else:
            crop = vis

        tile = Image.fromarray(crop).resize((tw, th), Image.Resampling.LANCZOS)
        r, c = divmod(idx, cols)
        sheet.paste(tile, (c * tw, r * th))
        t_sec = fidx / float(fps)
        cov = 100.0 * float((gm > 127).sum()) / float(max(1, int((lm > 127).sum())))
        lbl = f"t={t_sec:05.2f}s (f{fidx:04d}) g={cov:.0f}%"
        dr.rectangle([c * tw + 2, r * th + 2, c * tw + 218, r * th + 20], fill=(0, 0, 0))
        dr.text((c * tw + 6, r * th + 5), lbl, fill=(255, 255, 0))

    tmp_out = make_unique_temp_sibling(out_path)
    try:
        sheet.save(tmp_out, format="PNG")
        os.replace(tmp_out, out_path)
    except BaseException:
        tmp_out.unlink(missing_ok=True)
        raise


def compute_segment_generation(
    roi_frames_digest: str,
    seed_idx: int,
    init_left: dict[str, Any],
    init_right: dict[str, Any],
    config: SegmentConfig,
) -> tuple[str, str]:
    """Return `(sam2_generation, segment_generation)` bound to live ROI bytes & all behavior configs."""
    sam2_payload = json.dumps(
        {
            "roi_frames_digest": roi_frames_digest,
            "model_id": config.model_id,
            "device": config.device,
            "seed_idx": seed_idx,
            "init_left": init_left,
            "init_right": init_right,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "handoff_iou_threshold": config.handoff_iou_threshold,
        },
        sort_keys=True,
    )
    sam2_gen = hashlib.sha256(sam2_payload.encode()).hexdigest()[:24]

    seg_payload = json.dumps(
        {
            "sam2_generation": sam2_gen,
            "erode_px": config.erode_px,
            "exact_erode_px": config.exact_erode_px,
            "glare_dilate_px": config.glare_dilate_px,
            "glare_feather_sigma": config.glare_feather_sigma,
            "temporal_radius": config.temporal_radius,
            "glare_sensitivity": config.glare_sensitivity,
        },
        sort_keys=True,
    )
    seg_gen = hashlib.sha256(seg_payload.encode()).hexdigest()[:24]
    return sam2_gen, seg_gen


def run_segment_stage(
    roi_dir: Path,
    roi_meta: dict[str, Any],
    masks_root: Path,
    config: SegmentConfig,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """
    Execute Stage 2 end-to-end with:
    - Live ROI content hashing & upstream generation verification.
    - Non-zero seed-frame bidirectional SAM2.1 tracking, CPU processor/state placement, OOM backoff.
    - O(1)-RAM disk-streamed consistency-gated postprocessing & single-generation atomic publish.
    """
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    roi_dir = Path(roi_dir)
    masks_root = Path(masks_root)
    reject_symlink_path(roi_dir, "roi_dir")
    reject_symlink_path(masks_root.parent, "workspace root")
    reject_symlink_path(masks_root, "masks_root")

    frame_records = roi_meta["frames"]
    n = len(frame_records)
    if n == 0:
        raise ValueError("roi_meta contains 0 frames.")

    filenames = [validate_safe_basename(str(rec["filename"])) for rec in frame_records]
    # ALWAYS hash live ROI frames rather than blindly trusting roi_meta's recorded digest
    live_roi_digest = hash_files_manifest(roi_dir, filenames)
    recorded_roi_digest = roi_meta.get("roi_frames_digest")
    if recorded_roi_digest and recorded_roi_digest != live_roi_digest:
        raise ValueError(
            "ROI frames on disk do not match prep roi_boxes.json digest; rerun prep first."
        )

    roi_size = int(roi_meta.get("roi_size", 512))
    fps = float(roi_meta.get("fps", 30.0))
    seed_idx = max(0, min(n - 1, int(roi_meta.get("init_frame_idx", 0))))
    prep_gen = roi_meta.get("prep_generation")

    init_left, init_right = build_initial_lens_prompts(
        roi_meta.get("init_eye_geometry_roi"),
        roi_size=roi_size,
        override_prompts=config.lens_points,
    )
    sam2_gen, seg_gen = compute_segment_generation(
        live_roi_digest, seed_idx, init_left, init_right, config
    )

    configure_sdpa_guard(torch)
    device = resolve_torch_device(torch, config.device)
    dtype = preferred_inference_dtype(torch, device)

    # Keep resumable SAM2 chunk checkpoints in a sibling dir so masks_root commits atomically as one unit
    ckpt_dir = ensure_confined_dir(
        masks_root.parent, masks_root.parent / f".{masks_root.name}_sam2_chunks", create=True
    )

    with gpu_file_lock(config.gpu_lock):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        sam_state: dict[str, Any] = {"processor": None, "model": None}

        def _load_pil(idx: int) -> Image.Image:
            fp = ensure_safe_child_file(roi_dir, filenames[idx])
            return Image.open(fp).convert("RGB")

        def _chunk_runner(
            frames_pil: list[Image.Image],
            *,
            left_prompts: dict[str, Any] | None,
            right_prompts: dict[str, Any] | None,
            init_masks: tuple[np.ndarray, np.ndarray] | None,
        ) -> tuple[list[np.ndarray], list[np.ndarray]]:
            if sam_state["model"] is None:
                sam_state["processor"] = Sam2VideoProcessor.from_pretrained(config.model_id)
                sam_state["model"] = (
                    Sam2VideoModel.from_pretrained(config.model_id, dtype=dtype)
                    .to(device)
                    .eval()
                )
            return _run_sam2_single_chunk(
                sam_state["model"],
                sam_state["processor"],
                frames_pil,
                device=device,
                dtype=dtype,
                left_prompts=left_prompts,
                right_prompts=right_prompts,
                init_masks=init_masks,
            )

        try:
            _, _, fwd_logs, bwd_logs = track_bidirectional_from_seed(
                _chunk_runner,
                _load_pil,
                n,
                seed_idx,
                init_left,
                init_right,
                chunk_size=config.chunk_size,
                overlap=config.chunk_overlap,
                handoff_iou_threshold=config.handoff_iou_threshold,
                ckpt_dir=ckpt_dir,
                resume=resume,
                fingerprint=sam2_gen,
                store_in_memory=False,
                oom_cleanup_fn=lambda: cuda_empty_cache_if_needed(torch, device),
            )
            peak_gb = cuda_peak_vram_gib(torch, device)
        finally:
            sam_state.clear()
            gc.collect()
            cuda_empty_cache_if_needed(torch, device)

    def _stream_raw_pairs() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if bwd_logs:
            yield from _iter_stitched_pass_masks(ckpt_dir, bwd_logs, reverse_order=True)
        yield from _iter_stitched_pass_masks(ckpt_dir, fwd_logs, reverse_order=False)

    # Stage ALL published mask dirs + mask_stats.json into ONE generation dir for single atomic swap
    masks_root.parent.mkdir(parents=True, exist_ok=True)
    stage_masks = make_unique_temp_dir(masks_root.parent, prefix=f".stage_{masks_root.name}_")
    stage_lens_dir = stage_masks / "roi_lens_interior"
    stage_exact_dir = stage_masks / "roi_lens_exact"
    stage_glare_dir = stage_masks / "roi_glare_hotspot"
    for d in (stage_lens_dir, stage_exact_dir, stage_glare_dir):
        d.mkdir(parents=True, exist_ok=True)

    radius = config.temporal_radius
    stats_list: list[dict[str, Any]] = []

    try:
        raw_iter = _stream_raw_pairs()
        win_bufs: dict[str, dict[int, tuple[np.ndarray | None, np.ndarray, bool]]] = {
            "lc": {},
            "rc": {},
            "le": {},
            "re": {},
        }
        loaded_up_to = 0

        def _ensure_win(target_excl: int) -> None:
            nonlocal loaded_up_to
            while loaded_up_to < target_excl and loaded_up_to < n:
                l_raw, r_raw = next(raw_iter)
                win_bufs["lc"][loaded_up_to] = _capped_sdf_entry(
                    clean_single_lens_mask(l_raw, config.erode_px)
                )
                win_bufs["rc"][loaded_up_to] = _capped_sdf_entry(
                    clean_single_lens_mask(r_raw, config.erode_px)
                )
                win_bufs["le"][loaded_up_to] = _capped_sdf_entry(
                    clean_single_lens_mask(l_raw, config.exact_erode_px)
                )
                win_bufs["re"][loaded_up_to] = _capped_sdf_entry(
                    clean_single_lens_mask(r_raw, config.exact_erode_px)
                )
                loaded_up_to += 1

        for i in range(n):
            lo = max(0, i - radius)
            hi = min(n, i + radius + 1)
            _ensure_win(hi)

            lc_i = _resolve_consistent_window_mask(win_bufs["lc"], i, lo, hi, radius=radius)
            rc_i = _resolve_consistent_window_mask(win_bufs["rc"], i, lo, hi, radius=radius)
            le_i = _resolve_consistent_window_mask(win_bufs["le"], i, lo, hi, radius=radius)
            re_i = _resolve_consistent_window_mask(win_bufs["re"], i, lo, hi, radius=radius)

            for ch_buf in win_bufs.values():
                for k in list(ch_buf.keys()):
                    if k < i - radius:
                        del ch_buf[k]

            fn = filenames[i]
            roi_fp = ensure_safe_child_file(roi_dir, fn)
            bgr = cv2.imread(str(roi_fp))
            if bgr is None:
                raise FileNotFoundError(f"Missing ROI frame: {roi_fp}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            lens_roi = np.maximum(lc_i, rc_i)
            exact_roi = np.maximum(le_i, re_i)
            glare_roi = compute_glare_hotspot_mask(
                rgb,
                lens_roi,
                dilate_px=config.glare_dilate_px,
                feather_sigma=config.glare_feather_sigma,
                sensitivity=config.glare_sensitivity,
            )

            cv2.imwrite(str(stage_lens_dir / fn), lens_roi)
            cv2.imwrite(str(stage_exact_dir / fn), exact_roi)
            cv2.imwrite(str(stage_glare_dir / fn), glare_roi)

            l_px = int((lc_i > 127).sum())
            r_px = int((rc_i > 127).sum())
            le_px = int((le_i > 127).sum())
            re_px = int((re_i > 127).sum())
            g_px = int((glare_roi > 127).sum())
            gl_px = int(((glare_roi > 127) & (lc_i > 127)).sum())
            gr_px = int(((glare_roi > 127) & (rc_i > 127)).sum())
            stats_list.append(
                {
                    "frame_idx": i,
                    "left_lens_roi_px": l_px,
                    "right_lens_roi_px": r_px,
                    "left_exact_roi_px": le_px,
                    "right_exact_roi_px": re_px,
                    "glare_hotspot_roi_px": g_px,
                    "glare_left_pct": round(100.0 * gl_px / max(1, l_px), 2),
                    "glare_right_pct": round(100.0 * gr_px / max(1, r_px), 2),
                    "glare_pct_of_lenses": round(100.0 * g_px / max(1, l_px + r_px), 2),
                }
            )

        lens_digest = hash_files_manifest(stage_lens_dir, filenames)
        exact_digest = hash_files_manifest(stage_exact_dir, filenames)
        glare_digest = hash_files_manifest(stage_glare_dir, filenames)
        masks_digest = hashlib.sha256(
            f"{lens_digest}:{exact_digest}:{glare_digest}".encode()
        ).hexdigest()

        l_areas = [s["left_lens_roi_px"] for s in stats_list]
        r_areas = [s["right_lens_roi_px"] for s in stats_list]
        l_med = float(np.median(l_areas)) if l_areas else 0.0
        r_med = float(np.median(r_areas)) if r_areas else 0.0
        anomalies = [
            s["frame_idx"]
            for s in stats_list
            if (l_med > 0 and not (0.65 * l_med <= s["left_lens_roi_px"] <= 1.40 * l_med))
            or (r_med > 0 and not (0.65 * r_med <= s["right_lens_roi_px"] <= 1.40 * r_med))
        ]
        g_pcts = [s["glare_pct_of_lenses"] for s in stats_list]
        summary = {
            "model_id": config.model_id,
            "prep_generation": prep_gen,
            "sam2_generation": sam2_gen,
            "segment_generation": seg_gen,
            "roi_frames_digest": live_roi_digest,
            "masks_digest": masks_digest,
            "seed_frame_idx": seed_idx,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "handoff_iou_threshold": config.handoff_iou_threshold,
            "erode_px_roi": config.erode_px,
            "exact_erode_px_roi": config.exact_erode_px,
            "glare_dilate_px_roi": config.glare_dilate_px,
            "glare_sensitivity": config.glare_sensitivity,
            "num_frames": n,
            "init_prompts": {"left": init_left, "right": init_right},
            "left_lens_px_min_med_max": [min(l_areas), int(l_med), max(l_areas)],
            "right_lens_px_min_med_max": [min(r_areas), int(r_med), max(r_areas)],
            "area_anomaly_frames": anomalies,
            "glare_pct_min_med_max": [min(g_pcts), round(float(np.median(g_pcts)), 2), max(g_pcts)],
            "peak_vram_gib": round(peak_gb, 2),
            "chunks": fwd_logs + bwd_logs,
            "frames": stats_list,
        }
        atomic_write_text(stage_masks / "mask_stats.json", json.dumps(summary, indent=2))

        if config.write_qa_grid:
            sample_idxs = np.linspace(0, n - 1, min(config.qa_grid_samples, n), dtype=int).tolist()
            render_mask_overlay_grid(
                roi_dir,
                frame_records,
                stage_exact_dir,
                stage_glare_dir,
                sample_idxs=sample_idxs,
                fps=fps,
                out_path=stage_masks / "mask_qa_grid.png",
            )

        # Single atomic generation commit: mask dirs + manifest appear together or not at all!
        atomic_swap_directory(stage_masks, masks_root)
    except BaseException:
        if stage_masks.exists():
            shutil.rmtree(stage_masks, ignore_errors=True)
        raise

    return summary
