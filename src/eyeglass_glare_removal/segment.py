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


def _capped_sdf(mask_u8: np.ndarray, max_sdf: float = 6.0) -> tuple[np.ndarray | None, bool]:
    """
    Compute a safely capped signed distance field in [-max_sdf, +max_sdf].

    Explicitly handles empty masks (returns (None, True)) and all-foreground masks
    (returns constant +max_sdf) so OpenCV distanceTransform never receives an all-ones
    mask and produces extreme unnormalized distances that wipe neighboring frames.
    """
    b = (mask_u8 > 127).astype(np.uint8)
    fg = int(b.sum())
    if fg == 0:
        return None, True
    if fg == b.size:
        return np.full(b.shape, max_sdf, dtype=np.float32), False
    d_in = cv2.distanceTransform(b, cv2.DIST_L2, 5)
    d_out = cv2.distanceTransform(1 - b, cv2.DIST_L2, 5)
    sdf = np.clip(d_in - d_out, -max_sdf, max_sdf).astype(np.float32)
    return sdf, False


def iter_temporal_smoothed_masks(
    mask_stream: Iterator[np.ndarray],
    n_frames: int,
    radius: int = 1,
    max_sdf: float = 6.0,
) -> Iterator[np.ndarray]:
    """
    O(1)-RAM streaming majority-gated capped-SDF temporal binary mask smoother.

    Holds at most `2 * radius + 1` frames in memory at any instant.
    - Isolated 1-frame empty dropouts do NOT erase or shrink valid neighboring frames.
    - Genuine empty regions (majority empty in window) remain empty.
    - Handles all-empty and all-full masks explicitly without OpenCV distanceTransform overflow.
    """
    if n_frames <= 0:
        return
    if radius <= 0 or n_frames < 3:
        for m in mask_stream:
            yield ((m > 127).astype(np.uint8) * 255)
        return

    buf: dict[int, tuple[np.ndarray | None, bool, tuple[int, int]]] = {}
    next_read = 0

    def _ensure_loaded(up_to_excl: int) -> None:
        nonlocal next_read
        while next_read < up_to_excl and next_read < n_frames:
            m = next(mask_stream)
            sdf, is_empty = _capped_sdf(m, max_sdf=max_sdf)
            buf[next_read] = (sdf, is_empty, (m.shape[0], m.shape[1]))
            next_read += 1

    for i in range(n_frames):
        lo = max(0, i - radius)
        hi = min(n_frames, i + radius + 1)
        _ensure_loaded(hi)

        window_len = hi - lo
        valid_sdfs = [buf[j][0] for j in range(lo, hi) if not buf[j][1] and buf[j][0] is not None]
        shape = buf[i][2]

        # Majority-gated: if majority of window is empty, keep empty; else smooth over valid masks
        if len(valid_sdfs) * 2 < window_len or not valid_sdfs:
            yield np.zeros(shape, dtype=np.uint8)
        else:
            avg_sdf = sum(valid_sdfs) / float(len(valid_sdfs))
            yield (avg_sdf > 0.0).astype(np.uint8) * 255

        # Evict frames outside the sliding window to maintain O(1) RAM
        for k in list(buf.keys()):
            if k < i - radius:
                del buf[k]


def temporal_smooth_binary_masks(masks: list[np.ndarray], radius: int = 1) -> list[np.ndarray]:
    """Smooth a list of binary masks temporally via majority-gated capped SDF window."""
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
    """
    General photometric glare detector strictly inside `lens_interior`.

    Detects:
      1) Bright specular & low-saturation neutral reflections
      2) Broad uniform/washed-out neutral glare over darker skin tones (distinguished from
         small natural sclera lobes via per-lens connected-component area fraction)
      3) Cool bluish-white screen reflections
      4) Warm incandescent / orange-red lamp reflections
      5) Green/cyan and purple/violet/magenta AR-coating reflections

    Guarantees output is strictly 0 everywhere outside `lens_interior`.
    """
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

    # 1) Bright specular & low-saturation neutral washout
    bright_spec_thresh = min(242.0, max(skin_L85 + 26.0 * scale, 202.0 - 10.0 * (1.0 - scale)))
    washout_L_thresh = min(234.0, max(skin_L85 + 34.0 * scale, 194.0 - 8.0 * (1.0 - scale)))
    bright_specular = (L >= bright_spec_thresh) | (
        (L >= washout_L_thresh) & (chroma <= 15.0 / max(0.5, scale))
    )

    # 2) Broad neutral washed-out reflection over dark/medium skin (e.g. RGB 175 sheet filling lens)
    #    Natural sclera forms small lobes (< 10% of lens area) adjacent to dark iris;
    #    a broad neutral reflection spans >= 12% of a lens component with strong luminance jump.
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

    # 3) Cool / bluish screen reflection
    cool_b_thresh = min(skin_b - 10.0 * scale, 125.0)
    cool_screen = (
        ((b <= cool_b_thresh) & (L >= max(65.0, skin_L + 8.0 * scale)))
        | (((B - R) >= 6.0 * scale) & (L >= max(85.0, skin_L + 8.0 * scale)))
    )

    # 4) Warm orange/amber/red lamp reflection
    warm_a_delta = max(6.0, 2.2 * skin_a_mad) * scale
    warm_b_delta = max(10.0, 2.5 * skin_b_mad) * scale
    warm_lamp = (
        ((a >= skin_a + warm_a_delta) & (L >= skin_L + 4.0 * scale))
        | ((b >= skin_b + warm_b_delta) & (a >= skin_a + 3.0 * scale) & (L >= skin_L + 8.0 * scale))
    )

    # 5) Green / cyan AR-coating reflection
    ar_min_L = max(48.0, skin_L - 25.0)
    green_ar = (
        ((a <= min(skin_a - 10.0 * scale, 123.0)) & (L >= ar_min_L))
        | (((G - R) >= 10.0 * scale) & ((G - B) >= 6.0 * scale) & (L >= 52.0))
    )

    # 6) Purple / violet / magenta AR-coating reflection
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
                left=np.stack(left_masks, axis=0),
                right=np.stack(right_masks, axis=0),
                iou_l=np.float32(iou_l),
                iou_r=np.float32(iou_r),
                used_fallback=np.bool_(used_fallback),
                fingerprint=np.array(fingerprint),
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, ckpt_path)
    except Exception:
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
    Verify per-lens IoU across the overlap (seed, splice, and full overlap) AND detect
    left/right identity swaps around the actual splice point.
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

    s_idx = min(max(0, splice_i), ov_len - 1)
    l_ious = [mask_iou(prev_l_ov[k], cand_l_ov[k]) for k in range(ov_len)]
    r_ious = [mask_iou(prev_r_ov[k], cand_r_ov[k]) for k in range(ov_len)]
    cross_lr = [mask_iou(prev_l_ov[k], cand_r_ov[k]) for k in range(ov_len)]
    cross_rl = [mask_iou(prev_r_ov[k], cand_l_ov[k]) for k in range(ov_len)]

    splice_l = l_ious[s_idx]
    splice_r = r_ious[s_idx]
    splice_clr = cross_lr[s_idx]
    splice_crl = cross_rl[s_idx]

    # Identity swap detected if cross-overlap beats direct overlap at splice or across overlap
    swapped = (
        (splice_clr > splice_l + 0.05)
        or (splice_crl > splice_r + 0.05)
        or (float(np.mean(cross_lr)) > float(np.mean(l_ious)) + 0.05)
        or (float(np.mean(cross_rl)) > float(np.mean(r_ious)) + 0.05)
    )
    min_iou = min(l_ious[0], r_ious[0], splice_l, splice_r, min(l_ious), min(r_ious))
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
    - Safe zero-overlap handling (independent re-seeding) AND multi-frame overlap verification
      with explicit identity-swap rejection at the actual splice point.
    - Automatic OOM retry halving chunk length cleanly.
    - Atomic unique-temp `.npz` checkpoints with fingerprint validation.
    - Optional disk-only mode (`store_in_memory=False`, requiring `ckpt_dir`) for O(1) host RAM.
    """
    if not store_in_memory and ckpt_dir is None:
        raise ValueError("store_in_memory=False requires ckpt_dir.")
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    if chunk_size < 2:
        raise ValueError(f"chunk_size must be >= 2, got {chunk_size}")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < chunk_size ({chunk_size}), got {overlap}")

    left_all: list[np.ndarray | None] = [None] * n_frames if store_in_memory else []
    right_all: list[np.ndarray | None] = [None] * n_frames if store_in_memory else []
    chunk_logs: list[dict[str, Any]] = []

    if ckpt_dir is not None:
        reject_symlink_path(ckpt_dir, "SAM2 checkpoint directory")
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    fallback_pair = (init_left_prompts, init_right_prompts)

    # Keep only the previous chunk's overlap tail in RAM so even long videos stay O(1) RAM!
    prev_tail_start: int | None = None
    prev_tail_l: list[np.ndarray] = []
    prev_tail_r: list[np.ndarray] = []

    cur_chunk_size = chunk_size
    cs = 0
    c_idx = 0

    while cs < n_frames:
        ce = min(n_frames, cs + cur_chunk_size)
        chunk_len = ce - cs
        tc = time.time()

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
                with np.load(ckpt_p) as data:
                    stored_fp = str(data["fingerprint"].item()) if "fingerprint" in data else ""
                    fp_ok = (not fingerprint) or (stored_fp == fingerprint)
                    if (
                        fp_ok
                        and data["left"].shape[0] == chunk_len
                        and data["right"].shape[0] == chunk_len
                    ):
                        c_left = [data["left"][i].astype(bool) for i in range(chunk_len)]
                        c_right = [data["right"][i].astype(bool) for i in range(chunk_len)]
                        handoff_iou_l = float(data["iou_l"]) if "iou_l" in data else 1.0
                        handoff_iou_r = float(data["iou_r"]) if "iou_r" in data else 1.0
                        used_fallback = bool(data["used_fallback"]) if "used_fallback" in data else False
                        loaded_from_ckpt = True
            except Exception as exc:
                logger.warning("Corrupt SAM2 chunk checkpoint %s (%s); recomputing.", ckpt_p, exc)

        if not loaded_from_ckpt:
            try:
                chunk_frames = [frame_loader(i) for i in range(cs, ce)]
                actual_ov = 0 if c_idx == 0 else max(0, (prev_tail_start or 0) + len(prev_tail_l) - cs)

                if c_idx == 0 or actual_ov == 0:
                    # First chunk or zero-overlap disjoint chunk -> seed cleanly from initial prompts
                    c_left, c_right = chunk_runner(
                        chunk_frames,
                        left_prompts=init_left_prompts,
                        right_prompts=init_right_prompts,
                        init_masks=None,
                    )
                else:
                    assert prev_tail_start is not None
                    ov_offset = cs - prev_tail_start
                    prev_l_ov = prev_tail_l[ov_offset : ov_offset + actual_ov]
                    prev_r_ov = prev_tail_r[ov_offset : ov_offset + actual_ov]
                    prev_l0 = prev_l_ov[0]
                    prev_r0 = prev_r_ov[0]
                    splice_i = actual_ov // 2

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

                        # Select strictly non-swapped better candidate; if both fail, reject swap
                        candidates = [
                            (not ev1["swapped"], ev1["min_iou"], False, c_left, c_right, ev1),
                            (not ev2["swapped"], ev2["min_iou"], True, fb_left, fb_right, ev2),
                        ]
                        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
                        best_non_swapped, best_min_iou, best_fb, c_left, c_right, best_ev = candidates[0]
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
                        fingerprint=fingerprint,
                    )
            except Exception as exc:
                min_allowed = max(overlap + 2, 8)
                if _is_oom_error(exc) and cur_chunk_size > min_allowed:
                    new_sz = max(min_allowed, cur_chunk_size // 2)
                    logger.warning(
                        "CUDA OOM on SAM2 chunk [%d:%d] (size=%d); retrying with chunk_size=%d.",
                        cs,
                        ce,
                        cur_chunk_size,
                        new_sz,
                    )
                    cur_chunk_size = new_sz
                    if oom_cleanup_fn is not None:
                        oom_cleanup_fn()
                    gc.collect()
                    continue
                raise

        start_write = 0 if c_idx == 0 else min(overlap // 2, chunk_len - 1)
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

        # Retain ONLY the tail needed for next chunk's overlap
        keep_tail = min(chunk_len, max(1, overlap))
        prev_tail_start = ce - keep_tail
        prev_tail_l = c_left[chunk_len - keep_tail :]
        prev_tail_r = c_right[chunk_len - keep_tail :]

        if ce >= n_frames:
            break
        next_cs = ce - overlap
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
    """
    Stream `(left_bool, right_bool)` from chunk `.npz` checkpoints holding only one chunk at a time.
    """
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
        # Backward sub-pass stored frames in order [seed_idx, seed_idx-1, ..., 0];
        # iterate chunks in reverse to emit global frames [0, 1, ..., seed_idx-1] in forward order!
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
                    # Skip local 0 of chunk 0 (= seed_idx, already covered by forward pass)
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
    """
    Seed SAM2.1 directly on `seed_idx` (the actual first frame where face/eyes were detected),
    propagating forward `[seed_idx .. n-1]` and (when `seed_idx > 0`) backward `[seed_idx .. 0]`.

    Returns `(left_all, right_all, fwd_logs, bwd_logs)`.
    """
    if not (0 <= seed_idx < n_frames):
        raise ValueError(f"seed_idx {seed_idx} out of range [0, {n_frames})")

    # Forward pass: [seed_idx .. n-1]
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

    # Backward pass: [seed_idx, seed_idx-1, ..., 0] (length seed_idx + 1, local 0 == seed_idx)
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

    # Reassemble [0 .. seed_idx-1] from reversed backward pass + [seed_idx .. n-1] from forward pass
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
    except Exception:
        tmp_out.unlink(missing_ok=True)
        raise


def compute_segment_generation(
    roi_frames_digest: str,
    seed_idx: int,
    init_left: dict[str, Any],
    init_right: dict[str, Any],
    config: SegmentConfig,
) -> tuple[str, str]:
    """Return `(sam2_generation, segment_generation)` bound to upstream ROI bytes + config."""
    sam2_payload = json.dumps(
        {
            "roi_frames_digest": roi_frames_digest,
            "model_id": config.model_id,
            "seed_idx": seed_idx,
            "init_left": init_left,
            "init_right": init_right,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
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
    - Safe basename validation & symlink confinement.
    - Non-zero seed-frame bidirectional SAM2.1 tracking & CPU processor/state placement.
    - O(1)-RAM disk-streamed temporal post-processing & atomic staged directory publishing.
    """
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    roi_dir = Path(roi_dir)
    masks_root = Path(masks_root)
    reject_symlink_path(roi_dir, "roi_dir")
    reject_symlink_path(masks_root, "masks_root")
    masks_root.mkdir(parents=True, exist_ok=True)

    frame_records = roi_meta["frames"]
    n = len(frame_records)
    if n == 0:
        raise ValueError("roi_meta contains 0 frames.")

    filenames = [validate_safe_basename(str(rec["filename"])) for rec in frame_records]
    roi_frames_digest = str(roi_meta.get("roi_frames_digest") or "") or hash_files_manifest(
        roi_dir, filenames
    )

    roi_size = int(roi_meta.get("roi_size", 512))
    fps = float(roi_meta.get("fps", 30.0))
    seed_idx = max(0, min(n - 1, int(roi_meta.get("init_frame_idx", 0))))

    init_left, init_right = build_initial_lens_prompts(
        roi_meta.get("init_eye_geometry_roi"),
        roi_size=roi_size,
        override_prompts=config.lens_points,
    )
    sam2_gen, seg_gen = compute_segment_generation(
        roi_frames_digest, seed_idx, init_left, init_right, config
    )

    configure_sdpa_guard(torch)
    device = resolve_torch_device(torch, config.device)
    dtype = preferred_inference_dtype(torch, device)

    ckpt_dir = ensure_confined_dir(masks_root, masks_root / "_sam2_chunks", create=True)

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

    # Bounded O(1)-RAM disk-streamed temporal smoothing & glare hotspot export
    def _stream_raw_pairs() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if bwd_logs:
            yield from _iter_stitched_pass_masks(ckpt_dir, bwd_logs, reverse_order=True)
        yield from _iter_stitched_pass_masks(ckpt_dir, fwd_logs, reverse_order=False)

    # Stream each of the 4 cleaned mask channels through a bounded sliding window (<= 2*r+1 in RAM)
    # Using a single pass over raw pairs by keeping a tiny shared sliding buffer of cleaned masks!
    stage_lens_dir = make_unique_temp_dir(masks_root, prefix=".stage_lens_interior_")
    stage_exact_dir = make_unique_temp_dir(masks_root, prefix=".stage_lens_exact_")
    stage_glare_dir = make_unique_temp_dir(masks_root, prefix=".stage_glare_hotspot_")

    radius = config.temporal_radius
    stats_list: list[dict[str, Any]] = []

    try:
        raw_iter = _stream_raw_pairs()
        win_buf: dict[int, dict[str, tuple[np.ndarray | None, bool, tuple[int, int]]]] = {}
        loaded_up_to = 0

        def _ensure_win(target_excl: int) -> None:
            nonlocal loaded_up_to
            while loaded_up_to < target_excl and loaded_up_to < n:
                l_raw, r_raw = next(raw_iter)
                lc = clean_single_lens_mask(l_raw, config.erode_px)
                rc = clean_single_lens_mask(r_raw, config.erode_px)
                le = clean_single_lens_mask(l_raw, config.exact_erode_px)
                re_ = clean_single_lens_mask(r_raw, config.exact_erode_px)
                shape = (lc.shape[0], lc.shape[1])
                win_buf[loaded_up_to] = {
                    "lc": (*_capped_sdf(lc), shape),
                    "rc": (*_capped_sdf(rc), shape),
                    "le": (*_capped_sdf(le), shape),
                    "re": (*_capped_sdf(re_), shape),
                }
                loaded_up_to += 1

        def _resolve_smooth(i: int, key: str, lo: int, hi: int) -> np.ndarray:
            shape = win_buf[i][key][2]
            if radius <= 0 or n < 3:
                sdf_i, empty_i, _ = win_buf[i][key]
                if empty_i or sdf_i is None:
                    return np.zeros(shape, dtype=np.uint8)
                return (sdf_i > 0.0).astype(np.uint8) * 255
            valid_sdfs = [
                win_buf[j][key][0]
                for j in range(lo, hi)
                if not win_buf[j][key][1] and win_buf[j][key][0] is not None
            ]
            if len(valid_sdfs) * 2 < (hi - lo) or not valid_sdfs:
                return np.zeros(shape, dtype=np.uint8)
            avg_sdf = sum(valid_sdfs) / float(len(valid_sdfs))
            return (avg_sdf > 0.0).astype(np.uint8) * 255

        for i in range(n):
            lo = max(0, i - radius)
            hi = min(n, i + radius + 1)
            _ensure_win(hi)

            lc_i = _resolve_smooth(i, "lc", lo, hi)
            rc_i = _resolve_smooth(i, "rc", lo, hi)
            le_i = _resolve_smooth(i, "le", lo, hi)
            re_i = _resolve_smooth(i, "re", lo, hi)

            for k in list(win_buf.keys()):
                if k < i - radius:
                    del win_buf[k]

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

        roi_lens_dir = ensure_confined_dir(masks_root, masks_root / "roi_lens_interior", create=False)
        roi_exact_dir = ensure_confined_dir(masks_root, masks_root / "roi_lens_exact", create=False)
        roi_glare_dir = ensure_confined_dir(masks_root, masks_root / "roi_glare_hotspot", create=False)

        atomic_swap_directory(stage_lens_dir, roi_lens_dir)
        atomic_swap_directory(stage_exact_dir, roi_exact_dir)
        atomic_swap_directory(stage_glare_dir, roi_glare_dir)
    except Exception:
        for sd in (stage_lens_dir, stage_exact_dir, stage_glare_dir):
            if sd.exists():
                shutil.rmtree(sd, ignore_errors=True)
        raise

    lens_digest = hash_files_manifest(roi_lens_dir, filenames)
    exact_digest = hash_files_manifest(roi_exact_dir, filenames)
    glare_digest = hash_files_manifest(roi_glare_dir, filenames)
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
        "sam2_generation": sam2_gen,
        "segment_generation": seg_gen,
        "roi_frames_digest": roi_frames_digest,
        "masks_digest": masks_digest,
        "seed_frame_idx": seed_idx,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
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
    atomic_write_text(masks_root / "mask_stats.json", json.dumps(summary, indent=2))

    if config.write_qa_grid:
        sample_idxs = np.linspace(0, n - 1, min(config.qa_grid_samples, n), dtype=int).tolist()
        render_mask_overlay_grid(
            roi_dir,
            frame_records,
            roi_exact_dir,
            roi_glare_dir,
            sample_idxs=sample_idxs,
            fps=fps,
            out_path=masks_root / "mask_qa_grid.png",
        )

    return summary
