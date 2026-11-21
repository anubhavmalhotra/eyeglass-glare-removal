"""Stage 5: General rim-safe bounded-RAM streaming compositor and QA/compare renderer."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from eyeglass_glare_removal._workspace import (
    assert_distinct_output_paths,
    ensure_safe_child_file,
    make_unique_temp_sibling,
    reject_symlink_path,
    validate_safe_basename,
)
from eyeglass_glare_removal.config import CompositeConfig
from eyeglass_glare_removal.media import close_video_writer_pipe, open_video_writer_pipe

logger = logging.getLogger(__name__)

_QA_TILE_W = 360
_QA_TILE_H = 200


def smoothstep(x: np.ndarray) -> np.ndarray:
    """C1-smooth Hermite ramp mapped onto [0, 1]."""
    t = np.clip(x, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def inward_feather_alpha(binary_mask: np.ndarray, feather_px: float) -> np.ndarray:
    """Compute inward distance-transform feather strictly inside binary_mask."""
    m = (binary_mask > 127).astype(np.uint8)
    if feather_px <= 0.0:
        return m.astype(np.float32)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    return smoothstep(dist / float(feather_px))


def suppress_chroma_glint(
    clean_roi: np.ndarray,
    glare_mask: np.ndarray | None,
    *,
    purple_threshold: float = 10.0,
    green_threshold: float = 14.0,
) -> np.ndarray:
    """Strictly inside `glare_mask`, softly neutralize residual purple/bright-green AR glints."""
    if glare_mask is None:
        return clean_roi

    g_w = smoothstep((glare_mask.astype(np.float32) - 40.0) / 80.0)
    if float(g_w.max()) <= 0.0:
        return clean_roi

    out = clean_roi.astype(np.float32).copy()
    R, G, B = out[:, :, 0], out[:, :, 1], out[:, :, 2]
    luma = 0.299 * R + 0.587 * G + 0.114 * B

    purple_excess = np.minimum(R - G, B - G)
    w_purple = smoothstep((purple_excess - purple_threshold) / 10.0) * g_w

    green_excess = np.minimum(G - R, G - B)
    bright_gate = smoothstep((luma - 125.0) / 25.0)
    w_green = smoothstep((green_excess - green_threshold) / 12.0) * bright_gate * g_w

    w = np.clip(np.maximum(w_purple, w_green), 0.0, 1.0)
    if float(w.max()) <= 0.0:
        return clean_roi

    t_R = np.clip(luma * 1.04, 0.0, 255.0)
    t_G = np.clip(luma * 0.99, 0.0, 255.0)
    t_B = np.clip(luma * 0.96, 0.0, 255.0)

    out[:, :, 0] = R * (1.0 - w) + t_R * w
    out[:, :, 1] = G * (1.0 - w) + t_G * w
    out[:, :, 2] = B * (1.0 - w) + t_B * w
    return np.clip(out, 0.0, 255.0)


def subtle_boundary_color_match(
    orig_roi: np.ndarray,
    clean_roi: np.ndarray,
    lens_mask: np.ndarray,
    glare_mask: np.ndarray | None,
    strength: float = 0.0,
) -> np.ndarray:
    """Optional glare-excluded per-lens exposure alignment (disabled when strength == 0)."""
    if strength <= 0.0:
        return clean_roi

    m = (lens_mask > 127).astype(np.uint8)
    orig_f = orig_roi.astype(np.float32)
    orig_luma = 0.299 * orig_f[:, :, 0] + 0.587 * orig_f[:, :, 1] + 0.114 * orig_f[:, :, 2]

    p85 = float(np.percentile(orig_luma[m > 0], 85)) if int(m.sum()) > 0 else 200.0
    non_glare = (m > 0) & (orig_luma < min(210.0, p85))
    if glare_mask is not None:
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        glare_excl = cv2.dilate((glare_mask > 50).astype(np.uint8), kd) > 0
        non_glare = non_glare & (~glare_excl)

    num_labels, labels = cv2.connectedComponents(m)
    out = clean_roi.astype(np.float32).copy()
    for lbl in range(1, num_labels):
        comp = labels == lbl
        ref_region = comp & non_glare
        if int(ref_region.sum()) < 80:
            continue
        for ch in range(3):
            o_med = float(np.median(orig_f[:, :, ch][ref_region]))
            c_med = float(np.median(out[:, :, ch][ref_region]))
            shift = float(np.clip(o_med - c_med, -14.0, 10.0)) * strength
            out[:, :, ch][comp] = np.clip(out[:, :, ch][comp] + shift, 0.0, 255.0)

    return out


def compute_rim_safe_alpha(
    orig_roi_rgb: np.ndarray,
    clean_roi_rgb: np.ndarray,
    lens_mask_roi: np.ndarray,
    glare_mask_roi: np.ndarray | None,
    *,
    lens_expand_px: int = 0,
    feather_px: float = 2.0,
    bevel_band_px: int = 4,
    rim_protect_strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute `(alpha_roi [0..1 float32], active_support_roi [uint8 0/255])`.

    Key guarantees:
    1. When `glare_mask_roi` is provided and empty (zero glare), returns all-zero alpha and
       all-zero support so clean frames & clean lenses are preserved bit-for-bit untouched.
    2. When `glare_mask_roi` is provided, interior replacement is gated to the active glare
       edit support (clean areas far from glare and clean opposite lenses stay untouched).
    3. Morphological top-hat / black-hat + dark + chromatic rim protection applies on BOTH
       sides of the lens boundary (outermost 2px inside `base_bin` AND `bevel_ring` outside),
       protecting dark/black/bronze, bright neutral silver/gold metallic wires (even if clean
       darkened them heavily), and tortoiseshell/colored frames while allowing broad bright
       glare on a rimless bevel to be suppressed cleanly.
    """
    H, W = lens_mask_roi.shape[:2]
    if glare_mask_roi is not None and int((glare_mask_roi > 0).sum()) == 0:
        return np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.uint8)

    base_bin = (lens_mask_roi > 127).astype(np.uint8)
    if int(base_bin.sum()) == 0:
        return np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.uint8)

    if lens_expand_px > 0:
        ke = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * lens_expand_px + 1, 2 * lens_expand_px + 1)
        )
        base_bin = cv2.dilate(base_bin, ke, iterations=1)

    # Gate base lens interior to active edit support when glare mask is provided
    if glare_mask_roi is not None:
        edit_pad = max(8, int(round(feather_px * 3.0)) + 5)
        k_edit = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * edit_pad + 1, 2 * edit_pad + 1)
        )
        glare_support = cv2.dilate((glare_mask_roi > 0).astype(np.uint8), k_edit, iterations=1)
        active_base = ((base_bin > 0) & (glare_support > 0)).astype(np.uint8)
        if int(active_base.sum()) == 0:
            return np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.uint8)
        alpha_interior = inward_feather_alpha(active_base * 255, feather_px=max(feather_px, 2.0))
    else:
        active_base = base_bin
        alpha_interior = inward_feather_alpha(active_base * 255, feather_px=feather_px)

    # Compute structural & photometric rim protection (used on BOTH sides of lens boundary!)
    o = orig_roi_rgb.astype(np.float32)
    c = clean_roi_rgb.astype(np.float32)
    o_luma = 0.299 * o[:, :, 0] + 0.587 * o[:, :, 1] + 0.114 * o[:, :, 2]
    c_luma = 0.299 * c[:, :, 0] + 0.587 * c[:, :, 1] + 0.114 * c[:, :, 2]

    # Estimate local face reference outside bevel band (color-agnostic)
    ref_in_r = max(3, bevel_band_px + 3)
    ref_out_r = max(14, bevel_band_px + 18)
    k_ref_in = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * ref_in_r + 1, 2 * ref_in_r + 1)
    )
    k_ref_out = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * ref_out_r + 1, 2 * ref_out_r + 1)
    )
    face_ring = (cv2.dilate(base_bin, k_ref_out) > 0) & (cv2.dilate(base_bin, k_ref_in) == 0)
    local_luma = (
        float(np.median(o_luma[face_ring])) if int(face_ring.sum()) >= 50 else float(np.median(o_luma))
    )

    # Inner rim guard band: outermost 2px INSIDE base_bin also respects structural rim protection
    k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    strict_core_all = cv2.erode(base_bin, k_inner, iterations=1) > 0
    strict_core = strict_core_all & (active_base > 0)
    inner_rim_zone = (active_base > 0) & (~strict_core)

    # Cue 1: Color-agnostic dark rim shield (black, charcoal, brown, bronze, navy, dark tortoise)
    dark_thresh = max(45.0, min(115.0, local_luma - 14.0))
    dark_rim_prob = smoothstep((dark_thresh - o_luma) / 16.0)

    # Cue 2: Scale-selective morphological top-hat/black-hat thin-wire shield + Voronoi contour-step
    # rim shield. Protects dark, bright neutral silver/gold/chrome, and straddling metallic rims
    # (even when clean_roi drastically darkens them!) while staying 0 on continuous rimless bevel glare.
    k_rim = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tophat = o_luma - cv2.morphologyEx(o_luma, cv2.MORPH_OPEN, k_rim)
    blackhat = cv2.morphologyEx(o_luma, cv2.MORPH_CLOSE, k_rim) - o_luma
    thin_wire_prob = np.maximum(
        smoothstep((tophat - 16.0) / 18.0),
        smoothstep((blackhat - 16.0) / 18.0),
    )

    gx = cv2.Sobel(o_luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(o_luma, cv2.CV_32F, 0, 1, ksize=3)
    grad_o = cv2.magnitude(gx, gy) / 4.0
    grad_local = cv2.dilate(grad_o, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Compare boundary pixels against their nearest interior pixel inside strict_core (or base_bin)
    ref_core = strict_core_all if int(strict_core_all.sum()) > 0 else (base_bin > 0)
    ys_in, xs_in = np.where(ref_core)
    if ys_in.size > 0:
        _, nearest_lbl = cv2.distanceTransformWithLabels(
            (~ref_core).astype(np.uint8),
            cv2.DIST_L2,
            5,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        idx_map = np.clip(nearest_lbl - 1, 0, ys_in.size - 1)
        nearest_rgb = o[ys_in[idx_map], xs_in[idx_map]]
        nearest_luma = o_luma[ys_in[idx_map], xs_in[idx_map]]
        inner_diff = np.maximum(
            np.abs(o_luma - nearest_luma),
            np.max(np.abs(o - nearest_rgb), axis=2),
        )
        distinct_rim_prob = smoothstep((inner_diff - 7.0) / 8.0) * smoothstep(
            (grad_local - 14.0) / 16.0
        )
    else:
        distinct_rim_prob = np.zeros_like(o_luma, dtype=np.float32)

    structural_rim_prob = np.maximum(thin_wire_prob, distinct_rim_prob)

    # Cue 3: Chromatic/tortoiseshell acetate shield (high non-AR saturation relative to local face)
    o_purple = np.minimum(o[:, :, 0] - o[:, :, 1], o[:, :, 2] - o[:, :, 1])
    o_green = np.minimum(o[:, :, 1] - o[:, :, 0], o[:, :, 1] - o[:, :, 2])
    o_chroma = np.max(o, axis=2) - np.min(o, axis=2)
    face_chroma = (
        float(np.percentile(o_chroma[face_ring], 85)) if int(face_ring.sum()) >= 50 else 45.0
    )
    not_ar = (o_purple < 12.0) & (o_green < 12.0)
    chroma_rim_prob = np.where(
        not_ar,
        smoothstep((o_chroma - max(45.0, face_chroma + 22.0)) / 20.0)
        * (1.0 - smoothstep((o_luma - 210.0) / 25.0)),
        0.0,
    ).astype(np.float32)

    rim_prob = np.clip(
        np.maximum(dark_rim_prob, np.maximum(structural_rim_prob, chroma_rim_prob)),
        0.0,
        1.0,
    )
    w_rim_protect = np.clip(1.0 - float(rim_protect_strength) * rim_prob, 0.0, 1.0).astype(
        np.float32
    )

    if bevel_band_px <= 0:
        alpha_guarded = np.where(inner_rim_zone, alpha_interior * w_rim_protect, alpha_interior)
        support = ((active_base > 0) & (alpha_guarded > 0.01)).astype(np.uint8)
        return np.where(support > 0, alpha_guarded, 0.0).astype(np.float32), (support * 255).astype(
            np.uint8
        )

    kb = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * bevel_band_px + 1, 2 * bevel_band_px + 1)
    )
    outer_bin = cv2.dilate(active_base, kb, iterations=1)
    bevel_ring = (outer_bin > 0) & (base_bin == 0)

    dist_out = cv2.distanceTransform(1 - active_base, cv2.DIST_L2, 5)
    w_dist = 1.0 - smoothstep((dist_out - bevel_band_px * 0.55) / max(1.0, bevel_band_px * 0.55))
    w_dist = np.where(bevel_ring, w_dist, 0.0).astype(np.float32)

    if glare_mask_roi is not None:
        gate_r = max(12, bevel_band_px + 10)
        kg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gate_r + 1, 2 * gate_r + 1))
        glare_near = cv2.dilate((glare_mask_roi > 64).astype(np.uint8), kg, iterations=1)
        w_gate = cv2.GaussianBlur(glare_near.astype(np.float32), (9, 9), 2.5)
    else:
        w_gate = np.ones_like(alpha_interior)

    dL = o_luma - c_luma
    d_max_ch = np.maximum(
        o[:, :, 0] - c[:, :, 0],
        np.maximum(o[:, :, 1] - c[:, :, 1], o[:, :, 2] - c[:, :, 2]),
    )
    c_purple = np.minimum(c[:, :, 0] - c[:, :, 1], c[:, :, 2] - c[:, :, 1])
    c_green = np.minimum(c[:, :, 1] - c[:, :, 0], c[:, :, 1] - c[:, :, 2])
    ar_excess = np.maximum(o_purple - c_purple, o_green - c_green)
    glare_excess = np.maximum(dL, np.maximum(0.80 * d_max_ch, 0.90 * ar_excess))
    w_glare = smoothstep((glare_excess - 10.0) / 16.0)

    bevel_alpha = w_dist * w_gate * w_glare * w_rim_protect
    inner_glare_boost = np.where(active_base > 0, w_gate * w_glare * w_rim_protect, 0.0).astype(
        np.float32
    )

    # Apply rim protection on BOTH sides of the lens boundary (inner_rim_zone AND bevel_ring)
    alpha_guarded_interior = np.where(inner_rim_zone, alpha_interior * w_rim_protect, alpha_interior)
    alpha_combined = np.maximum(np.maximum(alpha_guarded_interior, inner_glare_boost), bevel_alpha)

    boundary_band = inner_rim_zone | bevel_ring
    alpha_combined = np.where(boundary_band, alpha_combined * w_rim_protect, alpha_combined)

    active_support = (((active_base > 0) & (alpha_combined > 0.01)) | (bevel_alpha > 0.02)).astype(
        np.uint8
    )
    alpha_smooth = cv2.GaussianBlur(alpha_combined.astype(np.float32), (3, 3), 0.7)
    alpha_smooth = np.where(boundary_band, alpha_smooth * w_rim_protect, alpha_smooth)

    alpha_roi = np.where(active_support > 0, np.clip(alpha_smooth, 0.0, 1.0), 0.0)
    # Restore 1.0 strictly in deep core interior (never overriding rim pixels in boundary_band)
    alpha_roi = np.maximum(
        alpha_roi, np.where(strict_core & (alpha_interior >= 0.99), 1.0, 0.0)
    )

    return alpha_roi.astype(np.float32), (active_support * 255).astype(np.uint8)


def composite_single_frame(
    orig_full_rgb: np.ndarray,
    clean_roi_rgb: np.ndarray,
    lens_mask_roi: np.ndarray,
    glare_mask_roi: np.ndarray | None,
    box: tuple[int, int, int, int],
    *,
    feather_px: float = 2.0,
    color_match_strength: float = 0.0,
    lens_expand_px: int = 0,
    bevel_band_px: int = 4,
    suppress_chroma: bool = True,
    rim_protect_strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Composite a single cleaned ROI frame back onto `orig_full_rgb`.

    Guarantees pre-encode RGB bit-exactness (`max_outside_diff == 0`) everywhere outside
    the active glare-gated lens/bevel support mask (and across the entire frame when glare is empty).
    """
    H, W = orig_full_rgb.shape[:2]
    if glare_mask_roi is not None and int((glare_mask_roi > 0).sum()) == 0:
        return orig_full_rgb.copy(), np.zeros((H, W), dtype=np.uint8)

    x1, y1, x2, y2 = box
    cw, ch = x2 - x1, y2 - y1
    roi_h, roi_w = clean_roi_rgb.shape[:2]

    orig_crop = orig_full_rgb[y1:y2, x1:x2]
    orig_roi = cv2.resize(orig_crop, (roi_w, roi_h), interpolation=cv2.INTER_LANCZOS4)

    if lens_mask_roi.shape[:2] != (roi_h, roi_w):
        lens_mask_roi = cv2.resize(lens_mask_roi, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    if glare_mask_roi is not None and glare_mask_roi.shape[:2] != (roi_h, roi_w):
        glare_mask_roi = cv2.resize(glare_mask_roi, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

    clean_proc = clean_roi_rgb.astype(np.float32)
    if suppress_chroma:
        clean_proc = suppress_chroma_glint(clean_proc, glare_mask_roi)

    clean_proc = subtle_boundary_color_match(
        orig_roi, clean_proc, lens_mask_roi, glare_mask_roi, strength=color_match_strength
    )

    alpha_roi, support_roi = compute_rim_safe_alpha(
        orig_roi,
        clean_proc,
        lens_mask_roi,
        glare_mask_roi,
        lens_expand_px=lens_expand_px,
        feather_px=feather_px,
        bevel_band_px=bevel_band_px,
        rim_protect_strength=rim_protect_strength,
    )
    if int((support_roi > 127).sum()) == 0:
        return orig_full_rgb.copy(), np.zeros((H, W), dtype=np.uint8)

    clean_crop = cv2.resize(
        np.clip(clean_proc, 0, 255).astype(np.uint8), (cw, ch), interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    alpha_crop = cv2.resize(alpha_roi, (cw, ch), interpolation=cv2.INTER_AREA)
    hard_support = (
        cv2.resize((support_roi > 127).astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST)
        > 0
    )

    alpha_3 = np.where(hard_support, np.clip(alpha_crop, 0.0, 1.0), 0.0)[:, :, None]
    blended_f = orig_crop.astype(np.float32) * (1.0 - alpha_3) + clean_crop * alpha_3
    blended_u8 = np.clip(np.rint(blended_f), 0, 255).astype(np.uint8)
    blended_crop = np.where(hard_support[:, :, None], blended_u8, orig_crop)

    out_full = orig_full_rgb.copy()
    out_full[y1:y2, x1:x2] = blended_crop

    full_support = np.zeros((H, W), dtype=np.uint8)
    full_support[y1:y2, x1:x2] = hard_support.astype(np.uint8) * 255
    return out_full, full_support


def extract_glasses_zoom_crop(
    full_rgb: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    """Resolution-independent 9:5 glasses zoom crop returned as an independent copy."""
    H, W = full_rgb.shape[:2]
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    crop_w = max(32, min(W, int(round(0.92 * (x2 - x1)))))
    crop_h = max(20, min(H, int(round(crop_w * 0.56))))
    gx1 = max(0, min(W - crop_w, cx - crop_w // 2))
    gy1 = max(0, min(H - crop_h, cy - crop_h // 2))
    return full_rgb[gy1 : gy1 + crop_h, gx1 : gx1 + crop_w].copy()


def _draw_label_bar(
    img_rgb: np.ndarray, text: str, x: int, y: int, fg: tuple[int, int, int] = (255, 255, 255)
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(1.0, img_rgb.shape[1] / 1400.0))
    thick = 1 if scale < 0.75 else 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(img_rgb, (x, y), (x + tw + 12, y + th + 10), (0, 0, 0), -1)
    cv2.putText(img_rgb, text, (x + 6, y + th + 5), font, scale, fg, thick, cv2.LINE_AA)


def compute_compare_dimensions(full_w: int, full_h: int) -> tuple[int, int, int]:
    """Compute even `(compare_w, compare_h, zoom_strip_h)` for any portrait/landscape/square video."""
    zoom_h = int(round(min(full_h * 0.45, full_w * 0.55)))
    zoom_h = max(96, zoom_h - (zoom_h % 2))
    comp_w = full_w * 2
    comp_h = full_h + zoom_h
    if comp_w % 2 != 0:
        comp_w += 1
    if comp_h % 2 != 0:
        comp_h += 1
    return comp_w, comp_h, zoom_h


def compute_effective_batch_size(
    full_w: int,
    full_h: int,
    *,
    compare: bool,
    requested_batch_size: int,
    max_batch_mem_mb: int = 256,
) -> int:
    """
    Bound per-batch pixel memory by `max_batch_mem_mb` so 4K/8K comparison renders never
    exhaust host RAM while smaller resolutions keep high parallelism.
    """
    main_bytes = full_w * full_h * 3
    if compare:
        cmp_w, cmp_h, _ = compute_compare_dimensions(full_w, full_h)
        per_frame_bytes = main_bytes * 2 + (cmp_w * cmp_h * 3)
    else:
        per_frame_bytes = main_bytes * 2
    budget_bytes = max(32, int(max_batch_mem_mb)) * 1024 * 1024
    max_by_mem = max(1, budget_bytes // max(1, per_frame_bytes))
    return max(1, min(int(requested_batch_size), int(max_by_mem)))


def build_compare_frame(
    orig_full_rgb: np.ndarray,
    comp_full_rgb: np.ndarray,
    box: tuple[int, int, int, int],
    frame_idx: int,
    fps: float,
) -> np.ndarray:
    """Build resolution-independent side-by-side Original vs Cleaned frame with zoomed eye strip."""
    H, W = orig_full_rgb.shape[:2]
    comp_w, comp_h, zoom_h = compute_compare_dimensions(W, H)

    oz = extract_glasses_zoom_crop(orig_full_rgb, box)
    cz = extract_glasses_zoom_crop(comp_full_rgb, box)
    oz2 = cv2.resize(oz, (W, zoom_h), interpolation=cv2.INTER_LANCZOS4)
    cz2 = cv2.resize(cz, (W, zoom_h), interpolation=cv2.INTER_LANCZOS4)

    canvas = np.zeros((comp_h, comp_w, 3), dtype=np.uint8)
    canvas[0:H, 0:W] = orig_full_rgb
    canvas[0:H, W : 2 * W] = comp_full_rgb
    canvas[H : H + zoom_h, 0:W] = oz2
    canvas[H : H + zoom_h, W : 2 * W] = cz2

    canvas[:, max(0, W - 1) : min(comp_w, W + 1)] = (40, 40, 40)
    canvas[max(0, H - 2) : min(comp_h, H + 2), :] = (40, 40, 40)

    t_sec = frame_idx / float(fps)
    _draw_label_bar(canvas, f"ORIGINAL  t={t_sec:05.2f}s (f{frame_idx:04d})", 10, 10, fg=(255, 255, 255))
    _draw_label_bar(canvas, f"CLEANED   t={t_sec:05.2f}s (f{frame_idx:04d})", W + 10, 10, fg=(120, 255, 140))
    _draw_label_bar(canvas, "GLASSES ZOOM (ORIGINAL)", 10, H + 8, fg=(255, 255, 255))
    _draw_label_bar(canvas, "GLASSES ZOOM (CLEANED)", W + 10, H + 8, fg=(120, 255, 140))
    return canvas


def build_compare_qa_grid(
    orig_zooms: list[tuple[int, float, np.ndarray]],
    comp_zooms: list[tuple[int, float, np.ndarray]],
    out_path: Path,
) -> None:
    """Save a before/after zoomed glasses contact sheet atomically via unique temp sibling."""
    n = len(orig_zooms)
    if n == 0:
        return
    tile_w, tile_h = _QA_TILE_W, _QA_TILE_H
    pair_gap = 6
    col_gap = 18
    row_gap = 8
    pair_w = tile_w * 2 + pair_gap
    macro_cols = min(2, n)
    rows = (n + macro_cols - 1) // macro_cols

    sheet_w = macro_cols * pair_w + (macro_cols - 1) * col_gap + 16
    sheet_h = rows * tile_h + (rows - 1) * row_gap + 16
    sheet = Image.new("RGB", (sheet_w, sheet_h), (18, 18, 18))
    dr = ImageDraw.Draw(sheet)

    for i in range(n):
        fidx, t_sec, oz = orig_zooms[i]
        _, _, cz = comp_zooms[i]
        r, mc = divmod(i, macro_cols)
        x0 = 8 + mc * (pair_w + col_gap)
        y0 = 8 + r * (tile_h + row_gap)

        oz_im = (
            Image.fromarray(oz)
            if oz.shape[:2] == (tile_h, tile_w)
            else Image.fromarray(cv2.resize(oz, (tile_w, tile_h), interpolation=cv2.INTER_LANCZOS4))
        )
        cz_im = (
            Image.fromarray(cz)
            if cz.shape[:2] == (tile_h, tile_w)
            else Image.fromarray(cv2.resize(cz, (tile_w, tile_h), interpolation=cv2.INTER_LANCZOS4))
        )
        sheet.paste(oz_im, (x0, y0))
        sheet.paste(cz_im, (x0 + tile_w + pair_gap, y0))

        l1 = f"ORIG t={t_sec:.2f}s (f{fidx:04d})"
        l2 = f"CLEAN t={t_sec:.2f}s"
        dr.rectangle([x0 + 4, y0 + 4, x0 + 168, y0 + 22], fill=(0, 0, 0))
        dr.text((x0 + 8, y0 + 6), l1, fill=(255, 255, 255))
        dr.rectangle(
            [x0 + tile_w + pair_gap + 4, y0 + 4, x0 + tile_w + pair_gap + 125, y0 + 22],
            fill=(0, 0, 0),
        )
        dr.text((x0 + tile_w + pair_gap + 8, y0 + 6), l2, fill=(120, 255, 140))

    tmp_out = make_unique_temp_sibling(out_path)
    try:
        sheet.save(tmp_out, format="PNG")
        os.replace(tmp_out, out_path)
    except Exception:
        tmp_out.unlink(missing_ok=True)
        raise


def run_composite_stage(
    *,
    orig_frames_dir: Path,
    cleaned_roi_dir: Path,
    lens_masks_dir: Path,
    roi_meta: dict[str, Any],
    output_mp4: Path,
    config: CompositeConfig,
    glare_masks_dir: Path | None = None,
    audio_path: Path | None = None,
    compare_mp4: Path | None = None,
    qa_grid_path: Path | None = None,
    source_video_path: Path | None = None,
    frame_has_edit: list[bool] | None = None,
) -> dict[str, Any]:
    """
    Execute Stage 5 bounded-RAM streaming composite directly into ffmpeg stdin.

    - Checks source/output/compare/qa path identity before opening any writers.
    - Encodes into unique temporary sibling files and atomically publishes only on success.
    - Preserves source SAR & color metadata and supports optional `--lossless` RGB encoding.
    - Uses authoritative CFR video clock (never `-shortest`).
    - Adapts micro-batch size by byte budget so 4K/8K compare renders stay bounded in RAM.
    """
    t0 = time.time()
    fps = float(roi_meta["fps"])
    full_w = int(roi_meta["full_width"])
    full_h = int(roi_meta["full_height"])
    frame_records = roi_meta["frames"]
    n_frames = len(frame_records)
    if n_frames == 0:
        raise ValueError("roi_meta contains 0 frames.")

    orig_dir = Path(orig_frames_dir)
    clean_dir = Path(cleaned_roi_dir)
    reject_symlink_path(orig_dir, "orig_frames_dir")
    reject_symlink_path(clean_dir, "cleaned_roi_dir")

    masks_dir = Path(lens_masks_dir)
    reject_symlink_path(masks_dir, "lens_masks_dir")
    lens_expand_px = config.lens_expand_px
    if not masks_dir.exists():
        fallback = masks_dir.parent / "roi_lens_interior"
        if fallback.exists():
            logger.info("%s missing; falling back to %s with lens_expand_px=4", masks_dir, fallback)
            masks_dir = fallback
            if lens_expand_px == 0:
                lens_expand_px = 4
        else:
            raise FileNotFoundError(f"Lens masks directory not found: {masks_dir}")

    glare_dir = (
        Path(glare_masks_dir)
        if glare_masks_dir is not None and Path(glare_masks_dir).is_dir()
        else None
    )
    if glare_dir is not None:
        reject_symlink_path(glare_dir, "glare_masks_dir")

    output_mp4 = Path(output_mp4)
    resolved_compare: Path | None = None
    if config.compare:
        resolved_compare = (
            Path(compare_mp4)
            if compare_mp4 is not None
            else output_mp4.with_name(f"{output_mp4.stem}_compare.mp4")
        )

    resolved_qa_grid: Path | None = None
    if config.qa_grid:
        resolved_qa_grid = (
            Path(qa_grid_path)
            if qa_grid_path is not None
            else output_mp4.with_name(f"{output_mp4.stem}_qa_grid.png")
        )

    # Verify no destination aliases source media (incl. symlinks/hardlinks) or collides with another output
    src_check = source_video_path
    if src_check is None and roi_meta.get("source_video"):
        src_check = Path(str(roi_meta["source_video"]))
    assert_distinct_output_paths(
        src_check,
        [
            ("output", output_mp4),
            ("compare", resolved_compare),
            ("qa_grid", resolved_qa_grid),
        ],
    )

    resolved_audio: Path | None = None
    if audio_path is not None and Path(audio_path).exists() and Path(audio_path).stat().st_size > 0:
        resolved_audio = Path(audio_path)

    save_dir = Path(config.save_frames_dir) if config.save_frames_dir is not None else None
    if save_dir is not None:
        reject_symlink_path(save_dir, "save_frames_dir")
        save_dir.mkdir(parents=True, exist_ok=True)

    # Validate all manifest filenames up front before launching ffmpeg
    for rec in frame_records:
        validate_safe_basename(str(rec["filename"]))

    grid_indices: set[int] = set()
    if resolved_qa_grid is not None:
        grid_indices = set(
            np.linspace(0, n_frames - 1, min(config.qa_grid_samples, n_frames), dtype=int).tolist()
        )

    vmeta = roi_meta.get("video_metadata") or {}
    src_sar = vmeta.get("sample_aspect_ratio")
    src_primaries = vmeta.get("color_primaries")
    src_trc = vmeta.get("color_trc")
    src_colorspace = vmeta.get("colorspace")

    tmp_main = make_unique_temp_sibling(output_mp4)
    tmp_cmp = make_unique_temp_sibling(resolved_compare) if resolved_compare is not None else None

    eff_bs = compute_effective_batch_size(
        full_w,
        full_h,
        compare=resolved_compare is not None,
        requested_batch_size=config.batch_size,
        max_batch_mem_mb=config.max_batch_mem_mb,
    )
    eff_workers = min(config.workers, eff_bs)

    proc_main = open_video_writer_pipe(
        full_w,
        full_h,
        fps,
        tmp_main,
        audio_path=resolved_audio,
        crf=config.crf,
        preset=config.preset,
        lossless=config.lossless,
        num_frames=n_frames,
        sample_aspect_ratio=src_sar,
        color_primaries=src_primaries,
        color_trc=src_trc,
        colorspace=src_colorspace,
    )
    proc_cmp = None
    if tmp_cmp is not None:
        cmp_w, cmp_h, _ = compute_compare_dimensions(full_w, full_h)
        proc_cmp = open_video_writer_pipe(
            cmp_w,
            cmp_h,
            fps,
            tmp_cmp,
            audio_path=resolved_audio,
            crf=config.crf,
            preset=config.preset,
            lossless=False,
            num_frames=n_frames,
            color_primaries=src_primaries,
            color_trc=src_trc,
            colorspace=src_colorspace,
        )

    orig_zooms: list[tuple[int, float, np.ndarray]] = []
    comp_zooms: list[tuple[int, float, np.ndarray]] = []

    def _work_one(rec: dict[str, Any]) -> tuple[int, np.ndarray, np.ndarray | None, Any]:
        i = int(rec["frame_idx"])
        fn = str(rec["filename"])
        box = (int(rec["x1"]), int(rec["y1"]), int(rec["x2"]), int(rec["y2"]))

        orig_fp = ensure_safe_child_file(orig_dir, fn)
        orig_bgr = cv2.imread(str(orig_fp))
        if orig_bgr is None:
            raise FileNotFoundError(f"Missing original frame: {orig_fp}")
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

        # Fast-path bypass: if frame has zero edit support, skip ROI resampling completely
        skip_edit = frame_has_edit is not None and i < len(frame_has_edit) and not frame_has_edit[i]
        glare_mask = None
        if not skip_edit and glare_dir is not None:
            gp = ensure_safe_child_file(glare_dir, fn, allow_missing=True)
            if gp.exists():
                glare_mask = cv2.imread(str(gp), cv2.IMREAD_GRAYSCALE)
                if glare_mask is not None and int((glare_mask > 0).sum()) == 0:
                    skip_edit = True

        if skip_edit:
            comp_rgb = orig_rgb
        else:
            clean_fp = ensure_safe_child_file(clean_dir, fn)
            mask_fp = ensure_safe_child_file(masks_dir, fn)
            clean_bgr = cv2.imread(str(clean_fp))
            lens_mask = cv2.imread(str(mask_fp), cv2.IMREAD_GRAYSCALE)
            if clean_bgr is None:
                raise FileNotFoundError(f"Missing cleaned ROI frame: {clean_fp}")
            if lens_mask is None:
                raise FileNotFoundError(f"Missing lens mask: {mask_fp}")
            clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
            comp_rgb, _ = composite_single_frame(
                orig_rgb,
                clean_rgb,
                lens_mask,
                glare_mask,
                box,
                feather_px=config.feather_px,
                color_match_strength=config.color_match_strength,
                lens_expand_px=lens_expand_px,
                bevel_band_px=config.bevel_band_px,
                suppress_chroma=config.suppress_chroma_glint,
                rim_protect_strength=config.rim_protect_strength,
            )

        if save_dir is not None:
            cv2.imwrite(str(save_dir / fn), cv2.cvtColor(comp_rgb, cv2.COLOR_RGB2BGR))

        cmp_rgb = (
            build_compare_frame(orig_rgb, comp_rgb, box, i, fps) if proc_cmp is not None else None
        )

        zoom_pair = None
        if i in grid_indices:
            t_sec = i / float(fps)
            # Downscale QA tiles immediately so full-resolution frame arrays are never pinned in RAM
            oz_tile = cv2.resize(
                extract_glasses_zoom_crop(orig_rgb, box),
                (_QA_TILE_W, _QA_TILE_H),
                interpolation=cv2.INTER_LANCZOS4,
            )
            cz_tile = cv2.resize(
                extract_glasses_zoom_crop(comp_rgb, box),
                (_QA_TILE_W, _QA_TILE_H),
                interpolation=cv2.INTER_LANCZOS4,
            )
            zoom_pair = ((i, t_sec, oz_tile), (i, t_sec, cz_tile))

        return i, comp_rgb, cmp_rgb, zoom_pair

    try:
        with ThreadPoolExecutor(max_workers=eff_workers) as ex:
            for b_start in range(0, n_frames, eff_bs):
                batch = frame_records[b_start : min(n_frames, b_start + eff_bs)]
                results = list(ex.map(_work_one, batch))
                for _, comp_rgb, cmp_rgb, zoom_pair in results:
                    assert proc_main.stdin is not None
                    proc_main.stdin.write(np.ascontiguousarray(comp_rgb).tobytes())
                    if proc_cmp is not None and cmp_rgb is not None:
                        assert proc_cmp.stdin is not None
                        proc_cmp.stdin.write(np.ascontiguousarray(cmp_rgb).tobytes())
                    if zoom_pair is not None:
                        orig_zooms.append(zoom_pair[0])
                        comp_zooms.append(zoom_pair[1])
                results.clear()

        close_video_writer_pipe(proc_main, "main")
        if proc_cmp is not None:
            close_video_writer_pipe(proc_cmp, "compare")

        # Publish completed outputs atomically only after ffmpeg succeeded cleanly
        os.replace(tmp_main, output_mp4)
        if tmp_cmp is not None and resolved_compare is not None:
            os.replace(tmp_cmp, resolved_compare)

        if resolved_qa_grid is not None:
            build_compare_qa_grid(orig_zooms, comp_zooms, resolved_qa_grid)

        return {
            "output_mp4": str(output_mp4),
            "compare_mp4": str(resolved_compare) if resolved_compare is not None else None,
            "qa_grid": str(resolved_qa_grid) if resolved_qa_grid is not None else None,
            "has_audio": resolved_audio is not None,
            "lossless": config.lossless,
            "effective_batch_size": eff_bs,
            "num_frames": n_frames,
            "elapsed_sec": round(time.time() - t0, 2),
        }
    except Exception:
        for p in (proc_main, proc_cmp):
            if p is not None and p.poll() is None:
                p.kill()
                p.wait()
        tmp_main.unlink(missing_ok=True)
        if tmp_cmp is not None:
            tmp_cmp.unlink(missing_ok=True)
        raise
