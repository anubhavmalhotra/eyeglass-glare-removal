"""General face/eye/glasses auto-detection and glare-robust adaptive ROI tracking."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d

from eyeglass_glare_removal._workspace import (
    assert_source_outside_managed_dir,
    atomic_swap_directory,
    atomic_write_text,
    ensure_confined_dir,
    ensure_safe_child_file,
    hash_file_sha256,
    hash_files_manifest,
    make_unique_temp_dir,
    make_unique_temp_sibling,
    reject_symlink_path,
    validate_safe_basename,
)
from eyeglass_glare_removal.config import PrepConfig
from eyeglass_glare_removal.media import compute_audio_offset_sec, extract_cfr_frames

logger = logging.getLogger(__name__)

# MediaPipe FaceMesh eye landmark indices (468+ point topology)
_MP_LEFT_EYE_IDXS = (33, 133, 159, 145)
_MP_RIGHT_EYE_IDXS = (362, 263, 386, 374)


@dataclass
class EyeGeometry:
    """Detected eye/glasses geometry in full-frame pixel coordinates."""

    left_eye: tuple[float, float]
    right_eye: tuple[float, float]
    bridge: tuple[float, float]
    inter_ocular_px: float
    face_box: tuple[float, float, float, float]
    source: str
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EyeGeometry:
        return cls(
            left_eye=(float(data["left_eye"][0]), float(data["left_eye"][1])),
            right_eye=(float(data["right_eye"][0]), float(data["right_eye"][1])),
            bridge=(float(data["bridge"][0]), float(data["bridge"][1])),
            inter_ocular_px=float(data["inter_ocular_px"]),
            face_box=(
                float(data["face_box"][0]),
                float(data["face_box"][1]),
                float(data["face_box"][2]),
                float(data["face_box"][3]),
            ),
            source=str(data.get("source", "unknown")),
            confidence=float(data.get("confidence", 1.0)),
        )

    def roi_center(self) -> tuple[float, float]:
        """Preferred square ROI center (slightly below brow, centered on glasses bridge)."""
        return (self.bridge[0], self.bridge[1] + 0.05 * self.inter_ocular_px)

    def project_to_roi(
        self,
        box: tuple[int, int, int, int],
        roi_size: int,
        shift: tuple[float, float] = (0.0, 0.0),
    ) -> dict[str, Any]:
        """Map full-frame eye coordinates into roi_size x roi_size ROI pixel coordinates."""
        x1, y1, x2, y2 = box
        cw = max(1, x2 - x1)
        ch = max(1, y2 - y1)
        sx = roi_size / float(cw)
        sy = roi_size / float(ch)
        dx, dy = shift

        lx = (self.left_eye[0] + dx - x1) * sx
        ly = (self.left_eye[1] + dy - y1) * sy
        rx = (self.right_eye[0] + dx - x1) * sx
        ry = (self.right_eye[1] + dy - y1) * sy
        bx = 0.5 * (lx + rx)
        by = 0.5 * (ly + ry)
        iod = math.hypot(rx - lx, ry - ly)

        return {
            "left_eye": [round(lx, 2), round(ly, 2)],
            "right_eye": [round(rx, 2), round(ry, 2)],
            "bridge": [round(bx, 2), round(by, 2)],
            "inter_ocular_px": round(iod, 2),
            "roi_size": int(roi_size),
            "source": self.source,
        }


def geometry_from_roi_box(
    roi_box: tuple[int, int, int, int], frame_w: int, frame_h: int
) -> EyeGeometry:
    """Create EyeGeometry from an explicit user-provided (x1, y1, x2, y2) ROI box."""
    x1, y1, x2, y2 = roi_box
    x1 = max(0, min(frame_w - 2, x1))
    y1 = max(0, min(frame_h - 2, y1))
    x2 = max(x1 + 2, min(frame_w, x2))
    y2 = max(y1 + 2, min(frame_h, y2))

    bw = float(x2 - x1)
    bh = float(y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    iod = max(8.0, 0.34 * min(bw, bh))
    bridge_y = cy - 0.05 * iod
    left_eye = (cx - 0.5 * iod, bridge_y)
    right_eye = (cx + 0.5 * iod, bridge_y)
    return EyeGeometry(
        left_eye=left_eye,
        right_eye=right_eye,
        bridge=(cx, bridge_y),
        inter_ocular_px=iod,
        face_box=(float(x1), float(y1), float(x2), float(y2)),
        source="manual",
        confidence=1.0,
    )


def _make_geometry_from_eyes(
    eye_a: tuple[float, float],
    eye_b: tuple[float, float],
    face_box: tuple[float, float, float, float],
    source: str,
    confidence: float,
) -> EyeGeometry:
    if eye_a[0] <= eye_b[0]:
        left_eye, right_eye = eye_a, eye_b
    else:
        left_eye, right_eye = eye_b, eye_a
    bridge = (0.5 * (left_eye[0] + right_eye[0]), 0.5 * (left_eye[1] + right_eye[1]))
    iod = max(8.0, math.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]))
    return EyeGeometry(
        left_eye=left_eye,
        right_eye=right_eye,
        bridge=bridge,
        inter_ocular_px=iod,
        face_box=face_box,
        source=source,
        confidence=confidence,
    )


def detect_eyes_mediapipe(frame_bgr: np.ndarray) -> EyeGeometry | None:
    """Detect primary face and eyes using MediaPipe (FaceMesh or FaceLandmarker tasks API)."""
    try:
        import mediapipe as mp
    except ImportError:
        return None

    H, W = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    solutions = getattr(mp, "solutions", None)
    face_mesh_mod = getattr(solutions, "face_mesh", None) if solutions is not None else None
    if face_mesh_mod is not None:
        with face_mesh_mod.FaceMesh(
            static_image_mode=True,
            max_num_faces=3,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as mesh:
            result = mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        best_geom: EyeGeometry | None = None
        best_area = -1.0
        for lm_list in result.multi_face_landmarks:
            pts = lm_list.landmark
            xs = [p.x * W for p in pts]
            ys = [p.y * H for p in pts]
            fx1, fx2 = max(0.0, min(xs)), min(float(W), max(xs))
            fy1, fy2 = max(0.0, min(ys)), min(float(H), max(ys))
            area = (fx2 - fx1) * (fy2 - fy1)
            if area <= best_area:
                continue
            best_area = area
            le = (
                float(np.mean([pts[i].x * W for i in _MP_LEFT_EYE_IDXS])),
                float(np.mean([pts[i].y * H for i in _MP_LEFT_EYE_IDXS])),
            )
            re = (
                float(np.mean([pts[i].x * W for i in _MP_RIGHT_EYE_IDXS])),
                float(np.mean([pts[i].y * H for i in _MP_RIGHT_EYE_IDXS])),
            )
            best_geom = _make_geometry_from_eyes(
                le, re, (fx1, fy1, fx2, fy2), source="mediapipe", confidence=0.95
            )
        return best_geom

    model_path = os.environ.get("GLARE_REMOVE_MEDIAPIPE_MODEL")
    if model_path and Path(model_path).exists():
        try:
            vision = mp.tasks.vision
            base_opts = mp.tasks.BaseOptions(model_asset_path=model_path)
            opts = vision.FaceLandmarkerOptions(
                base_options=base_opts,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=3,
            )
            with vision.FaceLandmarker.create_from_options(opts) as landmarker:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = landmarker.detect(mp_img)
            if not res.face_landmarks:
                return None
            best_geom = None
            best_area = -1.0
            for pts in res.face_landmarks:
                xs = [p.x * W for p in pts]
                ys = [p.y * H for p in pts]
                fx1, fx2 = max(0.0, min(xs)), min(float(W), max(xs))
                fy1, fy2 = max(0.0, min(ys)), min(float(H), max(ys))
                area = (fx2 - fx1) * (fy2 - fy1)
                if area <= best_area:
                    continue
                best_area = area
                le = (
                    float(np.mean([pts[i].x * W for i in _MP_LEFT_EYE_IDXS])),
                    float(np.mean([pts[i].y * H for i in _MP_LEFT_EYE_IDXS])),
                )
                re = (
                    float(np.mean([pts[i].x * W for i in _MP_RIGHT_EYE_IDXS])),
                    float(np.mean([pts[i].y * H for i in _MP_RIGHT_EYE_IDXS])),
                )
                best_geom = _make_geometry_from_eyes(
                    le, re, (fx1, fy1, fx2, fy2), source="mediapipe", confidence=0.95
                )
            return best_geom
        except Exception as exc:
            logger.debug("MediaPipe FaceLandmarker failed, falling back: %s", exc)

    return None


def _load_haar_cascade(filename: str) -> cv2.CascadeClassifier | None:
    path = Path(cv2.data.haarcascades) / filename
    if not path.exists():
        return None
    clf = cv2.CascadeClassifier(str(path))
    if clf.empty():
        return None
    return clf


def detect_eyes_haar(frame_bgr: np.ndarray) -> EyeGeometry | None:
    """Detect primary face and eyes using built-in OpenCV Haar cascades (zero downloads)."""
    H, W = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    p92 = float(np.percentile(gray, 92))
    gray_clipped = np.minimum(gray, max(p92, 120.0)).astype(np.uint8)

    face_cascades = [
        c
        for name in ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml")
        if (c := _load_haar_cascade(name)) is not None
    ]
    if not face_cascades:
        return None

    min_face = max(32, int(round(0.08 * min(W, H))))
    faces: list[tuple[int, int, int, int]] = []
    for clf in face_cascades:
        for img_g in (gray_eq, gray_clipped, gray):
            det = clf.detectMultiScale(
                img_g,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(min_face, min_face),
            )
            if len(det) > 0:
                faces = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in det]
                break
        if faces:
            break

    if not faces:
        return None

    fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
    face_box = (float(fx), float(fy), float(fx + fw), float(fy + fh))

    upper_y1 = max(0, fy + int(round(0.12 * fh)))
    upper_y2 = min(H, fy + int(round(0.62 * fh)))
    upper_gray = gray_eq[upper_y1:upper_y2, fx : fx + fw]

    eye_cascades = [
        c
        for name in ("haarcascade_eye_tree_eyeglasses.xml", "haarcascade_eye.xml")
        if (c := _load_haar_cascade(name)) is not None
    ]
    eye_candidates: list[tuple[float, float, float]] = []
    if upper_gray.size > 0:
        min_eye = max(10, int(round(0.10 * fw)))
        max_eye = max(min_eye + 4, int(round(0.48 * fw)))
        for eclf in eye_cascades:
            det_e = eclf.detectMultiScale(
                upper_gray,
                scaleFactor=1.08,
                minNeighbors=3,
                minSize=(min_eye, min_eye),
                maxSize=(max_eye, max_eye),
            )
            if len(det_e) > 0:
                for ex, ey, ew, eh in det_e:
                    ecx = float(fx + ex + 0.5 * ew)
                    ecy = float(upper_y1 + ey + 0.5 * eh)
                    eye_candidates.append((ecx, ecy, float(ew * eh)))
                break

    fx_mid = fx + 0.5 * fw
    if len(eye_candidates) >= 2:
        best_pair = None
        best_score = float("inf")
        for i in range(len(eye_candidates)):
            for j in range(i + 1, len(eye_candidates)):
                ax, ay, _ = eye_candidates[i]
                bx, by, _ = eye_candidates[j]
                dx = abs(ax - bx)
                dy = abs(ay - by)
                if dx < 0.18 * fw or dx > 0.72 * fw or dy > 0.18 * fh:
                    continue
                pair_mid = 0.5 * (ax + bx)
                score = abs(pair_mid - fx_mid) + 1.5 * dy + abs(dx - 0.38 * fw)
                if score < best_score:
                    best_score = score
                    best_pair = ((ax, ay), (bx, by))
        if best_pair is not None:
            return _make_geometry_from_eyes(
                best_pair[0], best_pair[1], face_box, source="haar", confidence=0.85
            )

    if len(eye_candidates) >= 1:
        valid_single = [
            c for c in eye_candidates if 0.08 * fw <= abs(c[0] - fx_mid) <= 0.36 * fw
        ]
        if valid_single:
            ex, ey, _ = max(valid_single, key=lambda c: c[2])
            mirror_x = 2.0 * fx_mid - ex
            return _make_geometry_from_eyes(
                (ex, ey), (mirror_x, ey), face_box, source="haar", confidence=0.70
            )

    left_eye = (fx + 0.32 * fw, fy + 0.38 * fh)
    right_eye = (fx + 0.68 * fw, fy + 0.38 * fh)
    return _make_geometry_from_eyes(
        left_eye, right_eye, face_box, source="haar", confidence=0.55
    )


def detect_eye_geometry(
    frame_bgr: np.ndarray,
    *,
    detector: str = "auto",
    roi_box: tuple[int, int, int, int] | None = None,
) -> EyeGeometry | None:
    """Detect face/eye/glasses geometry using the configured backend hierarchy."""
    H, W = frame_bgr.shape[:2]
    if detector == "manual" or (detector == "auto" and roi_box is not None):
        if roi_box is None:
            raise ValueError("detector='manual' requires roi_box.")
        return geometry_from_roi_box(roi_box, W, H)

    if detector == "mediapipe":
        geom = detect_eyes_mediapipe(frame_bgr)
        if geom is None:
            logger.warning("MediaPipe did not detect a face on frame.")
        return geom

    if detector == "haar":
        return detect_eyes_haar(frame_bgr)

    geom = detect_eyes_mediapipe(frame_bgr)
    if geom is not None:
        return geom
    return detect_eyes_haar(frame_bgr)


def compute_adaptive_crop_size(
    inter_ocular_px: float,
    frame_w: int,
    frame_h: int,
    *,
    scale: float = 3.0,
    explicit_crop_size: int | None = None,
) -> int:
    """Compute an even square crop_size dynamically from inter-ocular distance."""
    max_dim = min(frame_w, frame_h)
    if max_dim < 2:
        raise ValueError(f"Frame dimensions too small: {frame_w}x{frame_h}")
    max_even = max_dim - (max_dim % 2)
    min_even = min(64, max_even)
    if min_even < 2:
        min_even = max_even

    if explicit_crop_size is not None:
        size = int(explicit_crop_size)
        size = size - (size % 2)
        return max(min(32, max_even), min(max_even, size))

    raw = int(round(scale * max(8.0, inter_ocular_px)))
    raw = raw + (raw % 2)
    return max(min_even, min(max_even, raw))


def make_roi_box(
    cx: float, cy: float, crop_size: int, frame_w: int, frame_h: int
) -> tuple[int, int, int, int]:
    """Build a strictly in-bounds square (x1, y1, x2, y2) crop box of size crop_size."""
    max_dim = min(frame_w, frame_h)
    size = max(2, min(crop_size, max_dim))
    half = size // 2
    x1 = int(round(cx)) - half
    y1 = int(round(cy)) - half
    x1 = max(0, min(frame_w - size, x1))
    y1 = max(0, min(frame_h - size, y1))
    return (x1, y1, x1 + size, y1 + size)


def extract_tracking_feature(frame_bgr: np.ndarray, glare_clip_pct: float = 82.0) -> np.ndarray:
    """Build a glare-suppressed, lighting-invariant intensity + gradient tracking feature map."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    med = float(np.median(gray))
    hi = max(float(np.percentile(gray, glare_clip_pct)), med + 15.0, 1.0)
    clamped = np.minimum(gray, hi)

    gx = cv2.Sobel(clamped, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(clamped, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad_scale = max(float(np.percentile(grad, 90)), 10.0)

    feat = 0.65 * (clamped / hi) + 0.35 * np.tanh(grad / grad_scale)
    return feat.astype(np.float32)


def _derive_template_box(
    geom: EyeGeometry, frame_w: int, frame_h: int
) -> tuple[int, int, int, int]:
    """Compute a dynamic template box covering eyebrows, nose bridge, and upper frames."""
    iod = max(12.0, geom.inter_ocular_px)
    bx, by = geom.bridge
    half_w = max(10.0, 0.85 * iod)
    top_h = max(8.0, 0.48 * iod)
    bot_h = max(8.0, 0.42 * iod)

    tx1 = int(round(bx - half_w))
    ty1 = int(round(by - top_h))
    tx2 = int(round(bx + half_w))
    ty2 = int(round(by + bot_h))

    tx1 = max(0, min(frame_w - 16, tx1))
    ty1 = max(0, min(frame_h - 16, ty1))
    tx2 = max(tx1 + 16, min(frame_w, tx2))
    ty2 = max(ty1 + 16, min(frame_h, ty2))
    return (tx1, ty1, tx2, ty2)


def _match_step(
    feat: np.ndarray,
    tmpl: np.ndarray,
    prev_tx1: int,
    prev_ty1: int,
    search_pad: int,
) -> tuple[float, float, int, int, float]:
    """
    Run local NCC template match with clamped parabolic sub-pixel refinement.

    Rejects flat/low-variance templates or search patches and flat ambiguous correlation
    surfaces (returning score=0.0 instead of OpenCV's degenerate 1.0 at the top-left corner).
    """
    H, W = feat.shape[:2]
    th, tw = tmpl.shape[:2]

    sx1 = max(0, prev_tx1 - search_pad)
    sy1 = max(0, prev_ty1 - search_pad)
    sx2 = min(W, prev_tx1 + tw + search_pad)
    sy2 = min(H, prev_ty1 + th + search_pad)

    if sx2 - sx1 < tw:
        sx1 = max(0, min(W - tw, prev_tx1))
        sx2 = sx1 + tw
    if sy2 - sy1 < th:
        sy1 = max(0, min(H - th, prev_ty1))
        sy2 = sy1 + th

    patch = feat[sy1:sy2, sx1:sx2]

    # Reject flat/blank templates or search regions where TM_CCOEFF_NORMED degenerates to 1.0
    if float(np.std(tmpl)) < 1e-3 or float(np.std(patch)) < 1e-3:
        return (prev_tx1 + tw / 2.0, prev_ty1 + th / 2.0, prev_tx1, prev_ty1, 0.0)

    res = cv2.matchTemplate(patch, tmpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if not math.isfinite(max_val):
        return (prev_tx1 + tw / 2.0, prev_ty1 + th / 2.0, prev_tx1, prev_ty1, 0.0)

    # Reject flat/ambiguous correlation surfaces with no distinct peak
    if res.size > 9 and (float(max_val) - float(np.median(res))) < 0.03:
        return (prev_tx1 + tw / 2.0, prev_ty1 + th / 2.0, prev_tx1, prev_ty1, 0.0)

    mx, my = max_loc
    best_x = sx1 + mx
    best_y = sy1 + my

    # Also verify the matched sub-window itself has non-trivial variance
    matched_win = feat[best_y : best_y + th, best_x : best_x + tw]
    if float(np.std(matched_win)) < 1e-3:
        return (prev_tx1 + tw / 2.0, prev_ty1 + th / 2.0, prev_tx1, prev_ty1, 0.0)

    dx = 0.0
    dy = 0.0
    if 0 < mx < res.shape[1] - 1:
        l_val, c_val, r_val = float(res[my, mx - 1]), float(res[my, mx]), float(res[my, mx + 1])
        denom = l_val - 2.0 * c_val + r_val
        if abs(denom) > 1e-6:
            dx = float(np.clip(0.5 * (l_val - r_val) / denom, -0.5, 0.5))
    if 0 < my < res.shape[0] - 1:
        u_val, c_val, d_val = float(res[my - 1, mx]), float(res[my, mx]), float(res[my + 1, mx])
        denom = u_val - 2.0 * c_val + d_val
        if abs(denom) > 1e-6:
            dy = float(np.clip(0.5 * (u_val - d_val) / denom, -0.5, 0.5))

    return (best_x + dx + tw / 2.0, best_y + dy + th / 2.0, best_x, best_y, float(max_val))


def track_roi_centers(
    frame_loader: Callable[[int], np.ndarray],
    num_frames: int,
    init_geom: EyeGeometry,
    *,
    init_idx: int = 0,
    smooth_sigma: float = 6.0,
    ncc_redetect_threshold: float = 0.55,
    redetect_fn: Callable[[np.ndarray], EyeGeometry | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Track glasses ROI center forward and backward from init_idx using glare-masked
    gradient NCC, with flat-template rejection, automatic re-detection fallback,
    and Gaussian temporal smoothing.
    """
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    if not (0 <= init_idx < num_frames):
        raise ValueError(f"init_idx {init_idx} out of range [0, {num_frames})")

    init_img = frame_loader(init_idx)
    H, W = init_img.shape[:2]
    tx1, ty1, tx2, ty2 = _derive_template_box(init_geom, W, H)
    tw, th = tx2 - tx1, ty2 - ty1
    init_cx, init_cy = init_geom.roi_center()
    offset_x = init_cx - 0.5 * (tx1 + tx2)
    offset_y = init_cy - 0.5 * (ty1 + ty2)

    search_pad = max(28, int(round(0.65 * init_geom.inter_ocular_px)))
    init_feat = extract_tracking_feature(init_img)
    base_tmpl = init_feat[ty1:ty2, tx1:tx2].copy()

    raw_centers = np.zeros((num_frames, 2), dtype=np.float64)
    ncc_scores = np.zeros(num_frames, dtype=np.float64)
    redetected = np.zeros(num_frames, dtype=bool)

    raw_centers[init_idx] = [init_cx, init_cy]
    ncc_scores[init_idx] = 1.0 if float(np.std(base_tmpl)) >= 1e-3 else 0.0

    def _run_direction(indices: list[int]) -> None:
        prev_tx1, prev_ty1 = tx1, ty1
        prev_cx, prev_cy = init_cx, init_cy
        cur_tmpl = base_tmpl
        cur_off_x, cur_off_y = offset_x, offset_y

        for idx in indices:
            img = frame_loader(idx)
            feat = extract_tracking_feature(img)
            tcx, tcy, best_x, best_y, score = _match_step(
                feat, cur_tmpl, prev_tx1, prev_ty1, search_pad
            )
            cx = tcx + cur_off_x
            cy = tcy + cur_off_y

            if score < ncc_redetect_threshold:
                recovered = redetect_fn(img) if redetect_fn is not None else None
                if recovered is not None:
                    redetected[idx] = True
                    cx, cy = recovered.roi_center()
                    rtx1, rty1, rtx2, rty2 = _derive_template_box(recovered, W, H)
                    cand_tmpl = feat[rty1:rty2, rtx1:rtx2]
                    if (
                        (rtx2 - rtx1) >= 16
                        and (rty2 - rty1) >= 16
                        and float(np.std(cand_tmpl)) >= 1e-3
                    ):
                        cur_tmpl = cand_tmpl.copy()
                        cur_off_x = cx - 0.5 * (rtx1 + rtx2)
                        cur_off_y = cy - 0.5 * (rty1 + rty2)
                        prev_tx1, prev_ty1 = rtx1, rty1
                    else:
                        prev_tx1 = int(np.clip(round(cx - cur_off_x - tw / 2.0), 0, max(0, W - tw)))
                        prev_ty1 = int(np.clip(round(cy - cur_off_y - th / 2.0), 0, max(0, H - th)))
                else:
                    # Flat/blank/occluded frame: hold last reliable center strictly without drifting
                    cx, cy = prev_cx, prev_cy
            else:
                prev_tx1, prev_ty1 = best_x, best_y

            raw_centers[idx] = [cx, cy]
            ncc_scores[idx] = score
            prev_cx, prev_cy = cx, cy

    if init_idx + 1 < num_frames:
        _run_direction(list(range(init_idx + 1, num_frames)))
    if init_idx > 0:
        _run_direction(list(range(init_idx - 1, -1, -1)))

    if smooth_sigma > 0.0 and num_frames > 3:
        sm_x = gaussian_filter1d(raw_centers[:, 0], sigma=smooth_sigma, mode="nearest")
        sm_y = gaussian_filter1d(raw_centers[:, 1], sigma=smooth_sigma, mode="nearest")
        smoothed = np.stack([sm_x, sm_y], axis=1)
    else:
        smoothed = raw_centers.copy()

    return raw_centers, smoothed, ncc_scores, redetected


def find_initial_eye_geometry(
    frame_paths: list[Path],
    config: PrepConfig,
) -> tuple[int, EyeGeometry]:
    """Locate the earliest confident face/eye detection across sampled frames."""
    n = len(frame_paths)
    if config.roi_box is not None and config.detector in ("auto", "manual"):
        img0 = cv2.imread(str(frame_paths[0]))
        if img0 is None:
            raise FileNotFoundError(f"Could not read extracted frame: {frame_paths[0]}")
        return 0, geometry_from_roi_box(config.roi_box, img0.shape[1], img0.shape[0])

    probe_indices = [0]
    if n > 1:
        extra = np.linspace(0, n - 1, min(12, n), dtype=int).tolist()
        for idx in extra:
            if idx not in probe_indices:
                probe_indices.append(idx)

    for idx in probe_indices:
        img = cv2.imread(str(frame_paths[idx]))
        if img is None:
            continue
        geom = detect_eye_geometry(img, detector=config.detector, roi_box=config.roi_box)
        if geom is not None:
            return idx, geom

    raise RuntimeError(
        f"Could not auto-detect a face/eyes using detector={config.detector!r} across "
        f"{len(probe_indices)} sampled frames. Try passing --roi-box X1,Y1,X2,Y2 explicitly "
        "or installing optional MediaPipe support (`pip install 'eyeglass-glare-removal[mediapipe]'`)."
    )


def compute_prep_generation(source_sha256: str, config: PrepConfig) -> str:
    """Content-bound generation hash identifying source bytes + effective prep settings."""
    payload = json.dumps(
        {
            "source_sha256": source_sha256,
            "start_sec": config.start_sec,
            "duration_sec": config.duration_sec,
            "fps": config.fps,
            "roi_size": config.roi_size,
            "crop_size": config.crop_size,
            "roi_box": list(config.roi_box) if config.roi_box is not None else None,
            "detector": config.detector,
            "smooth_sigma": config.smooth_sigma,
            "ncc_redetect_threshold": config.ncc_redetect_threshold,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def build_roi_track_qa_grid(
    roi_dir: Path,
    frame_records: list[dict[str, Any]],
    fps: float,
    out_path: Path,
    n_samples: int = 24,
) -> None:
    """Save a contact sheet verifying ROI centering and tracking stability across the clip."""
    n = len(frame_records)
    if n == 0:
        return
    idxs = np.linspace(0, n - 1, min(n_samples, n), dtype=int).tolist()
    cols = min(6, len(idxs))
    rows = (len(idxs) + cols - 1) // cols
    tw = th = 256
    sheet = Image.new("RGB", (cols * tw, rows * th), (15, 15, 15))
    dr = ImageDraw.Draw(sheet)

    for k, i in enumerate(idxs):
        rec = frame_records[i]
        fp = ensure_safe_child_file(roi_dir, str(rec["filename"]))
        im = Image.open(fp).convert("RGB").resize((tw, th), Image.Resampling.LANCZOS)
        r, c = divmod(k, cols)
        sheet.paste(im, (c * tw, r * th))
        cx, cy = c * tw + tw // 2, r * th + th // 2
        dr.line([(cx - 12, cy), (cx + 12, cy)], fill=(0, 255, 255), width=1)
        dr.line([(cx, cy - 12), (cx, cy + 12)], fill=(0, 255, 255), width=1)
        t_sec = i / float(fps)
        lbl = f"t={t_sec:04.1f}s (f{i:04d})"
        dr.rectangle([c * tw + 2, r * th + 2, c * tw + 135, r * th + 18], fill=(0, 0, 0))
        dr.text((c * tw + 5, r * th + 4), lbl, fill=(255, 255, 0))

    tmp_out = make_unique_temp_sibling(out_path)
    try:
        sheet.save(tmp_out, format="JPEG", quality=90)
        os.replace(tmp_out, out_path)
    except Exception:
        tmp_out.unlink(missing_ok=True)
        raise


def run_prep_stage(
    source_video: Path | str,
    prep_dir: Path,
    config: PrepConfig,
) -> dict[str, Any]:
    """
    Execute Stage 1 end-to-end with source confinement guard, canonical source path
    recording, content hashing, and single-generation atomic staging.
    """
    source_path = Path(source_video)
    if not source_path.exists():
        raise FileNotFoundError(f"Input video not found: {source_path}")
    canonical_source = source_path.resolve(strict=True)

    prep_dir = Path(prep_dir)
    reject_symlink_path(prep_dir, "prep_dir")
    # Guard against source aliasing or residing inside prep_dir BEFORE any write or unlink
    assert_source_outside_managed_dir(canonical_source, prep_dir)

    source_sha256 = hash_file_sha256(canonical_source)
    prep_gen = compute_prep_generation(source_sha256, config)

    # Stage all prep outputs into a unique sibling dir and atomically commit prep_dir as one generation!
    prep_dir.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_path(prep_dir.parent, "workspace root")
    stage_prep = make_unique_temp_dir(prep_dir.parent, prefix=f".stage_{prep_dir.name}_")

    try:
        frame_paths, audio_path, meta, fps = extract_cfr_frames(
            canonical_source,
            stage_prep,
            start_sec=config.start_sec,
            duration_sec=config.duration_sec,
            target_fps=config.fps,
        )
        img0 = cv2.imread(str(frame_paths[0]))
        if img0 is None:
            raise RuntimeError(f"Failed to read extracted frame: {frame_paths[0]}")
        H, W = img0.shape[:2]
        n_frames = len(frame_paths)

        init_idx, init_geom = find_initial_eye_geometry(frame_paths, config)

        if config.roi_box is not None and config.crop_size is None:
            bx1, by1, bx2, by2 = config.roi_box
            user_side = max(bx2 - bx1, by2 - by1)
            crop_size = compute_adaptive_crop_size(
                init_geom.inter_ocular_px, W, H, explicit_crop_size=user_side
            )
        else:
            crop_size = compute_adaptive_crop_size(
                init_geom.inter_ocular_px, W, H, explicit_crop_size=config.crop_size
            )

        def _load_frame(i: int) -> np.ndarray:
            im = cv2.imread(str(frame_paths[i]))
            if im is None:
                raise RuntimeError(f"Failed to read frame {frame_paths[i]}")
            return im

        redetect_fn = (
            None
            if config.detector == "manual" or config.roi_box is not None
            else (lambda bgr: detect_eye_geometry(bgr, detector=config.detector))
        )

        raw_c, sm_c, ncc, redetected = track_roi_centers(
            _load_frame,
            n_frames,
            init_geom,
            init_idx=init_idx,
            smooth_sigma=config.smooth_sigma,
            ncc_redetect_threshold=config.ncc_redetect_threshold,
            redetect_fn=redetect_fn,
        )

        roi_dir = ensure_confined_dir(stage_prep, stage_prep / "roi_frames", create=True)
        frame_records: list[dict[str, Any]] = []
        work_items: list[tuple[Path, Path, tuple[int, int, int, int]]] = []
        filenames: list[str] = []
        for i, fp in enumerate(frame_paths):
            fn = validate_safe_basename(fp.name)
            filenames.append(fn)
            box = make_roi_box(float(sm_c[i, 0]), float(sm_c[i, 1]), crop_size, W, H)
            work_items.append((fp, roi_dir / fn, box))
            frame_records.append(
                {
                    "frame_idx": i,
                    "filename": fn,
                    "x1": box[0],
                    "y1": box[1],
                    "x2": box[2],
                    "y2": box[3],
                    "raw_cx": round(float(raw_c[i, 0]), 2),
                    "raw_cy": round(float(raw_c[i, 1]), 2),
                    "smooth_cx": round(float(sm_c[i, 0]), 2),
                    "smooth_cy": round(float(sm_c[i, 1]), 2),
                    "track_ncc": round(float(ncc[i]), 4),
                    "redetected": bool(redetected[i]),
                }
            )

        def _save_roi(item: tuple[Path, Path, tuple[int, int, int, int]]) -> None:
            src_p, dst_p, (x1, y1, x2, y2) = item
            bgr = cv2.imread(str(src_p))
            if bgr is None:
                raise RuntimeError(f"Failed to read frame during ROI export: {src_p}")
            crop = bgr[y1:y2, x1:x2]
            roi = cv2.resize(
                crop, (config.roi_size, config.roi_size), interpolation=cv2.INTER_LANCZOS4
            )
            cv2.imwrite(str(dst_p), roi)

        with ThreadPoolExecutor(max_workers=config.workers) as ex:
            list(ex.map(_save_roi, work_items))

        seed_rec = frame_records[init_idx]
        box_seed = (seed_rec["x1"], seed_rec["y1"], seed_rec["x2"], seed_rec["y2"])
        eye_geom_roi_seed = init_geom.project_to_roi(box_seed, config.roi_size, shift=(0.0, 0.0))

        frames_dir = stage_prep / "frames"
        frames_digest = hash_files_manifest(frames_dir, filenames)
        roi_frames_digest = hash_files_manifest(roi_dir, filenames)
        audio_sha256 = hash_file_sha256(audio_path) if audio_path is not None else None
        audio_offset = compute_audio_offset_sec(meta, start_sec=config.start_sec)

        final_audio_path = (prep_dir / audio_path.name) if audio_path is not None else None
        payload = {
            "source_video": str(canonical_source),
            "source_sha256": source_sha256,
            "prep_generation": prep_gen,
            "frames_digest": frames_digest,
            "roi_frames_digest": roi_frames_digest,
            "audio_sha256": audio_sha256,
            "audio_filename": audio_path.name if audio_path is not None else None,
            "audio_offset_sec": audio_offset if audio_path is not None else 0.0,
            "start_sec": config.start_sec,
            "requested_duration_sec": config.duration_sec,
            "requested_fps": config.fps,
            "requested_crop_size": config.crop_size,
            "requested_roi_box": list(config.roi_box) if config.roi_box is not None else None,
            "requested_detector": config.detector,
            "duration_sec": round(n_frames / float(fps), 4),
            "fps": fps,
            "num_frames": n_frames,
            "full_width": W,
            "full_height": H,
            "crop_size": crop_size,
            "roi_size": config.roi_size,
            "audio_path": str(final_audio_path) if final_audio_path is not None else None,
            "detector": init_geom.source,
            "init_frame_idx": init_idx,
            "init_eye_geometry_full": init_geom.to_dict(),
            "init_eye_geometry_roi": eye_geom_roi_seed,
            "video_metadata": meta.to_dict(),
            "redetected_count": int(redetected.sum()),
            "frames": frame_records,
        }
        atomic_write_text(stage_prep / "roi_boxes.json", json.dumps(payload, indent=2))

        if config.write_qa_grid:
            build_roi_track_qa_grid(
                roi_dir,
                frame_records,
                fps=fps,
                out_path=stage_prep / "roi_track_qa_grid.jpg",
                n_samples=config.qa_grid_samples,
            )

        atomic_swap_directory(stage_prep, prep_dir)
    except BaseException:
        if stage_prep.exists():
            shutil.rmtree(stage_prep, ignore_errors=True)
        raise

    return payload
