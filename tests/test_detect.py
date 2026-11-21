"""Unit tests for face/eye geometry, adaptive ROI sizing, flat-frame rejection, and tracking."""

from __future__ import annotations

import cv2
import numpy as np

from eyeglass_glare_removal.detect import (
    EyeGeometry,
    compute_adaptive_crop_size,
    extract_tracking_feature,
    geometry_from_roi_box,
    make_roi_box,
    track_roi_centers,
)


def _render_synthetic_face_frame(
    W: int,
    H: int,
    cx: float,
    cy: float,
    iod: float = 100.0,
    glare_offset: tuple[float, float] | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(123)
    img = np.full((H, W, 3), (45, 55, 65), dtype=np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    img[:, :, 0] = np.clip(45 + 15 * np.sin(xx / 40.0), 0, 255).astype(np.uint8)

    icx, icy = int(round(cx)), int(round(cy))
    cv2.ellipse(img, (icx, icy + 15), (int(1.25 * iod), int(1.65 * iod)), 0, 0, 360, (115, 145, 185), -1)
    noise = rng.integers(-6, 7, size=(H, W, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    lx = int(round(cx - 0.5 * iod))
    rx = int(round(cx + 0.5 * iod))
    ey = int(round(cy))

    cv2.line(img, (lx - 32, ey - 32), (lx + 28, ey - 28), (35, 35, 45), 7)
    cv2.line(img, (rx - 28, ey - 28), (rx + 32, ey - 32), (35, 35, 45), 7)
    cv2.rectangle(img, (lx - 36, ey - 22), (lx + 34, ey + 24), (25, 25, 35), 4)
    cv2.rectangle(img, (rx - 34, ey - 22), (rx + 36, ey + 24), (25, 25, 35), 4)
    cv2.line(img, (lx + 34, ey - 4), (rx - 34, ey - 4), (25, 25, 35), 4)

    for ex in (lx, rx):
        cv2.ellipse(img, (ex, ey), (16, 9), 0, 0, 360, (200, 205, 210), -1)
        cv2.circle(img, (ex, ey), 7, (30, 35, 45), -1)

    cv2.line(img, (icx, ey + 8), (icx - 4, ey + 52), (90, 115, 150), 3)
    cv2.ellipse(img, (icx, ey + 88), (28, 8), 0, 0, 360, (80, 90, 145), -1)

    if glare_offset is not None:
        gx, gy = glare_offset
        cv2.rectangle(
            img,
            (int(lx - 18 + gx), int(ey - 12 + gy)),
            (int(lx + 16 + gx), int(ey + 10 + gy)),
            (255, 252, 248),
            -1,
        )
        cv2.rectangle(
            img,
            (int(rx - 18 + gx), int(ey - 12 + gy)),
            (int(rx + 16 + gx), int(ey + 10 + gy)),
            (255, 252, 248),
            -1,
        )
    return img


def test_adaptive_crop_size_clamping_and_evenness() -> None:
    cs = compute_adaptive_crop_size(110.0, 720, 1280)
    assert cs % 2 == 0
    assert 300 <= cs <= 360

    assert compute_adaptive_crop_size(400.0, 1920, 1080) == 1080
    assert compute_adaptive_crop_size(150.0, 301, 401) == 300
    assert compute_adaptive_crop_size(30.0, 48, 48) == 48


def test_make_roi_box_bounds_clamping() -> None:
    assert make_roi_box(10.0, 15.0, 200, 640, 480) == (0, 0, 200, 200)
    assert make_roi_box(635.0, 475.0, 200, 640, 480) == (440, 280, 640, 480)
    assert make_roi_box(320.0, 240.0, 200, 640, 480) == (220, 140, 420, 340)


def test_geometry_from_roi_box_and_roi_projection() -> None:
    geom = geometry_from_roi_box((100, 120, 420, 440), frame_w=640, frame_h=720)
    assert geom.source == "manual"
    assert geom.left_eye[0] < geom.bridge[0] < geom.right_eye[0]
    roi_proj = geom.project_to_roi((100, 120, 420, 440), roi_size=512)
    assert roi_proj["left_eye"][0] < roi_proj["bridge"][0] < roi_proj["right_eye"][0]


def test_track_roi_centers_rejects_flat_frames_with_zero_drift() -> None:
    """Regression test: flat/blank frames must return score=0.0 and hold center without drifting."""
    W, H = 200, 160
    flat_frames = [np.full((H, W, 3), 120, dtype=np.uint8) for _ in range(8)]
    init_geom = EyeGeometry(
        left_eye=(60.0, 70.0),
        right_eye=(100.0, 70.0),
        bridge=(80.0, 70.0),
        inter_ocular_px=40.0,
        face_box=(30.0, 20.0, 130.0, 120.0),
        source="test",
    )
    redetect_calls = 0

    def counting_redetect(_bgr: np.ndarray) -> EyeGeometry | None:
        nonlocal redetect_calls
        redetect_calls += 1
        return None

    raw_c, sm_c, ncc, _ = track_roi_centers(
        lambda i: flat_frames[i],
        len(flat_frames),
        init_geom,
        init_idx=0,
        smooth_sigma=0.0,
        redetect_fn=counting_redetect,
    )
    assert float(np.max(ncc)) == 0.0
    assert redetect_calls == len(flat_frames) - 1
    expected_cx, expected_cy = init_geom.roi_center()
    assert np.allclose(raw_c[:, 0], expected_cx, atol=1e-6)
    assert np.allclose(raw_c[:, 1], expected_cy, atol=1e-6)
    assert np.allclose(sm_c, raw_c, atol=1e-6)


def test_track_roi_centers_with_shifting_glare_and_smoothing() -> None:
    W, H = 640, 480
    n = 25
    true_xs = [300.0 + 1.5 * i + (2.0 if i % 2 == 0 else -2.0) for i in range(n)]
    true_ys = [220.0 + 0.5 * i for i in range(n)]
    frames = [
        _render_synthetic_face_frame(
            W, H, true_xs[i], true_ys[i], iod=100.0, glare_offset=(float((i % 5) - 2) * 3.0, 0.0)
        )
        for i in range(n)
    ]
    init_geom = EyeGeometry(
        left_eye=(true_xs[0] - 50.0, true_ys[0]),
        right_eye=(true_xs[0] + 50.0, true_ys[0]),
        bridge=(true_xs[0], true_ys[0]),
        inter_ocular_px=100.0,
        face_box=(180.0, 80.0, 420.0, 360.0),
        source="test",
    )

    raw_c, sm_c, ncc, redet = track_roi_centers(
        lambda i: frames[i], n, init_geom, init_idx=0, smooth_sigma=3.0
    )
    assert float(np.median(ncc)) > 0.75
    assert not redet.any()
    assert float(np.max(np.abs(raw_c[:, 0] - np.array(true_xs)))) < 5.0
    assert float(np.max(np.abs(raw_c[:, 1] - (np.array(true_ys) + 5.0)))) < 5.0
    assert float(np.mean(np.abs(np.diff(sm_c[:, 0], n=2)))) < float(
        np.mean(np.abs(np.diff(raw_c[:, 0], n=2)))
    )


def test_track_roi_centers_redetect_fallback_and_occlusion_hold() -> None:
    W, H = 640, 480
    frames = [_render_synthetic_face_frame(W, H, 280.0, 220.0, iod=100.0) for _ in range(10)]
    frames[4] = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(5, 10):
        frames[i] = _render_synthetic_face_frame(W, H, 420.0, 270.0, iod=100.0)

    init_geom = EyeGeometry(
        left_eye=(230.0, 220.0),
        right_eye=(330.0, 220.0),
        bridge=(280.0, 220.0),
        inter_ocular_px=100.0,
        face_box=(160.0, 80.0, 400.0, 360.0),
        source="test",
    )

    def mock_redetect(bgr: np.ndarray) -> EyeGeometry | None:
        if float(bgr.mean()) < 5.0:
            return None
        return EyeGeometry(
            left_eye=(370.0, 270.0),
            right_eye=(470.0, 270.0),
            bridge=(420.0, 270.0),
            inter_ocular_px=100.0,
            face_box=(300.0, 130.0, 540.0, 410.0),
            source="mock_redetect",
        )

    raw_c, _, _, redet = track_roi_centers(
        lambda i: frames[i],
        len(frames),
        init_geom,
        init_idx=0,
        smooth_sigma=0.0,
        ncc_redetect_threshold=0.65,
        redetect_fn=mock_redetect,
    )
    assert abs(raw_c[4, 0] - raw_c[3, 0]) < 1e-6
    assert not redet[4]
    assert redet[5]
    assert abs(raw_c[5, 0] - 420.0) < 2.0
    assert abs(raw_c[9, 0] - 420.0) < 3.0


def test_track_roi_centers_backward_from_nonzero_init_idx() -> None:
    W, H = 512, 512
    frames = [_render_synthetic_face_frame(W, H, 240.0 + 2.0 * i, 240.0, iod=90.0) for i in range(12)]
    init_idx = 6
    init_geom = EyeGeometry(
        left_eye=(240.0 + 2.0 * init_idx - 45.0, 240.0),
        right_eye=(240.0 + 2.0 * init_idx + 45.0, 240.0),
        bridge=(240.0 + 2.0 * init_idx, 240.0),
        inter_ocular_px=90.0,
        face_box=(120.0, 100.0, 380.0, 380.0),
        source="test",
    )
    raw_c, _, ncc, _ = track_roi_centers(
        lambda i: frames[i], len(frames), init_geom, init_idx=init_idx, smooth_sigma=0.0
    )
    assert abs(raw_c[0, 0] - 240.0) < 4.0
    assert abs(raw_c[11, 0] - 262.0) < 4.0
    assert float(np.min(ncc)) > 0.8


def test_extract_tracking_feature_output_dtype_and_range() -> None:
    img = _render_synthetic_face_frame(256, 256, 128.0, 128.0, iod=60.0)
    feat = extract_tracking_feature(img)
    assert feat.dtype == np.float32
    assert feat.shape == (256, 256)
    assert np.isfinite(feat).all()
