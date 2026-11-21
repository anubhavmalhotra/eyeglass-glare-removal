"""Unit tests for glare detection, SAM2 chunk handoff/swap rejection, OOM retry, and smoothing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from eyeglass_glare_removal.segment import (
    _build_init_session_kwargs,
    build_initial_lens_prompts,
    clean_single_lens_mask,
    compute_glare_hotspot_mask,
    compute_sam2_chunk_ranges,
    derive_prompts_from_masks,
    mask_iou,
    run_chunked_tracking_loop,
    temporal_smooth_binary_masks,
    track_bidirectional_from_seed,
)


def _make_synthetic_lens_scene(
    skin_rgb: tuple[int, int, int],
    *,
    sclera_rgb: tuple[int, int, int] = (155, 150, 146),
    glare_color: tuple[int, int, int] | None = None,
    ring_glare: bool = False,
    fill_both_lenses: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    H = W = 256
    rgb = np.full((H, W, 3), skin_rgb, dtype=np.uint8)
    lens_mask = np.zeros((H, W), dtype=np.uint8)
    gt_glare = np.zeros((H, W), dtype=np.uint8)

    cv2.ellipse(lens_mask, (80, 128), (38, 26), 0, 0, 360, 255, -1)
    cv2.ellipse(lens_mask, (176, 128), (38, 26), 0, 0, 360, 255, -1)

    for cx in (80, 176):
        cv2.ellipse(rgb, (cx, 128), (15, 7), 0, 0, 360, sclera_rgb, -1)
        cv2.circle(rgb, (cx, 128), 6, (38, 32, 28), -1)

    if glare_color is not None:
        if fill_both_lenses:
            rgb[lens_mask > 127] = glare_color
            gt_glare = lens_mask.copy()
        elif ring_glare:
            cv2.circle(rgb, (80, 128), 16, glare_color, 4)
            cv2.circle(gt_glare, (80, 128), 16, 255, -1)
        else:
            cv2.rectangle(rgb, (62, 114), (96, 138), glare_color, -1)
            cv2.rectangle(gt_glare, (62, 114), (96, 138), 255, -1)
        gt_glare = np.where(lens_mask > 127, gt_glare, 0).astype(np.uint8)

    return rgb, lens_mask, gt_glare


@pytest.mark.parametrize(
    "label,skin_rgb,sclera_rgb",
    [
        ("light_skin", (222, 188, 168), (210, 206, 204)),
        ("medium_skin", (165, 124, 96), (175, 170, 166)),
        ("dark_skin", (94, 65, 48), (145, 140, 136)),
    ],
)
def test_glare_detector_clean_eyes_low_false_positives_across_skin_tones(
    label: str, skin_rgb: tuple[int, int, int], sclera_rgb: tuple[int, int, int]
) -> None:
    rgb, lens_mask, _ = _make_synthetic_lens_scene(skin_rgb, sclera_rgb=sclera_rgb, glare_color=None)
    glare = compute_glare_hotspot_mask(rgb, lens_mask, dilate_px=2, feather_sigma=1.0)
    lens_px = int((lens_mask > 127).sum())
    fp_pct = 100.0 * float((glare > 127).sum()) / float(lens_px)
    assert fp_pct < 2.0, f"{label}: unexpected false positive glare {fp_pct:.1f}%"
    assert int(glare[lens_mask == 0].max()) == 0


def test_glare_detector_catches_uniform_neutral_rgb175_over_dark_skin() -> None:
    """Regression test: uniform neutral RGB 175 reflection filling both lenses over dark skin."""
    rgb, lens_mask, gt = _make_synthetic_lens_scene(
        (90, 62, 46), glare_color=(175, 175, 175), fill_both_lenses=True
    )
    pred = compute_glare_hotspot_mask(rgb, lens_mask, dilate_px=2, feather_sigma=1.0)
    recall = float(((pred > 127) & (gt > 127)).sum()) / float(int((gt > 127).sum()))
    assert recall > 0.90


@pytest.mark.parametrize(
    "skin_rgb",
    [(222, 188, 168), (165, 124, 96), (94, 65, 48)],
)
@pytest.mark.parametrize(
    "glare_kind,glare_rgb",
    [
        ("white_specular", (250, 252, 255)),
        ("cool_screen", (210, 224, 248)),
        ("warm_orange_lamp", (245, 160, 105)),
        ("green_ar_coating", (125, 195, 135)),
        ("purple_ar_coating", (195, 130, 210)),
    ],
)
def test_glare_detector_high_recall_across_skins_and_glare_types(
    skin_rgb: tuple[int, int, int],
    glare_kind: str,
    glare_rgb: tuple[int, int, int],
) -> None:
    if glare_kind == "warm_orange_lamp" and skin_rgb[0] > 210:
        glare_rgb = (255, 195, 135)
    rgb, lens_mask, gt = _make_synthetic_lens_scene(skin_rgb, glare_color=glare_rgb)
    pred = compute_glare_hotspot_mask(rgb, lens_mask, dilate_px=3, feather_sigma=1.5)
    recall = float(((pred > 127) & (gt > 127)).sum()) / float(max(1, int((gt > 127).sum())))
    assert recall > 0.85, f"Low recall {recall:.2f} for {glare_kind} on skin {skin_rgb}"
    assert int(pred[lens_mask == 0].max()) == 0


def test_glare_detector_fills_ring_reflection_holes() -> None:
    rgb, lens_mask, _ = _make_synthetic_lens_scene(
        (150, 115, 92), glare_color=(125, 200, 135), ring_glare=True
    )
    pred = compute_glare_hotspot_mask(rgb, lens_mask, dilate_px=2, feather_sigma=1.0)
    assert pred[128, 80] > 200


def test_build_initial_lens_prompts_auto_and_override() -> None:
    eye_roi = {
        "left_eye": [170.0, 250.0],
        "right_eye": [342.0, 252.0],
        "bridge": [256.0, 251.0],
        "inter_ocular_px": 172.0,
        "roi_size": 512,
    }
    lp, rp = build_initial_lens_prompts(eye_roi, roi_size=512)
    assert lp["box"][2] < 256.0 < rp["box"][0]
    assert sum(lp["labels"]) == 5

    ov = {"left": {"box": [90, 200, 220, 300]}, "right": {"box": [290, 200, 420, 300]}}
    lp2, rp2 = build_initial_lens_prompts(None, roi_size=512, override_prompts=ov)
    assert lp2["box"] == [90.0, 200.0, 220.0, 300.0]
    assert len(lp2["points"]) == len(lp2["labels"])


def test_sam2_chunk_ranges_coverage() -> None:
    assert compute_sam2_chunk_ranges(50, chunk_size=280, overlap=10) == [(0, 50)]
    ranges = compute_sam2_chunk_ranges(600, chunk_size=280, overlap=10)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 600
    for (s1, e1), (s2, e2) in zip(ranges, ranges[1:], strict=False):
        assert e1 - s2 == 10
        assert s2 > s1 and e2 > e1


def test_temporal_smooth_isolated_empty_dropout_and_full_mask_regression() -> None:
    """Regression test: isolated empty mask must NEVER erase valid neighboring masks."""
    valid = np.zeros((64, 64), dtype=np.uint8)
    valid[16:48, 16:48] = 255
    empty = np.zeros((64, 64), dtype=np.uint8)
    full = np.full((64, 64), 255, dtype=np.uint8)

    # [valid, valid, empty, valid, valid] -> neighbors MUST retain full 1024 area!
    out = temporal_smooth_binary_masks([valid, valid, empty, valid, valid], radius=1)
    areas = [int((m > 127).sum()) for m in out]
    assert areas[0] == 1024
    assert areas[1] == 1024
    assert areas[2] == 1024  # isolated 1-frame dropout bridged
    assert areas[3] == 1024
    assert areas[4] == 1024

    # [empty, empty, valid, valid] -> fade-in empty frames stay empty; valid frames stay intact!
    out_fade = temporal_smooth_binary_masks([empty, empty, valid, valid], radius=1)
    assert [int((m > 127).sum()) for m in out_fade] == [0, 0, 1024, 1024]

    # All-empty stays all-empty; all-full stays valid without OpenCV overflow
    out_empty = temporal_smooth_binary_masks([empty, empty, empty], radius=1)
    assert all(int((m > 127).sum()) == 0 for m in out_empty)
    out_full = temporal_smooth_binary_masks([full, valid, full], radius=1)
    assert all(int((m > 127).sum()) > 0 for m in out_full)


def test_mask_iou_and_clean() -> None:
    empty = np.zeros((64, 64), dtype=bool)
    assert mask_iou(empty, empty) == 0.0
    a = np.zeros((64, 64), dtype=np.uint8)
    cv2.circle(a, (32, 32), 18, 255, -1)
    assert mask_iou(a, a) == pytest.approx(1.0)
    noisy = a.copy()
    cv2.circle(noisy, (32, 32), 4, 0, -1)
    noisy[2:5, 2:5] = 255
    cleaned = clean_single_lens_mask(noisy > 0, erode_px=2)
    assert cleaned[32, 32] == 255 and cleaned[3, 3] == 0


def test_chunked_tracking_detects_post_seed_identity_swap_and_zero_overlap(
    tmp_path: Path,
) -> None:
    H = W = 128
    base_l_u8 = np.zeros((H, W), dtype=np.uint8)
    base_r_u8 = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(base_l_u8, (38, 64), 18, 255, -1)
    cv2.circle(base_r_u8, (90, 64), 18, 255, -1)
    base_l = base_l_u8 > 0
    base_r = base_r_u8 > 0
    init_l, init_r = derive_prompts_from_masks(
        base_l, base_r, fallback_prompts=build_initial_lens_prompts(None, roi_size=128)
    )

    # 1) Zero-overlap disjoint chunks work without crashing
    l_z, r_z, logs_z = run_chunked_tracking_loop(
        lambda frames, **_kw: ([base_l.copy() for _ in frames], [base_r.copy() for _ in frames]),
        lambda i: i,
        n_frames=16,
        init_left_prompts=init_l,
        init_right_prompts=init_r,
        chunk_size=8,
        overlap=0,
    )
    assert len(l_z) == 16 and len(logs_z) == 2

    # 2) Post-seed identity swap (frame 0 matches, frames 1..end swap L<->R!) must trigger fallback!
    calls: list[str] = []

    def swap_on_box_runner(frames, *, left_prompts, right_prompts, init_masks):
        L = len(frames)
        if init_masks is not None:
            calls.append("fallback_masks")
            return [base_l.copy() for _ in range(L)], [base_r.copy() for _ in range(L)]
        calls.append("box_pts")
        if len(calls) == 2:
            # Frame 0 matches (IoU=1.0), frames 1..end SWAPPED!
            return (
                [base_l.copy()] + [base_r.copy() for _ in range(L - 1)],
                [base_r.copy()] + [base_l.copy() for _ in range(L - 1)],
            )
        return [base_l.copy() for _ in range(L)], [base_r.copy() for _ in range(L)]

    l_ok, r_ok, logs = run_chunked_tracking_loop(
        swap_on_box_runner,
        lambda i: i,
        n_frames=20,
        init_left_prompts=init_l,
        init_right_prompts=init_r,
        chunk_size=12,
        overlap=4,
        ckpt_dir=tmp_path / "ckpts_swap",
        fingerprint="fp_swap",
    )
    assert calls == ["box_pts", "box_pts", "fallback_masks"]
    assert logs[1]["used_fallback"] is True
    assert mask_iou(l_ok[-1], base_l) == pytest.approx(1.0)
    assert mask_iou(r_ok[-1], base_r) == pytest.approx(1.0)

    # 3) If BOTH box+pts and fallback_masks stay swapped -> MUST raise RuntimeError!
    def always_swap_chunk1(frames, *, left_prompts, right_prompts, init_masks):
        L = len(frames)
        if frames[0] == 0:
            return [base_l.copy() for _ in range(L)], [base_r.copy() for _ in range(L)]
        return (
            [base_l.copy()] + [base_r.copy() for _ in range(L - 1)],
            [base_r.copy()] + [base_l.copy() for _ in range(L - 1)],
        )

    with pytest.raises(RuntimeError, match="identity swap"):
        run_chunked_tracking_loop(
            always_swap_chunk1,
            lambda i: i,
            n_frames=20,
            init_left_prompts=init_l,
            init_right_prompts=init_r,
            chunk_size=12,
            overlap=4,
        )


def test_bidirectional_seed_frame_and_oom_halving_and_cpu_session_kwargs(tmp_path: Path) -> None:
    H = W = 64
    bl = np.zeros((H, W), dtype=bool)
    br = np.zeros((H, W), dtype=bool)
    bl[20:44, 8:28] = True
    br[20:44, 36:56] = True
    init_l, init_r = build_initial_lens_prompts(None, roi_size=64)

    # Non-zero seed_idx=5: both forward and backward passes MUST receive frame 5 as their frame 0!
    seed_frames_seen: list[int] = []

    def tracking_runner(frames, **_kw):
        seed_frames_seen.append(frames[0])
        # Return masks encoding frame idx presence
        return [bl.copy() for _ in frames], [br.copy() for _ in frames]

    l_all, r_all, fwd_logs, bwd_logs = track_bidirectional_from_seed(
        tracking_runner,
        lambda i: i,
        n_frames=12,
        seed_idx=5,
        init_left_prompts=init_l,
        init_right_prompts=init_r,
        chunk_size=10,
        overlap=2,
        ckpt_dir=tmp_path / "bi_ckpts",
        fingerprint="fp_bi",
    )
    assert len(l_all) == 12 and len(r_all) == 12
    assert seed_frames_seen == [5, 5]  # Forward and backward both seeded on frame 5, never blank frame 0!
    assert len(fwd_logs) == 1 and len(bwd_logs) == 1

    # OOM retry halves chunk_size cleanly
    attempted_lens: list[int] = []

    def oom_once_runner(frames, **_kw):
        attempted_lens.append(len(frames))
        if len(frames) > 10:
            raise RuntimeError("CUDA out of memory")
        return [bl.copy() for _ in frames], [br.copy() for _ in frames]

    l_oom, _, _ = run_chunked_tracking_loop(
        oom_once_runner,
        lambda i: i,
        n_frames=18,
        init_left_prompts=init_l,
        init_right_prompts=init_r,
        chunk_size=16,
        overlap=2,
    )
    assert len(l_oom) == 18
    assert attempted_lens[0] == 16 and attempted_lens[1] == 8

    # _build_init_session_kwargs passes CPU processing/state kwargs when processor accepts them
    class FakeProcessor:
        def init_video_session(
            self,
            video=None,
            inference_device="cpu",
            inference_state_device=None,
            processing_device=None,
            video_storage_device=None,
            max_vision_features_cache_size=1,
            dtype=None,
        ):
            pass

    kw = _build_init_session_kwargs(FakeProcessor(), [], "cuda", "bf16")
    assert kw["processing_device"] == "cpu"
    assert kw["inference_state_device"] == "cpu"
    assert kw["video_storage_device"] == "cpu"
    assert kw["max_vision_features_cache_size"] == 1


def test_segment_round3_p1_regressions_handoff_smoothing_oom_and_npz_loads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regressions P1-1, P1-7, P1-8, P1-9, P1-10."""
    from eyeglass_glare_removal import segment
    from eyeglass_glare_removal.config import SegmentConfig

    # --- P1-1: handoff_iou_threshold & device must alter segment generation ---
    il, ir = build_initial_lens_prompts(None, roi_size=64)
    g1 = segment.compute_segment_generation("dig", 0, il, ir, SegmentConfig(handoff_iou_threshold=0.5))
    g2 = segment.compute_segment_generation("dig", 0, il, ir, SegmentConfig(handoff_iou_threshold=0.7))
    g3 = segment.compute_segment_generation("dig", 0, il, ir, SegmentConfig(device="cpu"))
    assert g1[0] != g2[0] and g1[1] != g2[1]
    assert g1[0] != g3[0] and g1[1] != g3[1]

    H = W = 64
    bl = np.zeros((H, W), dtype=bool)
    br = np.zeros((H, W), dtype=bool)
    bl[20:44, 8:26] = True
    br[20:44, 38:56] = True

    # --- P1-7: overlap=1 rejected for multi-chunk; matched mutual occlusion passes without abort ---
    with pytest.raises(ValueError, match="overlap=1 cannot verify"):
        run_chunked_tracking_loop(
            lambda frames, **_kw: ([bl.copy() for _ in frames], [br.copy() for _ in frames]),
            lambda i: i,
            n_frames=16,
            init_left_prompts=il,
            init_right_prompts=ir,
            chunk_size=10,
            overlap=1,
        )

    empty_b = np.zeros((H, W), dtype=bool)

    def occluded_overlap_runner(frames, **_kw):
        # Frames 6..10 have right eye occluded (empty); reappears on frame 11+!
        ls, rs = [], []
        for fi in frames:
            ls.append(bl.copy())
            rs.append(empty_b.copy() if 6 <= fi <= 10 else br.copy())
        return ls, rs

    l_occ, r_occ, logs_occ = run_chunked_tracking_loop(
        occluded_overlap_runner,
        lambda i: i,
        n_frames=16,
        init_left_prompts=il,
        init_right_prompts=ir,
        chunk_size=10,
        overlap=3,
    )
    assert len(l_occ) == 16 and len(r_occ) == 16 and len(logs_occ) == 2
    assert mask_iou(r_occ[-1], br) == pytest.approx(1.0)

    # --- P1-1 & P1-10: predecessor overlap change invalidates chunk 1; .npz loads each array ONCE ---
    ckpt_dir = tmp_path / "npz_ckpts"
    calls = 0

    def steady_runner(frames, **_kw):
        nonlocal calls
        calls += 1
        return [bl.copy() for _ in frames], [br.copy() for _ in frames]

    run_chunked_tracking_loop(
        steady_runner,
        lambda i: i,
        n_frames=16,
        init_left_prompts=il,
        init_right_prompts=ir,
        chunk_size=10,
        overlap=3,
        ckpt_dir=ckpt_dir,
        fingerprint="fp_npz",
    )
    assert calls == 2

    # Resume with counting np.load wrapper: each array key read EXACTLY ONCE per chunk (not per frame!)
    real_np_load = np.load
    key_reads: list[str] = []

    class CountingNpz:
        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return self._inner.__exit__(*a)

        def __contains__(self, k):
            return k in self._inner

        def __getitem__(self, k):
            if k in ("left", "right"):
                key_reads.append(k)
            return self._inner[k]

    monkeypatch.setattr(segment.np, "load", lambda *a, **kw: CountingNpz(real_np_load(*a, **kw)))
    run_chunked_tracking_loop(
        steady_runner,
        lambda i: i,
        n_frames=16,
        init_left_prompts=il,
        init_right_prompts=ir,
        chunk_size=10,
        overlap=3,
        ckpt_dir=ckpt_dir,
        fingerprint="fp_npz",
    )
    monkeypatch.setattr(segment.np, "load", real_np_load)
    assert calls == 2  # resumed both chunks cleanly
    assert key_reads.count("left") == 2 and key_reads.count("right") == 2  # 1 read per chunk, NOT 16!

    # Now mutate chunk 0's overlap mask on disk -> chunk 1 checkpoint MUST invalidate & re-run!
    bl_shifted = np.zeros((H, W), dtype=bool)
    bl_shifted[16:40, 12:30] = True
    c0_path = sorted(ckpt_dir.glob("chunk_0000_*.npz"))[0]
    with real_np_load(c0_path, allow_pickle=False) as d0:
        r0 = np.array(d0["right"])
        fp0 = str(d0["fingerprint"].item())
    np.savez_compressed(
        c0_path,
        left=np.stack([bl_shifted] * 10, axis=0),
        right=r0,
        fingerprint=fp0,
    )

    def shifted_runner(frames, **_kw):
        nonlocal calls
        calls += 1
        return [bl_shifted.copy() for _ in frames], [br.copy() for _ in frames]

    run_chunked_tracking_loop(
        shifted_runner,
        lambda i: i,
        n_frames=16,
        init_left_prompts=il,
        init_right_prompts=ir,
        chunk_size=10,
        overlap=3,
        ckpt_dir=ckpt_dir,
        fingerprint="fp_npz",
    )
    assert calls == 3  # chunk 1 recomputed because predecessor overlap digest changed!

    # --- P1-8: moving 100px mask + empty dropout keeps valid frame at 100px; full outlier keeps 256px ---
    m0 = np.zeros((64, 64), dtype=np.uint8)
    m0[20:30, 10:20] = 255  # 100px
    m1 = np.zeros((64, 64), dtype=np.uint8)
    m1[20:30, 22:32] = 255  # shifted 12px (IoU=0 with m0)
    m_empty = np.zeros((64, 64), dtype=np.uint8)
    m3 = np.zeros((64, 64), dtype=np.uint8)
    m3[20:30, 34:44] = 255
    out_motion = temporal_smooth_binary_masks([m0, m1, m_empty, m3], radius=1)
    assert int((out_motion[1] > 127).sum()) == 100  # valid frame NEVER erased by dropout + motion!

    stat = np.zeros((64, 64), dtype=np.uint8)
    stat[24:40, 24:40] = 255  # 256px
    full_out = np.full((64, 64), 255, dtype=np.uint8)
    out_spike = temporal_smooth_binary_masks([stat, full_out, stat], radius=1)
    assert [int((m > 127).sum()) for m in out_spike] == [256, 256, 256]

    # --- P1-9: OOM retry with chunk_size=16, overlap=10 shrinks BOTH chunk_size AND overlap ---
    seen_oom_lens: list[int] = []

    def tight_vram_runner(frames, **_kw):
        seen_oom_lens.append(len(frames))
        if len(frames) > 10:
            raise RuntimeError("CUDA out of memory")
        return [bl.copy() for _ in frames], [br.copy() for _ in frames]

    l_tight, _, _ = run_chunked_tracking_loop(
        tight_vram_runner,
        lambda i: i,
        n_frames=24,
        init_left_prompts=il,
        init_right_prompts=ir,
        chunk_size=16,
        overlap=10,
    )
    assert len(l_tight) == 24
    assert any(L <= 10 for L in seen_oom_lens)


def test_handoff_rejects_swapped_lenses_on_first_visible_frame_after_empty_overlap() -> None:
    left = np.zeros((64, 64), dtype=bool)
    right = left.copy()
    left[20:40, 7:24] = True
    right[20:40, 40:57] = True
    absent = np.zeros_like(left)
    init_l, init_r = build_initial_lens_prompts(None, roi_size=64)

    def run(swapped):
        def tracker(frames, **_kwargs):
            pair = (right, left) if swapped and frames[0] > 0 else (left, right)
            return tuple([absent if 4 <= i <= 5 else mask for i in frames] for mask in pair)

        return run_chunked_tracking_loop(
            tracker, lambda i: i, 10, init_l, init_r, chunk_size=6, overlap=2,
        )

    with pytest.raises(RuntimeError, match="identity swap"):
        run(swapped=True)
    output_left, output_right, logs = run(swapped=False)
    assert not output_left[5].any() and not output_right[5].any()
    assert np.array_equal(output_left[6], left)
    assert np.array_equal(output_right[6], right)
    assert len(logs) == 2


def test_oom_backoff_reaches_three_frame_sessions_with_two_frame_overlap() -> None:
    left = np.zeros((64, 64), dtype=bool)
    right = left.copy()
    left[20:40, 7:24] = True
    right[20:40, 40:57] = True
    init_l, init_r = build_initial_lens_prompts(None, roi_size=64)
    attempted = []

    def runner(frames, **_kwargs):
        attempted.append(len(frames))
        if len(frames) > 3:
            raise RuntimeError("CUDA out of memory")
        return [left] * len(frames), [right] * len(frames)

    l_out, r_out, logs = run_chunked_tracking_loop(
        runner, lambda i: i, 21, init_l, init_r, chunk_size=16, overlap=2,
    )
    assert attempted[:4] == [16, 8, 4, 3]
    assert len(l_out) == len(r_out) == 21
    assert all(np.array_equal(mask, left) for mask in l_out)
    assert all(np.array_equal(mask, right) for mask in r_out)
    assert all(old["end"] - new["start"] == 2 for old, new in zip(logs, logs[1:], strict=False))
