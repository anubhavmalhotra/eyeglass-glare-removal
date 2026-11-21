"""End-to-end pipeline orchestration and canonical work-directory layout."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eyeglass_glare_removal._workspace import (
    ensure_confined_existing_file,
    ensure_safe_child_file,
    hash_file_sha256,
    hash_files_manifest,
    reject_symlink_path,
    validate_safe_basename,
    verify_image_file,
    workdir_lock,
)
from eyeglass_glare_removal.composite import run_composite_stage
from eyeglass_glare_removal.config import PipelineConfig
from eyeglass_glare_removal.detect import compute_prep_generation, run_prep_stage
from eyeglass_glare_removal.inpaint import run_inpaint_chunks
from eyeglass_glare_removal.reference import build_clean_reference
from eyeglass_glare_removal.segment import (
    build_initial_lens_prompts,
    compute_segment_generation,
    run_segment_stage,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkDirLayout:
    """Canonical subdirectory and file layout inside a pipeline work_dir."""

    root: Path

    @property
    def prep_dir(self) -> Path:
        return self.root / "prep"

    @property
    def frames_dir(self) -> Path:
        return self.prep_dir / "frames"

    @property
    def roi_frames_dir(self) -> Path:
        return self.prep_dir / "roi_frames"

    @property
    def roi_boxes_json(self) -> Path:
        return self.prep_dir / "roi_boxes.json"

    @property
    def masks_dir(self) -> Path:
        return self.root / "masks"

    @property
    def roi_lens_interior_dir(self) -> Path:
        return self.masks_dir / "roi_lens_interior"

    @property
    def roi_lens_exact_dir(self) -> Path:
        return self.masks_dir / "roi_lens_exact"

    @property
    def roi_glare_hotspot_dir(self) -> Path:
        return self.masks_dir / "roi_glare_hotspot"

    @property
    def mask_stats_json(self) -> Path:
        return self.masks_dir / "mask_stats.json"

    @property
    def reference_dir(self) -> Path:
        return self.root / "reference"

    @property
    def clean_eye_ref_png(self) -> Path:
        return self.reference_dir / "clean_eye_ref.png"

    @property
    def reference_meta_json(self) -> Path:
        return self.reference_dir / "reference_meta.json"

    @property
    def inpaint_dir(self) -> Path:
        return self.root / "inpaint"

    @property
    def roi_cleaned_dir(self) -> Path:
        return self.inpaint_dir / "roi_cleaned"

    @property
    def edit_support_json(self) -> Path:
        return self.inpaint_dir / "edit_support.json"


def resolve_confined_audio_path(prep_dir: Path, roi_meta: dict[str, Any]) -> Path | None:
    """Resolve optional audio path strictly confined inside `prep_dir`."""
    audio_fn = roi_meta.get("audio_filename")
    if isinstance(audio_fn, str) and audio_fn:
        p = ensure_safe_child_file(prep_dir, audio_fn, allow_missing=True)
        return p if p.exists() else None
    raw_audio = roi_meta.get("audio_path")
    if isinstance(raw_audio, str) and raw_audio:
        p = ensure_confined_existing_file(prep_dir, Path(raw_audio))
        return p if p.exists() else None
    return None


def _prep_matches_config(layout: WorkDirLayout, config: PipelineConfig) -> dict[str, Any] | None:
    """
    Return loaded `roi_boxes.json` iff prep output exists, is strictly content-verified
    against the current source file bytes & PrepConfig, and all frames/audio hashes match.
    """
    if not layout.roi_boxes_json.exists() or layout.roi_boxes_json.is_symlink():
        return None
    try:
        meta = json.loads(layout.roi_boxes_json.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not config.input_path.exists():
        return None

    source_sha = hash_file_sha256(config.input_path)
    expected_gen = compute_prep_generation(source_sha, config.prep)
    if meta.get("source_sha256") != source_sha or meta.get("prep_generation") != expected_gen:
        return None

    frames = meta.get("frames") or []
    n = int(meta.get("num_frames", -1))
    if not frames or len(frames) != n:
        return None

    try:
        filenames = [validate_safe_basename(str(r["filename"])) for r in frames]
        if hash_files_manifest(layout.frames_dir, filenames) != meta.get("frames_digest"):
            return None
        if hash_files_manifest(layout.roi_frames_dir, filenames) != meta.get("roi_frames_digest"):
            return None

        roi_size = int(meta["roi_size"])
        full_sz = (int(meta["full_width"]), int(meta["full_height"]))
        for check_idx in {0, n // 2, n - 1}:
            if not verify_image_file(layout.frames_dir / filenames[check_idx], expected_size=full_sz):
                return None
            if not verify_image_file(
                layout.roi_frames_dir / filenames[check_idx], expected_size=(roi_size, roi_size)
            ):
                return None

        expected_audio_sha = meta.get("audio_sha256")
        audio_p = resolve_confined_audio_path(layout.prep_dir, meta)
        if expected_audio_sha is not None:
            if audio_p is None or hash_file_sha256(audio_p) != expected_audio_sha:
                return None
        elif audio_p is not None:
            return None
    except Exception:
        return None

    return meta


def _segment_is_complete(
    layout: WorkDirLayout, roi_meta: dict[str, Any], config: PipelineConfig
) -> dict[str, Any] | None:
    """
    Return loaded `mask_stats.json` iff Stage 2 masks match the exact upstream ROI content
    digest, SegmentConfig, and live file-level mask content digest.
    """
    if not layout.mask_stats_json.exists() or layout.mask_stats_json.is_symlink():
        return None
    try:
        stats = json.loads(layout.mask_stats_json.read_text(encoding="utf-8"))
    except Exception:
        return None

    n = int(roi_meta["num_frames"])
    if int(stats.get("num_frames", -1)) != n:
        return None

    roi_frames_digest = str(roi_meta.get("roi_frames_digest") or "")
    if not roi_frames_digest or stats.get("roi_frames_digest") != roi_frames_digest:
        return None

    roi_size = int(roi_meta.get("roi_size", 512))
    seed_idx = max(0, min(n - 1, int(roi_meta.get("init_frame_idx", 0))))
    init_left, init_right = build_initial_lens_prompts(
        roi_meta.get("init_eye_geometry_roi"),
        roi_size=roi_size,
        override_prompts=config.segment.lens_points,
    )
    _, expected_seg_gen = compute_segment_generation(
        roi_frames_digest, seed_idx, init_left, init_right, config.segment
    )
    if stats.get("segment_generation") != expected_seg_gen:
        return None

    try:
        filenames = [validate_safe_basename(str(r["filename"])) for r in roi_meta["frames"]]
        d_lens = hash_files_manifest(layout.roi_lens_interior_dir, filenames)
        d_exact = hash_files_manifest(layout.roi_lens_exact_dir, filenames)
        d_glare = hash_files_manifest(layout.roi_glare_hotspot_dir, filenames)
        live_digest = hashlib.sha256(f"{d_lens}:{d_exact}:{d_glare}".encode()).hexdigest()
        if live_digest != stats.get("masks_digest"):
            return None
        for check_idx in {0, n // 2, n - 1}:
            for d in (
                layout.roi_lens_interior_dir,
                layout.roi_lens_exact_dir,
                layout.roi_glare_hotspot_dir,
            ):
                if not verify_image_file(d / filenames[check_idx], expected_size=(roi_size, roi_size)):
                    return None
    except Exception:
        return None

    return stats


def build_dry_run_plan(config: PipelineConfig) -> dict[str, Any]:
    """Build a JSON-serializable execution plan without invoking heavy media/GPU stages."""
    assert config.work_dir is not None
    layout = WorkDirLayout(config.work_dir)
    return {
        "dry_run": True,
        "input_path": str(config.input_path),
        "input_exists": config.input_path.exists(),
        "output_path": str(config.output_path),
        "work_dir": str(layout.root),
        "resume": config.resume,
        "stages": [
            {
                "name": "prep",
                "out_dir": str(layout.prep_dir),
                "start_sec": config.prep.start_sec,
                "duration_sec": config.prep.duration_sec,
                "fps": config.prep.fps,
                "roi_size": config.prep.roi_size,
                "crop_size": config.prep.crop_size,
                "roi_box": list(config.prep.roi_box) if config.prep.roi_box else None,
                "detector": config.prep.detector,
            },
            {
                "name": "segment",
                "out_dir": str(layout.masks_dir),
                "model_id": config.segment.model_id,
                "chunk_size": config.segment.chunk_size,
                "chunk_overlap": config.segment.chunk_overlap,
                "glare_sensitivity": config.segment.glare_sensitivity,
                "device": config.segment.device,
            },
            {
                "name": "reference",
                "out_dir": str(layout.reference_dir),
                "mode": config.reference.mode,
                "ref_image_path": str(config.reference.ref_image_path)
                if config.reference.ref_image_path
                else None,
                "allow_mirror": config.reference.allow_mirror,
            },
            {
                "name": "inpaint",
                "out_dir": str(layout.inpaint_dir),
                "model_id": config.inpaint.model_id,
                "quantize_4bit": config.inpaint.quantize_4bit,
                "chunk_len": config.inpaint.chunk_len,
                "min_overlap": config.inpaint.min_overlap,
                "steps": config.inpaint.steps,
                "device": config.inpaint.device,
            },
            {
                "name": "composite",
                "output_mp4": str(config.output_path),
                "feather_px": config.composite.feather_px,
                "bevel_band_px": config.composite.bevel_band_px,
                "suppress_chroma_glint": config.composite.suppress_chroma_glint,
                "lossless": config.composite.lossless,
                "compare": config.composite.compare,
                "qa_grid": config.composite.qa_grid,
            },
        ],
    }


def run_pipeline(config: PipelineConfig, *, dry_run: bool = False) -> dict[str, Any]:
    """Execute the full 5-stage eyeglass glare removal pipeline under an exclusive workdir lock."""
    if dry_run:
        return build_dry_run_plan(config)

    if not config.input_path.exists():
        raise FileNotFoundError(f"Input video not found: {config.input_path}")

    assert config.work_dir is not None
    layout = WorkDirLayout(config.work_dir)
    reject_symlink_path(layout.root, "work_dir")

    with workdir_lock(layout.root):
        # Stage 1: Prep & Face/Eye ROI Tracking
        roi_meta = _prep_matches_config(layout, config) if config.resume else None
        if roi_meta is not None:
            logger.info("[1/5] Prep verified & up-to-date in %s; reusing.", layout.prep_dir)
        else:
            logger.info("[1/5] Running prep & ROI tracking...")
            roi_meta = run_prep_stage(config.input_path, layout.prep_dir, config.prep)

        # Stage 2: Chunked SAM2.1 Lens Segmentation & Photometric Glare Detection
        mask_stats = _segment_is_complete(layout, roi_meta, config) if config.resume else None
        if mask_stats is not None:
            logger.info("[2/5] Segmentation verified & up-to-date in %s; reusing.", layout.masks_dir)
        else:
            logger.info("[2/5] Running chunked SAM2.1 lens segmentation & glare detection...")
            mask_stats = run_segment_stage(
                layout.roi_frames_dir,
                roi_meta,
                layout.masks_dir,
                config.segment,
                resume=config.resume,
            )

        # Stage 3: Clean-Eye Reference Builder
        logger.info("[3/5] Building clean-eye reference (mode=%s)...", config.reference.mode)
        ref_result = build_clean_reference(
            config=config.reference,
            out_dir=layout.reference_dir,
            roi_dir=layout.roi_frames_dir,
            roi_meta=roi_meta,
            mask_stats=mask_stats,
            lens_masks_dir=layout.roi_lens_interior_dir,
            glare_masks_dir=layout.roi_glare_hotspot_dir,
        )
        ref_paths = [ref_result.ref_path] if ref_result.ref_path is not None else None

        # Stage 4: Resumable Chunked Wan2.1-VACE Inpainting
        logger.info("[4/5] Running chunked Wan2.1-VACE inpainting...")
        roi_frame_paths = [
            ensure_safe_child_file(layout.roi_frames_dir, str(rec["filename"]))
            for rec in roi_meta["frames"]
        ]
        inpaint_summary = run_inpaint_chunks(
            roi_frame_paths,
            layout.roi_glare_hotspot_dir,
            layout.inpaint_dir,
            config.inpaint,
            clamp_mask_dir=layout.roi_lens_interior_dir,
            ref_image_paths=ref_paths,
            resume=config.resume,
        )

        # Load per-frame edit flags so clean frames bypass ROI resampling completely
        frame_has_edit: list[bool] | None = None
        if layout.edit_support_json.exists():
            es = json.loads(layout.edit_support_json.read_text(encoding="utf-8"))
            if isinstance(es.get("frame_has_edit"), list):
                frame_has_edit = [bool(v) for v in es["frame_has_edit"]]

        # Stage 5: Rim-Safe Streaming Compositor
        logger.info("[5/5] Running rim-safe streaming compositor -> %s", config.output_path)
        audio_p = resolve_confined_audio_path(layout.prep_dir, roi_meta)
        comp_summary = run_composite_stage(
            orig_frames_dir=layout.frames_dir,
            cleaned_roi_dir=layout.roi_cleaned_dir,
            lens_masks_dir=layout.roi_lens_exact_dir,
            roi_meta=roi_meta,
            output_mp4=config.output_path,
            config=config.composite,
            glare_masks_dir=layout.roi_glare_hotspot_dir,
            audio_path=audio_p,
            source_video_path=config.input_path,
            frame_has_edit=frame_has_edit,
        )

    return {
        "work_dir": str(layout.root),
        "prep": {
            "num_frames": roi_meta["num_frames"],
            "fps": roi_meta["fps"],
            "crop_size": roi_meta["crop_size"],
            "detector": roi_meta["detector"],
            "has_audio": audio_p is not None,
        },
        "segment": {
            "chunks": len(mask_stats.get("chunks", [])),
            "glare_pct_min_med_max": mask_stats.get("glare_pct_min_med_max"),
            "peak_vram_gib": mask_stats.get("peak_vram_gib"),
        },
        "reference": ref_result.to_dict(),
        "inpaint": inpaint_summary,
        "composite": comp_summary,
    }
