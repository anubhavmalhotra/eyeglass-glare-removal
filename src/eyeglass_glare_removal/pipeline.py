"""End-to-end pipeline orchestration and canonical work-directory layout."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eyeglass_glare_removal._workspace import (
    assert_source_outside_managed_dir,
    ensure_confined_existing_file,
    ensure_safe_child_file,
    hash_file_sha256,
    hash_files_manifest,
    reject_symlink_components_under,
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


def resolve_confined_audio_path(
    prep_dir: Path,
    roi_meta: dict[str, Any],
    *,
    workspace_root: Path | None = None,
) -> Path | None:
    """Resolve optional audio path strictly confined inside `prep_dir` with lexical symlink checks."""
    if workspace_root is not None:
        reject_symlink_components_under(workspace_root, prep_dir)
    else:
        reject_symlink_path(prep_dir, "prep_dir")
    audio_fn = roi_meta.get("audio_filename")
    if isinstance(audio_fn, str) and audio_fn:
        p = ensure_safe_child_file(
            prep_dir, audio_fn, allow_missing=True, workspace_root=workspace_root
        )
        return p if p.exists() else None
    raw_audio = roi_meta.get("audio_path")
    if isinstance(raw_audio, str) and raw_audio:
        p = ensure_confined_existing_file(prep_dir, Path(raw_audio))
        if workspace_root is not None:
            reject_symlink_components_under(workspace_root, p)
        return p if p.exists() else None
    return None


def _load_json_dict(path: Path, label: str) -> dict[str, Any]:
    """Load a JSON file and require its top-level value to be a JSON object (`dict`)."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Malformed JSON in {label} ({path}): {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid schema in {label} ({path}): expected JSON object, got {type(data).__name__}"
        )
    return data


def _extract_safe_filenames(meta: dict[str, Any], label: str) -> list[str]:
    """Extract and validate canonical frame basenames from `roi_boxes.json`."""
    raw_n = meta.get("num_frames")
    if isinstance(raw_n, bool) or not isinstance(raw_n, int) or raw_n < 1:
        raise ValueError(f"Invalid num_frames in {label}: {raw_n!r}")
    frames = meta.get("frames")
    if not isinstance(frames, list) or len(frames) != raw_n:
        raise ValueError(f"Invalid frames list in {label}: expected {raw_n} entries")
    filenames: list[str] = []
    for i, rec in enumerate(frames):
        if not isinstance(rec, dict) or not isinstance(rec.get("filename"), str):
            raise ValueError(f"Invalid frame record at index {i} in {label}")
        filenames.append(validate_safe_basename(rec["filename"]))
    return filenames


def verify_prep_artifacts(
    layout: WorkDirLayout,
) -> tuple[dict[str, Any], list[str], str, str]:
    """
    Strictly verify Stage 1 (`prep/`) confinement, schema, and live file content digests.
    Returns `(roi_meta, filenames, live_frames_digest, live_roi_digest)`.
    """
    reject_symlink_components_under(layout.root, layout.prep_dir)
    reject_symlink_components_under(layout.root, layout.roi_boxes_json)
    reject_symlink_components_under(layout.root, layout.frames_dir)
    reject_symlink_components_under(layout.root, layout.roi_frames_dir)

    meta = _load_json_dict(layout.roi_boxes_json, "prep manifest roi_boxes.json")
    filenames = _extract_safe_filenames(meta, "roi_boxes.json")

    prep_gen = meta.get("prep_generation")
    expected_frames_digest = meta.get("frames_digest")
    expected_roi_digest = meta.get("roi_frames_digest")
    if (
        not isinstance(prep_gen, str)
        or not prep_gen
        or not isinstance(expected_frames_digest, str)
        or not expected_frames_digest
        or not isinstance(expected_roi_digest, str)
        or not expected_roi_digest
    ):
        raise ValueError(
            "Prep manifest roi_boxes.json is missing required generation/digest fields; "
            "rerun `glare-remove prep`."
        )

    live_frames_digest = hash_files_manifest(
        layout.frames_dir, filenames, workspace_root=layout.root
    )
    if live_frames_digest != expected_frames_digest:
        raise ValueError(
            "Prep frames directory content no longer matches roi_boxes.json frames_digest; "
            "rerun `glare-remove prep`."
        )
    live_roi_digest = hash_files_manifest(
        layout.roi_frames_dir, filenames, workspace_root=layout.root
    )
    if live_roi_digest != expected_roi_digest:
        raise ValueError(
            "Prep roi_frames directory content no longer matches roi_boxes.json roi_frames_digest; "
            "rerun `glare-remove prep`."
        )

    expected_audio_sha = meta.get("audio_sha256")
    audio_p = resolve_confined_audio_path(layout.prep_dir, meta, workspace_root=layout.root)
    if expected_audio_sha is not None:
        if audio_p is None or hash_file_sha256(audio_p) != expected_audio_sha:
            raise ValueError(
                "Prep audio track is missing or does not match roi_boxes.json audio_sha256; "
                "rerun `glare-remove prep`."
            )
    elif audio_p is not None:
        raise ValueError(
            "Unexpected audio file found in prep/ when roi_boxes.json records no audio track; "
            "rerun `glare-remove prep`."
        )

    return meta, filenames, live_frames_digest, live_roi_digest


def verify_segment_artifacts(
    layout: WorkDirLayout,
    roi_meta: dict[str, Any],
    filenames: list[str],
    live_roi_digest: str,
) -> tuple[dict[str, Any], str]:
    """
    Strictly verify Stage 2 (`masks/`) confinement, upstream prep generation binding,
    and live mask content digests. Returns `(mask_stats, live_masks_digest)`.
    """
    reject_symlink_components_under(layout.root, layout.masks_dir)
    reject_symlink_components_under(layout.root, layout.mask_stats_json)
    reject_symlink_components_under(layout.root, layout.roi_lens_interior_dir)
    reject_symlink_components_under(layout.root, layout.roi_lens_exact_dir)
    reject_symlink_components_under(layout.root, layout.roi_glare_hotspot_dir)

    stats = _load_json_dict(layout.mask_stats_json, "segment manifest mask_stats.json")
    raw_n = stats.get("num_frames")
    if isinstance(raw_n, bool) or not isinstance(raw_n, int) or raw_n != len(filenames):
        raise ValueError(
            f"Segment manifest mask_stats.json frame count ({raw_n!r}) != prep frames "
            f"({len(filenames)}); rerun `glare-remove segment`."
        )

    prep_gen = roi_meta.get("prep_generation")
    seg_prep_gen = stats.get("prep_generation")
    seg_gen = stats.get("segment_generation")
    if not isinstance(seg_gen, str) or not seg_gen:
        raise ValueError(
            "Segment manifest mask_stats.json is missing segment_generation; "
            "rerun `glare-remove segment`."
        )
    if seg_prep_gen != prep_gen:
        raise ValueError(
            f"Workspace generation mismatch: masks prep_generation ({seg_prep_gen!r}) != "
            f"prep roi_boxes.json ({prep_gen!r}). Rerun `glare-remove segment`."
        )
    if stats.get("roi_frames_digest") != live_roi_digest:
        raise ValueError(
            "Workspace digest mismatch: masks were generated from different ROI frames than "
            "current prep/roi_frames. Rerun `glare-remove segment`."
        )

    d_lens = hash_files_manifest(
        layout.roi_lens_interior_dir, filenames, workspace_root=layout.root
    )
    d_exact = hash_files_manifest(
        layout.roi_lens_exact_dir, filenames, workspace_root=layout.root
    )
    d_glare = hash_files_manifest(
        layout.roi_glare_hotspot_dir, filenames, workspace_root=layout.root
    )
    live_masks_digest = hashlib.sha256(f"{d_lens}:{d_exact}:{d_glare}".encode()).hexdigest()
    if live_masks_digest != stats.get("masks_digest"):
        raise ValueError(
            "Mask files on disk do not match mask_stats.json masks_digest; "
            "rerun `glare-remove segment`."
        )

    return stats, live_masks_digest


def verify_reference_artifact(
    layout: WorkDirLayout,
    roi_meta: dict[str, Any],
    mask_stats: dict[str, Any],
) -> Path | None:
    """
    Validate an existing workspace `reference/clean_eye_ref.png` against the current prep
    and segment generations before standalone `inpaint` consumes it automatically.
    """
    if layout.reference_dir.exists() or layout.reference_dir.is_symlink():
        reject_symlink_components_under(layout.root, layout.reference_dir)
    if not (layout.clean_eye_ref_png.exists() or layout.clean_eye_ref_png.is_symlink()):
        return None

    reject_symlink_components_under(layout.root, layout.clean_eye_ref_png)
    reject_symlink_components_under(layout.root, layout.reference_meta_json)
    ref_meta = _load_json_dict(layout.reference_meta_json, "reference manifest reference_meta.json")

    prep_gen = roi_meta.get("prep_generation")
    seg_gen = mask_stats.get("segment_generation")
    if ref_meta.get("prep_generation") != prep_gen:
        raise ValueError(
            f"Stale reference/clean_eye_ref.png: prep_generation ({ref_meta.get('prep_generation')!r}) "
            f"!= current prep ({prep_gen!r}). Rerun `glare-remove reference`, or pass `--no-ref` / `--ref-image`."
        )
    if ref_meta.get("mode") == "auto" and ref_meta.get("segment_generation") != seg_gen:
        raise ValueError(
            f"Stale reference/clean_eye_ref.png: segment_generation ({ref_meta.get('segment_generation')!r}) "
            f"!= current masks ({seg_gen!r}). Rerun `glare-remove reference`, or pass `--no-ref` / `--ref-image`."
        )
    expected_sha = ref_meta.get("ref_sha256")
    if not isinstance(expected_sha, str) or hash_file_sha256(layout.clean_eye_ref_png) != expected_sha:
        raise ValueError(
            "reference/clean_eye_ref.png does not match reference_meta.json ref_sha256; "
            "rerun `glare-remove reference`."
        )
    return layout.clean_eye_ref_png


def verify_inpaint_artifacts(
    layout: WorkDirLayout,
    roi_meta: dict[str, Any],
    mask_stats: dict[str, Any],
    filenames: list[str],
    live_roi_digest: str,
    live_masks_digest: str,
) -> tuple[dict[str, Any], list[bool]]:
    """
    Strictly verify Stage 4 (`inpaint/`) confinement, upstream prep & segment generation
    binding, and live cleaned ROI content digest before Stage 5 composite reads them.
    """
    reject_symlink_components_under(layout.root, layout.inpaint_dir)
    reject_symlink_components_under(layout.root, layout.roi_cleaned_dir)
    reject_symlink_components_under(layout.root, layout.edit_support_json)

    es = _load_json_dict(layout.edit_support_json, "inpaint manifest edit_support.json")
    raw_n = es.get("num_frames")
    if isinstance(raw_n, bool) or not isinstance(raw_n, int) or raw_n != len(filenames):
        raise ValueError(
            f"Inpaint manifest edit_support.json frame count ({raw_n!r}) != prep frames "
            f"({len(filenames)}); rerun `glare-remove inpaint`."
        )

    prep_gen = roi_meta.get("prep_generation")
    seg_gen = mask_stats.get("segment_generation")
    if es.get("prep_generation") != prep_gen:
        raise ValueError(
            f"Workspace generation mismatch: inpaint prep_generation ({es.get('prep_generation')!r}) "
            f"!= prep roi_boxes.json ({prep_gen!r}). Rerun `glare-remove inpaint`."
        )
    if es.get("segment_generation") != seg_gen:
        raise ValueError(
            f"Workspace generation mismatch: inpaint segment_generation ({es.get('segment_generation')!r}) "
            f"!= masks mask_stats.json ({seg_gen!r}). Rerun `glare-remove inpaint`."
        )
    if es.get("roi_frames_digest") != live_roi_digest:
        raise ValueError(
            "Workspace digest mismatch: inpaint outputs were generated from different ROI frames "
            "than current prep/roi_frames. Rerun `glare-remove inpaint`."
        )
    if es.get("masks_digest") != live_masks_digest:
        raise ValueError(
            "Workspace digest mismatch: inpaint outputs were generated from different masks "
            "than current masks/. Rerun `glare-remove inpaint`."
        )

    raw_flags = es.get("frame_has_edit")
    if not isinstance(raw_flags, list) or len(raw_flags) != len(filenames):
        raise ValueError(
            "Inpaint manifest edit_support.json has invalid frame_has_edit list; "
            "rerun `glare-remove inpaint`."
        )
    frame_has_edit = [bool(v) for v in raw_flags]

    live_cleaned_digest = hash_files_manifest(
        layout.roi_cleaned_dir, filenames, workspace_root=layout.root
    )
    if live_cleaned_digest != es.get("cleaned_roi_digest"):
        raise ValueError(
            "inpaint/roi_cleaned files on disk do not match edit_support.json cleaned_roi_digest; "
            "rerun `glare-remove inpaint`."
        )

    return es, frame_has_edit


def _prep_matches_config(layout: WorkDirLayout, config: PipelineConfig) -> dict[str, Any] | None:
    """
    Return loaded `roi_boxes.json` iff prep output exists, is strictly lexically confined
    under `layout.root`, content-verified against current source bytes & PrepConfig, and
    all frames/audio hashes match. Malformed schemas return None cleanly; symlink escapes raise.
    """
    # Lexical symlink check FIRST (always raises on symlinked prep/ or ancestors!)
    reject_symlink_components_under(layout.root, layout.prep_dir)
    reject_symlink_components_under(layout.root, layout.roi_boxes_json)
    reject_symlink_components_under(layout.root, layout.frames_dir)
    reject_symlink_components_under(layout.root, layout.roi_frames_dir)

    if not layout.roi_boxes_json.exists():
        return None
    if not config.input_path.exists():
        return None

    try:
        meta = json.loads(layout.roi_boxes_json.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            return None

        source_sha = hash_file_sha256(config.input_path)
        expected_gen = compute_prep_generation(source_sha, config.prep)
        if meta.get("source_sha256") != source_sha or meta.get("prep_generation") != expected_gen:
            return None

        filenames = _extract_safe_filenames(meta, "roi_boxes.json")
        n = len(filenames)
        if (
            hash_files_manifest(layout.frames_dir, filenames, workspace_root=layout.root)
            != meta.get("frames_digest")
        ):
            return None
        if (
            hash_files_manifest(layout.roi_frames_dir, filenames, workspace_root=layout.root)
            != meta.get("roi_frames_digest")
        ):
            return None

        roi_size = int(meta["roi_size"])
        full_sz = (int(meta["full_width"]), int(meta["full_height"]))
        if roi_size < 1 or full_sz[0] < 1 or full_sz[1] < 1:
            return None
        for check_idx in {0, n // 2, n - 1}:
            if not verify_image_file(layout.frames_dir / filenames[check_idx], expected_size=full_sz):
                return None
            if not verify_image_file(
                layout.roi_frames_dir / filenames[check_idx], expected_size=(roi_size, roi_size)
            ):
                return None

        expected_audio_sha = meta.get("audio_sha256")
        audio_p = resolve_confined_audio_path(layout.prep_dir, meta, workspace_root=layout.root)
        if expected_audio_sha is not None:
            if audio_p is None or hash_file_sha256(audio_p) != expected_audio_sha:
                return None
        elif audio_p is not None:
            return None
    except ValueError as exc:
        if "Symlink" in str(exc) or "confined" in str(exc) or "escapes" in str(exc):
            raise
        return None
    except Exception:
        return None

    return meta


def _segment_is_complete(
    layout: WorkDirLayout, roi_meta: dict[str, Any], config: PipelineConfig
) -> dict[str, Any] | None:
    """
    Return loaded `mask_stats.json` iff Stage 2 masks are lexically confined under `layout.root`,
    match the exact upstream ROI content digest, prep_generation, SegmentConfig, and live
    mask content digest. Malformed schemas return None cleanly; symlink escapes raise.
    """
    # Lexical symlink check FIRST (always raises on symlinked masks/ or ancestors!)
    reject_symlink_components_under(layout.root, layout.masks_dir)
    reject_symlink_components_under(layout.root, layout.mask_stats_json)
    reject_symlink_components_under(layout.root, layout.roi_lens_interior_dir)
    reject_symlink_components_under(layout.root, layout.roi_lens_exact_dir)
    reject_symlink_components_under(layout.root, layout.roi_glare_hotspot_dir)

    if not layout.mask_stats_json.exists():
        return None

    try:
        stats = json.loads(layout.mask_stats_json.read_text(encoding="utf-8"))
        if not isinstance(stats, dict):
            return None

        filenames = _extract_safe_filenames(roi_meta, "roi_boxes.json")
        n = len(filenames)
        raw_sn = stats.get("num_frames")
        if isinstance(raw_sn, bool) or not isinstance(raw_sn, int) or raw_sn != n:
            return None

        prep_gen = roi_meta.get("prep_generation")
        if prep_gen and stats.get("prep_generation") != prep_gen:
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

        d_lens = hash_files_manifest(
            layout.roi_lens_interior_dir, filenames, workspace_root=layout.root
        )
        d_exact = hash_files_manifest(
            layout.roi_lens_exact_dir, filenames, workspace_root=layout.root
        )
        d_glare = hash_files_manifest(
            layout.roi_glare_hotspot_dir, filenames, workspace_root=layout.root
        )
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
    except ValueError as exc:
        if "Symlink" in str(exc) or "confined" in str(exc) or "escapes" in str(exc):
            raise
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
    assert_source_outside_managed_dir(config.input_path, layout.root)

    with workdir_lock(layout.root):
        # Stage 1: Prep & Face/Eye ROI Tracking
        roi_meta = _prep_matches_config(layout, config) if config.resume else None
        if roi_meta is not None:
            logger.info("[1/5] Prep verified & up-to-date in %s; reusing.", layout.prep_dir)
        else:
            reject_symlink_components_under(layout.root, layout.prep_dir)
            logger.info("[1/5] Running prep & ROI tracking...")
            roi_meta = run_prep_stage(config.input_path, layout.prep_dir, config.prep)

        # Stage 2: Chunked SAM2.1 Lens Segmentation & Photometric Glare Detection
        mask_stats = _segment_is_complete(layout, roi_meta, config) if config.resume else None
        if mask_stats is not None:
            logger.info("[2/5] Segmentation verified & up-to-date in %s; reusing.", layout.masks_dir)
        else:
            reject_symlink_components_under(layout.root, layout.masks_dir)
            logger.info("[2/5] Running chunked SAM2.1 lens segmentation & glare detection...")
            mask_stats = run_segment_stage(
                layout.roi_frames_dir,
                roi_meta,
                layout.masks_dir,
                config.segment,
                resume=config.resume,
            )

        # Stage 3: Clean-Eye Reference Builder
        reject_symlink_components_under(layout.root, layout.reference_dir)
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
        reject_symlink_components_under(layout.root, layout.inpaint_dir)
        logger.info("[4/5] Running chunked Wan2.1-VACE inpainting...")
        roi_frame_paths = [
            ensure_safe_child_file(
                layout.roi_frames_dir, str(rec["filename"]), workspace_root=layout.root
            )
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
            prep_generation=roi_meta.get("prep_generation"),
            segment_generation=mask_stats.get("segment_generation"),
            roi_frames_digest=roi_meta.get("roi_frames_digest"),
            masks_digest=mask_stats.get("masks_digest"),
        )

        # Load verified per-frame edit flags so clean frames bypass ROI resampling completely
        frame_has_edit: list[bool] | None = None
        if layout.edit_support_json.exists():
            reject_symlink_components_under(layout.root, layout.edit_support_json)
            es = _load_json_dict(layout.edit_support_json, "edit_support.json")
            if isinstance(es.get("frame_has_edit"), list):
                frame_has_edit = [bool(v) for v in es["frame_has_edit"]]

        # Stage 5: Rim-Safe Streaming Compositor
        logger.info("[5/5] Running rim-safe streaming compositor -> %s", config.output_path)
        audio_p = resolve_confined_audio_path(
            layout.prep_dir, roi_meta, workspace_root=layout.root
        )
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
