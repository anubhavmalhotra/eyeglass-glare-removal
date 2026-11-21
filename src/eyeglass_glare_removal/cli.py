"""Command-line interface (`glare-remove`) for end-to-end and modular execution."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from eyeglass_glare_removal import __version__
from eyeglass_glare_removal._workspace import ensure_safe_child_file, workdir_lock
from eyeglass_glare_removal.composite import run_composite_stage
from eyeglass_glare_removal.config import (
    DEFAULT_NEG_PROMPT,
    DEFAULT_PROMPT,
    VALID_DETECTORS,
    VALID_X264_PRESETS,
    CompositeConfig,
    InpaintConfig,
    PipelineConfig,
    PrepConfig,
    ReferenceConfig,
    SegmentConfig,
)
from eyeglass_glare_removal.detect import run_prep_stage
from eyeglass_glare_removal.inpaint import run_inpaint_chunks
from eyeglass_glare_removal.pipeline import (
    WorkDirLayout,
    resolve_confined_audio_path,
    run_pipeline,
)
from eyeglass_glare_removal.reference import build_clean_reference
from eyeglass_glare_removal.segment import run_segment_stage


def parse_roi_box(value: str) -> tuple[int, int, int, int]:
    """Parse 'x1,y1,x2,y2' into a validated 4-tuple of ints."""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"--roi-box must be 'x1,y1,x2,y2' (4 integers), got {value!r}"
        )
    try:
        x1, y1, x2, y2 = (int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--roi-box values must be integers, got {value!r}"
        ) from exc
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError(
            f"Invalid --roi-box {value!r}: require 0 <= x1 < x2 and 0 <= y1 < y2"
        )
    return (x1, y1, x2, y2)


def load_lens_points(value: str | None) -> dict[str, Any] | None:
    """Load --lens-points from a JSON file path or an inline JSON string."""
    if not value:
        return None
    p = Path(value)
    raw_text = p.read_text(encoding="utf-8") if p.exists() and p.is_file() else value
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse --lens-points JSON ({value!r}): {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--lens-points JSON must be an object with 'left' and 'right' keys.")
    return parsed


def _default_workers() -> int:
    return min(8, os.cpu_count() or 4)


def _add_prep_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Stage 1 (Prep & ROI Tracking)")
    g.add_argument("--start", type=float, default=0.0, help="Start timestamp in seconds (default: 0.0)")
    g.add_argument("--duration", type=float, default=None, help="Clip duration in seconds (default: full video)")
    g.add_argument(
        "--preview",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Shorthand for --duration SECONDS for fast preview runs (e.g. --preview 5)",
    )
    g.add_argument("--fps", type=float, default=None, help="Target constant frame rate (default: source FPS)")
    g.add_argument("--roi-size", type=int, default=512, help="Stabilized square ROI resolution (default: 512)")
    g.add_argument("--crop-size", type=int, default=None, help="Override auto square crop size in source pixels")
    g.add_argument(
        "--roi-box",
        type=parse_roi_box,
        default=None,
        metavar="X1,Y1,X2,Y2",
        help="Explicit initial ROI box override in display-oriented pixel coords",
    )
    g.add_argument(
        "--detector",
        choices=VALID_DETECTORS,
        default="auto",
        help="Face/eye detector backend (default: auto = mediapipe -> opencv haar)",
    )
    g.add_argument("--smooth-sigma", type=float, default=6.0, help="Gaussian smoothing sigma for ROI trajectory")
    g.add_argument(
        "--ncc-redetect-threshold",
        type=float,
        default=0.55,
        help="Re-detect face/eyes if template NCC falls below this score",
    )


def _add_segment_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Stage 2 (SAM2.1 Lens Segmentation & Glare Detection)")
    g.add_argument(
        "--sam2-model",
        default="facebook/sam2.1-hiera-large",
        help="Hugging Face model ID for SAM2.1 video segmentation",
    )
    g.add_argument("--sam2-chunk-size", type=int, default=280, help="Frames per SAM2 tracking session")
    g.add_argument("--sam2-overlap", type=int, default=10, help="Overlap frames between SAM2 chunks (>= 1)")
    g.add_argument("--erode-px", type=int, default=5, help="Inward erosion for conservative roi_lens_interior")
    g.add_argument("--exact-erode-px", type=int, default=1, help="Tight erosion for zero-outline roi_lens_exact")
    g.add_argument("--glare-dilate-px", type=int, default=5, help="Glare hotspot dilation in ROI pixels")
    g.add_argument("--glare-feather-sigma", type=float, default=2.0, help="Glare hotspot Gaussian feather sigma")
    g.add_argument("--glare-sensitivity", type=float, default=1.0, help="Photometric glare sensitivity multiplier")
    g.add_argument(
        "--lens-points",
        default=None,
        metavar="JSON_OR_PATH",
        help="Optional JSON file or string overriding initial left/right lens boxes/points",
    )


def _add_reference_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Stage 3 (Clean-Eye Reference Builder)")
    g.add_argument(
        "--ref-image",
        type=Path,
        default=None,
        help="Explicit clean-eye reference photo/ROI (overrides auto reference selection)",
    )
    g.add_argument(
        "--no-ref",
        action="store_true",
        help="Disable reference image conditioning and run reference-free VACE",
    )
    g.add_argument(
        "--max-clean-glare-pct",
        type=float,
        default=3.0,
        help="Max glare %% inside a lens to treat that eye as clean in auto mode (default: 3.0)",
    )
    g.add_argument(
        "--no-mirror-ref",
        action="store_true",
        help="In auto mode, do not mirror a single clean eye across the nose bridge",
    )


def _add_inpaint_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Stage 4 (Wan2.1-VACE Video Inpainting)")
    g.add_argument(
        "--vace-model",
        default="Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        help="Hugging Face model ID for Wan2.1-VACE diffusers pipeline",
    )
    g.add_argument("--text-encoder-model", default=None, help="Optional separate text encoder model ID")
    g.add_argument("--quantize-4bit", action="store_true", help="Enable 4-bit NF4 transformer quantization (bitsandbytes)")
    g.add_argument("--chunk-len", type=int, default=81, help="VACE chunk frame length (must satisfy (L-1)%%4==0)")
    g.add_argument("--min-overlap", type=int, default=12, help="Overlap frames between consecutive VACE chunks")
    g.add_argument("--mask-dilate", type=int, default=5, help="Inpaint mask dilation inside ROI")
    g.add_argument("--clamp-dilate", type=int, default=0, help="Optional clamp mask dilation")
    g.add_argument("--steps", type=int, default=20, help="Diffusion inference steps")
    g.add_argument("--guidance-scale", type=float, default=5.0, help="Classifier-free guidance scale")
    g.add_argument("--conditioning-scale", type=float, default=1.0, help="VACE conditioning scale")
    g.add_argument("--flow-shift", type=float, default=3.0, help="UniPC flow_shift parameter")
    g.add_argument("--prompt", default=DEFAULT_PROMPT, help="Positive text prompt for VACE")
    g.add_argument("--neg-prompt", default=DEFAULT_NEG_PROMPT, help="Negative text prompt for VACE")
    g.add_argument("--seed", type=int, default=42, help="Random seed")
    g.add_argument("--embed-cache-dir", type=Path, default=None, help="Directory to cache UMT5 prompt embeddings")


def _add_composite_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Stage 5 (Rim-Safe Streaming Compositor)")
    g.add_argument("--feather-px", type=float, default=2.0, help="Inward feather width in ROI pixels")
    g.add_argument("--bevel-band-px", type=int, default=4, help="Rim-safe glare-gated bevel band in ROI pixels")
    g.add_argument("--lens-expand-px", type=int, default=0, help="Optional lens mask expansion in ROI pixels")
    g.add_argument("--color-match-strength", type=float, default=0.0, help="Optional per-lens exposure match [0..1]")
    g.add_argument("--rim-protect-strength", type=float, default=1.0, help="General rim protection strength [0..1]")
    g.add_argument(
        "--suppress-chroma-glint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Neutralize residual purple/green AR chromatic glints inside glare mask (default: enabled)",
    )
    g.add_argument("--crf", type=int, default=16, help="H.264 CRF quality (0-51, lower is higher quality)")
    g.add_argument("--preset", choices=VALID_X264_PRESETS, default="medium", help="x264 encoding preset")
    g.add_argument(
        "--lossless",
        action="store_true",
        help="Encode with lossless libx264rgb (-crf 0 -pix_fmt rgb24) for decoded bit-exact RGB equality",
    )
    g.add_argument("--batch-size", type=int, default=32, help="Max streaming micro-batch size")
    g.add_argument("--max-batch-mem-mb", type=int, default=256, help="Per-batch RAM budget in MiB (bounds 4K/8K usage)")
    g.add_argument("--compare", action="store_true", help="Also render side-by-side Original vs Cleaned compare MP4")
    g.add_argument("--qa-grid", action="store_true", help="Write visual QA contact sheets")
    g.add_argument("--save-frames-dir", type=Path, default=None, help="Optionally save composited full-res PNGs")


def _add_common_runtime_args(p: argparse.ArgumentParser, *, include_gpu: bool = True, include_resume: bool = True) -> None:
    g = p.add_argument_group("Runtime Options")
    g.add_argument("--workers", type=int, default=_default_workers(), help="Parallel CPU worker threads")
    if include_gpu:
        g.add_argument("--device", default="auto", help="Compute device: auto, cuda, cuda:N, mps, or cpu")
        g.add_argument("--gpu-lock", type=Path, default=None, help="Optional flock file path to serialize GPU stages")
    if include_resume:
        g.add_argument(
            "--resume",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Reuse verified completed stages and chunk checkpoints (default: enabled)",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glare-remove",
        description="General-purpose eyeglass glare removal for video (SAM2.1 + Wan2.1-VACE + rim-safe compositor).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")

    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the full 5-stage eyeglass glare removal pipeline end-to-end.")
    p_run.add_argument("input", type=Path, help="Input video file path")
    p_run.add_argument("-o", "--output", type=Path, required=True, help="Output cleaned MP4 path")
    p_run.add_argument("--work-dir", type=Path, default=None, help="Workspace directory for intermediates & checkpoints")
    p_run.add_argument("--dry-run", action="store_true", help="Validate arguments and print resolved pipeline plan JSON")
    _add_prep_args(p_run)
    _add_segment_args(p_run)
    _add_reference_args(p_run)
    _add_inpaint_args(p_run)
    _add_composite_args(p_run)
    _add_common_runtime_args(p_run, include_gpu=True, include_resume=True)

    p_prep = sub.add_parser("prep", help="Stage 1: Extract CFR frames, detect eyes/glasses, track & crop ROIs.")
    p_prep.add_argument("input", type=Path, help="Input video file path")
    p_prep.add_argument("--work-dir", type=Path, required=True, help="Workspace directory")
    p_prep.add_argument("--qa-grid", action="store_true", help="Write ROI tracking QA grid")
    _add_prep_args(p_prep)
    _add_common_runtime_args(p_prep, include_gpu=False, include_resume=False)

    p_seg = sub.add_parser("segment", help="Stage 2: Chunked SAM2.1 lens tracking + photometric glare detection.")
    p_seg.add_argument("--work-dir", type=Path, required=True, help="Workspace directory containing prep/")
    p_seg.add_argument("--qa-grid", action="store_true", help="Write lens/glare overlay QA grid")
    _add_segment_args(p_seg)
    _add_common_runtime_args(p_seg, include_gpu=True, include_resume=True)

    p_ref = sub.add_parser("reference", help="Stage 3: Build clean-eye reference image (auto, user image, or none).")
    p_ref.add_argument("--work-dir", type=Path, required=True, help="Workspace directory containing prep/ & masks/")
    p_ref.add_argument("--roi-size", type=int, default=None, help="Reference output ROI size (default: from prep or 512)")
    _add_reference_args(p_ref)

    p_inp = sub.add_parser("inpaint", help="Stage 4: Resumable chunked Wan2.1-VACE inpainting + overlap blend.")
    p_inp.add_argument("--work-dir", type=Path, required=True, help="Workspace directory containing prep/ & masks/")
    p_inp.add_argument("--size", type=int, default=None, help="Inpainting square resolution (default: from prep or 512)")
    p_inp.add_argument("--ref-image", type=Path, default=None, help="Override clean-eye reference image path")
    p_inp.add_argument("--no-ref", action="store_true", help="Ignore reference/clean_eye_ref.png and run reference-free")
    _add_inpaint_args(p_inp)
    _add_common_runtime_args(p_inp, include_gpu=True, include_resume=True)

    p_cmp = sub.add_parser("composite", help="Stage 5: Rim-safe bounded-RAM streaming composite to MP4.")
    p_cmp.add_argument("--work-dir", type=Path, required=True, help="Workspace directory containing prep/, masks/, inpaint/")
    p_cmp.add_argument("-o", "--output", type=Path, required=True, help="Output cleaned MP4 path")
    _add_composite_args(p_cmp)
    _add_common_runtime_args(p_cmp, include_gpu=False, include_resume=False)

    return parser


def _resolve_duration(args: argparse.Namespace) -> float | None:
    if args.preview is not None and args.duration is not None:
        raise ValueError("Specify either --preview or --duration, not both.")
    return args.preview if args.preview is not None else args.duration


def _resolve_reference_config(args: argparse.Namespace, roi_size: int) -> ReferenceConfig:
    if args.ref_image is not None and args.no_ref:
        raise ValueError("Specify either --ref-image or --no-ref, not both.")
    if args.no_ref:
        mode = "none"
    elif args.ref_image is not None:
        mode = "image"
    else:
        mode = "auto"
    return ReferenceConfig(
        mode=mode,
        ref_image_path=args.ref_image,
        max_clean_glare_pct=args.max_clean_glare_pct,
        allow_mirror=not args.no_mirror_ref,
        roi_size=roi_size,
    )


def build_pipeline_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Construct and validate a PipelineConfig from parsed `run` CLI arguments."""
    duration = _resolve_duration(args)
    prep_cfg = PrepConfig(
        start_sec=args.start,
        duration_sec=duration,
        fps=args.fps,
        roi_size=args.roi_size,
        crop_size=args.crop_size,
        roi_box=args.roi_box,
        detector=args.detector,
        smooth_sigma=args.smooth_sigma,
        ncc_redetect_threshold=args.ncc_redetect_threshold,
        workers=args.workers,
        write_qa_grid=args.qa_grid,
    )
    seg_cfg = SegmentConfig(
        model_id=args.sam2_model,
        chunk_size=args.sam2_chunk_size,
        chunk_overlap=args.sam2_overlap,
        erode_px=args.erode_px,
        exact_erode_px=args.exact_erode_px,
        glare_dilate_px=args.glare_dilate_px,
        glare_feather_sigma=args.glare_feather_sigma,
        glare_sensitivity=args.glare_sensitivity,
        lens_points=load_lens_points(args.lens_points),
        device=args.device,
        workers=args.workers,
        gpu_lock=args.gpu_lock,
        write_qa_grid=args.qa_grid,
    )
    ref_cfg = _resolve_reference_config(args, roi_size=args.roi_size)
    inp_cfg = InpaintConfig(
        model_id=args.vace_model,
        text_encoder_model_id=args.text_encoder_model,
        quantize_4bit=args.quantize_4bit,
        size=args.roi_size,
        chunk_len=args.chunk_len,
        min_overlap=args.min_overlap,
        mask_dilate=args.mask_dilate,
        clamp_dilate=args.clamp_dilate,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        conditioning_scale=args.conditioning_scale,
        flow_shift=args.flow_shift,
        prompt=args.prompt,
        neg_prompt=args.neg_prompt,
        seed=args.seed,
        device=args.device,
        gpu_lock=args.gpu_lock,
        embed_cache_dir=args.embed_cache_dir,
    )
    cmp_cfg = CompositeConfig(
        feather_px=args.feather_px,
        bevel_band_px=args.bevel_band_px,
        lens_expand_px=args.lens_expand_px,
        color_match_strength=args.color_match_strength,
        suppress_chroma_glint=args.suppress_chroma_glint,
        rim_protect_strength=args.rim_protect_strength,
        crf=args.crf,
        preset=args.preset,
        lossless=args.lossless,
        workers=args.workers,
        batch_size=args.batch_size,
        max_batch_mem_mb=args.max_batch_mem_mb,
        compare=args.compare,
        qa_grid=args.qa_grid,
        save_frames_dir=args.save_frames_dir,
    )
    return PipelineConfig(
        input_path=args.input,
        output_path=args.output,
        work_dir=args.work_dir,
        resume=args.resume,
        prep=prep_cfg,
        segment=seg_cfg,
        reference=ref_cfg,
        inpaint=inp_cfg,
        composite=cmp_cfg,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        if args.command == "run":
            cfg = build_pipeline_config_from_args(args)
            result = run_pipeline(cfg, dry_run=args.dry_run)
            print(json.dumps(result, indent=2))
            return 0

        layout = WorkDirLayout(args.work_dir)
        with workdir_lock(layout.root):
            if args.command == "prep":
                prep_cfg = PrepConfig(
                    start_sec=args.start,
                    duration_sec=_resolve_duration(args),
                    fps=args.fps,
                    roi_size=args.roi_size,
                    crop_size=args.crop_size,
                    roi_box=args.roi_box,
                    detector=args.detector,
                    smooth_sigma=args.smooth_sigma,
                    ncc_redetect_threshold=args.ncc_redetect_threshold,
                    workers=args.workers,
                    write_qa_grid=args.qa_grid,
                )
                res = run_prep_stage(args.input, layout.prep_dir, prep_cfg)
                print(json.dumps({k: v for k, v in res.items() if k != "frames"}, indent=2))
                return 0

            if args.command == "segment":
                roi_meta = json.loads(layout.roi_boxes_json.read_text(encoding="utf-8"))
                seg_cfg = SegmentConfig(
                    model_id=args.sam2_model,
                    chunk_size=args.sam2_chunk_size,
                    chunk_overlap=args.sam2_overlap,
                    erode_px=args.erode_px,
                    exact_erode_px=args.exact_erode_px,
                    glare_dilate_px=args.glare_dilate_px,
                    glare_feather_sigma=args.glare_feather_sigma,
                    glare_sensitivity=args.glare_sensitivity,
                    lens_points=load_lens_points(args.lens_points),
                    device=args.device,
                    workers=args.workers,
                    gpu_lock=args.gpu_lock,
                    write_qa_grid=args.qa_grid,
                )
                res = run_segment_stage(
                    layout.roi_frames_dir, roi_meta, layout.masks_dir, seg_cfg, resume=args.resume
                )
                print(json.dumps({k: v for k, v in res.items() if k != "frames"}, indent=2))
                return 0

            if args.command == "reference":
                roi_meta = (
                    json.loads(layout.roi_boxes_json.read_text(encoding="utf-8"))
                    if layout.roi_boxes_json.exists()
                    else None
                )
                mask_stats = (
                    json.loads(layout.mask_stats_json.read_text(encoding="utf-8"))
                    if layout.mask_stats_json.exists()
                    else None
                )
                roi_size = (
                    args.roi_size
                    if args.roi_size is not None
                    else (int(roi_meta["roi_size"]) if roi_meta else 512)
                )
                ref_cfg = _resolve_reference_config(args, roi_size=roi_size)
                res = build_clean_reference(
                    config=ref_cfg,
                    out_dir=layout.reference_dir,
                    roi_dir=layout.roi_frames_dir if layout.roi_frames_dir.exists() else None,
                    roi_meta=roi_meta,
                    mask_stats=mask_stats,
                    lens_masks_dir=layout.roi_lens_interior_dir
                    if layout.roi_lens_interior_dir.exists()
                    else None,
                    glare_masks_dir=layout.roi_glare_hotspot_dir
                    if layout.roi_glare_hotspot_dir.exists()
                    else None,
                )
                print(json.dumps(res.to_dict(), indent=2))
                return 0

            if args.command == "inpaint":
                if args.ref_image is not None and args.no_ref:
                    raise ValueError("Specify either --ref-image or --no-ref, not both.")
                roi_meta = json.loads(layout.roi_boxes_json.read_text(encoding="utf-8"))
                frame_paths = [
                    ensure_safe_child_file(layout.roi_frames_dir, str(r["filename"]))
                    for r in roi_meta["frames"]
                ]
                ref_paths: list[Path] | None = None
                if not args.no_ref:
                    if args.ref_image is not None:
                        ref_paths = [args.ref_image]
                    elif layout.clean_eye_ref_png.exists():
                        ref_paths = [layout.clean_eye_ref_png]
                inp_size = (
                    args.size
                    if args.size is not None
                    else int(roi_meta.get("roi_size", 512))
                )
                inp_cfg = InpaintConfig(
                    model_id=args.vace_model,
                    text_encoder_model_id=args.text_encoder_model,
                    quantize_4bit=args.quantize_4bit,
                    size=inp_size,
                    chunk_len=args.chunk_len,
                    min_overlap=args.min_overlap,
                    mask_dilate=args.mask_dilate,
                    clamp_dilate=args.clamp_dilate,
                    steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    conditioning_scale=args.conditioning_scale,
                    flow_shift=args.flow_shift,
                    prompt=args.prompt,
                    neg_prompt=args.neg_prompt,
                    seed=args.seed,
                    device=args.device,
                    gpu_lock=args.gpu_lock,
                    embed_cache_dir=args.embed_cache_dir,
                )
                res = run_inpaint_chunks(
                    frame_paths,
                    layout.roi_glare_hotspot_dir,
                    layout.inpaint_dir,
                    inp_cfg,
                    clamp_mask_dir=layout.roi_lens_interior_dir,
                    ref_image_paths=ref_paths,
                    resume=args.resume,
                )
                print(json.dumps(res, indent=2))
                return 0

            if args.command == "composite":
                roi_meta = json.loads(layout.roi_boxes_json.read_text(encoding="utf-8"))
                audio_p = resolve_confined_audio_path(layout.prep_dir, roi_meta)
                frame_has_edit: list[bool] | None = None
                if layout.edit_support_json.exists():
                    es = json.loads(layout.edit_support_json.read_text(encoding="utf-8"))
                    if isinstance(es.get("frame_has_edit"), list):
                        frame_has_edit = [bool(v) for v in es["frame_has_edit"]]
                cmp_cfg = CompositeConfig(
                    feather_px=args.feather_px,
                    bevel_band_px=args.bevel_band_px,
                    lens_expand_px=args.lens_expand_px,
                    color_match_strength=args.color_match_strength,
                    suppress_chroma_glint=args.suppress_chroma_glint,
                    rim_protect_strength=args.rim_protect_strength,
                    crf=args.crf,
                    preset=args.preset,
                    lossless=args.lossless,
                    workers=args.workers,
                    batch_size=args.batch_size,
                    max_batch_mem_mb=args.max_batch_mem_mb,
                    compare=args.compare,
                    qa_grid=args.qa_grid,
                    save_frames_dir=args.save_frames_dir,
                )
                res = run_composite_stage(
                    orig_frames_dir=layout.frames_dir,
                    cleaned_roi_dir=layout.roi_cleaned_dir,
                    lens_masks_dir=layout.roi_lens_exact_dir,
                    roi_meta=roi_meta,
                    output_mp4=args.output,
                    config=cmp_cfg,
                    glare_masks_dir=layout.roi_glare_hotspot_dir,
                    audio_path=audio_p,
                    frame_has_edit=frame_has_edit,
                )
                print(json.dumps(res, indent=2))
                return 0

        parser.error(f"Unknown command: {args.command}")
        return 2
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
