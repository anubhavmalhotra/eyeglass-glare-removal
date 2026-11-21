"""Typed configuration dataclasses and validation for all pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from eyeglass_glare_removal._workspace import assert_distinct_output_paths

DEFAULT_PROMPT = (
    "Close-up natural video of a person wearing eyeglasses, with clear, sharp, natural eyes "
    "and eyelids fully visible through clean, transparent, anti-reflective eyeglass lenses. "
    "Consistent natural lighting and authentic skin texture around the eyes, preserving the "
    "original eyeglass frames, gaze direction, and natural blinking. Zero glare, zero specular "
    "reflections, zero screen reflection, zero lamp reflection."
)

DEFAULT_NEG_PROMPT = (
    "glare, reflection, specular highlight, screen reflection, lamp reflection, colored lens reflection, "
    "white rectangle on glasses, haze, fog, closed eyes when open, distorted pupils, cross-eyed, "
    "asymmetric eyes, blurry irises, overexposed, plastic skin, artifacts, flickering, warped frames"
)

VALID_DETECTORS = ("auto", "mediapipe", "haar", "manual")
VALID_REF_MODES = ("auto", "image", "none")
VALID_X264_PRESETS = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
    "placebo",
)


def _validate_device(device: str) -> None:
    if device in ("auto", "cuda", "mps", "cpu"):
        return
    if device.startswith("cuda:"):
        idx_str = device.split(":", 1)[1]
        if idx_str.isdigit():
            return
    raise ValueError(
        f"Invalid device {device!r}; expected 'auto', 'cuda', 'cuda:<index>', 'mps', or 'cpu'."
    )


def _validate_positive_finite(name: str, value: float, allow_zero: bool = False) -> None:
    if value != value or value == float("inf") or value == float("-inf"):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value!r}")
    else:
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value!r}")


@dataclass
class PrepConfig:
    """Configuration for Stage 1: media extraction, face/eye detection, and ROI tracking."""

    start_sec: float = 0.0
    duration_sec: float | None = None
    fps: float | None = None
    roi_size: int = 512
    crop_size: int | None = None
    roi_box: tuple[int, int, int, int] | None = None
    detector: Literal["auto", "mediapipe", "haar", "manual"] = "auto"
    smooth_sigma: float = 6.0
    ncc_redetect_threshold: float = 0.55
    workers: int = 8
    write_qa_grid: bool = False
    qa_grid_samples: int = 24

    def __post_init__(self) -> None:
        _validate_positive_finite("start_sec", self.start_sec, allow_zero=True)
        if self.duration_sec is not None:
            _validate_positive_finite("duration_sec", self.duration_sec, allow_zero=False)
        if self.fps is not None:
            _validate_positive_finite("fps", self.fps, allow_zero=False)
        if self.roi_size < 64 or self.roi_size % 16 != 0:
            raise ValueError(
                f"roi_size must be >= 64 and divisible by 16, got {self.roi_size}"
            )
        if self.crop_size is not None and (self.crop_size < 32 or self.crop_size % 2 != 0):
            raise ValueError(
                f"crop_size must be >= 32 and even when specified, got {self.crop_size}"
            )
        if self.roi_box is not None:
            if len(self.roi_box) != 4:
                raise ValueError(f"roi_box must have 4 ints (x1, y1, x2, y2), got {self.roi_box!r}")
            x1, y1, x2, y2 = self.roi_box
            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                raise ValueError(
                    f"Invalid roi_box {self.roi_box!r}: require 0 <= x1 < x2 and 0 <= y1 < y2"
                )
        if self.detector not in VALID_DETECTORS:
            raise ValueError(
                f"Invalid detector {self.detector!r}; expected one of {VALID_DETECTORS}"
            )
        if self.detector == "manual" and self.roi_box is None:
            raise ValueError("detector='manual' requires roi_box to be specified.")
        _validate_positive_finite("smooth_sigma", self.smooth_sigma, allow_zero=True)
        if not (0.0 < self.ncc_redetect_threshold <= 1.0):
            raise ValueError(
                f"ncc_redetect_threshold must be in (0, 1], got {self.ncc_redetect_threshold}"
            )
        if self.workers < 1:
            raise ValueError(f"workers must be >= 1, got {self.workers}")
        if self.qa_grid_samples < 1:
            raise ValueError(f"qa_grid_samples must be >= 1, got {self.qa_grid_samples}")


@dataclass
class SegmentConfig:
    """Configuration for Stage 2: chunked SAM2.1 lens segmentation & glare hotspot detection."""

    model_id: str = "facebook/sam2.1-hiera-large"
    chunk_size: int = 280
    chunk_overlap: int = 10
    erode_px: int = 5
    exact_erode_px: int = 1
    glare_dilate_px: int = 5
    glare_feather_sigma: float = 2.0
    temporal_radius: int = 1
    handoff_iou_threshold: float = 0.85
    glare_sensitivity: float = 1.0
    lens_points: dict[str, Any] | None = None
    device: str = "auto"
    workers: int = 8
    gpu_lock: Path | None = None
    write_qa_grid: bool = False
    qa_grid_samples: int = 24

    def __post_init__(self) -> None:
        if not self.model_id or not self.model_id.strip():
            raise ValueError("model_id must be non-empty.")
        if self.chunk_size < 2:
            raise ValueError(f"chunk_size must be >= 2, got {self.chunk_size}")
        if self.chunk_overlap < 1 or self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap must satisfy 1 <= chunk_overlap < chunk_size "
                f"({self.chunk_size}), got {self.chunk_overlap}"
            )
        if self.erode_px < 0:
            raise ValueError(f"erode_px must be >= 0, got {self.erode_px}")
        if self.exact_erode_px < 0:
            raise ValueError(f"exact_erode_px must be >= 0, got {self.exact_erode_px}")
        if self.glare_dilate_px < 0:
            raise ValueError(f"glare_dilate_px must be >= 0, got {self.glare_dilate_px}")
        _validate_positive_finite(
            "glare_feather_sigma", self.glare_feather_sigma, allow_zero=True
        )
        if self.temporal_radius < 0:
            raise ValueError(f"temporal_radius must be >= 0, got {self.temporal_radius}")
        if not (0.0 < self.handoff_iou_threshold <= 1.0):
            raise ValueError(
                f"handoff_iou_threshold must be in (0, 1], got {self.handoff_iou_threshold}"
            )
        _validate_positive_finite("glare_sensitivity", self.glare_sensitivity, allow_zero=False)
        _validate_device(self.device)
        if self.workers < 1:
            raise ValueError(f"workers must be >= 1, got {self.workers}")
        if self.gpu_lock is not None:
            self.gpu_lock = Path(self.gpu_lock)
        if self.qa_grid_samples < 1:
            raise ValueError(f"qa_grid_samples must be >= 1, got {self.qa_grid_samples}")


@dataclass
class ReferenceConfig:
    """Configuration for Stage 3: automatic clean-eye reference construction."""

    mode: Literal["auto", "image", "none"] = "auto"
    ref_image_path: Path | None = None
    max_clean_glare_pct: float = 3.0
    allow_mirror: bool = True
    min_lens_area_ratio: float = 0.7
    roi_size: int = 512

    def __post_init__(self) -> None:
        if self.mode not in VALID_REF_MODES:
            raise ValueError(f"Invalid mode {self.mode!r}; expected one of {VALID_REF_MODES}")
        if self.ref_image_path is not None:
            self.ref_image_path = Path(self.ref_image_path)
        if self.mode == "image" and self.ref_image_path is None:
            raise ValueError("ReferenceConfig(mode='image') requires ref_image_path.")
        if not (0.0 <= self.max_clean_glare_pct <= 100.0):
            raise ValueError(
                f"max_clean_glare_pct must be in [0, 100], got {self.max_clean_glare_pct}"
            )
        if not (0.0 < self.min_lens_area_ratio <= 1.0):
            raise ValueError(
                f"min_lens_area_ratio must be in (0, 1], got {self.min_lens_area_ratio}"
            )
        if self.roi_size < 64 or self.roi_size % 16 != 0:
            raise ValueError(
                f"roi_size must be >= 64 and divisible by 16, got {self.roi_size}"
            )


@dataclass
class InpaintConfig:
    """Configuration for Stage 4: resumable chunked Wan2.1-VACE video inpainting."""

    model_id: str = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    text_encoder_model_id: str | None = None
    quantize_4bit: bool = False
    size: int = 512
    chunk_len: int = 81
    min_overlap: int = 12
    mask_dilate: int = 5
    clamp_dilate: int = 0
    steps: int = 20
    guidance_scale: float = 5.0
    conditioning_scale: float = 1.0
    flow_shift: float = 3.0
    prompt: str = DEFAULT_PROMPT
    neg_prompt: str = DEFAULT_NEG_PROMPT
    seed: int = 42
    device: str = "auto"
    gpu_lock: Path | None = None
    embed_cache_dir: Path | None = None

    def __post_init__(self) -> None:
        if not self.model_id or not self.model_id.strip():
            raise ValueError("model_id must be non-empty.")
        if self.text_encoder_model_id is None:
            self.text_encoder_model_id = self.model_id
        if self.size < 64 or self.size % 16 != 0:
            raise ValueError(f"size must be >= 64 and divisible by 16, got {self.size}")
        if self.chunk_len < 5 or (self.chunk_len - 1) % 4 != 0:
            raise ValueError(
                f"chunk_len must be >= 5 and satisfy (chunk_len - 1) % 4 == 0, got {self.chunk_len}"
            )
        if self.min_overlap < 0 or self.min_overlap * 2 >= self.chunk_len:
            raise ValueError(
                f"min_overlap must satisfy 0 <= min_overlap < chunk_len / 2 "
                f"({self.chunk_len / 2:.1f}), got {self.min_overlap}"
            )
        if self.mask_dilate < 0:
            raise ValueError(f"mask_dilate must be >= 0, got {self.mask_dilate}")
        if self.clamp_dilate < 0:
            raise ValueError(f"clamp_dilate must be >= 0, got {self.clamp_dilate}")
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        _validate_positive_finite("guidance_scale", self.guidance_scale, allow_zero=True)
        _validate_positive_finite(
            "conditioning_scale", self.conditioning_scale, allow_zero=True
        )
        _validate_positive_finite("flow_shift", self.flow_shift, allow_zero=False)
        _validate_device(self.device)
        if self.gpu_lock is not None:
            self.gpu_lock = Path(self.gpu_lock)
        if self.embed_cache_dir is not None:
            self.embed_cache_dir = Path(self.embed_cache_dir)


@dataclass
class CompositeConfig:
    """Configuration for Stage 5: rim-safe bounded-RAM streaming compositing."""

    feather_px: float = 2.0
    bevel_band_px: int = 4
    lens_expand_px: int = 0
    color_match_strength: float = 0.0
    suppress_chroma_glint: bool = True
    rim_protect_strength: float = 1.0
    crf: int = 16
    preset: str = "medium"
    lossless: bool = False
    workers: int = 8
    batch_size: int = 32
    max_batch_mem_mb: int = 256
    compare: bool = False
    qa_grid: bool = False
    qa_grid_samples: int = 12
    save_frames_dir: Path | None = None

    def __post_init__(self) -> None:
        _validate_positive_finite("feather_px", self.feather_px, allow_zero=True)
        if self.bevel_band_px < 0:
            raise ValueError(f"bevel_band_px must be >= 0, got {self.bevel_band_px}")
        if self.lens_expand_px < 0:
            raise ValueError(f"lens_expand_px must be >= 0, got {self.lens_expand_px}")
        if not (0.0 <= self.color_match_strength <= 1.0):
            raise ValueError(
                f"color_match_strength must be in [0, 1], got {self.color_match_strength}"
            )
        if not (0.0 <= self.rim_protect_strength <= 1.0):
            raise ValueError(
                f"rim_protect_strength must be in [0, 1], got {self.rim_protect_strength}"
            )
        if not (0 <= self.crf <= 51):
            raise ValueError(f"crf must be in [0, 51], got {self.crf}")
        if self.preset not in VALID_X264_PRESETS:
            raise ValueError(
                f"Invalid x264 preset {self.preset!r}; expected one of {VALID_X264_PRESETS}"
            )
        if self.workers < 1:
            raise ValueError(f"workers must be >= 1, got {self.workers}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.max_batch_mem_mb < 32:
            raise ValueError(f"max_batch_mem_mb must be >= 32, got {self.max_batch_mem_mb}")
        if self.qa_grid_samples < 1:
            raise ValueError(f"qa_grid_samples must be >= 1, got {self.qa_grid_samples}")
        if self.save_frames_dir is not None:
            self.save_frames_dir = Path(self.save_frames_dir)


@dataclass
class PipelineConfig:
    """End-to-end configuration for the full eyeglass glare removal pipeline."""

    input_path: Path
    output_path: Path
    work_dir: Path | None = None
    resume: bool = True
    prep: PrepConfig = field(default_factory=PrepConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    inpaint: InpaintConfig = field(default_factory=InpaintConfig)
    composite: CompositeConfig = field(default_factory=CompositeConfig)

    def __post_init__(self) -> None:
        self.input_path = Path(self.input_path)
        self.output_path = Path(self.output_path)
        if self.work_dir is not None:
            self.work_dir = Path(self.work_dir)
        else:
            self.work_dir = self.output_path.parent / f"{self.output_path.stem}.work"

        # Prevent output or derived compare/qa artifacts from clobbering input media
        derived_compare = (
            self.output_path.with_name(f"{self.output_path.stem}_compare.mp4")
            if self.composite.compare
            else None
        )
        derived_qa = (
            self.output_path.with_name(f"{self.output_path.stem}_qa_grid.png")
            if self.composite.qa_grid
            else None
        )
        assert_distinct_output_paths(
            self.input_path,
            [
                ("output", self.output_path),
                ("compare", derived_compare),
                ("qa_grid", derived_qa),
            ],
        )

        if self.reference.roi_size != self.prep.roi_size:
            raise ValueError(
                f"reference.roi_size ({self.reference.roi_size}) must match "
                f"prep.roi_size ({self.prep.roi_size})"
            )
