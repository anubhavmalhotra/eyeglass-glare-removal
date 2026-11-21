# eyeglass-glare-removal

General-purpose, open-source Python package and CLI (`glare-remove`) for removing eyeglass reflections and specular glare from single-subject talking-head, portrait, landscape, or square videos while preserving the subject's eyes, natural blinking, eyeglass frames, and pre-encode RGB pixels outside the active lens/bevel edit mask bit-for-bit.

Licensed under **Apache-2.0**.

---

## Architecture Overview

```text
Input Video (portrait / landscape / square, rotated or VFR, with or without audio)
  │
  ▼
[1] Prep & Adaptive ROI Tracking (`detect.py`, `media.py`)
    • Auto-rotates display-matrix rotation (-90/90/180/270) & conforms VFR -> CFR via ffmpeg
    • Auto-detects face & eyes via MediaPipe (if installed) or built-in OpenCV Haar cascades
      (or accepts explicit `--roi-box`)
    • Sizes square eye ROI dynamically from inter-ocular distance & tracks smoothly with
      glare-masked gradient NCC + Gaussian temporal smoothing + low-NCC re-detection guard
  │
  ▼
[2] Chunked SAM2.1 Lens Segmentation & Photometric Glare Detection (`segment.py`)
    • Auto-seeds left/right lens prompts from detected eye geometry (or `--lens-points`)
    • Tracks lenses in overlapping VRAM-safe chunks with IoU boundary verification &
      input_masks fallback + atomic `.npz` checkpoints
    • Adaptive per-frame skin-tone estimation outside lenses detects cool/white screen glare,
      warm lamp reflections, AND green/purple AR-coating reflections across all skin tones
    • Emits `roi_lens_interior`, tight `roi_lens_exact` (1px erode), and `roi_glare_hotspot`
  │
  ▼
[3] Automatic Clean-Eye Reference Builder (`reference.py`)
    • `--ref-image <photo>`: auto-crops user-provided clean photo to match ROI scale
    • `auto` (default): scans glare coverage across the video; picks cleanest single frame,
      stitches clean left+right frames across the bridge, mirrors a single clean eye across
      the nose bridge, or falls back cleanly to reference-free VACE (never feeds a glared
      frame as a clean reference)
    • `--no-ref`: explicit reference-free mode
  │
  ▼
[4] Resumable Chunked Wan2.1-VACE Video Inpainting (`inpaint.py`)
    • 2-stage VRAM-safe loading: caches UMT5 prompt embeddings & unloads text encoder
      before loading `AutoencoderKLWan` (fp32) + `WanVACETransformer3DModel` (bf16 / 4-bit NF4)
    • Exact `4k + 1` frame padding/unpadding for ANY clip length (<81, 81, or >81 frames)
    • Skips zero-glare chunks, writes atomic per-chunk checkpoints, and streams exact
      sum-to-1.0 raised-cosine overlap blending
  │
  ▼
[5] Rim-Safe Bounded-RAM Streaming Compositor (`composite.py`)
    • General morphological + photometric rim protection on both sides of the lens boundary
      (dark/black/bronze, bright silver/gold metallic wires, tortoiseshell/acetate, rimless)
    • Tight inward feathering + glare-gated edit support (zero-glare frames & clean lenses
      bypass compositing untouched) + optional `--suppress-chroma-glint`
    • Pre-encode RGB composite guarantees `max_outside_diff == 0` outside the active edit
      support mask; default encode uses H.264 `yuv420p` (`--crf 16`, preserving source SAR &
      color signaling) for broad player compatibility, or `--lossless` (`libx264rgb -crf 0`)
      when decoded bit-exact RGB equality is required
    • Streams RGB24 directly to ffmpeg via atomic sibling temp files (`+faststart`), cleanly
      handling audio-less or short-audio inputs without truncating video, plus `--compare` / `--qa-grid`
```

---

## Installation

### Prerequisites

- Python **3.10+**
- `ffmpeg` and `ffprobe` installed and available on `PATH`
- A CUDA GPU (recommended; see [Hardware Requirements](#hardware-requirements))

### Install Package

```bash
git clone https://github.com/anubhavmalhotra/eyeglass-glare-removal.git
cd eyeglass-glare-removal

# Core package (PyTorch, Diffusers, Transformers, OpenCV, SciPy, Pillow)
pip install -e .

# Optional: add MediaPipe face/eye landmark detector (falls back cleanly to OpenCV Haar if omitted)
pip install -e ".[mediapipe]"

# Optional: add bitsandbytes for --quantize-4bit VACE inference
pip install -e ".[quant]"

# Development & test dependencies
pip install -e ".[dev]"
```

---

## Quickstart & Recommended `--preview` Workflow

Before processing a long video, run a fast **5-second preview** with `--compare` and `--qa-grid` to verify ROI tracking, lens masks, glare detection, and inpainting quality:

```bash
# 1. Fast 5-second preview with visual QA grids and side-by-side comparison video
glare-remove run input.mp4 -o preview_clean.mp4 \
  --preview 5 \
  --work-dir work/preview \
  --compare \
  --qa-grid

# Inspect:
#   work/preview/prep/roi_track_qa_grid.jpg   (ROI centering stability)
#   work/preview/masks/mask_qa_grid.png       (green lens boundary, red glare mask)
#   preview_clean_qa_grid.png                 (zoomed before/after contact sheet)
#   preview_clean_compare.mp4                 (side-by-side original vs cleaned)
```

Once the preview looks sharp, process the full video with automatic chunk-level resumability (`--resume` is enabled by default):

```bash
# 2. Full video run (resumable if interrupted at any stage or chunk)
glare-remove run input.mp4 -o cleaned.mp4 \
  --work-dir work/full \
  --compare \
  --qa-grid
```

### Common Options

```bash
# Supply a clean reference photo of the subject (auto-cropped around eyes)
glare-remove run input.mp4 -o cleaned.mp4 --ref-image clean_portrait.jpg

# Force reference-free inpainting
glare-remove run input.mp4 -o cleaned.mp4 --no-ref

# Override initial ROI box manually (display-oriented x1,y1,x2,y2 pixels)
glare-remove run input.mp4 -o cleaned.mp4 --roi-box 140,220,460,540

# Reduce VRAM usage with 4-bit NF4 transformer quantization & GPU lock
glare-remove run input.mp4 -o cleaned.mp4 --quantize-4bit --gpu-lock /tmp/glare_gpu.lock

# Validate configuration and inspect resolved execution plan without running models
glare-remove run input.mp4 -o cleaned.mp4 --dry-run
```

### Modular Subcommands

Every pipeline stage can also be executed or re-run independently against a shared `--work-dir`:

```bash
glare-remove prep input.mp4 --work-dir work/clip --qa-grid
glare-remove segment --work-dir work/clip --glare-sensitivity 1.1 --qa-grid
glare-remove reference --work-dir work/clip
glare-remove inpaint --work-dir work/clip --steps 20 --guidance-scale 5.0
glare-remove composite --work-dir work/clip -o cleaned.mp4 --compare --qa-grid
```

You can also invoke the CLI via `python -m eyeglass_glare_removal ...`.

---

## Hardware Requirements

| Stage | Default Model | Peak VRAM (approx.) | Notes |
| --- | --- | --- | --- |
| Stage 1 (`prep`) | MediaPipe / OpenCV Haar | CPU only | Multi-threaded ROI export |
| Stage 2 (`segment`) | `facebook/sam2.1-hiera-large` | ~5–6 GiB | Chunked (`--sam2-chunk-size 280`) keeps VRAM bounded on arbitrarily long videos |
| Stage 3 (`reference`) | CPU photometric scan | CPU only | Automatic clean-eye selection / stitching / mirroring |
| Stage 4 (`inpaint`) | `Wan-AI/Wan2.1-VACE-1.3B-diffusers` | ~14–16 GiB (bf16) / ~10–12 GiB (`--quantize-4bit`) | 2-stage UMT5 precompute+unload avoids holding text encoder + DiT simultaneously |
| Stage 5 (`composite`) | OpenCV + ffmpeg pipe | CPU only, O(1) RAM | Streams RGB24 directly to `ffmpeg` without writing full-res intermediate PNGs |

- Larger models (e.g., `Wan-AI/Wan2.1-VACE-14B-diffusers`) are supported via `--vace-model` (typically combined with `--quantize-4bit` on high-VRAM GPUs).
- CPU and Apple Silicon MPS devices are detected cleanly via `--device auto|cpu|mps`, though Stage 2 and Stage 4 are significantly faster on CUDA GPUs.

---

## NVIDIA Blackwell (`sm_121` / GB10 / DGX Spark) & ARM64 Notes

- **Mandatory SDPA Guard:** On NVIDIA Blackwell `sm_121` GPUs (such as GB10 / DGX Spark), PyTorch's `mem_efficient_sdp` kernel can silently corrupt attention outputs. Both `segment.py` and `inpaint.py` automatically call:
  ```python
  torch.backends.cuda.enable_flash_sdp(True)
  torch.backends.cuda.enable_math_sdp(True)
  torch.backends.cuda.enable_mem_efficient_sdp(False)
  ```
  whenever CUDA is available (safe no-op on CPU/MPS).
- **Shared GPU Serialization:** Pass `--gpu-lock /tmp/glare_gpu.lock` on multi-job systems to serialize SAM2.1 and VACE GPU stages cleanly via `fcntl.flock`.
- **ARM64 (aarch64) Linux:** `opencv-python-headless`, `scipy`, `Pillow`, `torch` (CUDA 13+ wheels), `transformers`, `diffusers`, and `bitsandbytes` run natively on ARM64 Linux; if `mediapipe` wheels are unavailable for your ARM64 Python version, omit `[mediapipe]` and the pipeline automatically uses the built-in OpenCV Haar cascade detector.

---

## Development & Testing

```bash
pytest
ruff check src tests
```

The test suite runs completely offline using synthetic frames, mocked subprocesses, and dependency-injected chunk runners.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
