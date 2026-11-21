"""Safe ffprobe/ffmpeg wrappers for media probing, CFR extraction, and streaming encode."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from eyeglass_glare_removal._workspace import (
    atomic_swap_directory,
    ensure_confined_dir,
    make_unique_temp_dir,
    make_unique_temp_sibling,
    reject_symlink_path,
)

logger = logging.getLogger(__name__)

_SAFE_COLOR_VALUES = frozenset(
    {
        "bt709",
        "bt470m",
        "bt470bg",
        "smpte170m",
        "smpte240m",
        "film",
        "bt2020",
        "bt2020nc",
        "bt2020c",
        "bt2020-10",
        "bt2020-12",
        "smpte2084",
        "arib-std-b67",
        "iec61966-2-1",
        "fcc",
        "linear",
        "rgb",
    }
)
_SAR_PATTERN = re.compile(r"^([1-9]\d*):([1-9]\d*)$")


@dataclass
class VideoMetadata:
    """Probed media stream properties with display-matrix rotation and stream indices."""

    path: str
    width: int
    height: int
    display_width: int
    display_height: int
    rotation: int
    fps: float
    is_vfr: bool
    duration_sec: float | None
    has_audio: bool
    video_stream_index: int = 0
    audio_stream_index: int | None = None
    sample_aspect_ratio: str | None = None
    audio_codec: str | None = None
    video_codec: str | None = None
    pix_fmt: str | None = None
    color_primaries: str | None = None
    color_trc: str | None = None
    colorspace: str | None = None
    color_range: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def require_binary(name: str) -> str:
    """Locate an external binary on PATH or raise a clear RuntimeError."""
    found = shutil.which(name)
    if not found:
        raise RuntimeError(
            f"Required binary '{name}' not found on PATH. "
            f"Please install ffmpeg (which includes ffprobe) and retry."
        )
    return found


def ffmpeg_local_path(path: Path | str, *, must_exist: bool = False) -> str:
    """
    Normalize a local filesystem path into an unambiguous FFmpeg `file:` URI.

    Resolving to an absolute path and prefixing `file:` (paired with `-protocol_whitelist`)
    ensures FFmpeg/FFprobe cannot misparse leading `-` as options, literal `file:` basenames
    as protocol redirects, or embedded colons as network protocols.
    """
    p = Path(path)
    resolved = p.resolve(strict=must_exist)
    return f"file:{resolved}"


def run_subprocess(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Execute a subprocess using a list of arguments (never shell=True)."""
    if not isinstance(cmd, list) or not cmd:
        raise ValueError("cmd must be a non-empty list of strings.")
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if check and proc.returncode != 0:
        stderr_tail = (proc.stderr or "").strip()[-1000:]
        raise RuntimeError(
            f"Command failed (rc={proc.returncode}): {' '.join(cmd[:6])} ... | stderr: {stderr_tail}"
        )
    return proc


def parse_rate(rate_str: str | None) -> float | None:
    """Parse an ffprobe rational frame-rate string such as '30000/1001' or '30/1'."""
    if not rate_str:
        return None
    rate_str = rate_str.strip()
    if not rate_str or rate_str in ("0/0", "N/A"):
        return None
    try:
        frac = Fraction(rate_str)
    except (ValueError, ZeroDivisionError):
        return None
    if frac <= 0:
        return None
    val = float(frac)
    if not math.isfinite(val) or val <= 0:
        return None
    return val


def normalize_rotation(raw_rotation: Any) -> int:
    """Normalize display-matrix or tag rotation degrees into {0, 90, 180, 270}."""
    if raw_rotation is None:
        return 0
    try:
        deg = float(raw_rotation)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(deg):
        return 0
    snapped = int(round(deg / 90.0) * 90) % 360
    return snapped


def extract_stream_rotation(video_stream: dict[str, Any]) -> int:
    """Extract rotation from ffprobe side_data_list Display Matrix or tags.rotate."""
    for sd in video_stream.get("side_data_list") or []:
        if not isinstance(sd, dict):
            continue
        if "rotation" in sd:
            return normalize_rotation(sd["rotation"])
    tags = video_stream.get("tags") or {}
    if isinstance(tags, dict) and "rotate" in tags:
        return normalize_rotation(tags["rotate"])
    return 0


def _sanitize_sar(raw_sar: Any) -> str | None:
    if not isinstance(raw_sar, str):
        return None
    s = raw_sar.strip()
    if not _SAR_PATTERN.match(s):
        return None
    return s


def _sanitize_color_tag(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    v = raw.strip().lower()
    if v in _SAFE_COLOR_VALUES:
        return v
    return None


def _select_primary_video_stream(streams: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Select the primary real video stream explicitly, excluding attached cover-art pictures.

    Preference order: default disposition -> largest pixel area -> lowest stream index.
    """
    candidates = []
    for pos, s in enumerate(streams):
        if s.get("codec_type") != "video":
            continue
        disp = s.get("disposition") or {}
        if int(disp.get("attached_pic", 0) or 0) == 1:
            continue
        w = int(s.get("width") or 0)
        h = int(s.get("height") or 0)
        idx = int(s.get("index", pos))
        is_default = int(disp.get("default", 0) or 0) == 1
        candidates.append((is_default, w * h, -idx, s))

    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    return candidates[0][3]


def _select_primary_audio_stream(streams: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the primary audio stream (default disposition first, else lowest stream index)."""
    candidates = []
    for pos, s in enumerate(streams):
        if s.get("codec_type") != "audio":
            continue
        disp = s.get("disposition") or {}
        idx = int(s.get("index", pos))
        is_default = int(disp.get("default", 0) or 0) == 1
        candidates.append((is_default, -idx, s))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return candidates[0][2]


def parse_ffprobe_payload(path: Path | str, payload: dict[str, Any]) -> VideoMetadata:
    """Parse an ffprobe JSON payload into a validated VideoMetadata object."""
    streams = payload.get("streams") or []
    fmt = payload.get("format") or {}

    video_stream = _select_primary_video_stream(streams)
    if video_stream is None:
        raise ValueError(f"No video stream found in media file: {path}")

    audio_stream = _select_primary_audio_stream(streams)

    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid video dimensions ({width}x{height}) in: {path}")

    video_stream_index = int(video_stream.get("index", 0))
    audio_stream_index = (
        int(audio_stream.get("index", 1)) if audio_stream is not None else None
    )

    rotation = extract_stream_rotation(video_stream)
    if rotation in (90, 270):
        display_w, display_h = height, width
    else:
        display_w, display_h = width, height

    avg_fps = parse_rate(video_stream.get("avg_frame_rate"))
    r_fps = parse_rate(video_stream.get("r_frame_rate"))

    if avg_fps is not None and r_fps is not None:
        is_vfr = abs(avg_fps - r_fps) / max(avg_fps, r_fps) > 0.02
        fps = avg_fps
    elif avg_fps is not None:
        is_vfr = False
        fps = avg_fps
    elif r_fps is not None:
        is_vfr = False
        fps = r_fps
    else:
        is_vfr = False
        fps = 30.0

    if abs(fps - round(fps)) < 0.005:
        fps = float(round(fps))
    else:
        fps = round(fps, 3)

    dur_raw = video_stream.get("duration") or fmt.get("duration")
    duration_sec: float | None = None
    if dur_raw not in (None, "N/A", ""):
        try:
            d = float(dur_raw)
            if math.isfinite(d) and d > 0:
                duration_sec = d
        except (TypeError, ValueError):
            duration_sec = None

    return VideoMetadata(
        path=str(path),
        width=width,
        height=height,
        display_width=display_w,
        display_height=display_h,
        rotation=rotation,
        fps=fps,
        is_vfr=is_vfr,
        duration_sec=duration_sec,
        has_audio=audio_stream is not None,
        video_stream_index=video_stream_index,
        audio_stream_index=audio_stream_index,
        sample_aspect_ratio=_sanitize_sar(video_stream.get("sample_aspect_ratio")),
        audio_codec=audio_stream.get("codec_name") if audio_stream else None,
        video_codec=video_stream.get("codec_name"),
        pix_fmt=video_stream.get("pix_fmt"),
        color_primaries=_sanitize_color_tag(video_stream.get("color_primaries")),
        color_trc=_sanitize_color_tag(video_stream.get("color_transfer")),
        colorspace=_sanitize_color_tag(video_stream.get("color_space")),
        color_range=video_stream.get("color_range"),
    )


def probe_video(path: Path | str) -> VideoMetadata:
    """Run ffprobe on a local video file and return structured VideoMetadata."""
    video_path = Path(path)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    ffprobe = require_binary("ffprobe")
    proc = run_subprocess(
        [
            ffprobe,
            "-v",
            "error",
            "-protocol_whitelist",
            "file",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            ffmpeg_local_path(video_path, must_exist=True),
        ]
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse ffprobe JSON for {video_path}: {exc}") from exc
    return parse_ffprobe_payload(video_path, payload)


def probe_audio_has_packets(audio_path: Path) -> bool:
    """
    Verify an extracted audio file contains a valid audio stream with >= 1 decoded packet
    or positive stream duration (preventing empty/packetless `.m4a` files after trims).
    """
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        return False
    ffprobe = require_binary("ffprobe")
    proc = run_subprocess(
        [
            ffprobe,
            "-v",
            "error",
            "-protocol_whitelist",
            "file",
            "-print_format",
            "json",
            "-select_streams",
            "a:0",
            "-count_packets",
            "-show_streams",
            ffmpeg_local_path(audio_path, must_exist=True),
        ],
        check=False,
    )
    if proc.returncode != 0:
        return False
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return False
    streams = payload.get("streams") or []
    if not streams:
        return False
    st = streams[0]
    if st.get("codec_type") != "audio":
        return False
    for key in ("nb_read_packets", "nb_packets"):
        raw = st.get(key)
        if raw not in (None, "N/A", ""):
            try:
                if int(raw) > 0:
                    return True
                return False
            except ValueError:
                pass
    raw_dur = st.get("duration")
    if raw_dur not in (None, "N/A", ""):
        try:
            return float(raw_dur) > 0.0
        except ValueError:
            return False
    return False


def format_fps_arg(fps: float) -> str:
    """Format a frame rate cleanly for ffmpeg arguments."""
    if abs(fps - round(fps)) < 1e-4:
        return str(int(round(fps)))
    return f"{fps:.6f}".rstrip("0").rstrip(".")


def extract_cfr_frames(
    source_video: Path | str,
    out_dir: Path,
    *,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    target_fps: float | None = None,
    metadata: VideoMetadata | None = None,
) -> tuple[list[Path], Path | None, VideoMetadata, float]:
    """
    Extract display-oriented CFR PNG frames and optional audio track from source_video.

    - Uses unambiguous `file:` normalized local paths + `-protocol_whitelist file`.
    - Maps the exact probed primary video/audio stream index (`-map 0:<idx>`).
    - Preserves 1-frame and short clips across FFmpeg 6/7+ (`eof_action=pass` + 1-frame fallback).
    - Verifies extracted audio actually contains valid packets; handles silent videos cleanly.
    - Publishes frames atomically via a unique staging directory (rejecting symlinked workdirs).
    """
    ffmpeg = require_binary("ffmpeg")
    source_path = Path(source_video)
    meta = metadata or probe_video(source_path)

    fps = float(target_fps) if target_fps is not None else float(meta.fps)
    fps_str = format_fps_arg(fps)

    out_dir = Path(out_dir)
    reject_symlink_path(out_dir, "prep output directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = ensure_confined_dir(out_dir, out_dir / "frames", create=False)

    is_full = duration_sec is None
    if not is_full and duration_sec is not None and duration_sec <= 0:
        raise ValueError(f"duration_sec must be > 0 or None, got {duration_sec}")

    src_arg = ffmpeg_local_path(source_path, must_exist=source_path.exists())

    # 1. Extract and verify audio only if source has an audio stream
    audio_path: Path | None = None
    final_audio = out_dir / "audio.m4a"
    reject_symlink_path(final_audio, "audio output file")
    if meta.has_audio and meta.audio_stream_index is not None:
        tmp_audio = make_unique_temp_sibling(final_audio)
        try:
            can_stream_copy = (meta.audio_codec == "aac") and (is_full or start_sec == 0.0)
            audio_cmd = [ffmpeg, "-y", "-v", "error", "-protocol_whitelist", "file"]
            if start_sec > 0.0:
                audio_cmd += ["-ss", f"{start_sec:.6f}"]
            audio_cmd += ["-i", src_arg]
            if not is_full and duration_sec is not None:
                audio_cmd += ["-t", f"{duration_sec:.6f}"]
            audio_cmd += ["-vn", "-map", f"0:{meta.audio_stream_index}"]
            if can_stream_copy:
                audio_cmd += ["-c:a", "copy"]
            else:
                audio_cmd += ["-c:a", "aac", "-b:a", "192k"]
            audio_cmd += [ffmpeg_local_path(tmp_audio)]

            proc_a = run_subprocess(audio_cmd, check=False)
            if proc_a.returncode == 0 and probe_audio_has_packets(tmp_audio):
                os.replace(tmp_audio, final_audio)
                audio_path = final_audio
            else:
                logger.warning(
                    "Audio extraction yielded no usable audio packets (rc=%s); continuing audio-less.",
                    proc_a.returncode,
                )
                tmp_audio.unlink(missing_ok=True)
                if final_audio.exists():
                    final_audio.unlink()
        except Exception:
            tmp_audio.unlink(missing_ok=True)
            raise
    else:
        if final_audio.exists():
            final_audio.unlink()

    # 2. Extract auto-rotated CFR PNG frames into a unique staging directory, then swap atomically
    stage_frames_dir = make_unique_temp_dir(out_dir, prefix=".stage_frames_")
    try:
        out_pattern = ffmpeg_local_path(stage_frames_dir / "%06d.png")
        fps_filter = f"fps=fps={fps_str}:round=near:eof_action=pass"

        frame_cmd = [ffmpeg, "-y", "-v", "error", "-protocol_whitelist", "file"]
        if start_sec > 0.0:
            frame_cmd += ["-ss", f"{start_sec:.6f}"]
        frame_cmd += ["-i", src_arg, "-map", f"0:{meta.video_stream_index}"]

        if not is_full and duration_sec is not None:
            target_frames = max(1, int(round(duration_sec * fps)))
            frame_cmd += [
                "-t",
                f"{duration_sec + 0.25:.6f}",
                "-vf",
                fps_filter,
                "-frames:v",
                str(target_frames),
            ]
        else:
            frame_cmd += ["-vf", fps_filter]

        frame_cmd += ["-start_number", "0", out_pattern]
        run_subprocess(frame_cmd)

        staged_pngs = sorted(stage_frames_dir.glob("*.png"))
        if not staged_pngs:
            # Defensive 1-frame fallback when fps filter drops a single-frame clip at trim/EOF
            fallback_cmd = [ffmpeg, "-y", "-v", "error", "-protocol_whitelist", "file"]
            if start_sec > 0.0:
                fallback_cmd += ["-ss", f"{start_sec:.6f}"]
            fallback_cmd += [
                "-i",
                src_arg,
                "-map",
                f"0:{meta.video_stream_index}",
                "-frames:v",
                "1",
                "-start_number",
                "0",
                out_pattern,
            ]
            run_subprocess(fallback_cmd, check=False)
            staged_pngs = sorted(stage_frames_dir.glob("*.png"))

        if not staged_pngs:
            raise RuntimeError(
                f"No video frames extracted from {source_path} "
                f"(start_sec={start_sec}, duration_sec={duration_sec}). "
                "Check that --start is within the video duration."
            )

        atomic_swap_directory(stage_frames_dir, frames_dir)
    except Exception:
        if stage_frames_dir.exists():
            shutil.rmtree(stage_frames_dir, ignore_errors=True)
        raise

    frame_paths = sorted(frames_dir.glob("*.png"))
    return frame_paths, audio_path, meta, fps


def build_video_writer_cmd(
    width: int,
    height: int,
    fps: float,
    out_mp4: Path,
    *,
    audio_path: Path | None = None,
    crf: int = 16,
    preset: str = "medium",
    lossless: bool = False,
    num_frames: int | None = None,
    sample_aspect_ratio: str | None = None,
    color_primaries: str | None = None,
    color_trc: str | None = None,
    colorspace: str | None = None,
    ffmpeg_bin: str = "ffmpeg",
) -> list[str]:
    """
    Build the ffmpeg command list for streaming RGB24 frames to MP4.

    - Uses `-protocol_whitelist file,pipe`, `-i pipe:0`, and `file:` local paths.
    - Never uses `-shortest` (prevents short audio tracks from truncating valid video);
      uses the CFR video clock (`-t num_frames/fps`) to cap trailing audio cleanly.
    - Preserves non-1:1 SAR (`setsar`) and source color signaling when known.
    - Supports `lossless=True` (`libx264rgb -crf 0 -pix_fmt rgb24`) for decoded pixel equality.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid output dimensions: {width}x{height}")

    fps_str = format_fps_arg(fps)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-v",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        fps_str,
        "-i",
        "pipe:0",
    ]
    has_audio = (
        audio_path is not None
        and Path(audio_path).exists()
        and Path(audio_path).stat().st_size > 0
    )
    if has_audio:
        assert audio_path is not None
        cmd += ["-i", ffmpeg_local_path(audio_path, must_exist=True)]

    vf_parts: list[str] = []
    if (not lossless) and ((width % 2 != 0) or (height % 2 != 0)):
        vf_parts.append("pad=ceil(iw/2)*2:ceil(ih/2)*2")
    sar = _sanitize_sar(sample_aspect_ratio)
    if sar is not None and sar != "1:1":
        vf_parts.append(f"setsar={sar.replace(':', '/')}")
    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]

    cmd += ["-map", "0:v:0"]
    if has_audio:
        cmd += ["-map", "1:a:0"]

    if lossless:
        cmd += [
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            "-preset",
            preset,
            "-pix_fmt",
            "rgb24",
        ]
    else:
        cp = _sanitize_color_tag(color_primaries) or "bt709"
        ct = _sanitize_color_tag(color_trc) or "bt709"
        cs = _sanitize_color_tag(colorspace) or "bt709"
        cmd += [
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
            "-color_primaries",
            cp,
            "-color_trc",
            ct,
            "-colorspace",
            cs,
        ]

    if has_audio:
        cmd += ["-c:a", "copy"]
        if num_frames is not None and num_frames > 0 and fps > 0:
            # Cap output to authoritative CFR video duration (never truncate video via -shortest!)
            vid_dur = float(num_frames) / float(fps)
            cmd += ["-t", f"{vid_dur:.6f}"]

    cmd += ["-movflags", "+faststart", ffmpeg_local_path(out_mp4)]
    return cmd


def open_video_writer_pipe(
    width: int,
    height: int,
    fps: float,
    out_mp4: Path,
    *,
    audio_path: Path | None = None,
    crf: int = 16,
    preset: str = "medium",
    lossless: bool = False,
    num_frames: int | None = None,
    sample_aspect_ratio: str | None = None,
    color_primaries: str | None = None,
    color_trc: str | None = None,
    colorspace: str | None = None,
) -> subprocess.Popen[bytes]:
    """Open an ffmpeg stdin pipe expecting contiguous RGB24 frame buffers."""
    ffmpeg = require_binary("ffmpeg")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_video_writer_cmd(
        width=width,
        height=height,
        fps=fps,
        out_mp4=out_mp4,
        audio_path=audio_path,
        crf=crf,
        preset=preset,
        lossless=lossless,
        num_frames=num_frames,
        sample_aspect_ratio=sample_aspect_ratio,
        color_primaries=color_primaries,
        color_trc=color_trc,
        colorspace=colorspace,
        ffmpeg_bin=ffmpeg,
    )
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def close_video_writer_pipe(proc: subprocess.Popen[bytes], label: str = "video") -> None:
    """Close stdin, wait for ffmpeg exit, and raise a descriptive error on non-zero rc."""
    if proc.stdin is not None and not proc.stdin.closed:
        proc.stdin.close()
    err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    if proc.stderr is not None and not proc.stderr.closed:
        proc.stderr.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg ({label}) failed with rc={rc}: {err.strip()}")
