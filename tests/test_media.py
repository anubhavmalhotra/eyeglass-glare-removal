"""Unit tests for media probing, stream selection, protocol safety, audio, and encoding."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from eyeglass_glare_removal import media


def test_parse_rate_variants() -> None:
    assert media.parse_rate("30/1") == pytest.approx(30.0)
    assert media.parse_rate("30000/1001") == pytest.approx(29.97002997, rel=1e-5)
    assert media.parse_rate("0/0") is None
    assert media.parse_rate("N/A") is None
    assert media.parse_rate(None) is None
    assert media.parse_rate("-30/1") is None


def test_normalize_rotation() -> None:
    assert media.normalize_rotation(-90) == 270
    assert media.normalize_rotation(90) == 90
    assert media.normalize_rotation(180) == 180
    assert media.normalize_rotation(-180) == 180
    assert media.normalize_rotation(270) == 270
    assert media.normalize_rotation(0) == 0
    assert media.normalize_rotation(None) == 0


def test_ffmpeg_local_path_neutralizes_dashes_and_protocols(tmp_path: Path) -> None:
    dash_file = tmp_path / "-dash.mp4"
    dash_file.write_bytes(b"x")
    arg1 = media.ffmpeg_local_path(dash_file, must_exist=True)
    assert arg1.startswith("file:/")
    assert arg1.endswith("/-dash.mp4")

    proto_file = tmp_path / "file:redirect.mp4"
    proto_file.write_bytes(b"x")
    arg2 = media.ffmpeg_local_path(proto_file, must_exist=True)
    assert arg2 == f"file:{proto_file.resolve()}"


def test_parse_ffprobe_rotated_and_primary_stream_selection() -> None:
    # Stream 0 is an attached cover-art picture; stream 1 is low-res aux; stream 2 is main default HD!
    payload = {
        "format": {"duration": "12.500"},
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "mjpeg",
                "width": 500,
                "height": 500,
                "disposition": {"attached_pic": 1},
            },
            {
                "index": 1,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 320,
                "height": 180,
                "avg_frame_rate": "15/1",
                "r_frame_rate": "15/1",
                "disposition": {"default": 0},
            },
            {
                "index": 2,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1280,
                "height": 720,
                "sample_aspect_ratio": "4:3",
                "avg_frame_rate": "29500/1000",
                "r_frame_rate": "30/1",
                "pix_fmt": "yuv420p",
                "color_primaries": "smpte170m",
                "color_transfer": "smpte170m",
                "color_space": "smpte170m",
                "disposition": {"default": 1},
                "side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}],
            },
            {
                "index": 3,
                "codec_type": "audio",
                "codec_name": "aac",
                "disposition": {"default": 1},
            },
        ],
    }
    meta = media.parse_ffprobe_payload("clip.mp4", payload)
    assert meta.video_stream_index == 2
    assert meta.audio_stream_index == 3
    assert meta.width == 1280
    assert meta.height == 720
    assert meta.rotation == 270
    assert meta.display_width == 720
    assert meta.display_height == 1280
    assert meta.sample_aspect_ratio == "4:3"
    assert meta.color_primaries == "smpte170m"
    assert meta.has_audio is True


def test_parse_ffprobe_vfr_flag_and_tag_rotation_and_no_audio() -> None:
    payload = {
        "format": {"duration": "5.0"},
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "hevc",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "24/1",
                "r_frame_rate": "60/1",
                "tags": {"rotate": "90"},
            }
        ],
    }
    meta = media.parse_ffprobe_payload("silent.mov", payload)
    assert meta.rotation == 90
    assert meta.display_width == 1080
    assert meta.display_height == 1920
    assert meta.is_vfr is True
    assert meta.fps == pytest.approx(24.0)
    assert meta.has_audio is False
    assert meta.audio_stream_index is None


def test_parse_ffprobe_missing_video_stream_raises() -> None:
    with pytest.raises(ValueError, match="No video stream"):
        media.parse_ffprobe_payload(
            "cover_only.mp3",
            {"streams": [{"codec_type": "video", "disposition": {"attached_pic": 1}}]},
        )


def test_require_binary_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="not found on PATH"):
        media.require_binary("ffmpeg")


def test_extract_cfr_frames_maps_stream_checks_audio_packets_and_1frame_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorded_cmds: list[list[str]] = []

    def fake_run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        recorded_cmds.append(cmd)
        out_arg = cmd[-1]
        if out_arg.endswith("%06d.png"):
            raw_path = out_arg[5:] if out_arg.startswith("file:") else out_arg
            fdir = Path(raw_path).parent
            fdir.mkdir(parents=True, exist_ok=True)
            # Simulate FFmpeg fps filter dropping frame on first try only when testing fallback
            if any("fps=fps=10" in c for c in cmd):
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            for i in range(3):
                (fdir / f"{i:06d}.png").write_bytes(b"fake_png")
        elif ".m4a" in out_arg:
            raw_path = out_arg[5:] if out_arg.startswith("file:") else out_arg
            Path(raw_path).write_bytes(b"fake_aac_bytes")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(media, "require_binary", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(media, "run_subprocess", fake_run)

    # 1) Opus audio with valid packets -> transcoded to AAC, explicit -map 0:<idx> & protocol_whitelist
    monkeypatch.setattr(media, "probe_audio_has_packets", lambda _p: True)
    src = tmp_path / "web.mkv"
    src.write_bytes(b"mkv")
    opus_meta = media.VideoMetadata(
        path=str(src),
        width=1280,
        height=720,
        display_width=1280,
        display_height=720,
        rotation=0,
        fps=25.0,
        is_vfr=False,
        duration_sec=5.0,
        has_audio=True,
        video_stream_index=2,
        audio_stream_index=3,
        audio_codec="opus",
    )
    frames, audio_p, _, _ = media.extract_cfr_frames(
        src, tmp_path / "out_opus", duration_sec=2.0, metadata=opus_meta
    )
    assert len(frames) == 3
    assert audio_p is not None and audio_p.name == "audio.m4a"
    audio_cmd, video_cmd = recorded_cmds[0], recorded_cmds[1]
    assert "-protocol_whitelist" in audio_cmd and "file" in audio_cmd
    assert audio_cmd[audio_cmd.index("-map") + 1] == "0:3"
    assert audio_cmd[audio_cmd.index("-c:a") + 1] == "aac"
    assert video_cmd[video_cmd.index("-map") + 1] == "0:2"
    assert "eof_action=pass" in video_cmd[video_cmd.index("-vf") + 1]

    # 2) Packetless .m4a after trim -> treated gracefully as audio-less!
    recorded_cmds.clear()
    monkeypatch.setattr(media, "probe_audio_has_packets", lambda _p: False)
    _, audio_empty, _, _ = media.extract_cfr_frames(
        src, tmp_path / "out_no_pkt", start_sec=10.0, metadata=opus_meta
    )
    assert audio_empty is None

    # 3) Defensive 1-frame fallback when fps filter drops all frames (10fps test case above)
    recorded_cmds.clear()
    frames_1f, _, _, _ = media.extract_cfr_frames(
        src, tmp_path / "out_1f", target_fps=10.0, metadata=opus_meta
    )
    assert len(frames_1f) >= 1
    assert any("-frames:v" in c and "1" in c for c in recorded_cmds)


def test_build_video_writer_cmd_preserves_sar_color_no_shortest_and_lossless(
    tmp_path: Path,
) -> None:
    out_mp4 = tmp_path / "out.mp4"
    audio_file = tmp_path / "audio.m4a"
    audio_file.write_bytes(b"aac")

    cmd = media.build_video_writer_cmd(
        641,
        481,
        30.0,
        out_mp4,
        audio_path=audio_file,
        num_frames=300,
        sample_aspect_ratio="4:3",
        color_primaries="smpte170m",
        color_trc="smpte170m",
        colorspace="smpte170m",
    )
    assert "-protocol_whitelist" in cmd and "file,pipe" in cmd
    assert "pipe:0" in cmd
    assert "-shortest" not in cmd
    assert "-t" in cmd and cmd[cmd.index("-t") + 1] == "10.000000"
    vf = cmd[cmd.index("-vf") + 1]
    assert "pad=ceil(iw/2)*2:ceil(ih/2)*2" in vf
    assert "setsar=4/3" in vf
    assert cmd[cmd.index("-color_primaries") + 1] == "smpte170m"
    assert cmd[-1].startswith("file:/")

    cmd_lossless = media.build_video_writer_cmd(
        641, 481, 30.0, out_mp4, lossless=True
    )
    assert cmd_lossless[cmd_lossless.index("-c:v") + 1] == "libx264rgb"
    assert cmd_lossless[cmd_lossless.index("-crf") + 1] == "0"
    assert cmd_lossless[cmd_lossless.index("-pix_fmt") + 1] == "rgb24"
    assert "-vf" not in cmd_lossless


def test_run_subprocess_rejects_empty_cmd() -> None:
    with pytest.raises(ValueError):
        media.run_subprocess([])
