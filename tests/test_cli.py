"""Unit tests for CLI parsing, workspace confinement, and content-bound resume guards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from eyeglass_glare_removal import cli
from eyeglass_glare_removal._workspace import (
    ensure_confined_dir,
    ensure_safe_child_file,
    hash_file_sha256,
    hash_files_manifest,
    validate_safe_basename,
)
from eyeglass_glare_removal.config import PipelineConfig, PrepConfig, SegmentConfig
from eyeglass_glare_removal.detect import compute_prep_generation
from eyeglass_glare_removal.pipeline import (
    WorkDirLayout,
    _prep_matches_config,
    _segment_is_complete,
)
from eyeglass_glare_removal.segment import build_initial_lens_prompts, compute_segment_generation


def test_cli_run_dry_run_preview_and_options(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inp = tmp_path / "sample.mp4"
    out = tmp_path / "cleaned.mp4"
    rc = cli.main(
        [
            "run",
            str(inp),
            "-o",
            str(out),
            "--preview",
            "5",
            "--roi-box",
            "100,120,420,440",
            "--no-ref",
            "--quantize-4bit",
            "--lossless",
            "--compare",
            "--qa-grid",
            "--dry-run",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["output_path"] == str(out)
    stages = {s["name"]: s for s in payload["stages"]}
    assert stages["prep"]["duration_sec"] == 5.0
    assert stages["prep"]["roi_box"] == [100, 120, 420, 440]
    assert stages["reference"]["mode"] == "none"
    assert stages["inpaint"]["quantize_4bit"] is True
    assert stages["composite"]["lossless"] is True
    assert stages["composite"]["compare"] is True
    assert stages["composite"]["qa_grid"] is True


def test_cli_conflicting_flags_return_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc1 = cli.main(
        [
            "run",
            str(tmp_path / "in.mp4"),
            "-o",
            str(tmp_path / "out.mp4"),
            "--preview",
            "5",
            "--duration",
            "10",
            "--dry-run",
        ]
    )
    assert rc1 == 1
    assert "preview" in capsys.readouterr().err.lower()

    rc2 = cli.main(
        [
            "run",
            str(tmp_path / "in.mp4"),
            "-o",
            str(tmp_path / "out.mp4"),
            "--ref-image",
            str(tmp_path / "ref.png"),
            "--no-ref",
            "--dry-run",
        ]
    )
    assert rc2 == 1
    assert "ref-image" in capsys.readouterr().err.lower()


def test_cli_lens_points_file_and_inline_json(tmp_path: Path) -> None:
    raw = {"left": {"box": [80, 180, 210, 290]}, "right": {"box": [290, 180, 420, 290]}}
    assert cli.load_lens_points(json.dumps(raw)) == raw
    p = tmp_path / "points.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    assert cli.load_lens_points(str(p)) == raw
    with pytest.raises(ValueError, match="Failed to parse"):
        cli.load_lens_points("{not valid json")


def test_cli_subcommands_parse_cleanly(tmp_path: Path) -> None:
    parser = cli.build_parser()
    for argv in [
        ["prep", str(tmp_path / "in.mp4"), "--work-dir", str(tmp_path / "w"), "--preview", "3"],
        ["segment", "--work-dir", str(tmp_path / "w"), "--sam2-chunk-size", "120"],
        ["reference", "--work-dir", str(tmp_path / "w"), "--no-mirror-ref"],
        ["inpaint", "--work-dir", str(tmp_path / "w"), "--steps", "16", "--no-ref"],
        [
            "composite",
            "--work-dir",
            str(tmp_path / "w"),
            "-o",
            str(tmp_path / "out.mp4"),
            "--lossless",
            "--no-suppress-chroma-glint",
        ],
    ]:
        ns = parser.parse_args(argv)
        assert ns.command == argv[0]


def test_workspace_confinement_rejects_traversal_absolute_and_symlinks(tmp_path: Path) -> None:
    """Regression P0-4: reject absolute filenames, '..' traversal, and symlinked workdir subdirs."""
    for bad in ("/etc/passwd", "/tmp/evil.png", "../escape.png", "a/b.png", "-flag.png", "..", ""):
        with pytest.raises(ValueError):
            validate_safe_basename(bad)

    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (work / "frames").symlink_to(outside)

    with pytest.raises(ValueError, match="Symlinked"):
        ensure_confined_dir(work, work / "frames", create=False)
    with pytest.raises(ValueError, match="Symlinked"):
        ensure_safe_child_file(work / "frames", "000000.png", allow_missing=True)


def test_content_bound_resume_invalidates_on_replaced_video_or_corrupt_artifacts(
    tmp_path: Path,
) -> None:
    """Regression P0-2 & P1-10: replacing source clip at same path or corrupting PNGs invalidates resume."""
    inp = tmp_path / "video.mp4"
    inp.write_bytes(b"video_bytes_v1")
    work = tmp_path / "work"
    layout = WorkDirLayout(work)
    layout.frames_dir.mkdir(parents=True)
    layout.roi_frames_dir.mkdir(parents=True)

    Image.fromarray(np.full((120, 160, 3), 80, dtype=np.uint8)).save(layout.frames_dir / "000000.png")
    Image.fromarray(np.full((64, 64, 3), 90, dtype=np.uint8)).save(layout.roi_frames_dir / "000000.png")

    prep_cfg = PrepConfig(duration_sec=5.0, roi_size=64)
    src_sha = hash_file_sha256(inp)
    frames_d = hash_files_manifest(layout.frames_dir, ["000000.png"])
    roi_d = hash_files_manifest(layout.roi_frames_dir, ["000000.png"])
    roi_meta = {
        "source_video": str(inp),
        "source_sha256": src_sha,
        "prep_generation": compute_prep_generation(src_sha, prep_cfg),
        "frames_digest": frames_d,
        "roi_frames_digest": roi_d,
        "audio_sha256": None,
        "audio_filename": None,
        "full_width": 160,
        "full_height": 120,
        "roi_size": 64,
        "num_frames": 1,
        "init_frame_idx": 0,
        "init_eye_geometry_roi": None,
        "frames": [{"frame_idx": 0, "filename": "000000.png"}],
    }
    layout.roi_boxes_json.write_text(json.dumps(roi_meta), encoding="utf-8")

    cfg_preview = PipelineConfig(
        input_path=inp,
        output_path=tmp_path / "out.mp4",
        work_dir=work,
        prep=prep_cfg,
        reference=cli.ReferenceConfig(roi_size=64),
    )
    assert _prep_matches_config(layout, cfg_preview) is not None

    # 1) Switching duration from 5s preview to full video -> invalidated!
    cfg_full = PipelineConfig(
        input_path=inp,
        output_path=tmp_path / "out.mp4",
        work_dir=work,
        prep=PrepConfig(duration_sec=None, roi_size=64),
        reference=cli.ReferenceConfig(roi_size=64),
    )
    assert _prep_matches_config(layout, cfg_full) is None

    # 2) Replacing clip bytes at the EXACT SAME path -> invalidated!
    inp.write_bytes(b"video_bytes_v2_completely_different_clip")
    assert _prep_matches_config(layout, cfg_preview) is None
    inp.write_bytes(b"video_bytes_v1")
    assert _prep_matches_config(layout, cfg_preview) is not None

    # 3) Segment content digest verification catches both sensitivity change AND mask corruption
    for d in (
        layout.roi_lens_interior_dir,
        layout.roi_lens_exact_dir,
        layout.roi_glare_hotspot_dir,
    ):
        d.mkdir(parents=True)
        Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(d / "000000.png")

    seg_cfg = SegmentConfig()
    il, ir = build_initial_lens_prompts(None, roi_size=64)
    sam2_g, seg_g = compute_segment_generation(roi_d, 0, il, ir, seg_cfg)
    d_l = hash_files_manifest(layout.roi_lens_interior_dir, ["000000.png"])
    d_e = hash_files_manifest(layout.roi_lens_exact_dir, ["000000.png"])
    d_g = hash_files_manifest(layout.roi_glare_hotspot_dir, ["000000.png"])
    masks_d = hashlib.sha256(f"{d_l}:{d_e}:{d_g}".encode()).hexdigest()

    layout.mask_stats_json.write_text(
        json.dumps(
            {
                "sam2_generation": sam2_g,
                "segment_generation": seg_g,
                "roi_frames_digest": roi_d,
                "masks_digest": masks_d,
                "num_frames": 1,
            }
        ),
        encoding="utf-8",
    )
    assert _segment_is_complete(layout, roi_meta, cfg_preview) is not None

    # Corrupting a mask PNG with garbage invalidates Stage 2 resume!
    (layout.roi_glare_hotspot_dir / "000000.png").write_bytes(b"corrupt")
    assert _segment_is_complete(layout, roi_meta, cfg_preview) is None
