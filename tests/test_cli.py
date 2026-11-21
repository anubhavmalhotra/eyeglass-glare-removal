"""Unit tests for CLI parsing, workspace confinement, and content-bound resume guards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

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


_CHUNK_RECORD = {
    "chunk_idx": 0, "start": 0, "end": 1, "start_write": 0,
    "from_checkpoint": False, "used_fallback": False,
}
_MASK_FRAME = {
    "frame_idx": 0, "left_lens_roi_px": 0, "right_lens_roi_px": 0,
    "glare_left_pct": 0.0, "glare_right_pct": 0.0,
}


def test_content_bound_resume_invalidates_on_replaced_video_or_corrupt_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
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
        "crop_size": 64,
        "detector": "manual",
        "fps": 30.0,
        "num_frames": 1,
        "init_frame_idx": 0,
        "init_eye_geometry_roi": None,
        "frames": [{"frame_idx": 0, "filename": "000000.png", "x1": 0, "y1": 0, "x2": 64, "y2": 64}],
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
    import sys

    from eyeglass_glare_removal import pipeline

    runtime = ("cpu", "float32", "local:fixed", "test")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__=runtime[-1]))
    monkeypatch.setattr(pipeline, "resolve_segment_identity", lambda *_: runtime[:3])
    sam2_g, seg_g = compute_segment_generation(roi_d, 0, il, ir, seg_cfg, runtime_identity=runtime)
    d_l = hash_files_manifest(layout.roi_lens_interior_dir, ["000000.png"])
    d_e = hash_files_manifest(layout.roi_lens_exact_dir, ["000000.png"])
    d_g = hash_files_manifest(layout.roi_glare_hotspot_dir, ["000000.png"])
    masks_d = hashlib.sha256(f"{d_l}:{d_e}:{d_g}".encode()).hexdigest()

    layout.mask_stats_json.write_text(
        json.dumps(
            {
                "prep_generation": roi_meta["prep_generation"],
                "sam2_generation": sam2_g,
                "segment_generation": seg_g,
                "roi_frames_digest": roi_d,
                "masks_digest": masks_d,
                "num_frames": 1,
                "chunks": [_CHUNK_RECORD],
                "frames": [_MASK_FRAME],
            }
        ),
        encoding="utf-8",
    )
    assert _segment_is_complete(layout, roi_meta, cfg_preview) is not None

    # All artifact hashes and generations stay valid while consumed fields are corrupted.
    valid_stats = layout.mask_stats_json.read_text()
    for malformed in (None, [{"start": 0}], [{**_CHUNK_RECORD, "used_fallback": "false"}]):
        layout.mask_stats_json.write_text(json.dumps({**json.loads(valid_stats), "chunks": malformed}))
        assert _segment_is_complete(layout, roi_meta, cfg_preview) is None
    layout.mask_stats_json.write_text(valid_stats)
    assert _segment_is_complete(layout, roi_meta, cfg_preview) is not None

    # Corrupting a mask PNG with garbage invalidates Stage 2 resume!
    (layout.roi_glare_hotspot_dir / "000000.png").write_bytes(b"corrupt")
    assert _segment_is_complete(layout, roi_meta, cfg_preview) is None


def test_round3_p0_p1_cli_generation_chain_symlink_ancestors_schema_and_atomic_rollback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regressions P0-1, P0-3, P0-4, P1-11, P1-12."""
    import os

    from eyeglass_glare_removal import _workspace
    from eyeglass_glare_removal.pipeline import verify_prep_artifacts

    # --- P0-1: CLI `prep` refuses input inside work_dir or aliased to audio.m4a ---
    w_p01 = tmp_path / "w_p01"
    (w_p01 / "prep").mkdir(parents=True)
    inside_vid = w_p01 / "prep" / "clip.mp4"
    inside_vid.write_bytes(b"PRECIOUS_VIDEO")
    rc = cli.main(["prep", str(inside_vid), "--work-dir", str(w_p01)])
    assert rc == 1
    assert inside_vid.read_bytes() == b"PRECIOUS_VIDEO"
    capsys.readouterr()

    # --- P0-4: symlinked `prep` or `masks` ANCESTOR rejected even when leaf manifests are ordinary ---
    w_sym = tmp_path / "w_sym"
    w_sym.mkdir()
    outside_prep = tmp_path / "outside_prep"
    (outside_prep / "frames").mkdir(parents=True)
    (outside_prep / "roi_frames").mkdir(parents=True)
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(outside_prep / "frames" / "000000.png")
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(
        outside_prep / "roi_frames" / "000000.png"
    )
    (outside_prep / "roi_boxes.json").write_text("{}", encoding="utf-8")
    (w_sym / "prep").symlink_to(outside_prep)

    sym_layout = WorkDirLayout(w_sym)
    dummy_in = tmp_path / "dummy.mp4"
    dummy_in.write_bytes(b"x")
    dummy_cfg = PipelineConfig(input_path=dummy_in, output_path=tmp_path / "o.mp4", work_dir=w_sym)
    with pytest.raises(ValueError, match="Symlinked"):
        _prep_matches_config(sym_layout, dummy_cfg)
    with pytest.raises(ValueError, match="Symlinked"):
        verify_prep_artifacts(sym_layout)

    # --- P1-12: valid JSON with invalid schema (null, [], bad types) returns None on resume ---
    w_schema = tmp_path / "w_schema"
    sl = WorkDirLayout(w_schema)
    sl.prep_dir.mkdir(parents=True)
    sl.masks_dir.mkdir(parents=True)
    scfg = PipelineConfig(input_path=dummy_in, output_path=tmp_path / "os.mp4", work_dir=w_schema)
    for bad in ("null", "[]", '"hello"', '{"num_frames": "bad"}', '{"num_frames": true}'):
        sl.roi_boxes_json.write_text(bad, encoding="utf-8")
        assert _prep_matches_config(sl, scfg) is None
        sl.mask_stats_json.write_text(bad, encoding="utf-8")
        assert _segment_is_complete(sl, {"num_frames": 1, "frames": [{"filename": "000000.png"}]}, scfg) is None

    # --- P0-3: Clip A artifacts in workspace MUST be rejected after prep runs for Clip B! ---
    w_mix = tmp_path / "w_mix"
    ml = WorkDirLayout(w_mix)
    for d in (
        ml.frames_dir,
        ml.roi_frames_dir,
        ml.roi_lens_interior_dir,
        ml.roi_lens_exact_dir,
        ml.roi_glare_hotspot_dir,
        ml.reference_dir,
        ml.roi_cleaned_dir,
    ):
        d.mkdir(parents=True)

    fn = "000000.png"
    # Populate Clip A across all 4 stages
    Image.fromarray(np.full((64, 64, 3), 10, dtype=np.uint8)).save(ml.frames_dir / fn)
    Image.fromarray(np.full((64, 64, 3), 20, dtype=np.uint8)).save(ml.roi_frames_dir / fn)
    for md in (ml.roi_lens_interior_dir, ml.roi_lens_exact_dir, ml.roi_glare_hotspot_dir):
        Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(md / fn)
    Image.fromarray(np.full((64, 64, 3), 30, dtype=np.uint8)).save(ml.clean_eye_ref_png)
    Image.fromarray(np.full((64, 64, 3), 40, dtype=np.uint8)).save(ml.roi_cleaned_dir / fn)

    roi_a_digest = hash_files_manifest(ml.roi_frames_dir, [fn])
    d_l = hash_files_manifest(ml.roi_lens_interior_dir, [fn])
    d_e = hash_files_manifest(ml.roi_lens_exact_dir, [fn])
    d_g = hash_files_manifest(ml.roi_glare_hotspot_dir, [fn])
    masks_a_digest = hashlib.sha256(f"{d_l}:{d_e}:{d_g}".encode()).hexdigest()
    cleaned_a_digest = hash_files_manifest(ml.roi_cleaned_dir, [fn])

    ml.mask_stats_json.write_text(
        json.dumps(
            {
                "prep_generation": "prep_CLIP_A",
                "segment_generation": "seg_CLIP_A",
                "roi_frames_digest": roi_a_digest,
                "masks_digest": masks_a_digest,
                "num_frames": 1,
                "chunks": [_CHUNK_RECORD],
                "frames": [
                    {
                        "frame_idx": 0,
                        "left_lens_roi_px": 100,
                        "right_lens_roi_px": 100,
                        "glare_left_pct": 0.0,
                        "glare_right_pct": 0.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    ml.reference_meta_json.write_text(
        json.dumps(
            {
                "mode": "auto",
                "strategy": "single_clean_frame",
                "prep_generation": "prep_CLIP_A",
                "segment_generation": "seg_CLIP_A",
                "ref_sha256": hash_file_sha256(ml.clean_eye_ref_png),
            }
        ),
        encoding="utf-8",
    )
    ml.edit_support_json.write_text(
        json.dumps(
            {
                "prep_generation": "prep_CLIP_A",
                "segment_generation": "seg_CLIP_A",
                "roi_frames_digest": roi_a_digest,
                "masks_digest": masks_a_digest,
                "cleaned_roi_digest": cleaned_a_digest,
                "num_frames": 1,
                "frame_has_edit": [True],
            }
        ),
        encoding="utf-8",
    )

    # Now overwrite prep/ with Clip B while leaving Clip A's masks/reference/inpaint behind!
    Image.fromarray(np.full((64, 64, 3), 210, dtype=np.uint8)).save(ml.frames_dir / fn)
    Image.fromarray(np.full((64, 64, 3), 220, dtype=np.uint8)).save(ml.roi_frames_dir / fn)
    frames_b_digest = hash_files_manifest(ml.frames_dir, [fn])
    roi_b_digest = hash_files_manifest(ml.roi_frames_dir, [fn])
    ml.roi_boxes_json.write_text(
        json.dumps(
            {
                "source_video": str((tmp_path / "clip_b.mp4").resolve()),
                "prep_generation": "prep_CLIP_B",
                "frames_digest": frames_b_digest,
                "roi_frames_digest": roi_b_digest,
                "audio_sha256": None,
                "audio_filename": None,
                "fps": 30.0,
                "full_width": 64,
                "full_height": 64,
                "roi_size": 64,
                "crop_size": 64,
                "detector": "manual",
                "num_frames": 1,
                "frames": [{"frame_idx": 0, "filename": fn, "x1": 0, "y1": 0, "x2": 64, "y2": 64}],
            }
        ),
        encoding="utf-8",
    )

    # ALL standalone downstream stages MUST fail cleanly with generation mismatch!
    for argv in [
        ["reference", "--work-dir", str(w_mix)],
        ["inpaint", "--work-dir", str(w_mix)],
        ["composite", "--work-dir", str(w_mix), "-o", str(tmp_path / "mix_out.mp4")],
    ]:
        assert cli.main(argv) == 1
        err = capsys.readouterr().err.lower()
        assert "generation mismatch" in err or "digest mismatch" in err

    # Now update masks to Clip B, leaving Clip A's reference/ & inpaint/ behind -> BOTH fail!
    ml.mask_stats_json.write_text(
        json.dumps(
            {
                "prep_generation": "prep_CLIP_B",
                "segment_generation": "seg_CLIP_B",
                "roi_frames_digest": roi_b_digest,
                "masks_digest": masks_a_digest,
                "num_frames": 1,
                "chunks": [_CHUNK_RECORD],
                "frames": [_MASK_FRAME],
            }
        ),
        encoding="utf-8",
    )
    assert cli.main(["inpaint", "--work-dir", str(w_mix)]) == 1
    assert "stale reference" in capsys.readouterr().err.lower()
    assert cli.main(["composite", "--work-dir", str(w_mix), "-o", str(tmp_path / "mix_out.mp4")]) == 1
    assert "generation mismatch" in capsys.readouterr().err.lower()

    # --- P1-11: atomic_swap_directory handles KeyboardInterrupt AND preserves backup on rollback error ---
    tgt = tmp_path / "tgt_dir"
    tgt.mkdir()
    (tgt / "data.txt").write_text("OLD_DATA", encoding="utf-8")
    stg = tmp_path / "stg_dir"
    stg.mkdir()
    (stg / "data.txt").write_text("NEW_DATA", encoding="utf-8")

    real_replace = os.replace
    calls = 0

    def interrupt_on_second_replace(src, dst):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("user aborted mid-swap")
        return real_replace(src, dst)

    monkeypatch.setattr(_workspace.os, "replace", interrupt_on_second_replace)
    with pytest.raises(KeyboardInterrupt):
        _workspace.atomic_swap_directory(stg, tgt)
    monkeypatch.setattr(_workspace.os, "replace", real_replace)
    assert (tgt / "data.txt").read_text(encoding="utf-8") == "OLD_DATA"

    # Now fail BOTH the second replace AND the rollback replace -> backup MUST stay on disk!
    calls = 0

    def fail_commit_and_rollback(src, dst):
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise OSError("disk error during commit/rollback")
        return real_replace(src, dst)

    monkeypatch.setattr(_workspace.os, "replace", fail_commit_and_rollback)
    with pytest.raises(OSError):
        _workspace.atomic_swap_directory(stg, tgt)
    monkeypatch.setattr(_workspace.os, "replace", real_replace)
    preserved_backups = list(tmp_path.glob(".bak_tgt_dir_*"))
    assert len(preserved_backups) == 1
    assert (preserved_backups[0] / "data.txt").read_text(encoding="utf-8") == "OLD_DATA"


@pytest.fixture
def verified_workspace(tmp_path: Path) -> tuple[WorkDirLayout, Path, dict, dict]:
    from eyeglass_glare_removal.config import InpaintConfig
    from eyeglass_glare_removal.inpaint import run_inpaint_chunks

    layout = WorkDirLayout(tmp_path / "work")
    source = tmp_path / "clip_a.mp4"
    source.write_bytes(b"clip A")
    fn = "000000.png"
    for directory in (layout.frames_dir, layout.roi_frames_dir, layout.roi_lens_interior_dir,
                      layout.roi_lens_exact_dir, layout.roi_glare_hotspot_dir):
        directory.mkdir(parents=True)
        Image.new("RGB" if directory.parent == layout.prep_dir else "L", (64, 64)).save(directory / fn)
    roi_digest = hash_files_manifest(layout.roi_frames_dir, [fn])
    mask_digests = [
        hash_files_manifest(d, [fn])
        for d in (layout.roi_lens_interior_dir, layout.roi_lens_exact_dir, layout.roi_glare_hotspot_dir)
    ]
    masks_digest = hashlib.sha256(":".join(mask_digests).encode()).hexdigest()
    prep = {
        "source_video": str(source), "source_sha256": hash_file_sha256(source),
        "prep_generation": "prep", "frames_digest": hash_files_manifest(layout.frames_dir, [fn]),
        "roi_frames_digest": roi_digest, "fps": 24, "full_width": 64, "full_height": 64,
        "roi_size": 64, "crop_size": 64, "detector": "manual", "num_frames": 1,
        "frames": [{"frame_idx": 0, "filename": fn, "x1": 0, "y1": 0, "x2": 64, "y2": 64}],
    }
    segment = {
        "prep_generation": "prep", "segment_generation": "segment", "num_frames": 1,
        "roi_frames_digest": roi_digest, "masks_digest": masks_digest,
        "frames": [_MASK_FRAME], "chunks": [_CHUNK_RECORD],
    }
    layout.roi_boxes_json.write_text(json.dumps(prep))
    layout.mask_stats_json.write_text(json.dumps(segment))
    run_inpaint_chunks(
        [layout.roi_frames_dir / fn], layout.roi_glare_hotspot_dir, layout.inpaint_dir,
        InpaintConfig(size=64), prep_generation="prep", segment_generation="segment",
        roi_frames_digest=roi_digest, masks_digest=masks_digest,
    )
    return layout, source, prep, segment


def test_composite_explicit_source_cannot_replace_recorded_source_or_select_another_clip(
    tmp_path: Path, verified_workspace, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    import io

    from eyeglass_glare_removal import composite

    layout, source, _, _ = verified_workspace
    other = tmp_path / "clip_b.mp4"
    other.write_bytes(b"clip B")
    opened = []

    def open_writer(*args, **kwargs):
        opened.append(args[3])
        return SimpleNamespace(stdin=io.BytesIO(), output=args[3])

    monkeypatch.setattr(composite, "open_video_writer_pipe", open_writer)
    monkeypatch.setattr(composite, "close_video_writer_pipe", lambda writer, _: writer.output.write_bytes(b"result"))
    cmd = ["composite", "--work-dir", str(layout.root), "--source", str(other)]
    assert cli.main(cmd + ["-o", str(source)]) == 1
    assert "aliases input" in capsys.readouterr().err
    assert source.read_bytes() == b"clip A" and not opened

    output = tmp_path / "output.mp4"
    assert cli.main(cmd + ["-o", str(output)]) == 1
    assert "source_sha256" in capsys.readouterr().err
    assert not output.exists() and not opened

    other.write_bytes(source.read_bytes())
    assert cli.main(cmd + ["-o", str(output)]) == 0
    assert output.read_bytes() == b"result" and len(opened) == 1


def test_saved_pngs_reject_symlinked_workspace_ancestor_before_encoding(
    tmp_path: Path, verified_workspace, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    from eyeglass_glare_removal import composite

    layout, _, _, _ = verified_workspace
    outside = tmp_path / "outside"
    publication = outside / "exports"
    publication.mkdir(parents=True)
    prior_frame = publication / "000000.png"
    prior_frame.write_bytes(b"previous")
    (layout.root / "link").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(composite, "open_video_writer_pipe", lambda *_a, **_k: pytest.fail("encoder opened"))
    assert cli.main([
        "composite", "--work-dir", str(layout.root), "-o", str(tmp_path / "output.mp4"),
        "--save-frames-dir", str(layout.root / "link" / "exports"),
    ]) == 1
    assert "Symlinked" in capsys.readouterr().err
    assert prior_frame.read_bytes() == b"previous"


def test_pipeline_rejects_output_in_replaceable_prep_before_rerun(
    tmp_path: Path, verified_workspace, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    from eyeglass_glare_removal import pipeline

    layout, source, _, _ = verified_workspace
    output = layout.prep_dir / "output.mp4"
    output.write_bytes(b"previous")
    source.write_bytes(b"changed")
    monkeypatch.setattr(pipeline, "run_prep_stage", lambda *_: pytest.fail("prep ran"))
    assert cli.main(["run", str(source), "-o", str(output), "--work-dir", str(layout.root)]) == 1
    assert "replaceable stage" in capsys.readouterr().err
    assert output.read_bytes() == b"previous"


def test_reference_lineage_binds_live_roi_masks_and_reference_used_by_inpaint(
    verified_workspace, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    from eyeglass_glare_removal import composite, pipeline
    from eyeglass_glare_removal.config import InpaintConfig

    layout, _, prep, segment = verified_workspace
    fn = "000000.png"
    layout.reference_dir.mkdir()
    Image.new("RGB", (64, 64), (60, 80, 100)).save(layout.clean_eye_ref_png)
    reference = {
        "mode": "auto", "prep_generation": "prep", "segment_generation": "segment",
        "roi_frames_digest": prep["roi_frames_digest"], "masks_digest": segment["masks_digest"],
        "ref_sha256": hash_file_sha256(layout.clean_eye_ref_png),
    }
    layout.reference_meta_json.write_text(json.dumps(reference))
    assert pipeline.verify_reference_artifact(layout, prep, segment) == layout.clean_eye_ref_png

    Image.new("L", (64, 64), 1).save(layout.roi_glare_hotspot_dir / fn)
    digests = [hash_files_manifest(d, [fn]) for d in (
        layout.roi_lens_interior_dir, layout.roi_lens_exact_dir, layout.roi_glare_hotspot_dir
    )]
    segment["masks_digest"] = hashlib.sha256(":".join(digests).encode()).hexdigest()
    layout.mask_stats_json.write_text(json.dumps(segment))
    live_segment, live_mask_digest = pipeline.verify_segment_artifacts(layout, prep, [fn], prep["roi_frames_digest"])
    assert live_segment["segment_generation"] == reference["segment_generation"]
    with pytest.raises(ValueError, match="Stale reference.*digest"):
        pipeline.verify_reference_artifact(layout, prep, live_segment)

    reference["masks_digest"] = live_mask_digest
    layout.reference_meta_json.write_text(json.dumps(reference))
    pipeline.run_inpaint_chunks(
        [layout.roi_frames_dir / fn], layout.roi_glare_hotspot_dir, layout.inpaint_dir,
        InpaintConfig(size=64), ref_image_paths=[layout.clean_eye_ref_png], workspace_reference=True,
        prep_generation="prep", segment_generation="segment", roi_frames_digest=prep["roi_frames_digest"],
        masks_digest=live_mask_digest,
    )
    args = (layout, prep, segment, [fn], prep["roi_frames_digest"], live_mask_digest)
    assert pipeline.verify_inpaint_artifacts(*args)[1] == [False]

    Image.new("RGB", (64, 64), (70, 90, 110)).save(layout.clean_eye_ref_png)
    reference["ref_sha256"] = hash_file_sha256(layout.clean_eye_ref_png)
    layout.reference_meta_json.write_text(json.dumps(reference))
    assert pipeline.verify_reference_artifact(layout, prep, segment) == layout.clean_eye_ref_png
    monkeypatch.setattr(composite, "open_video_writer_pipe", lambda *_a, **_k: pytest.fail("encoder opened"))
    assert cli.main([
        "composite", "--work-dir", str(layout.root), "-o", str(layout.root / "out.mp4"),
    ]) == 1
    assert "Inpaint reference no longer matches" in capsys.readouterr().err


@pytest.mark.parametrize("invalid_flag", ["false", 1, None])
def test_composite_rejects_non_boolean_edit_flags_with_valid_digests(
    verified_workspace, invalid_flag,
) -> None:
    from eyeglass_glare_removal.pipeline import verify_inpaint_artifacts

    layout, _, prep, segment = verified_workspace
    args = (layout, prep, segment, ["000000.png"], prep["roi_frames_digest"], segment["masks_digest"])
    assert verify_inpaint_artifacts(*args)[1] == [False]
    edit_support = json.loads(layout.edit_support_json.read_text())
    edit_support["frame_has_edit"] = [invalid_flag]
    layout.edit_support_json.write_text(json.dumps(edit_support))
    with pytest.raises(ValueError, match="frame_has_edit"):
        verify_inpaint_artifacts(*args)


def test_atomic_commit_rejects_file_replacing_directory_and_vice_versa(tmp_path: Path) -> None:
    """Regression P0 #1: publishing a file must NEVER replace or delete an existing directory."""
    from eyeglass_glare_removal._workspace import atomic_commit_paths, make_unique_temp_sibling

    # 1) Regular file staged -> existing non-empty directory destination: MUST raise & keep directory intact!
    existing_dir = tmp_path / "out.mp4"
    existing_dir.mkdir()
    precious = existing_dir / "keep_me.txt"
    precious.write_text("DO_NOT_DELETE", encoding="utf-8")

    staged_file = tmp_path / ".staged_out.tmp"
    staged_file.write_bytes(b"new_mp4_bytes")

    with pytest.raises(ValueError, match="non-regular destination"):
        atomic_commit_paths([(staged_file, existing_dir)])
    assert existing_dir.is_dir()
    assert precious.read_text(encoding="utf-8") == "DO_NOT_DELETE"
    assert staged_file.exists()

    with pytest.raises(ValueError, match="existing directory"):
        make_unique_temp_sibling(existing_dir)

    # 2) Staged directory -> existing regular file destination: MUST raise & keep file intact!
    existing_file = tmp_path / "regular_file.txt"
    existing_file.write_text("ORIGINAL_FILE", encoding="utf-8")
    staged_dir = tmp_path / ".staged_dir"
    staged_dir.mkdir()
    (staged_dir / "child.txt").write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="non-directory destination"):
        atomic_commit_paths([(staged_dir, existing_file)])
    assert existing_file.is_file()
    assert existing_file.read_text(encoding="utf-8") == "ORIGINAL_FILE"


@pytest.mark.parametrize(
    "mutation",
    [
        {"init_frame_idx": None},
        {"init_frame_idx": True},
        {"init_frame_idx": 5},
        {"video_metadata": {"colorspace": ["bt709"]}},
        {"video_metadata": {"color_primaries": 709}},
        {"video_metadata": {"sar": ["1:1"]}},
        {"init_eye_geometry_roi": {"left_eye": "bad"}},
    ],
)
def test_prep_resume_invalidates_malformed_seed_and_video_metadata_fields(
    tmp_path: Path, mutation: dict[str, object]
) -> None:
    """Regression P1 #5: malformed init_frame_idx or non-string video_metadata fields invalidate prep resume."""
    inp = tmp_path / "clip.mp4"
    inp.write_bytes(b"bytes")
    work = tmp_path / "work"
    layout = WorkDirLayout(work)
    layout.frames_dir.mkdir(parents=True)
    layout.roi_frames_dir.mkdir(parents=True)
    Image.new("RGB", (160, 120), (80, 80, 80)).save(layout.frames_dir / "000000.png")
    Image.new("RGB", (64, 64), (90, 90, 90)).save(layout.roi_frames_dir / "000000.png")

    prep_cfg = PrepConfig(roi_size=64)
    src_sha = hash_file_sha256(inp)
    roi_meta = {
        "source_video": str(inp.resolve()),
        "source_sha256": src_sha,
        "prep_generation": compute_prep_generation(src_sha, prep_cfg),
        "frames_digest": hash_files_manifest(layout.frames_dir, ["000000.png"]),
        "roi_frames_digest": hash_files_manifest(layout.roi_frames_dir, ["000000.png"]),
        "audio_sha256": None,
        "audio_filename": None,
        "fps": 30.0,
        "detector": "haar",
        "crop_size": 64,
        "full_width": 160,
        "full_height": 120,
        "roi_size": 64,
        "num_frames": 1,
        "init_frame_idx": 0,
        "init_eye_geometry_roi": None,
        "video_metadata": {"colorspace": "bt709"},
        "frames": [{"frame_idx": 0, "filename": "000000.png", "x1": 0, "y1": 0, "x2": 64, "y2": 64}],
    }
    cfg = PipelineConfig(
        input_path=inp,
        output_path=tmp_path / "out.mp4",
        work_dir=work,
        prep=prep_cfg,
        reference=cli.ReferenceConfig(roi_size=64),
    )
    layout.roi_boxes_json.write_text(json.dumps(roi_meta), encoding="utf-8")
    assert _prep_matches_config(layout, cfg) is not None

    bad_meta = dict(roi_meta)
    bad_meta.update(mutation)
    layout.roi_boxes_json.write_text(json.dumps(bad_meta), encoding="utf-8")
    assert _prep_matches_config(layout, cfg) is None
