"""Unit tests for configuration dataclasses, path safety, and packaging metadata."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from eyeglass_glare_removal.config import (
    DEFAULT_NEG_PROMPT,
    DEFAULT_PROMPT,
    CompositeConfig,
    InpaintConfig,
    PipelineConfig,
    PrepConfig,
    ReferenceConfig,
    SegmentConfig,
)


def test_default_configs_are_valid(tmp_path: Path) -> None:
    cfg = PipelineConfig(input_path=tmp_path / "in.mp4", output_path=tmp_path / "out.mp4")
    assert cfg.work_dir == tmp_path / "out.work"
    assert cfg.prep.roi_size == 512
    assert cfg.inpaint.chunk_len == 81
    assert cfg.inpaint.min_overlap == 12
    assert cfg.composite.suppress_chroma_glint is True
    assert cfg.composite.lossless is False
    assert cfg.composite.max_batch_mem_mb == 256


def test_default_prompt_is_generic_and_neutral() -> None:
    assert "person wearing eyeglasses" in DEFAULT_PROMPT.lower()
    combined = (DEFAULT_PROMPT + " " + DEFAULT_NEG_PROMPT).lower()
    for specific in ("woman", "man", "male", "female", "elderly", "child", "silver", "gold"):
        assert specific not in combined.split()


def test_prep_config_validation() -> None:
    with pytest.raises(ValueError, match="start_sec"):
        PrepConfig(start_sec=-1.0)
    with pytest.raises(ValueError, match="duration_sec"):
        PrepConfig(duration_sec=0.0)
    with pytest.raises(ValueError, match="fps"):
        PrepConfig(fps=-30.0)
    with pytest.raises(ValueError, match="roi_size"):
        PrepConfig(roi_size=500)
    with pytest.raises(ValueError, match="crop_size"):
        PrepConfig(crop_size=301)
    with pytest.raises(ValueError, match="roi_box"):
        PrepConfig(roi_box=(100, 100, 100, 200))
    with pytest.raises(ValueError, match="manual"):
        PrepConfig(detector="manual", roi_box=None)
    with pytest.raises(ValueError, match="ncc_redetect_threshold"):
        PrepConfig(ncc_redetect_threshold=0.0)

    valid = PrepConfig(detector="manual", roi_box=(10, 20, 210, 220), crop_size=200)
    assert valid.roi_box == (10, 20, 210, 220)


def test_segment_config_validation() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        SegmentConfig(chunk_size=2)
    with pytest.raises(ValueError, match="chunk_overlap"):
        SegmentConfig(chunk_size=50, chunk_overlap=0)
    with pytest.raises(ValueError, match="chunk_overlap"):
        SegmentConfig(chunk_size=50, chunk_overlap=1)
    with pytest.raises(ValueError, match="chunk_overlap"):
        SegmentConfig(chunk_size=50, chunk_overlap=50)
    with pytest.raises(ValueError, match="glare_sensitivity"):
        SegmentConfig(glare_sensitivity=0.0)
    with pytest.raises(ValueError, match="Invalid device"):
        SegmentConfig(device="tpu:0")

    assert SegmentConfig(chunk_size=3, chunk_overlap=2).chunk_overlap == 2
    assert SegmentConfig(device="cuda:0").device == "cuda:0"
    assert SegmentConfig(device="mps").device == "mps"
    assert SegmentConfig(device="cpu").device == "cpu"


def test_reference_config_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ref_image_path"):
        ReferenceConfig(mode="image", ref_image_path=None)
    with pytest.raises(ValueError, match="max_clean_glare_pct"):
        ReferenceConfig(max_clean_glare_pct=105.0)
    with pytest.raises(ValueError, match="min_lens_area_ratio"):
        ReferenceConfig(min_lens_area_ratio=0.0)

    rc = ReferenceConfig(mode="image", ref_image_path=tmp_path / "clean.png")
    assert rc.ref_image_path == tmp_path / "clean.png"


def test_inpaint_config_validation() -> None:
    with pytest.raises(ValueError, match="chunk_len"):
        InpaintConfig(chunk_len=80)
    with pytest.raises(ValueError, match="chunk_len"):
        InpaintConfig(chunk_len=1)
    with pytest.raises(ValueError, match="min_overlap"):
        InpaintConfig(chunk_len=21, min_overlap=11)
    with pytest.raises(ValueError, match="size"):
        InpaintConfig(size=510)
    with pytest.raises(ValueError, match="steps"):
        InpaintConfig(steps=0)

    ic = InpaintConfig(chunk_len=21, min_overlap=10)
    assert ic.text_encoder_model_id == ic.model_id


def test_composite_and_pipeline_validation_and_source_alias_rejection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="crf"):
        CompositeConfig(crf=52)
    with pytest.raises(ValueError, match="preset"):
        CompositeConfig(preset="hyperfast")
    with pytest.raises(ValueError, match="color_match_strength"):
        CompositeConfig(color_match_strength=1.5)
    with pytest.raises(ValueError, match="max_batch_mem_mb"):
        CompositeConfig(max_batch_mem_mb=16)

    with pytest.raises(ValueError, match="roi_size"):
        PipelineConfig(
            input_path=tmp_path / "a.mp4",
            output_path=tmp_path / "b.mp4",
            prep=PrepConfig(roi_size=512),
            reference=ReferenceConfig(roi_size=256),
        )

    # Reject output aliasing input (direct, symlink, hardlink, and derived compare path)
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"video_data")
    with pytest.raises(ValueError, match="aliases input video"):
        PipelineConfig(input_path=src, output_path=src)

    sym = tmp_path / "clip_sym.mp4"
    sym.symlink_to(src)
    with pytest.raises(ValueError, match="aliases input video"):
        PipelineConfig(input_path=src, output_path=sym)

    hard = tmp_path / "clip_hard.mp4"
    os.link(src, hard)
    with pytest.raises(ValueError, match="aliases input video"):
        PipelineConfig(input_path=src, output_path=hard)

    # Derived compare path (`out_compare.mp4`) equals source!
    src_cmp = tmp_path / "out_compare.mp4"
    src_cmp.write_bytes(b"source_named_compare")
    with pytest.raises(ValueError, match="compare"):
        PipelineConfig(
            input_path=src_cmp,
            output_path=tmp_path / "out.mp4",
            composite=CompositeConfig(compare=True),
        )


def test_canonical_apache2_license_integrity() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    lic = (repo_root / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in lic
    assert "Version 2.0, January 2004" in lic
    assert lic.count('"Contributor" shall mean Licensor') == 1
    assert '"Source" form shall mean the preferred form' in lic
