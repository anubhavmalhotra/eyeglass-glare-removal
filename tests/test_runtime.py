"""Runtime and model identities exercised without downloading weights or requiring CUDA."""

from __future__ import annotations

import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from eyeglass_glare_removal import inpaint, segment
from eyeglass_glare_removal.config import InpaintConfig, SegmentConfig


class Device:
    def __init__(self, name):
        self.type, _, index = name.partition(":")
        self.index = int(index) if index else None

    def __str__(self):
        return self.type if self.index is None else f"{self.type}:{self.index}"


@pytest.fixture
def torch_stub(monkeypatch):
    state = SimpleNamespace(cuda=False, index=1, bf16=False, dtype_checks=[])
    torch = ModuleType("torch")
    torch.__version__ = "stub"
    torch.device = Device
    torch.float32, torch.float16, torch.bfloat16 = "fp32", "fp16", "bf16"
    torch.inference_mode = nullcontext
    torch.backends = SimpleNamespace()
    torch.Generator = lambda device: SimpleNamespace(manual_seed=lambda seed: (device, seed))

    @contextmanager
    def on_device(device):
        assert device.index is not None
        previous, state.index = state.index, device.index
        try:
            yield
        finally:
            state.index = previous

    def supports_bf16():
        state.dtype_checks.append(state.index)
        return state.bf16

    torch.cuda = SimpleNamespace(
        is_available=lambda: state.cuda, current_device=lambda: state.index, device=on_device,
        is_bf16_supported=supports_bf16, reset_peak_memory_stats=lambda _device: None,
        synchronize=lambda _device: None, empty_cache=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch, state


def test_segment_checkpoints_follow_auto_device_dtype_and_pinned_hub_commit(
    tmp_path: Path, monkeypatch, torch_stub,
) -> None:
    _, runtime = torch_stub
    current_revision = "a" * 40
    hub = ModuleType("huggingface_hub")
    hub.HfApi = lambda: SimpleNamespace(
        model_info=lambda model_id, timeout: SimpleNamespace(sha=current_revision)
    )
    hub.try_to_load_from_cache = lambda *_: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    loaded = []

    def load_model(model_id, **kwargs):
        loaded.append(("model", kwargs["revision"], kwargs["dtype"]))
        obj = SimpleNamespace()
        obj.to = lambda device: obj
        obj.eval = lambda: obj
        return obj

    def load_processor(model_id, **kwargs):
        loaded.append(("processor", kwargs["revision"], None))
        return object()

    transformers = ModuleType("transformers")
    transformers.Sam2VideoModel = SimpleNamespace(from_pretrained=load_model)
    transformers.Sam2VideoProcessor = SimpleNamespace(from_pretrained=load_processor)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    left, right = np.zeros((64, 64), dtype=bool), np.zeros((64, 64), dtype=bool)
    left[18:46, 5:27], right[18:46, 35:57] = True, True
    monkeypatch.setattr(
        segment, "_run_sam2_single_chunk",
        lambda _m, _p, frames, **_kw: ([left] * len(frames), [right] * len(frames)),
    )
    roi = tmp_path / "roi"
    roi.mkdir()
    Image.new("RGB", (64, 64), (110, 100, 90)).save(roi / "000000.png")
    meta = {"roi_size": 64, "frames": [{"filename": "000000.png", "frame_idx": 0}]}
    config = SegmentConfig(device="auto", write_qa_grid=False)

    def run():
        return segment.run_segment_stage(roi, meta, tmp_path / "masks", config)

    cpu = run()
    assert run()["chunks"][0]["from_checkpoint"]
    runtime.cuda = True
    gpu = run()
    assert not gpu["chunks"][0]["from_checkpoint"]
    runtime.bf16 = True
    bf16 = run()
    assert not bf16["chunks"][0]["from_checkpoint"]
    current_revision = "b" * 40
    new_model = run()
    assert not new_model["chunks"][0]["from_checkpoint"]
    assert len({stats["segment_generation"] for stats in (cpu, gpu, bf16, new_model)}) == 4
    assert loaded == [
        ("processor", "a" * 40, None), ("model", "a" * 40, "fp32"),
        ("processor", "a" * 40, None), ("model", "a" * 40, "fp16"),
        ("processor", "a" * 40, None), ("model", "a" * 40, "bf16"),
        ("processor", "b" * 40, None), ("model", "b" * 40, "bf16"),
    ]


def test_inpaint_and_embedding_caches_bind_auto_runtime_and_local_assets(
    tmp_path: Path, monkeypatch, torch_stub,
) -> None:
    from contextlib import nullcontext

    _, runtime = torch_stub
    model, encoder = tmp_path / "model", tmp_path / "encoder"
    for folder in (model, encoder):
        folder.mkdir()
        (folder / "config.json").write_text("{}")
        (folder / "model.safetensors").write_bytes(b"v1")
    tokenizer = encoder / "tokenizer.json"
    tokenizer.write_bytes(b"tokenizer 1")

    class Tensor:
        shape = (1, 512, 16)

        def cpu(self):
            return self

        def to(self, **_kw):
            return self

    class Encoder:
        text_encoder = SimpleNamespace(to=lambda _device: None)

        def encode_prompt(self, **_kw):
            return Tensor(), Tensor()

    monkeypatch.setattr(inpaint, "_load_text_encoder_pipeline", lambda *_: Encoder())
    monkeypatch.setattr(inpaint, "_resolve_expected_text_dim", lambda *_: 16)
    cached = {}
    monkeypatch.setattr(inpaint, "load_prompt_embeds_safetensors", lambda key, **_kw: cached.get(key))
    monkeypatch.setattr(inpaint, "save_prompt_embeds_safetensors", lambda key, *values: cached.update({key: values}))
    monkeypatch.setattr(inpaint, "gpu_file_lock", lambda _path: nullcontext())
    monkeypatch.setattr(
        inpaint, "load_vace_denoiser",
        lambda *_a, **_kw: lambda **kwargs: SimpleNamespace(
            frames=[np.full((len(kwargs["video"]), 64, 64, 3), 0.5, dtype=np.float32)]
        ),
    )
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    frame = tmp_path / "000000.png"
    Image.new("RGB", (64, 64)).save(frame)
    Image.new("L", (64, 64), 255).save(mask_dir / frame.name)
    cfg = InpaintConfig(size=64, device="auto", model_id=str(model), text_encoder_model_id=str(encoder))

    def run():
        return inpaint.run_inpaint_chunks([frame], mask_dir, tmp_path / "inpaint", cfg)

    fingerprints = {run()["config_fingerprint"]}
    assert run()["chunks_resumed"] == 1
    assert len(cached) == 1

    runtime.cuda = True
    gpu = run()
    fingerprints.add(gpu["config_fingerprint"])
    assert gpu["chunks_resumed"] == 0 and len(cached) == 2
    runtime.index = 2
    alternate_gpu = run()
    fingerprints.add(alternate_gpu["config_fingerprint"])
    assert alternate_gpu["chunks_resumed"] == 0 and len(cached) == 3

    (encoder / "model.safetensors").write_bytes(b"v2")
    new_weights = run()
    fingerprints.add(new_weights["config_fingerprint"])
    assert new_weights["chunks_resumed"] == 0 and len(cached) == 4
    tokenizer.write_bytes(b"tokenizer 2")
    new_tokenizer = run()
    fingerprints.add(new_tokenizer["config_fingerprint"])
    assert new_tokenizer["chunks_resumed"] == 0 and len(cached) == 5

    (model / "model.safetensors").write_bytes(b"v2")
    new_denoiser = run()
    fingerprints.add(new_denoiser["config_fingerprint"])
    assert new_denoiser["chunks_resumed"] == 0 and len(cached) == 5
    assert len(fingerprints) == 6


def test_nf4_current_cuda_ordinal_controls_dtype_probe_and_model_allocation(
    monkeypatch, torch_stub,
) -> None:
    torch, runtime = torch_stub
    runtime.cuda = True
    runtime.index = 1
    recorded = {}
    torch.zeros = lambda shape, device, dtype: recorded.update(probe=(str(device), dtype)) or "input"
    bnb = ModuleType("bitsandbytes")
    bnb.functional = SimpleNamespace(
        quantize_4bit=lambda *a, **kw: ("quantized", "state"), dequantize_4bit=lambda *a, **kw: None,
    )
    monkeypatch.setitem(sys.modules, "bitsandbytes", bnb)

    def load_vae(*_, **kwargs):
        recorded["vae_revision"] = kwargs["revision"]
        return SimpleNamespace(to=lambda device: recorded.update(vae=str(device)))

    def load_transformer(*_, **kwargs):
        recorded.update(transformer=kwargs["device_map"], dtype=kwargs["torch_dtype"])
        assert kwargs["revision"] == "b" * 40

    diffusers = ModuleType("diffusers")
    diffusers.AutoencoderKLWan = SimpleNamespace(from_pretrained=load_vae)
    diffusers.WanVACETransformer3DModel = SimpleNamespace(from_pretrained=load_transformer)
    diffusers.BitsAndBytesConfig = lambda **kwargs: kwargs
    diffusers.WanVACEPipeline = lambda **kwargs: kwargs
    scheduler = ModuleType("diffusers.schedulers.scheduling_unipc_multistep")
    scheduler.UniPCMultistepScheduler = SimpleNamespace(
        from_pretrained=lambda *a, **kw: SimpleNamespace(config=kw),
        from_config=lambda config, **kw: config,
    )
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, scheduler.__name__, scheduler)

    result = inpaint.load_vace_denoiser(
        "org/model", quantize_4bit=True, device=torch.device("cuda"), dtype="bf16",
        model_identity="hf:" + "b" * 40,
    )
    assert runtime.dtype_checks == [1]
    assert recorded == {
        "probe": ("cuda:1", "fp16"), "vae": "cuda:1", "vae_revision": "b" * 40,
        "transformer": {"": 1}, "dtype": "fp16",
    }
    assert result["scheduler"]["revision"] == "b" * 40
