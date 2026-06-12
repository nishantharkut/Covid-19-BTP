from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from covid_audio_btp.torch_embedding_features import _as_numpy


BACKEND_ALIASES = {
    "wav2vec2": "wav2vec2_torchaudio",
    "wav2vec2_base": "wav2vec2_torchaudio",
    "wav2vec2_torchaudio": "wav2vec2_torchaudio",
    "beats": "beats_official",
    "beats_official": "beats_official",
    "panns": "panns_cnn14_official",
    "panns_cnn14": "panns_cnn14_official",
    "panns_cnn14_official": "panns_cnn14_official",
}


def normalize_backend_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    if normalized not in BACKEND_ALIASES:
        raise ValueError(f"Unsupported SSL embedding backend: {name}")
    return BACKEND_ALIASES[normalized]


def embedding_array_from_model_output(output) -> np.ndarray:
    if isinstance(output, dict):
        for key in ("embedding", "features", "last_hidden_state", "clipwise_output"):
            if key in output:
                return embedding_array_from_model_output(output[key])
        raise ValueError(f"Unsupported model output dictionary keys: {sorted(output)}")
    if isinstance(output, tuple):
        if not output:
            raise ValueError("Empty tuple returned by model")
        return embedding_array_from_model_output(output[0])
    if isinstance(output, list):
        if not output:
            raise ValueError("Empty list returned by model")
        return embedding_array_from_model_output(output[-1])
    return _as_numpy(output)


def resolve_torch_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


class Wav2Vec2TorchaudioExtractor:
    representation_name = "wav2vec2_base"
    sample_rate = 16_000

    def __init__(self, device: str = "auto") -> None:
        import torch
        import torchaudio

        self.torch = torch
        self.device = resolve_torch_device(device)
        self.bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.model = self.bundle.get_model().to(self.device).eval()

    def embed_batch(self, waveforms: np.ndarray) -> np.ndarray:
        tensor = self.torch.as_tensor(waveforms, dtype=self.torch.float32, device=self.device)
        with self.torch.inference_mode():
            output = self.model.extract_features(tensor)
        return embedding_array_from_model_output(output)


class BEATsOfficialExtractor:
    representation_name = "beats_official"
    sample_rate = 16_000

    def __init__(self, checkpoint_path: str | Path, source_dir: str | Path, device: str = "auto") -> None:
        import torch

        self.torch = torch
        self.device = resolve_torch_device(device)
        source = str(Path(source_dir).resolve())
        if source not in sys.path:
            sys.path.insert(0, source)
        from BEATs import BEATs, BEATsConfig

        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        model = BEATs(BEATsConfig(checkpoint["cfg"]))
        model.load_state_dict(checkpoint["model"])
        self.model = model.to(self.device).eval()

    def embed_batch(self, waveforms: np.ndarray) -> np.ndarray:
        tensor = self.torch.as_tensor(waveforms, dtype=self.torch.float32, device=self.device)
        padding_mask = self.torch.zeros(tensor.shape, dtype=self.torch.bool, device=self.device)
        with self.torch.inference_mode():
            output = self.model.extract_features(tensor, padding_mask=padding_mask)
        return embedding_array_from_model_output(output)


class PannsCnn14OfficialExtractor:
    representation_name = "panns_cnn14"
    sample_rate = 32_000

    def __init__(self, checkpoint_path: str | Path, source_dir: str | Path, device: str = "auto") -> None:
        import torch

        self.torch = torch
        self.device = resolve_torch_device(device)
        source_path = Path(source_dir).resolve()
        for source in (source_path, source_path / "pytorch"):
            source_text = str(source)
            if source_text not in sys.path:
                sys.path.insert(0, source_text)
        from pytorch.models import Cnn14

        model = Cnn14(
            sample_rate=self.sample_rate,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14_000,
            classes_num=527,
        )
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        model.load_state_dict(state_dict, strict=False)
        self.model = model.to(self.device).eval()

    def embed_batch(self, waveforms: np.ndarray) -> np.ndarray:
        tensor = self.torch.as_tensor(waveforms, dtype=self.torch.float32, device=self.device)
        with self.torch.inference_mode():
            output = self.model(tensor)
        return embedding_array_from_model_output(output)


def create_ssl_extractor(
    backend: str,
    *,
    checkpoint_path: str | Path | None = None,
    source_dir: str | Path | None = None,
    device: str = "auto",
):
    normalized = normalize_backend_name(backend)
    if normalized == "wav2vec2_torchaudio":
        return Wav2Vec2TorchaudioExtractor(device=device)
    if normalized == "beats_official":
        if checkpoint_path is None or source_dir is None:
            raise ValueError("BEATs extraction requires --checkpoint-path and --source-dir")
        return BEATsOfficialExtractor(checkpoint_path=checkpoint_path, source_dir=source_dir, device=device)
    if normalized == "panns_cnn14_official":
        if checkpoint_path is None or source_dir is None:
            raise ValueError("PANNs CNN14 extraction requires --checkpoint-path and --source-dir")
        return PannsCnn14OfficialExtractor(checkpoint_path=checkpoint_path, source_dir=source_dir, device=device)
    raise ValueError(f"Unsupported SSL embedding backend: {backend}")
