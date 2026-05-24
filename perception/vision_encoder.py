"""Vision Tile Encoder — Compress webcam/screen frames into 512-dim embedding tiles.

Backends (in order of preference):
  1. SigLIP via transformers (best quality)
  2. CLIP via openai-clip (balanced)
  3. MobileViT or custom CNN (ultra-light)
  4. ONNXRuntime (edge / Jetson)

All backends converge to a 512-dim float32 embedding that feeds into
NerveTopology as a first-class vision tile.
"""
from __future__ import annotations

__all__ = ["VisionTileEncoder", "EncoderBackend"]

import hashlib
import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Backend availability detection ───────────────────────────

_HAS_TRANSFORMERS = False
try:
    import transformers
    from transformers import AutoImageProcessor, AutoModel
    _HAS_TRANSFORMERS = True
except Exception:
    pass

_HAS_CLIP = False
try:
    import clip as _clip_module
    _HAS_CLIP = True
except Exception:
    pass

_HAS_ONNX = False
try:
    import onnxruntime as ort
    _HAS_ONNX = True
except Exception:
    pass

_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
except Exception:
    pass


class EncoderBackend(Enum):
    """Available vision encoder backends, ordered by quality."""
    SIGLIP = auto()      # transformers AutoModel — best quality
    CLIP = auto()        # openai-clip — balanced
    MOBILEVIT = auto()   # lightweight torch CNN
    ONNX = auto()        # onnxruntime edge inference
    RANDOM_PROJECTION = auto()  # deterministic fallback, no deps


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a vision model backend."""
    name: str
    embedding_dim: int
    input_size: tuple[int, int]  # (H, W)
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


# Predefined specs for known models
MODEL_SPECS: dict[str, ModelSpec] = {
    "siglip": ModelSpec(
        name="google/siglip-base-patch16-224",
        embedding_dim=768,  # SigLIP base outputs 768; we project to 512
        input_size=(224, 224),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "clip": ModelSpec(
        name="ViT-B/32",
        embedding_dim=512,
        input_size=(224, 224),
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
    "mobilevit": ModelSpec(
        name="mobilevit_s",
        embedding_dim=512,
        input_size=(256, 256),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    "onnx": ModelSpec(
        name="vision_encoder.onnx",
        embedding_dim=512,
        input_size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    "random_projection": ModelSpec(
        name="random_projection",
        embedding_dim=512,
        input_size=(224, 224),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
}


class VisionTileEncoder:
    """Compress vision frames into 512-dim embedding tiles for NerveTopology.

    Args:
        model: Backend identifier — 'siglip', 'clip', 'mobilevit', 'onnx',
               or 'random_projection' (deterministic fallback).
        device: Compute device — 'cpu', 'cuda', 'mps' (Apple Silicon).
        target_dim: Output embedding dimension (default 512).
    """

    def __init__(
        self,
        model: str = "siglip",
        device: str = "cpu",
        target_dim: int = 512,
    ):
        self.model_name = model
        self.device = device
        self.target_dim = target_dim
        self.spec = MODEL_SPECS.get(model, MODEL_SPECS["random_projection"])

        # Runtime state
        self._backend: EncoderBackend | None = None
        self._processor: Any = None
        self._model: Any = None
        self._session: Any = None          # ONNX session
        self._projection: np.ndarray | None = None  # for dim-mismatch projection
        self._frame_count = 0
        self._fps_window: list[float] = []

        self._init_backend()

    # ── Backend initialisation ────────────────────────────────

    def _init_backend(self) -> None:
        """Auto-detect and initialise the best available backend."""
        requested = self.model_name.lower()

        # Map requested name to backend enum with fallback chain
        preference_order = [
            ("siglip", EncoderBackend.SIGLIP, _HAS_TRANSFORMERS),
            ("clip", EncoderBackend.CLIP, _HAS_CLIP),
            ("mobilevit", EncoderBackend.MOBILEVIT, _HAS_TORCH),
            ("onnx", EncoderBackend.ONNX, _HAS_ONNX),
            ("random_projection", EncoderBackend.RANDOM_PROJECTION, True),
        ]

        # Try the requested one first, then walk the chain
        ordered = sorted(
            preference_order,
            key=lambda t: 0 if t[0] == requested else 1,
        )

        for name, backend, available in ordered:
            if not available:
                continue
            try:
                self._setup_backend(backend)
                self._backend = backend
                logger.info(
                    "VisionTileEncoder initialised: backend=%s device=%s dim=%d",
                    backend.name, self.device, self.target_dim,
                )
                return
            except Exception as exc:
                logger.warning("Backend %s failed: %s", backend.name, exc)
                continue

        # Absolute fallback — never raises
        self._backend = EncoderBackend.RANDOM_PROJECTION
        self._setup_random_projection()
        logger.info(
            "VisionTileEncoder fallback: backend=RANDOM_PROJECTION dim=%d",
            self.target_dim,
        )

    def _setup_backend(self, backend: EncoderBackend) -> None:
        """Concrete backend setup."""
        if backend == EncoderBackend.SIGLIP:
            self._setup_siglip()
        elif backend == EncoderBackend.CLIP:
            self._setup_clip()
        elif backend == EncoderBackend.MOBILEVIT:
            self._setup_mobilevit()
        elif backend == EncoderBackend.ONNX:
            self._setup_onnx()
        elif backend == EncoderBackend.RANDOM_PROJECTION:
            self._setup_random_projection()

    def _setup_siglip(self) -> None:
        if not _HAS_TRANSFORMERS:
            raise ImportError("transformers not installed")
        self._processor = AutoImageProcessor.from_pretrained(
            self.spec.name, use_fast=True
        )
        self._model = AutoModel.from_pretrained(self.spec.name)
        self._model.eval()
        if _HAS_TORCH:
            import torch
            self._model.to(self.device)
        self._maybe_build_projection(self.spec.embedding_dim)

    def _setup_clip(self) -> None:
        if not _HAS_CLIP:
            raise ImportError("clip not installed")
        import clip
        self._model, self._preprocess = clip.load(
            self.spec.name, device=self.device
        )
        self._model.eval()
        self._processor = self._preprocess
        # CLIP ViT-B/32 already emits 512-dim

    def _setup_mobilevit(self) -> None:
        if not _HAS_TORCH:
            raise ImportError("torch not installed")
        import torch
        import torchvision.models as models
        # Use MobileViT from timm if available, else MobileNetV3 as proxy
        try:
            import timm
            self._model = timm.create_model(
                "mobilevit_s", pretrained=True, num_classes=0
            )
            self.spec = MODEL_SPECS["mobilevit"]
        except Exception:
            # Fallback: MobileNetV3 feature extractor
            mn = models.mobilenet_v3_small(pretrained=True)
            self._model = torch.nn.Sequential(
                mn.features,
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
            )
            self.spec = ModelSpec(
                name="mobilenet_v3_proxy",
                embedding_dim=576,
                input_size=(224, 224),
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        self._model.to(self.device)
        self._model.eval()
        self._maybe_build_projection(self.spec.embedding_dim)

    def _setup_onnx(self) -> None:
        if not _HAS_ONNX:
            raise ImportError("onnxruntime not installed")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.device == "cuda":
            prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            prov = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(
            self.spec.name, opts, providers=prov
        )
        self._input_name = self._session.get_inputs()[0].name

    def _setup_random_projection(self) -> None:
        """Deterministic random projection — no neural network needed."""
        rng = np.random.RandomState(42)
        # Project flattened image patches to target_dim
        patch_dim = 224 * 224 * 3
        self._projection = rng.randn(patch_dim, self.target_dim).astype(np.float32)
        self._projection /= np.linalg.norm(self._projection, axis=0, keepdims=True)

    def _maybe_build_projection(self, in_dim: int) -> None:
        """Build a linear projection if model output dim != target dim."""
        if in_dim != self.target_dim:
            rng = np.random.RandomState(in_dim)
            proj = rng.randn(in_dim, self.target_dim).astype(np.float32)
            proj /= np.linalg.norm(proj, axis=0, keepdims=True)
            self._projection = proj
            logger.debug("Built projection %d → %d", in_dim, self.target_dim)

    # ── Public API ────────────────────────────────────────────

    @property
    def backend(self) -> str:
        return self._backend.name if self._backend else "none"

    @property
    def fps(self) -> float:
        """Rolling average FPS over the last 10 frames."""
        if not self._fps_window:
            return 0.0
        return float(np.mean(self._fps_window))

    def encode_frame(self, frame: np.ndarray) -> np.ndarray:
        """Encode a single frame (H×W×3 uint8) into a 512-dim float32 embedding.

        Args:
            frame: RGB image array, shape (H, W, 3), dtype uint8.

        Returns:
            np.ndarray of shape (512,), dtype float32, L2-normalised.
        """
        t0 = time.perf_counter()
        embedding = self._encode_impl(frame)
        # L2 normalise
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm

        # FPS bookkeeping
        elapsed = time.perf_counter() - t0
        self._fps_window.append(1.0 / max(elapsed, 1e-6))
        if len(self._fps_window) > 10:
            self._fps_window.pop(0)
        self._frame_count += 1

        return embedding.astype(np.float32)

    def encode_batch(self, frames: list[np.ndarray]) -> np.ndarray:
        """Batch encode for efficiency.

        Args:
            frames: List of RGB arrays, each (H, W, 3) uint8.

        Returns:
            np.ndarray of shape (N, 512), dtype float32.
        """
        if not frames:
            return np.zeros((0, self.target_dim), dtype=np.float32)

        # Batch processing per-backend for efficiency
        if self._backend == EncoderBackend.SIGLIP and _HAS_TRANSFORMERS:
            return self._batch_siglip(frames)
        if self._backend == EncoderBackend.CLIP and _HAS_CLIP:
            return self._batch_clip(frames)
        if self._backend in (EncoderBackend.MOBILEVIT,) and _HAS_TORCH:
            return self._batch_torch(frames)

        # Fallback: loop
        return np.stack([self.encode_frame(f) for f in frames], axis=0)

    def to_tile(
        self,
        embedding: np.ndarray,
        source: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convert embedding to a NerveTopology-compatible tile dict.

        Args:
            embedding: 512-dim np.ndarray from encode_frame / encode_batch.
            source: Identifier like 'webcam_0' or 'screen_capture'.
            extra_metadata: Optional additional metadata merged in.

        Returns:
            Tile dict: {
                'type': 'vision',
                'source': str,
                'embedding': np.ndarray,
                'timestamp': float,
                'metadata': dict,
            }
        """
        meta: dict[str, Any] = {
            "model": self.model_name,
            "backend": self.backend,
            "device": self.device,
            "fps": self.fps,
            "frame_count": self._frame_count,
        }
        if extra_metadata:
            meta.update(extra_metadata)

        return {
            "type": "vision",
            "source": source,
            "embedding": embedding,
            "timestamp": time.time(),
            "metadata": meta,
        }

    # ── Internal encoding implementations ────────────────────

    def _encode_impl(self, frame: np.ndarray) -> np.ndarray:
        """Dispatch to the active backend."""
        if self._backend == EncoderBackend.SIGLIP:
            return self._encode_siglip(frame)
        if self._backend == EncoderBackend.CLIP:
            return self._encode_clip(frame)
        if self._backend == EncoderBackend.MOBILEVIT:
            return self._encode_mobilevit(frame)
        if self._backend == EncoderBackend.ONNX:
            return self._encode_onnx(frame)
        return self._encode_random_projection(frame)

    # ── SigLIP ──────────────────────────────────────────────

    def _encode_siglip(self, frame: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image
        pil = Image.fromarray(frame).convert("RGB").resize(self.spec.input_size[::-1])
        inputs = self._processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.get_image_features(**inputs)
        emb = out.cpu().numpy().flatten()
        if self._projection is not None:
            emb = emb @ self._projection
        return emb

    def _batch_siglip(self, frames: list[np.ndarray]) -> np.ndarray:
        import torch
        from PIL import Image
        pils = [
            Image.fromarray(f).convert("RGB").resize(self.spec.input_size[::-1])
            for f in frames
        ]
        inputs = self._processor(images=pils, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.get_image_features(**inputs)
        emb = out.cpu().numpy()
        if self._projection is not None:
            emb = emb @ self._projection
        return emb.astype(np.float32)

    # ── CLIP ────────────────────────────────────────────────

    def _encode_clip(self, frame: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image
        import clip
        pil = Image.fromarray(frame).convert("RGB")
        image = self._preprocess(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self._model.encode_image(image).cpu().numpy().flatten()
        return emb

    def _batch_clip(self, frames: list[np.ndarray]) -> np.ndarray:
        import torch
        from PIL import Image
        pils = [Image.fromarray(f).convert("RGB") for f in frames]
        images = torch.stack([self._preprocess(p) for p in pils]).to(self.device)
        with torch.no_grad():
            emb = self._model.encode_image(images).cpu().numpy()
        return emb.astype(np.float32)

    # ── MobileViT / MobileNet proxy ───────────────────────────

    def _encode_mobilevit(self, frame: np.ndarray) -> np.ndarray:
        import torch
        from torchvision import transforms
        tensor = transforms.ToTensor()(frame)
        tensor = transforms.Normalize(
            mean=self.spec.mean, std=self.spec.std
        )(tensor)
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self._model(tensor).cpu().numpy().flatten()
        if self._projection is not None:
            emb = emb @ self._projection
        return emb

    def _batch_torch(self, frames: list[np.ndarray]) -> np.ndarray:
        import torch
        from torchvision import transforms
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=self.spec.mean, std=self.spec.std)
        tensors = [normalize(to_tensor(f)) for f in frames]
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            emb = self._model(batch).cpu().numpy()
        if self._projection is not None:
            emb = emb @ self._projection
        return emb.astype(np.float32)

    # ── ONNX ────────────────────────────────────────────────

    def _encode_onnx(self, frame: np.ndarray) -> np.ndarray:
        tensor = self._preprocess_numpy(frame)
        emb = self._session.run(None, {self._input_name: tensor})[0].flatten()
        if self._projection is not None:
            emb = emb @ self._projection
        return emb

    def _preprocess_numpy(self, frame: np.ndarray) -> np.ndarray:
        """Resize, normalise, CHW format."""
        from PIL import Image
        pil = Image.fromarray(frame).convert("RGB").resize(self.spec.input_size[::-1])
        arr = np.array(pil).astype(np.float32) / 255.0
        arr = (arr - np.array(self.spec.mean)) / np.array(self.spec.std)
        arr = np.transpose(arr, (2, 0, 1))  # HWC → CHW
        return np.expand_dims(arr, axis=0).astype(np.float32)

    # ── Random projection (fallback) ──────────────────────────

    def _encode_random_projection(self, frame: np.ndarray) -> np.ndarray:
        """Deterministic random projection of resized frame patches."""
        from PIL import Image
        pil = Image.fromarray(frame).convert("RGB").resize((224, 224))
        arr = np.array(pil).astype(np.float32) / 255.0
        flat = arr.flatten()
        # Pad or truncate to match projection matrix
        patch_dim = self._projection.shape[0]
        if len(flat) < patch_dim:
            flat = np.pad(flat, (0, patch_dim - len(flat)), mode="edge")
        elif len(flat) > patch_dim:
            flat = flat[:patch_dim]
        emb = flat @ self._projection
        return emb

    # ── Convenience: downsample to NerveTopology signal_dim ───

    def to_signal(
        self,
        embedding: np.ndarray,
        signal_dim: int = 64,
        seed: int | None = None,
    ) -> np.ndarray:
        """Project a 512-dim vision embedding down to a NerveTopology signal.

        This bridges the vision tile format to the existing 64-dim (or other)
        signal_dim used by RoomGrid / NerveTopology.

        Uses a deterministic random projection seeded by the embedding hash
        so the same image always maps to the same signal.
        """
        if seed is None:
            seed = int(hashlib.md5(embedding.tobytes()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        proj = rng.randn(len(embedding), signal_dim).astype(np.float32)
        proj /= np.linalg.norm(proj, axis=0, keepdims=True)
        signal = embedding @ proj
        # Normalise
        norm = np.linalg.norm(signal)
        if norm > 1e-8:
            signal /= norm
        return signal
