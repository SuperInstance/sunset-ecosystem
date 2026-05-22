#!/usr/bin/env python3
"""Demo: Audio tiles → NerveTopology RoomGrid.

Captures audio chunks (microphone, system audio, or synthetic), encodes them
with AudioTileEncoder, projects embeddings down to NerveTopology signal_dim,
and pushes them into a RoomGrid via the topology tick cycle.

Usage:
    python scripts/demo_audio_tiles.py --source microphone --ticks 100
    python scripts/demo_audio_tiles.py --source system --ticks 50
    python scripts/demo_audio_tiles.py --source synthetic --ticks 50
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np

# Ensure sunset-ecosystem is importable when run from repo root
sys.path.insert(0, ".")

from perception import AudioTileEncoder, MicrophoneCapture, SystemAudioCapture
from nerve.topology import NerveTopology

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("demo_audio_tiles")


def synthetic_audio(seed: int, sample_rate: int = 16000, duration_sec: float = 1.0) -> np.ndarray:
    """Generate a deterministic synthetic audio segment for testing without hardware."""
    rng = np.random.RandomState(seed)
    n_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    # Mix of sine waves
    freq1 = 220.0 + rng.uniform(-20, 20)
    freq2 = 440.0 + rng.uniform(-30, 30)
    signal = (
        0.5 * np.sin(2 * np.pi * freq1 * t)
        + 0.3 * np.sin(2 * np.pi * freq2 * t)
        + 0.05 * rng.randn(n_samples)
    )
    return signal.astype(np.float32)


def run_demo(
    source: str,
    ticks: int,
    signal_dim: int,
    model: str,
    n_rooms: int = 50,
    n_fibers: int = 4,
) -> None:
    """Run the audio tile → topology demo."""
    log.info("=== Audio Tile Demo ===")
    log.info("source=%s model=%s signal_dim=%d rooms=%d fibers=%d", source, model, signal_dim, n_rooms, n_fibers)

    # ── Encoder ──────────────────────────────────────────────
    encoder = AudioTileEncoder(model=model, device="cpu", target_dim=512)
    log.info("Encoder backend: %s", encoder.backend)

    # ── Capture source ───────────────────────────────────────
    capture: MicrophoneCapture | SystemAudioCapture | None = None
    if source == "microphone":
        try:
            capture = MicrophoneCapture()
            if not capture.open():
                log.error("Microphone open failed — falling back to synthetic")
                source = "synthetic"
        except ImportError as exc:
            log.error("Microphone not available: %s — falling back to synthetic", exc)
            source = "synthetic"
    elif source == "system":
        try:
            capture = SystemAudioCapture()
            if not capture.open():
                log.error("System audio open failed — falling back to synthetic")
                source = "synthetic"
        except ImportError as exc:
            log.error("System audio not available: %s — falling back to synthetic", exc)
            source = "synthetic"
    else:
        log.info("Synthetic source — no hardware needed")

    # ── Topology ─────────────────────────────────────────────
    topo = NerveTopology(
        n_fibers=n_fibers,
        n_rooms=n_rooms,
        signal_dim=signal_dim,
        chaos=0.3,
        adapt_threshold=0.8,
        learning_rate=0.05,
    )
    log.info("Topology initialised: %r", topo)

    # ── Main loop ────────────────────────────────────────────
    segment_idx = 0
    for tick in range(ticks):
        t0 = time.perf_counter()

        # Acquire audio chunk
        if source == "synthetic":
            audio = synthetic_audio(seed=segment_idx, sample_rate=16000, duration_sec=1.0)
        elif capture is not None:
            chunk = capture.read()
            if chunk is None:
                log.warning("Capture failed at tick %d — using synthetic fallback", tick)
                audio = synthetic_audio(seed=segment_idx, sample_rate=16000, duration_sec=1.0)
            else:
                audio = chunk
        else:
            audio = synthetic_audio(seed=segment_idx, sample_rate=16000, duration_sec=1.0)

        # Encode
        emb = encoder.encode_segment(audio, sample_rate=16000)
        tile = encoder.to_tile(emb, source=source)

        # Bridge: 512-dim audio embedding → signal_dim for topology
        signal = encoder.to_signal(emb, signal_dim=signal_dim)

        # Inject as fiber-0 signal (audio is a dedicated perception fiber)
        signals = {"fiber-0": signal}
        # Other fibers get small random noise (unattended channels)
        for i in range(1, n_fibers):
            signals[f"fiber-{i}"] = np.random.randn(signal_dim).astype(np.float32) * 0.05

        result = topo.tick(signals=signals)

        if tick % 10 == 0 or tick == ticks - 1:
            log.info(
                "Tick %3d | rooms=%d routes=%d compiled=%d novel=%d | "
                "enc_latency=%.1f ms tile=%s",
                tick,
                result.rooms_fired,
                result.routes_activated,
                result.routes_compiled,
                result.novel_signals,
                encoder.latency_ms,
                tile["source"],
            )

        segment_idx += 1

    # ── Summary ──────────────────────────────────────────────
    stats = topo.stats
    log.info("=== Final stats ===")
    log.info("Ticks: %d", stats["tick"])
    log.info("Rooms active: %d / %d", stats["rooms_active"], stats["rooms"])
    log.info("Fibers compiled: %d / %d", stats["fibers_compiled"], stats["fibers"])
    log.info("Routes: %d | Channels: %d", stats["routes"], stats["channels"])
    log.info("Chaos: %.4f", stats["chaos"])

    # Compiled pathways
    pathways = topo.compiled_pathways()
    log.info("Compiled pathways: %d", len(pathways))
    for p in pathways[:5]:
        log.info("  %s → %s (strength %.2f)", p["source"], p["destination"], p["strength"])

    # Cleanup
    if capture is not None:
        capture.close()

    log.info("Demo complete in %.2f s", (time.perf_counter() - t0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audio Tile → NerveTopology demo")
    parser.add_argument(
        "--source",
        choices=["microphone", "system", "synthetic"],
        default="synthetic",
        help="Audio source (default: synthetic — no hardware needed)",
    )
    parser.add_argument(
        "--ticks", type=int, default=50, help="Number of topology ticks"
    )
    parser.add_argument(
        "--signal-dim", type=int, default=64, help="NerveTopology signal dimension"
    )
    parser.add_argument(
        "--model",
        choices=["whisper", "wav2vec2", "clap", "speechbrain", "openai_whisper", "onnx", "random_projection"],
        default="random_projection",
        help="Audio encoder backend (default: random_projection — no heavy deps)",
    )
    parser.add_argument(
        "--rooms", type=int, default=50, help="Number of RoomGrid rooms"
    )
    parser.add_argument(
        "--fibers", type=int, default=4, help="Number of topology fibers"
    )
    args = parser.parse_args()

    run_demo(
        source=args.source,
        ticks=args.ticks,
        signal_dim=args.signal_dim,
        model=args.model,
        n_rooms=args.rooms,
        n_fibers=args.fibers,
    )


if __name__ == "__main__":
    main()
