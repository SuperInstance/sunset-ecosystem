#!/usr/bin/env python3
"""Demo: Vision tiles → NerveTopology RoomGrid.

Captures frames (webcam or synthetic), encodes them with VisionTileEncoder,
projects embeddings down to NerveTopology signal_dim, and pushes them into
a RoomGrid via the topology tick cycle.

Usage:
    python scripts/demo_vision_tiles.py --source webcam --ticks 100
    python scripts/demo_vision_tiles.py --source screen --region 0,0,640,480
    python scripts/demo_vision_tiles.py --source synthetic --ticks 50
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np

# Ensure sunset-ecosystem is importable when run from repo root
sys.path.insert(0, ".")

from perception import VisionTileEncoder, WebcamCapture, ScreenCapture
from nerve.topology import NerveTopology

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
log = logging.getLogger("demo_vision_tiles")


def synthetic_frame(seed: int, size: tuple[int, int] = (480, 640)) -> np.ndarray:
    """Generate a deterministic synthetic RGB frame for testing without hardware."""
    rng = np.random.RandomState(seed)
    h, w = size
    # Perlin-ish noise via smooth random walk
    base = rng.randint(0, 256, size=(h // 8, w // 8, 3), dtype=np.uint8)
    from PIL import Image

    img = Image.fromarray(base).resize((w, h), Image.BILINEAR)
    frame = np.array(img)
    # Add some high-frequency noise
    noise = rng.randint(0, 32, size=(h, w, 3), dtype=np.uint8)
    frame = np.clip(
        frame.astype(np.int16) + noise.astype(np.int16) - 16, 0, 255
    ).astype(np.uint8)
    return frame


def run_demo(
    source: str,
    ticks: int,
    signal_dim: int,
    model: str,
    region: tuple[int, int, int, int] | None = None,
    n_rooms: int = 50,
    n_fibers: int = 4,
) -> None:
    """Run the vision tile → topology demo."""
    log.info("=== Vision Tile Demo ===")
    log.info(
        "source=%s model=%s signal_dim=%d rooms=%d fibers=%d",
        source,
        model,
        signal_dim,
        n_rooms,
        n_fibers,
    )

    # ── Encoder ──────────────────────────────────────────────
    encoder = VisionTileEncoder(model=model, device="cpu", target_dim=512)
    log.info("Encoder backend: %s", encoder.backend)

    # ── Capture source ───────────────────────────────────────
    capture: WebcamCapture | ScreenCapture | None = None
    if source == "webcam":
        capture = WebcamCapture(device_id=0)
        if not capture.open():
            log.error("Webcam open failed — falling back to synthetic")
            source = "synthetic"
    elif source == "screen":
        if region:
            monitor = {
                "left": region[0],
                "top": region[1],
                "width": region[2],
                "height": region[3],
            }
        else:
            monitor = None
        capture = ScreenCapture(monitor=monitor)
        capture.open()
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
    frame_idx = 0
    for tick in range(ticks):
        t0 = time.perf_counter()

        # Acquire frame
        if source == "synthetic":
            frame = synthetic_frame(seed=frame_idx, size=(480, 640))
        elif capture is not None:
            frame = capture.read()
            if frame is None:
                log.warning(
                    "Capture failed at tick %d — using synthetic fallback", tick
                )
                frame = synthetic_frame(seed=frame_idx, size=(480, 640))
        else:
            frame = synthetic_frame(seed=frame_idx, size=(480, 640))

        # Encode
        emb = encoder.encode_frame(frame)
        tile = encoder.to_tile(emb, source=source)

        # Bridge: 512-dim vision embedding → signal_dim for topology
        signal = encoder.to_signal(emb, signal_dim=signal_dim)

        # Inject as fiber-0 signal (vision is a dedicated perception fiber)
        signals = {f"fiber-0": signal}
        # Other fibers get small random noise (unattended channels)
        for i in range(1, n_fibers):
            signals[f"fiber-{i}"] = (
                np.random.randn(signal_dim).astype(np.float32) * 0.05
            )

        result = topo.tick(signals=signals)

        if tick % 10 == 0 or tick == ticks - 1:
            log.info(
                "Tick %3d | rooms=%d routes=%d compiled=%d novel=%d | "
                "enc_fps=%.1f tile=%s",
                tick,
                result.rooms_fired,
                result.routes_activated,
                result.routes_compiled,
                result.novel_signals,
                encoder.fps,
                tile["source"],
            )

        frame_idx += 1

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
        log.info(
            "  %s → %s (strength %.2f)", p["source"], p["destination"], p["strength"]
        )

    # Cleanup
    if capture is not None:
        capture.close()

    log.info("Demo complete in %.2f s", (time.perf_counter() - t0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision Tile → NerveTopology demo")
    parser.add_argument(
        "--source",
        choices=["webcam", "screen", "synthetic"],
        default="synthetic",
        help="Frame source (default: synthetic — no hardware needed)",
    )
    parser.add_argument(
        "--ticks", type=int, default=50, help="Number of topology ticks"
    )
    parser.add_argument(
        "--signal-dim", type=int, default=64, help="NerveTopology signal dimension"
    )
    parser.add_argument(
        "--model",
        choices=["siglip", "clip", "mobilevit", "onnx", "random_projection"],
        default="random_projection",
        help="Vision encoder backend (default: random_projection — no heavy deps)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Screen capture region: left,top,width,height (e.g. 0,0,640,480)",
    )
    parser.add_argument(
        "--rooms", type=int, default=50, help="Number of RoomGrid rooms"
    )
    parser.add_argument(
        "--fibers", type=int, default=4, help="Number of topology fibers"
    )
    args = parser.parse_args()

    region: tuple[int, int, int, int] | None = None
    if args.region:
        parts = [int(x.strip()) for x in args.region.split(",")]
        if len(parts) != 4:
            raise ValueError("--region must be left,top,width,height")
        region = tuple(parts)  # type: ignore[assignment]

    run_demo(
        source=args.source,
        ticks=args.ticks,
        signal_dim=args.signal_dim,
        model=args.model,
        region=region,
        n_rooms=args.rooms,
        n_fibers=args.fibers,
    )


if __name__ == "__main__":
    main()
