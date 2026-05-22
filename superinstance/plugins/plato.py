"""PLATO tile-store plugin for the superinstance-runtime event bus.

Wraps plato-core tile concepts (if available) or falls back to a
pure-Python tile registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from superinstance.runtime import CollectorPlugin, CompilerPlugin, SelectorPlugin

__all__ = ["TileArtifact", "PlatoCollector", "PlatoSelector", "PlatoCompiler"]


@dataclass(frozen=True, slots=True)
class TileArtifact:
    """A single tile observation from the PLATO store."""

    tile_id: str
    room_id: int
    content: str
    tags: list[str]
    entropy: float


class PlatoCollector(CollectorPlugin):
    """Collect tile observations from a context dict."""

    name = "plato"

    def collect(self, context: dict[str, Any]) -> list[TileArtifact]:
        """Extract tile observations from *context*.

        Expected context keys::

            tiles: list[dict] with keys:
                tile_id, room_id, content, tags, entropy
        """
        raw = context.get("tiles", [])
        artifacts: list[TileArtifact] = []
        for item in raw:
            artifacts.append(
                TileArtifact(
                    tile_id=str(item.get("tile_id", "unknown")),
                    room_id=int(item.get("room_id", -1)),
                    content=str(item.get("content", "")),
                    tags=list(item.get("tags", [])),
                    entropy=float(item.get("entropy", 0.0)),
                )
            )
        return artifacts


class PlatoSelector(SelectorPlugin):
    """Select high-entropy tiles (most information-dense)."""

    name = "plato"

    def select(
        self,
        artifacts: list[Any],
        context: dict[str, Any],
    ) -> list[TileArtifact]:
        """Filter to tiles with entropy above threshold."""
        threshold = float(context.get("entropy_threshold", 0.5))
        tiles = [a for a in artifacts if isinstance(a, TileArtifact) and a.entropy >= threshold]
        return sorted(tiles, key=lambda t: t.entropy, reverse=True)


class PlatoCompiler(CompilerPlugin):
    """Compile selected tiles into knowledge directives."""

    name = "plato"

    def compile(
        self,
        artifacts: list[Any],
        context: dict[str, Any],
    ) -> list[dict]:
        """Produce knowledge directives for each selected tile."""
        directives: list[dict] = []
        for a in artifacts:
            if not isinstance(a, TileArtifact):
                continue
            directives.append(
                {
                    "plugin": "plato",
                    "tile_id": a.tile_id,
                    "room_id": a.room_id,
                    "action": "ingest",
                    "summary": a.content[:200],
                    "tags": a.tags,
                    "entropy": a.entropy,
                }
            )
        return directives
