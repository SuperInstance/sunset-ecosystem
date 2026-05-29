"""Text diff engine for comparing agent outputs.

Unified diff-style comparison with word-level granularity. Produces
human-readable deltas for evaluating breeding candidate quality.

Usage:
    diff = DiffEngine()
    result = diff.compare("hello world", "hello fleet")
    # result.ratio ~ 0.5, result.chunks show differences
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DiffChunk:
    """A single diff chunk."""

    tag: str  # "equal", "insert", "delete"
    old_start: int
    old_end: int
    new_start: int
    new_end: int
    old_text: str = ""
    new_text: str = ""


@dataclass
class DiffResult:
    """Result of a diff comparison."""

    chunks: List[DiffChunk] = field(default_factory=list)

    def ratio(self) -> float:
        """Similarity ratio 0.0-1.0."""
        if not self.chunks:
            return 1.0
        equal_chars = sum(
            len(c.old_text)
            for c in self.chunks
            if c.tag == "equal"
        )
        total_chars = sum(
            max(len(c.old_text), len(c.new_text))
            for c in self.chunks
        )
        if total_chars == 0:
            return 1.0
        return equal_chars / total_chars

    def unified_format(self, old_label: str = "old", new_label: str = "new") -> str:
        """Render as unified diff text."""
        lines: List[str] = []
        for chunk in self.chunks:
            if chunk.tag == "equal":
                for line in chunk.old_text.splitlines():
                    lines.append(f" {line}")
            elif chunk.tag == "delete":
                for line in chunk.old_text.splitlines():
                    lines.append(f"-{line}")
            elif chunk.tag == "insert":
                for line in chunk.new_text.splitlines():
                    lines.append(f"+{line}")
        return "\n".join(lines)


class DiffEngine:
    """
    Myers' diff algorithm (simplified O(ND)) for word-level comparison.
    """

    def __init__(self):
        pass

    def compare(self, old: str, new: str) -> DiffResult:
        """Compare two strings and return diff result."""
        old_words = old.split() if old else []
        new_words = new.split() if new else []
        chunks = self._myers_diff(old_words, new_words)
        return DiffResult(chunks=chunks)

    def compare_lines(self, old: str, new: str) -> DiffResult:
        """Compare two strings line by line."""
        old_lines = old.splitlines() if old else []
        new_lines = new.splitlines() if new else []
        chunks = self._myers_diff(old_lines, new_lines)
        return DiffResult(chunks=chunks)

    # ------------------------------------------------------------------
    # Myers diff (simplified)
    # ------------------------------------------------------------------

    def _myers_diff(
        self,
        old: List[str],
        new: List[str],
    ) -> List[DiffChunk]:
        """Simplified O(ND) diff."""
        n, m = len(old), len(new)
        max_d = n + m
        if max_d == 0:
            return []

        # Initialize forward path
        size = 2 * max_d + 1
        v: List[int] = [0] * size
        trace: List[List[int]] = []

        for d in range(max_d + 1):
            trace.append(v[:])
            for k in range(-d, d + 1, 2):
                if k == -d or (k != d and v[k - 1 + max_d] < v[k + 1 + max_d]):
                    x = v[k + 1 + max_d]
                else:
                    x = v[k - 1 + max_d] + 1
                y = x - k
                while x < n and y < m and old[x] == new[y]:
                    x += 1
                    y += 1
                v[k + max_d] = x
                if x >= n and y >= m:
                    return self._backtrack(trace, old, new, d, k)

        return []

    def _backtrack(
        self,
        trace: List[List[int]],
        old: List[str],
        new: List[str],
        d_final: int,
        k_final: int,
    ) -> List[DiffChunk]:
        """Reconstruct diff from trace."""
        x, y = len(old), len(new)
        chunks: List[DiffChunk] = []
        old_buf: List[str] = []
        new_buf: List[str] = []

        for d in range(d_final, -1, -1):
            v = trace[d]
            k = k_final
            if k == -d or (k != d and v[k - 1 + len(old) + len(new)] < v[k + 1 + len(old) + len(new)]):
                prev_k = k + 1
            else:
                prev_k = k - 1
            prev_x = v[prev_k + len(old) + len(new)]
            prev_y = prev_x - prev_k

            # Equal elements (diagonal moves)
            while x > prev_x and y > prev_y:
                x -= 1
                y -= 1
                old_buf.insert(0, old[x])
                new_buf.insert(0, new[y])

            if old_buf or new_buf:
                text = " ".join(old_buf)
                chunks.insert(
                    0,
                    DiffChunk(
                        tag="equal",
                        old_start=x,
                        old_end=x + len(old_buf),
                        new_start=y,
                        new_end=y + len(new_buf),
                        old_text=text,
                        new_text=text,
                    ),
                )
                old_buf, new_buf = [], []

            if d > 0:
                if x == prev_x:
                    # Insertion
                    new_buf.insert(0, new[y - 1])
                    y -= 1
                    chunks.insert(
                        0,
                        DiffChunk(
                            tag="insert",
                            old_start=x,
                            old_end=x,
                            new_start=y,
                            new_end=y + 1,
                            new_text=" ".join(new_buf),
                        ),
                    )
                    new_buf = []
                else:
                    # Deletion
                    old_buf.insert(0, old[x - 1])
                    x -= 1
                    chunks.insert(
                        0,
                        DiffChunk(
                            tag="delete",
                            old_start=x,
                            old_end=x + 1,
                            new_start=y,
                            new_end=y,
                            old_text=" ".join(old_buf),
                        ),
                    )
                    old_buf = []

            k_final = prev_k

        return chunks
