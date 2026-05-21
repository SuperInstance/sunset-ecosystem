"""Trinity scorer — three-body selection pressure.

Ethos (values), Pathos (emotional resonance), Logos (logic/relevance).
Score is the product: if any connection is zero, the agent sunsets.
"""

from __future__ import annotations


def trinity_score(ethos_conn: float, pathos_conn: float, logos_conn: float) -> float:
    """Compute the trinity score as the product of three connections.

    Each connection should be in [0.0, 1.0]. If any is 0, the score is 0.

    Args:
        ethos_conn: Connection to values/ethos dimension [0, 1].
        pathos_conn: Connection to emotional resonance [0, 1].
        logos_conn: Connection to logical relevance [0, 1].

    Returns:
        Product of the three connections, a float in [0, 1].
    """
    return ethos_conn * pathos_conn * logos_conn


def normalize_connection(raw: float, clamp: bool = True) -> float:
    """Normalize a raw connection value to [0, 1].

    Args:
        raw: Raw connection value.
        clamp: Whether to clamp to [0, 1].

    Returns:
        Normalized value.
    """
    if clamp:
        return max(0.0, min(1.0, raw))
    return raw


def trinity_score_raw(
    ethos_raw: float,
    pathos_raw: float,
    logos_raw: float,
) -> float:
    """Compute trinity score from raw values (auto-normalized).

    Convenience wrapper that normalizes inputs first.
    """
    e = normalize_connection(ethos_raw)
    p = normalize_connection(pathos_raw)
    l = normalize_connection(logos_raw)
    return trinity_score(e, p, l)


__all__ = ["trinity_score", "normalize_connection", "trinity_score_raw"]
