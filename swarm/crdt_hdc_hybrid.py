"""CRDT-HDC hybrid merge: consensus + diversity preservation."""

from __future__ import annotations

import numpy as np

from swarm.crdt_merge import CRDTMergeEngine
from swarm.hdc_novelty import HDCDiversityScorer
from swarm.vector_table import FluxVectorTable


def merge_with_diversity(
    local_crdt: FluxVectorTable,
    remote_crdt: FluxVectorTable,
    diversity_threshold: float = 0.3,
) -> FluxVectorTable:
    """Merge two ship vector tables, preserving diversity.

    1. CRDT merge: resolve conflicts via LWW (latest write wins)
    2. HDC filter: if merged agent is too similar to existing, flag for review
    3. Return merged table with diversity annotations
    """
    # 1. CRDT LWW merge
    engine = CRDTMergeEngine(local_crdt)
    merged = engine.sync_vector_table(local_crdt, remote_crdt)

    # 2. HDC diversity check
    ids = sorted(merged._meta.keys())
    if len(ids) < 2:
        return merged

    dim = merged.dim
    scorer = HDCDiversityScorer(dim)

    # Extract vectors and encode to binary hypervectors
    vecs = []
    for aid in ids:
        v = engine._extract_vector(merged, aid)
        vecs.append(v if v is not None else [0.0] * dim)
    packed = scorer.encoder.encode_batch(np.array(vecs, dtype=np.float32))

    # Pairwise novelty: agent i vs all others
    scores = scorer.score_batch(packed, packed)
    for i, aid in enumerate(ids):
        others = np.concatenate([scores[i, :i], scores[i, i + 1:]])
        min_dist = float(others.min()) if len(others) else 1.0
        merged._meta[aid].extra["novelty_score"] = round(min_dist, 4)
        merged._meta[aid].extra["diversity_flag"] = min_dist < diversity_threshold

    return merged
