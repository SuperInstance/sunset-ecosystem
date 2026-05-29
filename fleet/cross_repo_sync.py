from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SyncEntry:
    """A cross-repo sync entry."""
    repo_name: str
    repo_url: str
    commit_hash: str
    breeding_result: Dict[str, Any]
    timestamp: float
    sync_id: str = field(default_factory=lambda: str(int(time.time() * 1000000)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sync_id": self.sync_id,
            "repo_name": self.repo_name,
            "repo_url": self.repo_url,
            "commit_hash": self.commit_hash,
            "breeding_result": self.breeding_result,
            "timestamp": self.timestamp,
        }


class CrossRepoSync:
    """
    Synchronize breeding results across SuperInstance repositories.

    Tracks which repos have which breeding results and enables
    cross-pollination of successful strategies.
    """

    def __init__(self, fleet_node_id: str = "default"):
        self.fleet_node_id = fleet_node_id
        self.entries: List[SyncEntry] = []
        self._repo_index: Dict[str, List[SyncEntry]] = {}
        self._hash_index: Dict[str, SyncEntry] = {}

    def push(self, repo_name: str, repo_url: str, commit_hash: str,
             breeding_result: Dict[str, Any]) -> SyncEntry:
        """Push a breeding result to the sync log."""
        entry = SyncEntry(
            repo_name=repo_name,
            repo_url=repo_url,
            commit_hash=commit_hash,
            breeding_result=breeding_result,
            timestamp=time.time(),
        )
        self.entries.append(entry)
        self._repo_index.setdefault(repo_name, []).append(entry)
        self._hash_index[commit_hash] = entry
        return entry

    def get_by_repo(self, repo_name: str) -> List[SyncEntry]:
        """Get all entries for a specific repository."""
        return self._repo_index.get(repo_name, [])

    def get_by_commit(self, commit_hash: str) -> Optional[SyncEntry]:
        """Get entry by commit hash."""
        return self._hash_index.get(commit_hash)

    def get_latest(self, repo_name: str) -> Optional[SyncEntry]:
        """Get latest entry for a repository."""
        entries = self._repo_index.get(repo_name, [])
        if not entries:
            return None
        return max(entries, key=lambda e: e.timestamp)

    def get_all_repos(self) -> List[str]:
        """List all known repositories."""
        return list(self._repo_index.keys())

    def find_compatible(self, repo_name: str,
                        min_fitness: float = 0.0) -> List[SyncEntry]:
        """Find entries from other repos with good fitness."""
        results = []
        for name, entries in self._repo_index.items():
            if name == repo_name:
                continue
            for entry in entries:
                fitness = entry.breeding_result.get("best_fitness", 0.0)
                if fitness >= min_fitness:
                    results.append(entry)
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get sync statistics."""
        return {
            "total_entries": len(self.entries),
            "repos": len(self._repo_index),
            "latest_commit": self.entries[-1].commit_hash if self.entries else None,
        }

    def export_json(self) -> str:
        """Export sync log as JSON."""
        return json.dumps({
            "node": self.fleet_node_id,
            "entries": [e.to_dict() for e in self.entries],
            "stats": self.get_stats(),
        }, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stats": self.get_stats(),
            "repos": self.get_all_repos(),
        }
