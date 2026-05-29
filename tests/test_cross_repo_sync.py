import pytest
from fleet.cross_repo_sync import SyncEntry, CrossRepoSync


class TestSyncEntry:
    def test_to_dict(self):
        e = SyncEntry(
            repo_name="test",
            repo_url="https://github.com/test",
            commit_hash="abc123",
            breeding_result={"fitness": 0.9},
            timestamp=0.0,
        )
        d = e.to_dict()
        assert d["repo_name"] == "test"
        assert d["commit_hash"] == "abc123"


class TestCrossRepoSync:
    def test_init(self):
        crs = CrossRepoSync()
        assert crs.entries == []
        assert crs.fleet_node_id == "default"

    def test_push(self):
        crs = CrossRepoSync()
        e = crs.push("repo1", "https://github.com/repo1", "abc123", {"fitness": 0.9})
        assert e.repo_name == "repo1"
        assert len(crs.entries) == 1

    def test_get_by_repo(self):
        crs = CrossRepoSync()
        crs.push("repo1", "url1", "h1", {"fitness": 0.9})
        crs.push("repo1", "url1", "h2", {"fitness": 0.8})
        entries = crs.get_by_repo("repo1")
        assert len(entries) == 2

    def test_get_by_commit(self):
        crs = CrossRepoSync()
        crs.push("repo1", "url1", "h1", {"fitness": 0.9})
        e = crs.get_by_commit("h1")
        assert e is not None
        assert e.commit_hash == "h1"

    def test_get_latest(self):
        crs = CrossRepoSync()
        crs.push("repo1", "url1", "h1", {"fitness": 0.9})
        crs.push("repo1", "url1", "h2", {"fitness": 0.8})
        latest = crs.get_latest("repo1")
        assert latest.commit_hash == "h2"

    def test_get_latest_empty(self):
        crs = CrossRepoSync()
        assert crs.get_latest("missing") is None

    def test_get_all_repos(self):
        crs = CrossRepoSync()
        crs.push("repo1", "url1", "h1", {})
        crs.push("repo2", "url2", "h2", {})
        repos = crs.get_all_repos()
        assert "repo1" in repos
        assert "repo2" in repos

    def test_find_compatible(self):
        crs = CrossRepoSync()
        crs.push("repo1", "url1", "h1", {"best_fitness": 0.9})
        crs.push("repo2", "url2", "h2", {"best_fitness": 0.7})
        results = crs.find_compatible("repo1", min_fitness=0.8)
        assert len(results) == 0  # repo1 excluded, repo2 below threshold

    def test_find_compatible_includes(self):
        crs = CrossRepoSync()
        crs.push("repo1", "url1", "h1", {"best_fitness": 0.9})
        crs.push("repo2", "url2", "h2", {"best_fitness": 0.95})
        results = crs.find_compatible("repo1", min_fitness=0.8)
        assert len(results) == 1
        assert results[0].repo_name == "repo2"

    def test_get_stats(self):
        crs = CrossRepoSync()
        crs.push("repo1", "url1", "h1", {})
        crs.push("repo2", "url2", "h2", {})
        stats = crs.get_stats()
        assert stats["total_entries"] == 2
        assert stats["repos"] == 2

    def test_export_json(self):
        crs = CrossRepoSync()
        crs.push("repo1", "url1", "h1", {"fitness": 0.9})
        j = crs.export_json()
        assert "repo1" in j
        assert "entries" in j

    def test_to_dict(self):
        crs = CrossRepoSync()
        crs.push("repo1", "url1", "h1", {})
        d = crs.to_dict()
        assert d["stats"]["total_entries"] == 1
        assert "repo1" in d["repos"]
