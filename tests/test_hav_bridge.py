"""
Tests for HAV Bridge.

Covers: VocabularyEntry, HAVBridge.
"""

import pytest

from fleet.hav_bridge import VocabularyEntry, HAVBridge


class TestVocabularyEntry:
    def test_to_dict(self):
        entry = VocabularyEntry(
            term="breeding",
            human_definition="Evolutionary optimization",
            agent_schema={"type": "object"},
            context="genetics",
        )
        d = entry.to_dict()
        assert d["term"] == "breeding"
        assert d["context"] == "genetics"


class TestHAVBridge:
    def test_init(self):
        bridge = HAVBridge()
        assert len(bridge.vocabulary) == 0

    def test_teach_agent(self):
        bridge = HAVBridge()
        bridge.teach_agent("breeding", "Evolutionary optimization")
        assert "breeding" in bridge.vocabulary
        assert (
            bridge.vocabulary["breeding"].human_definition
            == "Evolutionary optimization"
        )

    def test_teach_human(self):
        bridge = HAVBridge()
        bridge.teach_human("flux_gate", "Constraint checker")
        assert "flux_gate" in bridge.vocabulary
        assert bridge.human_to_agent["Constraint checker"] == "flux_gate"

    def test_translate_direct_match(self):
        bridge = HAVBridge()
        bridge.teach_agent("breeding", "Evolutionary optimization")
        result = bridge.translate("breeding")
        assert result["confidence"] == 1.0
        assert result["term"] == "breeding"

    def test_translate_unknown(self):
        bridge = HAVBridge()
        result = bridge.translate("unknown_term")
        assert result["confidence"] == 0.0
        assert result["translation"] is None

    def test_translate_with_schema(self):
        bridge = HAVBridge()
        bridge.teach_agent(
            "breeding",
            "Evolutionary optimization",
            agent_schema={
                "type": "object",
                "properties": {"population_size": {"type": "integer"}},
            },
        )
        result = bridge.translate("breeding")
        assert result["translation"]["type"] == "object"

    def test_translate_synonym(self):
        bridge = HAVBridge()
        bridge.teach_agent("breeding", "Evolutionary optimization")
        bridge.add_synonym("breeding", "evolution")
        result = bridge.translate("evolution")
        assert result["confidence"] == 0.9
        assert result["term"] == "breeding"

    def test_translate_partial_match(self):
        bridge = HAVBridge()
        bridge.teach_agent("breeding_cycle", "One generation")
        result = bridge.translate("breeding")
        assert result["confidence"] == 0.7

    def test_translate_sentence(self):
        bridge = HAVBridge()
        bridge.teach_agent("breeding", "Evolution")
        bridge.teach_agent("cycle", "Round")
        results = bridge.translate_sentence("breeding cycle")
        assert len(results) == 2
        assert results[0]["term"] == "breeding"
        assert results[1]["term"] == "cycle"

    def test_get_vocabulary_by_context(self):
        bridge = HAVBridge()
        bridge.teach_agent("gene", "Unit of heredity", context="genetics")
        bridge.teach_agent("room", "Spatial area", context="spatial")
        entries = bridge.get_vocabulary_by_context("genetics")
        assert len(entries) == 1
        assert entries[0].term == "gene"

    def test_get_popular_terms(self):
        bridge = HAVBridge()
        bridge.teach_agent("a", "First")
        bridge.teach_agent("b", "Second")
        bridge.vocabulary["a"].frequency = 10
        bridge.vocabulary["b"].frequency = 5
        popular = bridge.get_popular_terms(n=1)
        assert len(popular) == 1
        assert popular[0].term == "a"

    def test_add_synonym(self):
        bridge = HAVBridge()
        bridge.teach_agent("breeding", "Evolution")
        bridge.add_synonym("breeding", "evolution")
        assert "evolution" in bridge.vocabulary["breeding"].synonyms

    def test_export_a2a_vocabulary_card(self):
        bridge = HAVBridge()
        bridge.teach_agent("gene", "Unit", context="genetics")
        card = bridge.export_a2a_vocabulary_card("genetics")
        assert card["type"] == "vocabulary_card"
        assert card["context"] == "genetics"
        assert card["total_terms"] == 1

    def test_import_a2a_vocabulary_card(self):
        bridge = HAVBridge()
        card = {
            "terms": [
                {
                    "term": "gene",
                    "human_definition": "Unit",
                    "agent_schema": {"type": "string"},
                    "context": "genetics",
                    "synonyms": [],
                }
            ]
        }
        bridge.import_a2a_vocabulary_card(card)
        assert "gene" in bridge.vocabulary

    def test_get_stats(self):
        bridge = HAVBridge()
        bridge.teach_agent("a", "First")
        bridge.teach_agent("b", "Second")
        stats = bridge.get_stats()
        assert stats["total_terms"] == 2
        assert "genetics" in stats["contexts"] or "general" in stats["contexts"]

    def test_to_dict(self):
        bridge = HAVBridge()
        bridge.teach_agent("a", "First")
        d = bridge.to_dict()
        assert "vocabulary" in d
        assert d["stats"]["total_terms"] == 1

    def test_generate_human_guide(self):
        bridge = HAVBridge()
        bridge.teach_agent("gene", "Unit of heredity", context="genetics")
        guide = bridge.generate_human_guide("genetics")
        assert "Human Guide: Genetics" in guide
        assert "gene" in guide
        assert "Unit of heredity" in guide

    def test_generate_agent_schema(self):
        bridge = HAVBridge()
        bridge.teach_agent(
            "gene", "Unit", agent_schema={"type": "string"}, context="genetics"
        )
        schema = bridge.generate_agent_schema("genetics")
        assert schema["context"] == "genetics"
        assert "gene" in schema["terms"]

    def test_frequency_tracking(self):
        bridge = HAVBridge()
        bridge.teach_agent("test", "A test")
        bridge.translate("test")
        bridge.translate("test")
        assert bridge.vocabulary["test"].frequency == 2
