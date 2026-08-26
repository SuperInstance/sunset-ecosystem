"""
HAV Bridge — Human-Agent Vocabulary Bridge

Translates between human natural language and agent-native data structures.
Acts as a bilingual dictionary that both humans and agents can query.

Key features:
- Bidirectional translation: human phrase ↔ agent schema
- Vocabulary learning: agents teach humans their terms, humans teach agents theirs
- Context-aware disambiguation
- A2A vocabulary cards for inter-agent communication

Usage:
    from fleet.hav_bridge import HAVBridge
    bridge = HAVBridge()
    bridge.teach_agent("breeding", "Evolutionary optimization of agent populations")
    bridge.teach_human("flux_gate", "Constraint-checking system before breeding")
    translation = bridge.translate("Run a breeding cycle")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np


@dataclass
class VocabularyEntry:
    """A single vocabulary entry."""

    term: str
    human_definition: str
    agent_schema: Dict[str, Any]
    context: str = "general"
    synonyms: List[str] = field(default_factory=list)
    # Usage examples
    human_examples: List[str] = field(default_factory=list)
    agent_examples: List[str] = field(default_factory=list)
    # Frequency of use (for ranking)
    frequency: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "term": self.term,
            "human_definition": self.human_definition,
            "agent_schema": self.agent_schema,
            "context": self.context,
            "synonyms": self.synonyms,
            "human_examples": self.human_examples,
            "agent_examples": self.agent_examples,
            "frequency": self.frequency,
        }


class HAVBridge:
    """
    Bidirectional translator between human and agent vocabularies.

    Maintains a shared lexicon that both humans and agents contribute to.
    """

    def __init__(self):
        self.vocabulary: Dict[str, VocabularyEntry] = {}
        self.human_to_agent: Dict[str, str] = {}  # human phrase -> agent term
        self.agent_to_human: Dict[str, str] = {}  # agent term -> human phrase
        self.contexts: Set[str] = set()

    def teach_agent(
        self,
        human_phrase: str,
        human_definition: str,
        agent_schema: Optional[Dict[str, Any]] = None,
        context: str = "general",
    ):
        """
        Teach the agent a human term.

        Args:
            human_phrase: The human expression
            human_definition: What it means to humans
            agent_schema: How to represent this in agent data structures
            context: Domain context (e.g., "breeding", "spatial", "health")
        """
        schema = agent_schema or {"type": "string", "value": human_phrase}
        entry = VocabularyEntry(
            term=human_phrase,
            human_definition=human_definition,
            agent_schema=schema,
            context=context,
        )
        self.vocabulary[human_phrase] = entry
        self.human_to_agent[human_phrase] = human_phrase
        self.agent_to_human[human_phrase] = human_phrase
        self.contexts.add(context)

    def teach_human(
        self,
        agent_term: str,
        human_friendly: str,
        agent_schema: Optional[Dict[str, Any]] = None,
        context: str = "general",
    ):
        """
        Teach humans an agent-native term.

        Args:
            agent_term: The agent's internal term
            human_friendly: Human-readable explanation
            agent_schema: Schema definition for the term
            context: Domain context
        """
        schema = agent_schema or {"type": "string", "value": agent_term}
        entry = VocabularyEntry(
            term=agent_term,
            human_definition=human_friendly,
            agent_schema=schema,
            context=context,
        )
        self.vocabulary[agent_term] = entry
        self.agent_to_human[agent_term] = human_friendly
        self.human_to_agent[human_friendly] = agent_term
        self.contexts.add(context)

    def translate(self, phrase: str, direction: str = "auto") -> Dict[str, Any]:
        """
        Translate a phrase.

        Args:
            phrase: The phrase to translate
            direction: "human_to_agent", "agent_to_human", or "auto"

        Returns:
            Translation result with confidence and alternatives
        """
        # Direct match
        if phrase in self.vocabulary:
            entry = self.vocabulary[phrase]
            entry.frequency += 1
            return {
                "original": phrase,
                "translation": entry.human_definition
                if direction == "agent_to_human"
                else entry.agent_schema,
                "confidence": 1.0,
                "context": entry.context,
                "term": entry.term,
            }

        # Synonym match
        for term, entry in self.vocabulary.items():
            if phrase in entry.synonyms:
                entry.frequency += 1
                return {
                    "original": phrase,
                    "translation": entry.human_definition
                    if direction == "agent_to_human"
                    else entry.agent_schema,
                    "confidence": 0.9,
                    "context": entry.context,
                    "term": entry.term,
                }

        # Partial match (substring)
        for term, entry in self.vocabulary.items():
            if phrase in term or term in phrase:
                entry.frequency += 1
                return {
                    "original": phrase,
                    "translation": entry.human_definition
                    if direction == "agent_to_human"
                    else entry.agent_schema,
                    "confidence": 0.7,
                    "context": entry.context,
                    "term": entry.term,
                }

        return {
            "original": phrase,
            "translation": None,
            "confidence": 0.0,
            "context": "unknown",
            "term": None,
        }

    def translate_sentence(self, sentence: str) -> List[Dict[str, Any]]:
        """Translate each word in a sentence."""
        words = sentence.split()
        return [self.translate(word) for word in words]

    def get_vocabulary_by_context(self, context: str) -> List[VocabularyEntry]:
        """Get all vocabulary entries for a context."""
        return [e for e in self.vocabulary.values() if e.context == context]

    def get_popular_terms(self, n: int = 10) -> List[VocabularyEntry]:
        """Get most frequently used terms."""
        sorted_terms = sorted(
            self.vocabulary.values(), key=lambda e: e.frequency, reverse=True
        )
        return sorted_terms[:n]

    def add_synonym(self, term: str, synonym: str):
        """Add a synonym to an existing term."""
        if term in self.vocabulary:
            if synonym not in self.vocabulary[term].synonyms:
                self.vocabulary[term].synonyms.append(synonym)

    def export_a2a_vocabulary_card(self, context: str = "general") -> Dict[str, Any]:
        """Export vocabulary as A2A card."""
        entries = self.get_vocabulary_by_context(context)
        return {
            "type": "vocabulary_card",
            "context": context,
            "terms": [e.to_dict() for e in entries],
            "total_terms": len(entries),
        }

    def import_a2a_vocabulary_card(self, card: Dict[str, Any]):
        """Import vocabulary from A2A card."""
        for term_data in card.get("terms", []):
            entry = VocabularyEntry(
                term=term_data["term"],
                human_definition=term_data["human_definition"],
                agent_schema=term_data["agent_schema"],
                context=term_data.get("context", "general"),
                synonyms=term_data.get("synonyms", []),
            )
            self.vocabulary[entry.term] = entry

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_terms": len(self.vocabulary),
            "contexts": list(self.contexts),
            "total_translations": sum(e.frequency for e in self.vocabulary.values()),
            "avg_confidence": np.mean(
                [len(e.synonyms) for e in self.vocabulary.values()]
            )
            if self.vocabulary
            else 0,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocabulary": {k: v.to_dict() for k, v in self.vocabulary.items()},
            "stats": self.get_stats(),
        }

    def generate_human_guide(self, context: str = "general") -> str:
        """Generate a human-readable guide for a context."""
        entries = self.get_vocabulary_by_context(context)
        lines = [
            f"# Human Guide: {context.title()}",
            "",
            f"This guide explains the {len(entries)} terms used in the {context} domain.",
            "",
        ]

        for entry in entries:
            lines.append(f"## {entry.term}")
            lines.append(f"**Definition:** {entry.human_definition}")
            if entry.synonyms:
                lines.append(f"**Also known as:** {', '.join(entry.synonyms)}")
            if entry.human_examples:
                lines.append("**Examples:**")
                for ex in entry.human_examples:
                    lines.append(f'  - "{ex}"')
            lines.append("")

        return "\n".join(lines)

    def generate_agent_schema(self, context: str = "general") -> Dict[str, Any]:
        """Generate agent-native schema for a context."""
        entries = self.get_vocabulary_by_context(context)
        return {
            "context": context,
            "schema_version": "1.0",
            "terms": {entry.term: entry.agent_schema for entry in entries},
        }
