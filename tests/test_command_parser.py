"""Tests for command_parser.py — Command-line style parser.

Run: python3 -m pytest tests/test_command_parser.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.command_parser import CommandParser


class TestCommandParser:
    def test_parse_simple(self):
        parser = CommandParser()
        cmd = parser.parse("breed --count 5")
        assert cmd is not None
        assert cmd.name == "breed"
        assert cmd.args["count"] == 5

    def test_parse_flags(self):
        parser = CommandParser()
        cmd = parser.parse("deploy -f --force")
        assert "f" in cmd.flags
        assert "force" in cmd.flags

    def test_parse_positional(self):
        parser = CommandParser()
        cmd = parser.parse("log alpha beta")
        assert cmd.positional == ["alpha", "beta"]

    def test_parse_equals_syntax(self):
        parser = CommandParser()
        cmd = parser.parse("set --key=value")
        assert cmd.args["key"] == "value"

    def test_parse_quoted(self):
        parser = CommandParser()
        cmd = parser.parse('echo "hello world"')
        assert cmd.positional == ["hello world"]

    def test_parse_empty(self):
        parser = CommandParser()
        assert parser.parse("") is None
        assert parser.parse("   ") is None

    def test_parse_coerce_bool(self):
        parser = CommandParser()
        cmd = parser.parse("flag --enable true")
        assert cmd.args["enable"] is True

    def test_parse_coerce_float(self):
        parser = CommandParser()
        cmd = parser.parse("scale --factor 1.5")
        assert cmd.args["factor"] == 1.5

    def test_parse_many(self):
        parser = CommandParser()
        cmds = parser.parse_many("cmd1; cmd2")
        assert len(cmds) == 2
        assert cmds[0].name == "cmd1"
        assert cmds[1].name == "cmd2"

    def test_repr(self):
        parser = CommandParser()
        assert "CommandParser" in repr(parser)
