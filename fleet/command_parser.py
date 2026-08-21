"""Command-line style command parser for agent directives.

Parses structured commands from strings. Used for agent shell
interfaces, fleet control commands, and bot directives.

Usage:
    parser = CommandParser()
    cmd = parser.parse("breed --count 5 --target alpha")
    assert cmd.name == "breed"
    assert cmd.args["count"] == "5"
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Command:
    """Parsed command result."""

    name: str
    args: Dict[str, Any]
    flags: List[str]
    positional: List[str]
    raw: str


class CommandParser:
    """
    Command parser with POSIX-style quoting.

    Supports:
    - --long-opt value
    - --long-opt=value
    - -f (short flags)
    - positional arguments
    - quoted strings
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse(self, text: str) -> Optional[Command]:
        """Parse a command string."""
        text = text.strip()
        if not text:
            return None
        try:
            tokens = shlex.split(text)
        except ValueError:
            tokens = text.split()
        if not tokens:
            return None
        name = tokens[0]
        args: Dict[str, Any] = {}
        flags: List[str] = []
        positional: List[str] = []
        i = 1
        while i < len(tokens):
            token = tokens[i]
            if token.startswith("--"):
                opt = token[2:]
                if "=" in opt:
                    key, value = opt.split("=", 1)
                    args[key] = self._coerce(value)
                elif i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                    args[opt] = self._coerce(tokens[i + 1])
                    i += 1
                else:
                    flags.append(opt)
            elif token.startswith("-") and len(token) > 1:
                flags.extend(token[1:])
            else:
                positional.append(token)
            i += 1
        return Command(
            name=name,
            args=args,
            flags=flags,
            positional=positional,
            raw=text,
        )

    def parse_many(self, text: str) -> List[Command]:
        """Parse multiple commands separated by newlines or semicolons."""
        lines = re.split(r"[;\n]", text)
        results: List[Command] = []
        for line in lines:
            cmd = self.parse(line)
            if cmd:
                results.append(cmd)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce(value: str) -> Any:
        """Coerce string to int/float/bool if possible."""
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    def __repr__(self) -> str:
        return "<CommandParser>"
