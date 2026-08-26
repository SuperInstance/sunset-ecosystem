"""tests/test_compiler_integration.py — RoomGridCompiler tests."""

import pytest
import numpy as np

# Mock compiler if not installed
pytestmark = pytest.mark.skipif(
    True,  # Skip by default — agentic-compiler is optional
    reason="agentic-compiler not installed in test environment",
)


def test_compiler_integration_stub():
    """Placeholder — real tests need agentic-compiler installed."""
    assert True
