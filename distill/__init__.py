"""Distill — The progressive hint removal loop."""

from .prompt_history import PromptHistory, PromptRecord
from .hint_schedule import HintSchedule, ExponentialBackoffSchedule
from .backtest_runner import BacktestRunner, BacktestResult
from .distillation_signal import DistillationSignal, DistillationGuidance
from .delta_tracker import DeltaSnapshot, DeltaTracker

__all__ = [
    "PromptHistory",
    "PromptRecord",
    "HintSchedule",
    "ExponentialBackoffSchedule",
    "BacktestRunner",
    "BacktestResult",
    "DistillationSignal",
    "DistillationGuidance",
    "DeltaSnapshot",
    "DeltaTracker",
]
