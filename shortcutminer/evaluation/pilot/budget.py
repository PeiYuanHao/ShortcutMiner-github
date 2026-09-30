"""Global, resumable accounting for a real-provider pilot."""

from __future__ import annotations

from shortcutminer.evaluation.pilot.protocol import PilotBudget


class PilotBudgetExceeded(RuntimeError):
    """Raised before a new scientific run would exceed the frozen global cap."""


class PilotBudgetAccount:
    def __init__(self, budget: PilotBudget, *, text_calls: int = 0, vlm_calls: int = 0,
                 vlm_images: int = 0, total_tokens: int = 0,
                 usage_complete: bool = True) -> None:
        self.budget = budget
        self.text_calls = text_calls
        self.vlm_calls = vlm_calls
        self.vlm_images = vlm_images
        self.total_tokens = total_tokens
        self.usage_complete = usage_complete

    def can_start_run(self) -> bool:
        return (
            self.text_calls + self.budget.max_text_calls_per_run <= self.budget.max_total_text_model_calls
            and self.vlm_calls + self.budget.max_vlm_calls_per_run <= self.budget.max_total_vlm_calls
            and self.vlm_images < self.budget.max_total_vlm_images
            and (self.budget.max_total_tokens is None or (
                self.usage_complete and self.total_tokens < self.budget.max_total_tokens
            ))
        )

    def require_run_capacity(self) -> None:
        if not self.can_start_run():
            raise PilotBudgetExceeded("global API budget cannot cover another bounded full run")

    def add(self, *, text_calls: int = 0, vlm_calls: int = 0, vlm_images: int = 0) -> None:
        if min(text_calls, vlm_calls, vlm_images) < 0:
            raise ValueError("budget increments must be non-negative")
        values = (
            self.text_calls + text_calls,
            self.vlm_calls + vlm_calls,
            self.vlm_images + vlm_images,
        )
        limits = (
            self.budget.max_total_text_model_calls,
            self.budget.max_total_vlm_calls,
            self.budget.max_total_vlm_images,
        )
        if any(value > limit for value, limit in zip(values, limits, strict=True)):
            raise PilotBudgetExceeded("global API budget was exceeded")
        self.text_calls, self.vlm_calls, self.vlm_images = values


__all__ = ["PilotBudgetAccount", "PilotBudgetExceeded"]
