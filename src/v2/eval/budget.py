"""
API spend budget enforcer for PaperAffilBench evaluation runs.

The MAX_USD environment variable sets a hard ceiling on estimated API cost.
Every LLM call must go through BudgetTracker.charge(); exceeding the ceiling
raises BudgetExceededError, aborting the run cleanly.

Pricing constants (approximate, per 1M tokens):
  claude-sonnet-4-6   input=$3.00  output=$15.00
  gpt-4o              input=$2.50  output=$10.00
  gemini-1.5-flash    input=$0.075 output=$0.30

Usage
-----
::

    tracker = BudgetTracker(max_usd=80.0)
    tracker.charge(model="claude-sonnet-4-6", input_tokens=1000, output_tokens=200)
    tracker.checkpoint("after_system_grobid")
    print(tracker.summary())
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_DEFAULT_MAX_USD = float(os.getenv("MAX_USD", "80.0"))
_BUDGET_LOG = Path(os.getenv("BUDGET_LOG", "output/eval/budget.json"))

# Pricing per 1M tokens (input, output)
_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6":    (3.00,  15.00),
    "claude-haiku-4-5-20251001": (0.25, 1.25),
    "gpt-4o":               (2.50,  10.00),
    "gpt-4o-mini":          (0.15,   0.60),
    "gemini-1.5-flash":     (0.075,  0.30),
    "gemini-1.5-pro":       (3.50,  10.50),
}
_DEFAULT_PRICING = (2.00, 8.00)  # fallback for unknown models


class BudgetExceededError(RuntimeError):
    """Raised when the estimated spend would exceed MAX_USD."""


@dataclass
class ChargeRecord:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    label: str = ""


@dataclass
class BudgetTracker:
    """Thread-safe accumulator of estimated API spend.

    Parameters
    ----------
    max_usd:
        Hard ceiling in USD.  Defaults to ``MAX_USD`` env var or $80.
    log_path:
        Path to persist spend log as JSON.
    """

    max_usd: float = field(default_factory=lambda: _DEFAULT_MAX_USD)
    log_path: Path = field(default_factory=lambda: _BUDGET_LOG)

    _total_usd: float = field(default=0.0, init=False, repr=False)
    _records: list[ChargeRecord] = field(default_factory=list, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def charge(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        label: str = "",
        dry_run: bool = False,
    ) -> float:
        """Record a charge and raise if over budget.

        Parameters
        ----------
        model:
            Model name (used for pricing lookup).
        input_tokens:
            Number of input/prompt tokens.
        output_tokens:
            Number of output/completion tokens.
        label:
            Optional description for the log.
        dry_run:
            If True, compute cost but do not accumulate or raise.

        Returns
        -------
        float
            Cost of this call in USD.

        Raises
        ------
        BudgetExceededError
            If adding this charge would exceed max_usd.
        """
        in_price, out_price = _PRICING.get(model, _DEFAULT_PRICING)
        cost = (input_tokens * in_price + output_tokens * out_price) / 1_000_000

        if dry_run:
            return cost

        with self._lock:
            new_total = self._total_usd + cost
            if new_total > self.max_usd:
                raise BudgetExceededError(
                    f"Budget exceeded: ${new_total:.4f} > max ${self.max_usd:.2f} "
                    f"(adding ${cost:.4f} for {model} [{label}])"
                )
            self._total_usd = new_total
            self._records.append(
                ChargeRecord(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                    label=label,
                )
            )
        return cost

    def checkpoint(self, label: str = "") -> None:
        """Persist current spend to log_path (thread-safe)."""
        with self._lock:
            data = {
                "total_usd": round(self._total_usd, 6),
                "max_usd": self.max_usd,
                "remaining_usd": round(self.max_usd - self._total_usd, 6),
                "checkpoint_label": label,
                "n_calls": len(self._records),
                "records": [
                    {
                        "model": r.model,
                        "input_tokens": r.input_tokens,
                        "output_tokens": r.output_tokens,
                        "cost_usd": round(r.cost_usd, 6),
                        "label": r.label,
                    }
                    for r in self._records
                ],
            }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @property
    def total_usd(self) -> float:
        with self._lock:
            return self._total_usd

    @property
    def remaining_usd(self) -> float:
        with self._lock:
            return self.max_usd - self._total_usd

    def summary(self) -> str:
        with self._lock:
            pct = 100 * self._total_usd / self.max_usd if self.max_usd > 0 else 0
            return (
                f"Budget: ${self._total_usd:.4f} / ${self.max_usd:.2f} "
                f"({pct:.1f}%)  remaining=${self.remaining_usd:.4f}  "
                f"calls={len(self._records)}"
            )

    def estimate_remaining_calls(self, model: str, avg_in: int = 2000, avg_out: int = 400) -> int:
        """Estimate how many more calls can be made within the remaining budget."""
        in_price, out_price = _PRICING.get(model, _DEFAULT_PRICING)
        cost_per_call = (avg_in * in_price + avg_out * out_price) / 1_000_000
        if cost_per_call <= 0:
            return 0
        return int(self.remaining_usd / cost_per_call)


# Module-level singleton (can be replaced in tests)
_global_tracker: BudgetTracker | None = None


def get_global_tracker(max_usd: float | None = None) -> BudgetTracker:
    """Return (or create) the module-level BudgetTracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = BudgetTracker(max_usd=max_usd or _DEFAULT_MAX_USD)
    return _global_tracker


def reset_global_tracker(max_usd: float | None = None) -> BudgetTracker:
    """Reset and return a fresh module-level BudgetTracker (useful in tests)."""
    global _global_tracker
    _global_tracker = BudgetTracker(max_usd=max_usd or _DEFAULT_MAX_USD)
    return _global_tracker
