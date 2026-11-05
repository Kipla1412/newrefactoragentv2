"""Runtime — the two ways to obtain evaluation events.

Both drive the same engine, registry and metrics; only the source of the event
differs:

- :class:`~evaluation.runtime.offline.OfflineRunner` — golden dataset.
- :class:`~evaluation.runtime.observer.EvaluationObserver` — a live execution,
  evaluated in the background without blocking the response.
"""

from evaluation.runtime.observer import EvaluationObserver
from evaluation.runtime.offline import OfflineRunner

__all__ = [
    "EvaluationObserver",
    "OfflineRunner",
]
