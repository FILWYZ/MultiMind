"""Small, dependency-free execution harness for bounded agent runs."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from evaluation.privacy import redact_text


def _positive_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


@dataclass
class RunHarness:
    """Collects a privacy-conscious event trace and enforces run budgets."""

    request_id: str
    message: str
    max_tool_calls: int = field(default_factory=lambda: _positive_int("ECHOMIND_MAX_TOOL_CALLS", 6))
    max_agent_calls: int = field(default_factory=lambda: _positive_int("ECHOMIND_MAX_AGENT_CALLS", 2))
    started_at: float = field(default_factory=time.monotonic)
    events: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: int = 0
    agent_calls: int = 0

    @property
    def message_hash(self) -> str:
        return hashlib.sha256(self.message.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def record(self, event: str, **data: Any) -> None:
        self.events.append({"event": event, "elapsed_ms": round((time.monotonic() - self.started_at) * 1000, 1), **data})

    def allow_agent(self) -> bool:
        if self.agent_calls >= self.max_agent_calls:
            self.record("budget_exhausted", budget="agent_calls", limit=self.max_agent_calls)
            return False
        self.agent_calls += 1
        return True

    def allow_tool(self) -> bool:
        if self.tool_calls >= self.max_tool_calls:
            self.record("budget_exhausted", budget="tool_calls", limit=self.max_tool_calls)
            return False
        self.tool_calls += 1
        return True

    def summary(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "message_hash": self.message_hash,
            "duration_ms": round((time.monotonic() - self.started_at) * 1000, 1),
            "budgets": {"agent_calls": self.max_agent_calls, "tool_calls": self.max_tool_calls},
            "usage": {"agent_calls": self.agent_calls, "tool_calls": self.tool_calls},
            "events": list(self.events),
        }


def replay_events(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministically reconstruct the recorded event timeline without tools/LLM."""
    events = trace.get("harness", {}).get("events", [])
    if not isinstance(events, list):
        raise ValueError("trace events must be a list")
    return {
        "request_id": trace.get("request_id"),
        "mode": "debug_event_replay",
        "side_effects": False,
        "events": [dict(event) for event in events if isinstance(event, dict)],
    }
