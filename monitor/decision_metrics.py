                                                                         

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Deque, Optional


@dataclass(frozen=True)
class DecisionMetricEvent:
    mode: str
    source: str
    latency_ms: float
    confidence: float
    fallback: bool = False
    intent_match: Optional[bool] = None
    route_match: Optional[bool] = None


class DecisionMetricsRegistry:
    def __init__(self, max_events: int = 10000):
        self._events: Deque[DecisionMetricEvent] = deque(maxlen=max_events)

    def record(self, event: DecisionMetricEvent) -> None:
        self._events.append(event)

    def summary(self) -> dict[str, Any]:
        events = list(self._events)
        if not events:
            return {
                "total": 0,
                "by_source": {},
                "provider_rate": 0.0,
                "fallback_rate": 0.0,
                "intent_agreement": None,
                "route_agreement": None,
            }
        by_source = Counter(event.source for event in events)
        intent = [event.intent_match for event in events if event.intent_match is not None]
        route = [event.route_match for event in events if event.route_match is not None]
        return {
            "total": len(events),
            "by_source": dict(by_source),
            "provider_rate": round(sum(event.source != "local" for event in events) / len(events), 4),
            "fallback_rate": round(sum(event.fallback for event in events) / len(events), 4),
            "intent_agreement": round(sum(bool(v) for v in intent) / len(intent), 4) if intent else None,
            "route_agreement": round(sum(bool(v) for v in route) / len(route), 4) if route else None,
        }
