                                                        

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

from core.decision_provider import DecisionResult
from core.intent_recognizer import IntentResult


@dataclass(frozen=True)
class ShadowEvent:
    message_hash: str
    local_intent: str
    candidate_intent: str
    local_route: str
    candidate_route: str
    local_confidence: float
    candidate_confidence: float
    needs_human_changed: bool


class Version3ShadowRegistry:
                                                                           

    def __init__(self, max_events: int = 5000):
        self._events: Deque[ShadowEvent] = deque(maxlen=max(100, int(max_events)))

    @staticmethod
    def _route(intent: str) -> str:
        if intent in {"technical", "technical_login", "technical_crash"}:
            return "technical"
        if intent in {"billing", "refund", "invoice", "payment_issue", "account"}:
            return "billing"
        if intent in {"complaint", "escalation", "human_handoff", "account_security"}:
            return "escalation"
        return "general"

    def record(self, message: str, local: IntentResult, candidate: DecisionResult) -> None:
        local_intent = local.intent.value
        candidate_intent = candidate.intent.value
        self._events.append(
            ShadowEvent(
                message_hash=hashlib.sha256((message or "").encode("utf-8")).hexdigest()[:16],
                local_intent=local_intent,
                candidate_intent=candidate_intent,
                local_route=self._route(local_intent),
                candidate_route=candidate.primary_agent or self._route(candidate_intent),
                local_confidence=round(float(local.confidence), 4),
                candidate_confidence=round(float(candidate.confidence), 4),
                needs_human_changed=(
                    bool(local.urgency.value >= 4)
                    or local.intent.value in {"complaint", "escalation", "human_handoff", "account_security"}
                ) != bool(candidate.needs_human),
            )
        )

    def summary(self) -> Dict[str, object]:
        total = len(self._events)
        intent_disagreements = sum(e.local_intent != e.candidate_intent for e in self._events)
        route_disagreements = sum(e.local_route != e.candidate_route for e in self._events)
        human_changes = sum(e.needs_human_changed for e in self._events)
        return {
            "events": total,
            "intent_agreement_rate": round(1 - intent_disagreements / total, 4) if total else 1.0,
            "route_disagreement_rate": round(route_disagreements / total, 4) if total else 0.0,
            "human_policy_change_rate": round(human_changes / total, 4) if total else 0.0,
            "raw_messages_retained": False,
            "buffer_limit": self._events.maxlen,
        }
