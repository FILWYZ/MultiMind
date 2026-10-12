
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from core.decision_provider import DecisionResult
from core.intent_recognizer import IntentCategory, UrgencyLevel


class DecisionEnvelope(BaseModel):
                                                                 

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="3.0", pattern=r"^3\.\d+$")
    intent: IntentCategory
    confidence: float = Field(ge=0.0, le=1.0)
    urgency: UrgencyLevel
    primary_agent: Optional[str] = None
    supporting_agents: List[str] = Field(default_factory=list, max_length=4)
    needs_human: bool = False
    needs_clarification: bool = False
    source: str = Field(default="v3_local", max_length=80)
    reason: str = Field(default="", max_length=500)
    evidence: Dict[str, float] = Field(default_factory=dict, max_length=32)

    @classmethod
    def from_result(cls, result: DecisionResult, *, evidence: Optional[Dict[str, float]] = None) -> "DecisionEnvelope":
        return cls(
            schema_version=getattr(result, "schema_version", "3.0"),
            intent=result.intent,
            confidence=result.confidence,
            urgency=result.urgency,
            primary_agent=result.primary_agent,
            supporting_agents=list(result.supporting_agents),
            needs_human=result.needs_human,
            needs_clarification=getattr(result, "needs_clarification", False),
            source=result.source,
            reason=result.reason,
            evidence=evidence or {},
        )

    def to_result(self) -> DecisionResult:
        return DecisionResult(
            intent=self.intent,
            confidence=self.confidence,
            urgency=self.urgency,
            needs_human=self.needs_human,
            primary_agent=self.primary_agent,
            supporting_agents=list(self.supporting_agents),
            source=self.source,
            reason=self.reason,
            schema_version=self.schema_version,
            needs_clarification=self.needs_clarification,
        )

    def to_public_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")
