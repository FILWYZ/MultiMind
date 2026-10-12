"""Validated structured contract for specialist-agent conclusions and evidence."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
import re


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fact_key: str = Field(min_length=1, max_length=64)
    value: str = Field(min_length=1, max_length=300)
    source: str = Field(min_length=1, max_length=200)
    quote: str = Field(min_length=1, max_length=500)


class AgentOutputContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=1, max_length=3000)
    facts: Dict[str, str] = Field(default_factory=dict, max_length=20)
    evidence: List[EvidenceItem] = Field(default_factory=list, max_length=30)
    uncertainties: List[str] = Field(default_factory=list, max_length=20)
    next_steps: List[str] = Field(default_factory=list, max_length=20)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def each_fact_has_matching_evidence(self):
        supported = {(item.fact_key, item.value) for item in self.evidence}
        unsupported = [key for key, value in self.facts.items() if (key, value) not in supported]
        if unsupported:
            raise ValueError(f"facts missing matching evidence: {', '.join(unsupported)}")
        return self


def parse_agent_output(raw: str) -> Optional[AgentOutputContract]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        payload = json.loads(text)
        if not isinstance(payload, dict):
            return None
        return AgentOutputContract.model_validate(payload)
    except (json.JSONDecodeError, ValueError):
        return None


def contract_evidence_is_grounded(contract: AgentOutputContract, message: str,
                                  context: str, tool_traces: list[dict[str, Any]]) -> bool:
    sources = {"user_message": message, "conversation_context": context}
    for trace in tool_traces:
        name = str(trace.get("tool_name", ""))
        excerpt = str(trace.get("evidence_text", ""))
        if name and excerpt:
            sources[name] = excerpt

    def normalized(value: str) -> str:
        return re.sub(r"\s+", "", value).lower()

    for evidence in contract.evidence:
        source_text = sources.get(evidence.source)
        if not source_text or normalized(evidence.quote) not in normalized(source_text):
            return False
    return True
