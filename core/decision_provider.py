from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol

from core.intent_recognizer import IntentCategory, IntentResult, UrgencyLevel


@dataclass(frozen=True)
class DecisionResult:
    intent: IntentCategory
    confidence: float
    urgency: UrgencyLevel
    needs_human: bool = False
    primary_agent: Optional[str] = None
    supporting_agents: List[str] = field(default_factory=list)
    source: str = "v2_local_policy"
    latency_ms: float = 0.0
    reason: str = ""
    schema_version: str = "2.0"
    needs_clarification: bool = False


class DecisionProvider(Protocol):
    async def decide(
        self,
        message: str,
        history: Optional[List[dict[str, str]]] = None,
        local_result: Optional[IntentResult] = None,
    ) -> DecisionResult:
        ...


class LocalPolicyDecisionProvider:
\
\
\
\
       

    _INTENT_AGENT = {
        IntentCategory.TECHNICAL: "technical",
        IntentCategory.TECHNICAL_LOGIN: "technical",
        IntentCategory.TECHNICAL_CRASH: "technical",
        IntentCategory.BILLING: "billing",
        IntentCategory.REFUND: "billing",
        IntentCategory.INVOICE: "billing",
        IntentCategory.PAYMENT_ISSUE: "billing",
        IntentCategory.ACCOUNT: "billing",
        IntentCategory.ACCOUNT_SECURITY: "escalation",
        IntentCategory.ESCALATION: "escalation",
        IntentCategory.HUMAN_HANDOFF: "escalation",
    }

    _HUMAN_KEYWORDS = (
        "转人工", "人工客服", "找人工", "投诉", "经理", "升级处理",
        "被盗", "异常登录", "隐私泄露", "退款纠纷", "chargeback",
        "不满意", "失望", "没人处理", "没人解决", "不能接受", "服务太差",
        "处理效率太低", "拖了很久",
        "人工处理", "人工审核", "专人处理", "人工申诉", "主管处理", "升级到专人",
        "账户安全", "安全信息", "异常操作", "风险账户", "冻结",
        "没有给我答复", "怀疑有人获取", "不认识的登录", "账号疑似被别人使用",
        "不要再给我机器人", "存在风险的账号", "怀疑有人拿到",
        "升级给专人",
    )
    _TECHNICAL_KEYWORDS = ("崩溃", "闪退", "报错", "error", "无法登录", "登录失败", "500", "401")
    _BILLING_KEYWORDS = ("退款", "扣款", "多扣", "发票", "账单", "支付", "订阅", "invoice", "refund")

    async def decide(
        self,
        message: str,
        history: Optional[List[dict[str, str]]] = None,
        local_result: Optional[IntentResult] = None,
    ) -> DecisionResult:
        started = time.monotonic()
        result = local_result
        if result is None:
            raise ValueError("LocalPolicyDecisionProvider 需要 Version 1 local_result")

        text = (message or "").lower()
        explicit_human = any(keyword in text for keyword in self._HUMAN_KEYWORDS)
        is_security = result.intent is IntentCategory.ACCOUNT_SECURITY or any(
        keyword in text
        for keyword in (
            "被盗", "异常登录", "隐私泄露", "账户安全", "安全信息", "异常操作",
            "风险账户", "冻结", "怀疑有人获取", "不认识的登录", "账号疑似被别人使用",
        )
        )
        needs_human = explicit_human or is_security or result.intent is IntentCategory.COMPLAINT

        primary = self._INTENT_AGENT.get(result.intent, "general")
        reason = "Version 1 intent preserved; local policy validated"
        if needs_human:
            primary = "escalation"
            reason = "high-risk or explicit human-handoff policy"

        supporting: List[str] = []
        has_technical = any(keyword in text for keyword in self._TECHNICAL_KEYWORDS)
        has_billing = any(keyword in text for keyword in self._BILLING_KEYWORDS)
        if has_technical and has_billing:
            supporting = ["technical" if primary == "billing" else "billing"]
            reason = "multi-domain request: primary route plus supporting specialist"

        confidence = max(0.0, min(float(result.confidence), 1.0))
        if needs_human or supporting:
            confidence = max(confidence, 0.75)

        return DecisionResult(
            intent=result.intent,
            confidence=confidence,
            urgency=result.urgency,
            needs_human=needs_human,
            primary_agent=primary,
            supporting_agents=supporting,
            latency_ms=(time.monotonic() - started) * 1000,
            reason=reason,
        )
