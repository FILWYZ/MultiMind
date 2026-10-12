                                                     

from __future__ import annotations

from typing import Iterable

from core.decision_provider import DecisionResult
from core.intent_recognizer import IntentCategory, IntentResult


def should_consult_provider(
    message: str,
    local_result: IntentResult,
    local_confidence_gate: float = 0.55,
) -> bool:
                                                                             
    if local_result.intent is IntentCategory.OTHER:
        return True
    if local_result.intent in {
        IntentCategory.COMPLAINT,
        IntentCategory.ACCOUNT_SECURITY,
        IntentCategory.HUMAN_HANDOFF,
        IntentCategory.ESCALATION,
    }:
        return True
    if float(local_result.confidence or 0.0) < local_confidence_gate:
        return True

    text = (message or "").lower()
    high_risk = (
        "投诉", "人工", "经理", "升级", "被盗", "异常登录", "隐私",
        "重复扣款", "多扣", "退款纠纷", "chargeback", "不满意", "失望",
        "没人处理", "没人解决", "不能接受", "服务太差", "处理效率太低",
        "拖了很久", "人工处理", "人工审核", "专人处理", "人工申诉", "主管处理",
        "升级到专人", "账户安全", "安全信息", "异常操作", "风险账户", "冻结",
        "没有给我答复", "怀疑有人获取", "不认识的登录", "账号疑似被别人使用",
    )
    if any(keyword in text for keyword in high_risk):
        return True

    technical = ("崩溃", "闪退", "报错", "error", "无法登录", "登录失败", "500", "401")
    billing = ("退款", "扣款", "多扣", "发票", "账单", "支付", "订阅", "invoice", "refund")
    if any(keyword in text for keyword in technical) and any(keyword in text for keyword in billing):
        return True
    return bool(local_result.source_scores.get("refined_by_pattern", 0.0))


def validate_decision(
    decision: DecisionResult,
    allowed_agents: Iterable[str],
    min_confidence: float = 0.65,
) -> bool:
                                                                         
    allowed = set(allowed_agents)
    if decision.confidence < min_confidence:
        return False
    if not decision.primary_agent:
        return False
    normalized = decision.primary_agent.removesuffix("_0")
    return normalized in {agent.removesuffix("_0") for agent in allowed}
