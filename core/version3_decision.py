
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from core.decision_provider import DecisionResult, LocalPolicyDecisionProvider
from core.decision_schema import DecisionEnvelope
from core.intent_recognizer import IntentCategory, IntentResult, UrgencyLevel


class Version3LocalDecisionProvider(LocalPolicyDecisionProvider):
                                                                     
    version = "3"
                                                                                                                                           
    _CUES: Dict[IntentCategory, Tuple[Tuple[str, float], ...]] = {
        IntentCategory.TECHNICAL_LOGIN: (
            ("无法登录", 3.0), ("登录失败", 3.0), ("登录不了", 3.0),
            ("登不上", 2.5), ("验证码", 2.0), ("认证失败", 2.5), ("登录认证", 2.5), ("401", 2.5),
        ),
        IntentCategory.TECHNICAL_CRASH: (
            ("崩溃", 3.0), ("闪退", 3.0), ("500", 2.5), ("错误码", 2.0),
            ("接口失败", 2.5), ("应用打不开", 2.5), ("卡死", 2.0),
        ),
        IntentCategory.PAYMENT_ISSUE: (
            ("重复扣款", 3.5), ("扣了两次", 3.5), ("多扣", 3.0),
            ("扣款异常", 3.0), ("支付失败", 2.5), ("金额不对", 2.5), ("多收", 2.5),
            ("银行卡", 2.0), ("账单金额", 2.5), ("订阅扣款", 3.0), ("支付显示成功", 2.5),
        ),
        IntentCategory.REFUND: (("退款", 2.5), ("退货退款", 3.0), ("售后退款", 3.0), ("退回金额", 3.0), ("办理退款", 3.0)),
        IntentCategory.INVOICE: (("发票", 3.0), ("发票抬头", 3.5), ("税号", 3.0)),
        IntentCategory.LOGISTICS: (("物流", 3.0), ("快递", 3.0), ("配送", 2.5), ("运单", 3.0), ("包裹", 2.5)),
        IntentCategory.ORDER_STATUS: (
            ("订单状态", 3.0), ("订单进度", 3.0), ("什么时候发货", 3.0), ("是否发货", 3.0),
            ("发货通知", 2.5), ("状态好几天没变化", 3.0), ("订单页面", 2.0),
        ),
        IntentCategory.ACCOUNT_SECURITY: (
            ("异常登录", 3.5), ("陌生设备", 3.5), ("账户被盗", 3.5),
            ("账号被别人", 3.5), ("账户安全", 3.0), ("冻结账户", 3.0), ("冻结", 2.5),
            ("不认识的登录", 3.5), ("怀疑有人拿到", 3.5), ("存在风险的账号", 3.0),
        ),
        IntentCategory.HUMAN_HANDOFF: (
            ("转人工", 3.5), ("人工客服", 3.5), ("人工处理", 3.0), ("找人工", 3.0),
            ("找你们经理", 3.5), ("升级处理", 3.0), ("升级给专人", 3.5),
            ("人工审核", 3.0), ("不要再给我机器人", 3.5),
        ),
        IntentCategory.COMPLAINT: (
            ("投诉", 3.5), ("不满意", 2.5), ("服务太差", 3.0), ("很失望", 2.5),
            ("不能接受", 2.5), ("没人处理", 3.0), ("没有给我答复", 3.0), ("处理让我", 2.0),
        ),
        IntentCategory.REQUEST: (
            ("取消订单", 2.5), ("修改地址", 2.5), ("关闭自动续费", 3.0),
            ("停止自动续费", 3.0), ("更新账户资料", 2.5), ("更新账户", 2.0),
            ("更换联系方式", 2.5), ("提交申请", 2.0), ("协助我", 1.5),
        ),
        IntentCategory.GREETING: (("你好", 2.0), ("嗨", 2.0), ("hello", 2.0)),
        IntentCategory.FEEDBACK: (("非常满意", 3.0), ("服务很棒", 3.0), ("给个好评", 3.0)),
        IntentCategory.QUERY: (("怎么", 1.0), ("如何", 1.0), ("哪里", 1.0), ("请问", 1.0), ("查询", 1.0)),
    }

    def _context_text(self, message: str, history: Optional[List[dict[str, str]]]) -> str:
        user_history = [
            str(item.get("content", ""))
            for item in (history or [])[-4:]
            if item.get("role") == "user"
        ]
        return " ".join([*user_history, message or ""]).lower()

    def _score(self, text: str) -> Dict[IntentCategory, float]:
        scores: Dict[IntentCategory, float] = {}
        for intent, cues in self._CUES.items():
            score = sum(weight for phrase, weight in cues if phrase.lower() in text)
            if score:
                scores[intent] = round(score, 4)
                                                                  
        for generic in (IntentCategory.QUERY, IntentCategory.REQUEST):
            if generic in scores and len(scores) > 1:
                scores[generic] *= 0.35
        return scores

    def _refine_intent(
        self,
        text: str,
        local_result: IntentResult,
    ) -> tuple[IntentCategory, float, Dict[str, float]]:
        scores = self._score(text)
        if not scores:
            return local_result.intent, max(float(local_result.confidence), 0.0), {}
        best_intent, best_score = max(scores.items(), key=lambda item: item[1])
                                                                           
                                                                            
                                               
        local_score = scores.get(local_result.intent, 0.0)
        if local_result.intent not in {IntentCategory.OTHER, IntentCategory.QUERY, IntentCategory.REQUEST}:
            if local_result.confidence >= 0.85 and best_intent != local_result.intent and best_score < local_score + 1.5:
                best_intent = local_result.intent
                best_score = local_score
        confidence = min(0.98, max(0.55, 0.55 + best_score * 0.1))
        evidence = {intent.value: score for intent, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:8]}
        return best_intent, confidence, evidence

    async def decide(
        self,
        message: str,
        history: Optional[List[dict[str, str]]] = None,
        local_result: Optional[IntentResult] = None,
    ) -> DecisionResult:
        started = time.monotonic()
        if local_result is None:
            raise ValueError("Version3LocalDecisionProvider 需要 Version 1 local_result")

        text = self._context_text(message, history)
        intent, confidence, evidence = self._refine_intent(text, local_result)
        refined = IntentResult(
            intent=intent,
            confidence=confidence,
            urgency=local_result.urgency,
            intent_group=local_result.intent_group,
            entities=local_result.entities,
            reasoning=local_result.reasoning,
            latency_ms=local_result.latency_ms,
            source_scores=local_result.source_scores,
        )
        result = await super().decide(message, history=history, local_result=refined)
        result = DecisionResult(
            intent=result.intent,
            confidence=result.confidence,
            urgency=result.urgency,
            needs_human=result.needs_human,
            primary_agent=result.primary_agent,
            supporting_agents=result.supporting_agents,
            source="v3_local_semantic",
            latency_ms=(time.monotonic() - started) * 1000,
            reason="Version 3 phrase scoring and policy validation",
            schema_version="3.0",
            needs_clarification=(result.intent is IntentCategory.OTHER or confidence < 0.6),
        )
                                                                        
        envelope = DecisionEnvelope.from_result(result, evidence=evidence)
        return envelope.to_result()
