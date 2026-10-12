import asyncio
import copy
import inspect
import json
import logging
import os
import time
import uuid
from collections import deque
from datetime import datetime
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic

from agents.tools import (
    AgentToolSpec,
    build_shared_rag_tools,
    billing_tools,
    escalation_tools,
    general_tools,
    technical_tools,
)
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel
from core.llm_utils import extract_text_content
from core.decision_policy import should_consult_provider, validate_decision
from core.decision_provider import DecisionProvider, DecisionResult, LocalPolicyDecisionProvider
from monitor.decision_metrics import DecisionMetricEvent, DecisionMetricsRegistry
from monitor.version3_shadow import Version3ShadowRegistry
from runtime.harness import RunHarness
from evaluation.privacy import redact_text
from agents.output_contract import parse_agent_output, contract_evidence_is_grounded

logger = logging.getLogger(__name__)


                                                                            

class AgentType(Enum):
    GENERAL   = "general"          
    TECHNICAL = "technical"        
    BILLING   = "billing"           
    ESCALATION = "escalation"          


@dataclass(frozen=True)
class AgentProfile:

    role: str
    mission: str
    workflow: Tuple[str, ...]
    input_contract: Tuple[str, ...]
    output_contract: Tuple[str, ...]
    handoff_conditions: Tuple[str, ...] = ()
    tool_scope: Tuple[str, ...] = ()
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1024


def _env_float(name: str, default: float) -> float:
                                
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("忽略非法浮点配置 %s=%r", name, os.getenv(name))
        return default


def _env_int(name: str, default: int) -> int:
                                
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("忽略非法整数配置 %s=%r", name, os.getenv(name))
        return default


@dataclass
class AgentStats:
                                        
    total:     int   = 0
    success:   int   = 0
    total_ms:  float = 0.0
    monitor_penalty: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total if self.total else 0.0

    def routing_score(self) -> float:
                                       
        latency_score = 1.0 / (1.0 + self.avg_ms / 1000)
        base_score = self.success_rate * 0.7 + latency_score * 0.3
        return base_score * max(0.0, 1.0 - self.monitor_penalty)


@dataclass
class AgentResponse:
    agent_type:  AgentType
    content:     str
    success:     bool
    confidence:  float = 1.0
    latency_ms:  float = 0.0
    escalate:    bool  = False           
    tools_used:  List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)
    error_code: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    output_contract: Optional[Dict[str, Any]] = None


@dataclass
class Request:
    message:     str
    user_id:     str
    conv_id:     str
    context:     str = ""                                  
    history:     Optional[List[Dict[str, str]]] = None               
    entities:    Dict[str, List[str]] = field(default_factory=dict)
    intent:      Optional[IntentCategory] = None
    intent_group: Optional[str] = None
    urgency:     Optional[UrgencyLevel]   = None
    intent_confidence: float = 1.0
    tenant_id: str = "default"
    rollout_variant: str = "stable"
    model_override: Optional[str] = None
    decision_source: str = "local"
    decision_intent: Optional[IntentCategory] = None
    decision_confidence: float = 0.0
    decision_latency_ms: float = 0.0
    decision_primary_agent: Optional[str] = None
    decision_supporting_agents: List[str] = field(default_factory=list)
    decision_needs_human: bool = False
    decision_needs_clarification: bool = False
    decision_fallback_reason: str = ""
                                        
    request_id:  str = field(default_factory=lambda: str(uuid.uuid4()))
    harness: Optional[RunHarness] = None


@dataclass
class OrchestratorResult:
    request_id:  str
    response:    str
    agent_type:  AgentType
    intent:      Optional[IntentCategory]
    escalated:   bool  = False
    latency_ms:  float = 0.0
    agent_types: List[AgentType] = field(default_factory=list)
    primary_agent: Optional[AgentType] = None
    supporting_agents: List[AgentType] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)
    routing_reason: str = ""
    routing_confidence: float = 0.0
    agent_failures: List[Dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    harness_trace: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
                       
    primary_agent: AgentType
    supporting_agents: List[AgentType] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0

    @property
    def agent_types(self) -> List[AgentType]:
        return [self.primary_agent] + self.supporting_agents

    @property
    def multi_agent(self) -> bool:
        return bool(self.supporting_agents)


class AgentExecutionError(RuntimeError):
                                 

    def __init__(
        self,
        message: str,
        *,
        tools_used: Optional[List[str]] = None,
        tool_traces: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(message)
        self.tools_used = list(tools_used or [])
        self.tool_traces = list(tool_traces or [])


                                                                              

class BaseAgent:
                                         

    agent_type: AgentType
    system_prompt: str
    profile: AgentProfile

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        skill_manager: Optional[Any] = None,
        profile: Optional[AgentProfile] = None,
    ):
        self._client = client
        self.profile = profile or self.profile
        self._model  = self.profile.model or model
        self._skill_manager = skill_manager
        self.stats   = AgentStats()
        self._shared_tools: Dict[str, AgentToolSpec] = {}

    def get_tools(self) -> Dict[str, AgentToolSpec]:
                               
        return dict(self._shared_tools)

    def set_shared_tools(self, tools: Optional[Dict[str, AgentToolSpec]]) -> None:
        self._shared_tools = dict(tools or {})

    async def handle(self, req: Request) -> AgentResponse:
        t0 = time.monotonic()
        self.stats.total += 1
        try:
            timeout_s = _env_float("ECHOMIND_AGENT_TIMEOUT_S", 60.0)
            content, tools_used, tool_traces, input_tokens, output_tokens = await asyncio.wait_for(
                self._call_llm(req),
                timeout=timeout_s,
            )
            ms = (time.monotonic() - t0) * 1000
            contract = parse_agent_output(content)
            if contract is not None and not contract_evidence_is_grounded(
                contract, req.message, req.context, tool_traces
            ):
                contract = None
            if contract is None and content.lstrip().startswith(("{", "```json")):
                raise AgentExecutionError("Agent 输出结构或事实证据校验失败", tools_used=tools_used,
                                          tool_traces=tool_traces)
            if contract is not None:
                content = contract.summary
            self.stats.success += 1
            self.stats.total_ms += ms
            escalate = self._needs_escalation(content)
            return AgentResponse(
                agent_type=self.agent_type,
                content=content,
                success=True,
                latency_ms=ms,
                escalate=escalate,
                tools_used=tools_used,
                tool_traces=tool_traces,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                output_contract=contract.model_dump() if contract is not None else None,
            )
        except AgentExecutionError as ex:
            ms = (time.monotonic() - t0) * 1000
            self.stats.total_ms += ms
            logger.error(f"{self.agent_type.value} 处理失败: {ex}")
            return AgentResponse(
                agent_type=self.agent_type,
                content="抱歉，处理您的请求时出现问题，请稍后重试。",
                success=False,
                latency_ms=ms,
                tool_traces=list(ex.tool_traces),
                tools_used=list(ex.tools_used),
                error_code="AGENT_EXECUTION_ERROR",
            )
        except asyncio.TimeoutError:
            ms = (time.monotonic() - t0) * 1000
            self.stats.total_ms += ms
            logger.error("%s 处理超时", self.agent_type.value)
            return AgentResponse(
                agent_type=self.agent_type,
                content="当前请求处理超时，请稍后重试。",
                success=False,
                latency_ms=ms,
                error_code="AGENT_TIMEOUT",
            )
        except Exception as ex:
            ms = (time.monotonic() - t0) * 1000
            self.stats.total_ms += ms
            logger.error(f"{self.agent_type.value} 处理失败: {ex}")
            return AgentResponse(
                agent_type=self.agent_type,
                content="抱歉，处理您的请求时出现问题，请稍后重试。",
                success=False,
                latency_ms=ms,
                tool_traces=[],
                error_code="AGENT_EXECUTION_FAILED",
            )

    async def _call_llm(self, req: Request) -> Tuple[str, List[str], List[Dict[str, Any]], int, int]:
        def _clean(s: str) -> str:
            return s.encode("utf-8", errors="ignore").decode("utf-8")

        messages = []
        if req.context:
            messages.append({"role": "user", "content": f"[背景信息]\n{_clean(req.context)}"})
            messages.append({"role": "assistant", "content": "好的，我已了解背景信息。"})
        if req.entities:
            entities_text = json.dumps(req.entities, ensure_ascii=False)
            messages.append({"role": "user", "content": f"[结构化实体]\n{_clean(entities_text)}"})
            messages.append({"role": "assistant", "content": "好的，我会结合这些结构化实体处理。"})
        role_packet = self._build_role_packet(req)
        if role_packet:
            messages.append({"role": "user", "content": f"[角色输入契约]\n{_clean(role_packet)}"})
            messages.append({"role": "assistant", "content": "好的，我会按照该角色的输入和输出契约处理。"})
        messages.append({"role": "user", "content": _clean(req.message)})

        tools = self.get_tools()
        tools_used: List[str] = []
        tool_traces: List[Dict[str, Any]] = []
        input_tokens = 0
        output_tokens = 0
        for _ in range(3):
            request_kwargs: Dict[str, Any] = {
                "model": req.model_override or self._model,
                "max_tokens": self.profile.max_tokens,
                "temperature": self.profile.temperature,
                "system": self._build_system_prompt(req),
                "messages": messages,
            }
            if tools:
                request_kwargs["tools"] = [
                    {
                        "name": spec.name,
                        "description": spec.description,
                        "input_schema": spec.input_schema,
                    }
                    for spec in tools.values()
                ]
            resp = await self._client.messages.create(**request_kwargs)
            usage = getattr(resp, "usage", None)
            if usage is not None:
                if isinstance(usage, dict):
                    input_tokens += int(usage.get("input_tokens", 0) or 0)
                    output_tokens += int(usage.get("output_tokens", 0) or 0)
                else:
                    input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
                    output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
            tool_uses = [block for block in (resp.content or []) if self._block_type(block) == "tool_use"]
            if not tool_uses:
                return (
                    extract_text_content(resp.content), list(tools_used), list(tool_traces),
                    input_tokens, output_tokens,
                )

            messages.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for block in tool_uses:
                name = self._block_value(block, "name")
                tool_use_id = self._block_value(block, "id")
                args = self._block_value(block, "input") or {}
                spec = tools.get(name)
                tool_t0 = time.monotonic()
                call_success = True
                result_success: Optional[bool] = None
                error_text = ""
                if req.harness is not None and not req.harness.allow_tool():
                    call_success = False
                    result = {"success": False, "error": "本次运行的工具调用预算已用完"}
                    error_text = result["error"]
                elif spec is None:
                    call_success = False
                    result: Any = {"success": False, "error": f"工具不在 {self.agent_type.value} Agent 白名单中"}
                    error_text = result["error"]
                else:
                    try:
                        self._validate_tool_input(spec, args)
                        result = spec.handler(req, args)
                        if inspect.isawaitable(result):
                            result = await result
                        tools_used.append(name)
                        if isinstance(result, dict) and "success" in result:
                            result_success = bool(result.get("success"))
                    except Exception as ex:
                        call_success = False
                        logger.warning("Agent 工具 %s 执行失败: %s", name, ex)
                        error_text = str(ex)
                        result = {"success": False, "error": error_text}
                tool_latency_ms = (time.monotonic() - tool_t0) * 1000
                if not error_text and isinstance(result, dict):
                    error_text = str(result.get("error", "") or "")
                tool_traces.append(
                    {
                        "agent_type": self.agent_type.value,
                        "tool_name": name,
                        "tool_use_id": tool_use_id,
                        "input": dict(args),
                        "success": call_success,
                        "result_success": result_success,
                        "latency_ms": round(tool_latency_ms, 1),
                        "cached": bool(result.get("cached")) if isinstance(result, dict) else False,
                        "reranked": bool(result.get("reranked")) if isinstance(result, dict) else False,
                        "error": error_text,
                        "evidence_text": json.dumps(result, ensure_ascii=False)[:1500],
                    }
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
            messages.append({"role": "user", "content": tool_results})

        raise AgentExecutionError(
            f"{self.agent_type.value} 工具调用超过最大轮数",
            tools_used=tools_used,
            tool_traces=tool_traces,
        )

    @staticmethod
    def _block_type(block: Any) -> Optional[str]:
        if isinstance(block, dict):
            return block.get("type")
        return getattr(block, "type", None)

    @staticmethod
    def _block_value(block: Any, key: str) -> Any:
        if isinstance(block, dict):
            return block.get(key)
        return getattr(block, key, None)

    @staticmethod
    def _validate_tool_input(spec: AgentToolSpec, args: Any) -> None:
        if not isinstance(args, dict):
            raise ValueError("工具参数必须是 JSON 对象")
        schema = spec.input_schema
        for field_name in schema.get("required", []):
            if field_name not in args:
                raise ValueError(f"缺少必需参数: {field_name}")
        properties = schema.get("properties", {})
        unknown = set(args) - set(properties)
        if unknown and schema.get("additionalProperties") is False:
            raise ValueError(f"不允许的工具参数: {', '.join(sorted(unknown))}")
        type_map = {"string": str, "number": (int, float), "integer": int, "boolean": bool}
        for key, value in args.items():
            expected = properties.get(key, {}).get("type")
            if expected in type_map and not isinstance(value, type_map[expected]):
                raise ValueError(f"参数 {key} 类型错误，期望 {expected}")

    def _build_system_prompt(self, req: Request) -> str:
                                               
        profile_prompt = (
            f"\n\n[角色契约]\n"
            f"角色：{self.profile.role}\n"
            f"职责：{self.profile.mission}\n"
            f"处理流程：{' -> '.join(self.profile.workflow)}\n"
            f"可用输入：{'；'.join(self.profile.input_contract)}\n"
            f"输出要求：{'；'.join(self.profile.output_contract)}\n"
            f"升级条件：{'；'.join(self.profile.handoff_conditions) or '无，按通用客服规则处理'}\n"
            f"允许的数据/工具范围：{'、'.join(self.profile.tool_scope) or '仅使用当前请求上下文'}\n"
            "不要声称执行了未提供的查询、修改或退款操作；缺少证据时明确说明需要核验。"
        )
        base_prompt = f"{self.system_prompt}{profile_prompt}"
        base_prompt += (
            "\n\n[机器可校验输出契约]\n必须只返回 JSON 对象，字段为 summary、facts、evidence、"
            "uncertainties、next_steps、confidence。facts 中每个事实都必须有 evidence 项逐一匹配，"
            "evidence 的 source 和 quote 必须来自当前对话或实际工具结果；无证据时 facts 和 evidence 留空，"
            "把未知内容写入 uncertainties。禁止编造来源。"
        )
        if self._skill_manager is None:
            return base_prompt
        skill_prompt = self._skill_manager.prompt_for(req.message, self.agent_type.value)
        if not skill_prompt:
            return base_prompt
        return f"{base_prompt}\n\n[动态 Skills]\n{skill_prompt}"

    def _build_role_packet(self, req: Request) -> str:
                                         
        packet = {
            "agent_type": self.agent_type.value,
            "intent": req.intent.value if req.intent else None,
            "intent_group": req.intent_group,
            "urgency": req.urgency.name if req.urgency else None,
            "intent_confidence": round(req.intent_confidence, 4),
            "available_entities": req.entities or {},
        }
        return json.dumps(packet, ensure_ascii=False)

    def _needs_escalation(self, content: str) -> bool:
                                       
        keywords = ["转人工", "人工客服", "escalate", "specialist", "无法处理"]
        return any(kw in content for kw in keywords)


class GeneralAgent(BaseAgent):
    agent_type    = AgentType.GENERAL
    profile = AgentProfile(
        role="通用客服分诊与首轮接待",
        mission="快速回答基础问题，澄清不完整需求，并识别是否需要专业 Agent 或人工处理。",
        workflow=("复述诉求", "判断业务范围", "直接回答或补充必要信息", "给出下一步"),
        input_contract=("对话历史", "用户画像", "意图与紧急度", "知识库上下文"),
        output_contract=("先回应核心问题", "信息不足时只询问必要字段", "明确下一步和边界"),
        handoff_conditions=("涉及权限、资金、隐私或复杂投诉", "用户明确要求人工"),
        tool_scope=("search_knowledge_base", "inspect_request_context", "suggest_required_fields"),
        temperature=0.3,
        max_tokens=900,
    )
    system_prompt = (
        "你是 EchoMind 智能客服。友好、简洁地回答用户问题。"
        "如果问题超出你的能力范围，明确说明并建议转接专业客服。"
    )

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["triage_targets"] = ["technical", "billing", "escalation"]
        packet["response_mode"] = "answer_or_clarify"
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(general_tools())
        return tools


class TechnicalAgent(BaseAgent):
    agent_type    = AgentType.TECHNICAL
    profile = AgentProfile(
        role="技术故障诊断与排障",
        mission="基于错误码、环境和复现信息缩小根因范围，给出低风险、可验证的排查步骤。",
        workflow=("确认现象", "判断影响范围", "按网络/权限/配置/依赖排查", "给出验证方式", "判断升级条件"),
        input_contract=("错误码", "问题发生时间", "运行环境", "影响范围", "最近变更", "知识库上下文"),
        output_contract=("现象复述", "可能原因", "编号排查步骤", "验证结果", "需要补充的信息"),
        handoff_conditions=("生产大面积不可用", "数据丢失或权限异常", "需要后台日志、数据库或人工操作"),
        tool_scope=("search_knowledge_base", "lookup_error_code", "build_diagnostic_plan"),
        temperature=0.1,
        max_tokens=1200,
    )
    system_prompt = (
        "你是技术支持专家。专注于：故障排查、错误诊断、系统配置。"
        "提供清晰的步骤化解决方案。遇到需要后台操作的问题，说明需要升级处理。"
    )

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["diagnostic_fields"] = {
            "error_codes": req.entities.get("error_code", []),
            "environment_hint": "请从用户消息和背景中确认设备、系统、版本、网络",
            "risk_boundary": "不得要求密码、验证码、完整密钥；不得建议破坏性操作",
        }
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(technical_tools())
        return tools


class BillingAgent(BaseAgent):
    agent_type    = AgentType.BILLING
    profile = AgentProfile(
        role="账单核验与售后处理",
        mission="区分扣款、退款、发票、订阅等资金场景，解释可判断事实，并明确核验和人工审核边界。",
        workflow=("确认账单场景", "收集必要核验字段", "区分订单/实付/退款金额", "说明处理路径与时效", "判断是否升级"),
        input_contract=("订单号", "金额与币种", "支付时间", "支付渠道", "用户期望", "知识库上下文"),
        output_contract=("需要核验的信息", "当前可判断内容", "下一步处理路径", "时效边界"),
        handoff_conditions=("实际退款或补偿", "重复扣款或支付成功但订单未生效", "发票作废/重开", "企业合同或大额订单"),
        tool_scope=("search_knowledge_base", "check_billing_fields", "compare_amounts"),
        temperature=0.0,
        max_tokens=1100,
    )
    system_prompt = (
        "你是账单服务专家。专注于：账单查询、退款申请、发票问题、订阅管理。"
        "对财务问题保持准确和专业。涉及实际退款操作时，说明需要人工审核。"
    )

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["verification_fields"] = {
            "order_id": req.entities.get("order_id", []),
            "amount": req.entities.get("amount", []),
            "date": req.entities.get("date", []),
            "missing_fields": [
                field for field, values in (
                    ("订单号或交易号", req.entities.get("order_id", [])),
                    ("支付金额", req.entities.get("amount", [])),
                ) if not values
            ],
            "risk_boundary": "不得承诺退款成功、立即到账或直接修改账单",
        }
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(billing_tools())
        return tools


class EscalationAgent(BaseAgent):
\
\
\
\
       

    agent_type = AgentType.ESCALATION
    profile = AgentProfile(
        role="人工升级与交接",
        mission="确认升级原因，整理已知上下文，告知用户下一步，不执行未经授权的业务操作。",
        workflow=("确认升级原因", "整理已知信息", "标记优先级", "生成交接摘要"),
        input_contract=("用户消息", "意图", "紧急度", "结构化实体", "对话背景"),
        output_contract=("升级原因", "已知信息摘要", "还需补充的信息", "保守的后续说明"),
        handoff_conditions=("用户明确要求人工", "紧急或高风险场景"),
        tool_scope=("search_knowledge_base", "create_handoff_summary"),
        temperature=0.0,
        max_tokens=500,
    )
    system_prompt = "你负责客服人工升级交接，不要继续模拟已完成的后台操作。"

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(escalation_tools())
        return tools

    async def handle(self, req: Request) -> AgentResponse:
        t0 = time.monotonic()
        self.stats.total += 1
        intent = req.intent.value if req.intent else "unknown"
        urgency = req.urgency.name if req.urgency else "UNKNOWN"
        entities = json.dumps(req.entities or {}, ensure_ascii=False)
        content = (
            "我已将这个问题标记为人工升级处理。\n\n"
            f"升级原因：意图={intent}，紧急度={urgency}\n"
            f"已记录信息：{entities}\n"
            "请不要发送密码、短信验证码或完整支付凭证；人工客服会根据会话记录继续核验。"
        )
        ms = (time.monotonic() - t0) * 1000
        self.stats.success += 1
        self.stats.total_ms += ms
        return AgentResponse(
            agent_type=self.agent_type,
            content=content,
            success=True,
            latency_ms=ms,
            escalate=True,
            tools_used=[],
        )


class ResponseComposer:
                                    

    def __init__(self, client: AsyncAnthropic, model: str, skill_manager: Optional[Any] = None):
        self._client = client
        self._model = model
        self._skill_manager = skill_manager

    async def compose(self, req: Request, responses: List[AgentResponse]) -> str:
        successful = [response for response in responses if response.success and response.content.strip()]
        if not successful:
            return "抱歉，所有 Agent 均处理失败。"
        if len(successful) == 1:
            return successful[0].content

        if self._has_material_conflict(successful):
            return (
                "不同的核验结果存在关键信息冲突，我先不合并成确定结论。"
                "建议由人工客服核对订单或账单记录后再继续处理。"
            )

        evidence = "\n\n".join(
            f"[{response.agent_type.value} Agent 输出]\n{response.content}\n"
            f"结构化契约：{json.dumps(response.output_contract, ensure_ascii=False) if response.output_contract else '未通过结构化解析；按未核验文本处理'}"
            for response in successful
        )
        prompt = (
            "你是客服 Response Composer，负责把多个专业 Agent 的结果合并成一条最终回复。\n"
            "要求：以主 Agent 的结论为主，按用户问题优先级组织内容；去掉重复和冲突表述；"
            "不能补造订单、退款、后台查询结果；如果结论冲突，明确说明需要核验；"
            "保留必要的排查步骤、核验字段和升级边界。只输出给用户看的中文回复，不要提及 Agent。\n\n"
            f"主 Agent：{successful[0].agent_type.value}\n"
            f"用户问题：{req.message}\n"
            f"候选结果：\n{evidence}"
        )
        if self._skill_manager is not None:
            skill = self._skill_manager.prompt_for(req.message, "general")
            if skill:
                prompt += f"\n\n[通用客服输出边界]\n{skill}"
        try:
            response = await self._client.messages.create(
                model=req.model_override or self._model,
                max_tokens=_env_int("ECHOMIND_COMPOSER_MAX_TOKENS", 1000),
                temperature=_env_float("ECHOMIND_COMPOSER_TEMPERATURE", 0.1),
                messages=[{"role": "user", "content": prompt}],
            )
            content = extract_text_content(response.content).strip()
            if content:
                return content
        except Exception as ex:
            logger.warning("Response Composer 失败，使用确定性合并: %s", ex)

                                            
        return "\n\n".join(
            f"{response.content}" if index == 0 else f"补充说明：\n{response.content}"
            for index, response in enumerate(successful)
        )

    @staticmethod
    def _has_material_conflict(responses: List[AgentResponse]) -> bool:
        """Conservative conflict check for contradictory numbers or polarity."""
        import re

        facts_by_key: Dict[str, set[str]] = {}
        for response in responses:
            contract = response.output_contract or {}
            for key, value in (contract.get("facts") or {}).items():
                facts_by_key.setdefault(str(key), set()).add(str(value).strip().lower())
        if any(len(values) > 1 for values in facts_by_key.values()):
            return True
        normalized = [re.sub(r"\s+", "", response.content.lower()) for response in responses]
        numeric_values = [set(re.findall(r"(?:￥|¥|\$)\d+(?:\.\d+)?|\d+(?:\.\d+)?(?:元|块|美元|人民币)", text)) for text in normalized]
        nonempty_numbers = [values for values in numeric_values if values]
        if len(nonempty_numbers) > 1 and any(values != nonempty_numbers[0] for values in nonempty_numbers[1:]):
            return True
        positive = ("已退款", "已发货", "已支付", "已开票", "处理成功")
        negative = ("未退款", "未发货", "未支付", "未开票", "处理失败")
        return any(any(term in text for term in positive) for text in normalized) and any(
            any(term in text for term in negative) for text in normalized
        )


                                                                             

class AgentOrchestrator:
\
\
\
\
\
\
\
       

                             
    _INTENT_ROUTING: Dict[IntentCategory, AgentType] = {
        IntentCategory.TECHNICAL:  AgentType.TECHNICAL,
        IntentCategory.TECHNICAL_LOGIN: AgentType.TECHNICAL,
        IntentCategory.TECHNICAL_CRASH: AgentType.TECHNICAL,
        IntentCategory.BILLING:    AgentType.BILLING,
        IntentCategory.REFUND:     AgentType.BILLING,
        IntentCategory.INVOICE:    AgentType.BILLING,
        IntentCategory.PAYMENT_ISSUE: AgentType.BILLING,
        IntentCategory.ACCOUNT:    AgentType.BILLING,
        IntentCategory.ACCOUNT_SECURITY: AgentType.BILLING,
        IntentCategory.ESCALATION: AgentType.ESCALATION,
        IntentCategory.HUMAN_HANDOFF: AgentType.ESCALATION,
                            
    }

    def __init__(
        self,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "claude-3-5-sonnet-20241022",
        skill_manager: Optional[Any] = None,
        rag_tool_manager: Optional[Any] = None,
        decision_provider: Optional[DecisionProvider] = None,
        decision_mode: str = "off",
        decision_metrics: Optional[DecisionMetricsRegistry] = None,
        shadow_registry: Optional[Version3ShadowRegistry] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = AsyncAnthropic(**kwargs)

        self._intent_recognizer = IntentRecognizer(api_key=api_key, base_url=base_url, model=model)
        self._decision_mode = decision_mode.strip().lower() if decision_mode else "off"
        if self._decision_mode not in {"off", "shadow", "active"}:
            self._decision_mode = "off"
        self._decision_provider = decision_provider if self._decision_mode != "off" else None
        self._decision_local_gate = _env_float("ECHOMIND_V2_LOCAL_CONFIDENCE_GATE", 0.55)
        self._decision_min_confidence = _env_float("ECHOMIND_V2_MIN_CONFIDENCE", 0.65)
        self._decision_metrics = decision_metrics or DecisionMetricsRegistry()
        self._decision_version = str(getattr(self._decision_provider, "version", "2")) if self._decision_provider else "1"
        self._shadow_registry = shadow_registry or Version3ShadowRegistry()
        self._decision_gate_counts = {"consulted": 0, "skipped": 0}
        self._skill_manager = skill_manager
        self._composer = ResponseComposer(client, model, skill_manager)
        self._shared_tools: Dict[str, AgentToolSpec] = {}
        self._recent_tool_traces = deque(maxlen=_env_int("ECHOMIND_TOOL_TRACE_MAX", 200))
        self._trace_inputs: Dict[str, str] = {}
        self._replay_enabled = os.getenv("ECHOMIND_TRACE_RERUN_ENABLED", "false").strip().lower() in {"1", "true", "yes"}
        self._replay_requests: Dict[str, Request] = {}
        self._replay_order = deque(maxlen=_env_int("ECHOMIND_TRACE_REPLAY_MAX", 50))

                                  
        self._pool: Dict[AgentType, List[BaseAgent]] = {
            AgentType.GENERAL: [self._make_agent(GeneralAgent, client, model, skill_manager)],
            AgentType.TECHNICAL: [self._make_agent(TechnicalAgent, client, model, skill_manager)],
            AgentType.BILLING: [self._make_agent(BillingAgent, client, model, skill_manager)],
            AgentType.ESCALATION: [self._make_agent(EscalationAgent, client, model, skill_manager)],
        }
        self.set_shared_tools(build_shared_rag_tools(rag_tool_manager))

    @staticmethod
    def _make_agent(
        agent_cls: type[BaseAgent],
        client: AsyncAnthropic,
        default_model: str,
        skill_manager: Optional[Any],
    ) -> BaseAgent:
\
\
\
           
        profile = agent_cls.profile
        env_name = f"ECHOMIND_{agent_cls.agent_type.value.upper()}_MODEL"
        model = os.getenv(env_name, "").strip() or profile.model
        configured_profile = replace(profile, model=model) if model else profile
        return agent_cls(client, default_model, skill_manager, profile=configured_profile)

    def set_skill_manager(self, skill_manager: Optional[Any]) -> None:
                                               
        self._skill_manager = skill_manager
        self._composer._skill_manager = skill_manager
        for agents in self._pool.values():
            for agent in agents:
                agent._skill_manager = skill_manager

    def set_shared_tools(self, tools: Optional[Dict[str, AgentToolSpec]]) -> None:
                                  
        self._shared_tools = dict(tools or {})
        for agents in self._pool.values():
            for agent in agents:
                agent.set_shared_tools(self._shared_tools)

    async def recognize_intent(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ):
                                                
        local_result = await self._intent_recognizer.recognize(message, history=history)
        if self._decision_provider is None:
            return local_result

        should_consult = (
            self._decision_version == "3"
            or should_consult_provider(message, local_result, self._decision_local_gate)
        )
        if not should_consult:
            self._decision_gate_counts["skipped"] += 1
            self._decision_metrics.record(
                DecisionMetricEvent(self._decision_mode, "local", 0.0, local_result.confidence)
            )
            return local_result

        self._decision_gate_counts["consulted"] += 1
        try:
            decision = await self._decision_provider.decide(
                message, history=history, local_result=local_result
            )
            source = f"v{self._decision_version}_{self._decision_mode}"
            self._shadow_registry.record(message, local_result, decision)
            self._decision_metrics.record(
                DecisionMetricEvent(
                    self._decision_mode,
                    source,
                    decision.latency_ms,
                    decision.confidence,
                    intent_match=decision.intent == local_result.intent,
                )
            )
            metadata = {
                "decision_intent": decision.intent,
                "decision_confidence": decision.confidence,
                "decision_latency_ms": decision.latency_ms,
                "decision_primary_agent": decision.primary_agent,
                "decision_supporting_agents": list(decision.supporting_agents),
                "decision_needs_human": decision.needs_human,
                "decision_needs_clarification": decision.needs_clarification,
                "decision_schema_version": decision.schema_version,
                "decision_source": source,
                "decision_fallback_reason": "",
            }
                                                                           
                                                                             
            if self._decision_version == "3" and self._decision_mode == "active":
                metadata.update(
                    intent=decision.intent,
                    confidence=decision.confidence,
                    intent_group=self._intent_recognizer._intent_group(decision.intent),
                )
            return replace(local_result, **metadata)
        except Exception as exc:
            self._decision_metrics.record(
                DecisionMetricEvent(
                    self._decision_mode,
                    "fallback",
                    0.0,
                    local_result.confidence,
                    fallback=True,
                )
            )
            return replace(
                local_result,
                decision_source="local",
                decision_fallback_reason=f"v{self._decision_version}_error:{type(exc).__name__}",
            )

    def get_decision_stats(self) -> Dict[str, Any]:
        consulted = self._decision_gate_counts["consulted"]
        skipped = self._decision_gate_counts["skipped"]
        total = consulted + skipped
        return {
            "mode": self._decision_mode,
            "enabled": self._decision_provider is not None,
            "version": self._decision_version,
            "selection": {
                "consulted": consulted,
                "skipped": skipped,
                "consult_rate": round(consulted / total, 4) if total else 0.0,
                "local_confidence_gate": self._decision_local_gate,
            },
            "metrics": self._decision_metrics.summary(),
            "shadow": self._shadow_registry.summary(),
        }

    def _record_tool_trace(self, result: OrchestratorResult) -> None:
        events = result.harness_trace.get("events", [])
        run_started = next((event for event in events if event.get("event") == "run_started"), {})
        trace = {
            "request_id": result.request_id,
            "timestamp": datetime.now().isoformat(),
            "intent": result.intent.value if result.intent else None,
            "primary_agent": result.primary_agent.value if result.primary_agent else None,
            "supporting_agents": [agent.value for agent in result.supporting_agents],
            "tools_used": list(result.tools_used),
            "tool_calls": self._redact_value(list(result.tool_traces)),
            "escalated": result.escalated,
            "latency_ms": round(result.latency_ms, 1),
            "agent_failures": list(result.agent_failures),
            "routing_reason": result.routing_reason,
            "routing_confidence": round(result.routing_confidence, 4),
            "agent_types": [agent.value for agent in result.agent_types],
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "harness": dict(result.harness_trace),
            "runtime_version": os.getenv("ECHOMIND_BUILD_VERSION", "dev"),
            "decision_mode": self._decision_mode,
            "rollout_variant": run_started.get("rollout_variant", "stable"),
            "model": run_started.get("model", "unknown"),
            "decision_source": run_started.get("decision_source", "local"),
        }
        self._recent_tool_traces.append(trace)

    def get_tool_trace(self, request_id: str) -> Optional[Dict[str, Any]]:
        for trace in reversed(self._recent_tool_traces):
            if trace.get("request_id") == request_id:
                return trace
        return None

    def get_feedback_input(self, request_id: str) -> Optional[str]:
        return self._trace_inputs.get(request_id)

    @staticmethod
    def _redact_value(value: Any) -> Any:
        if isinstance(value, str):
            return redact_text(value)
        if isinstance(value, dict):
            return {str(key): AgentOrchestrator._redact_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [AgentOrchestrator._redact_value(item) for item in value]
        return value

    async def replay_request(self, request_id: str) -> OrchestratorResult:
        if not self._replay_enabled:
            raise PermissionError("Trace rerun is disabled; set ECHOMIND_TRACE_RERUN_ENABLED=true")
        snapshot = self._replay_requests.get(request_id)
        if snapshot is None:
            raise KeyError(request_id)
        replay = copy.deepcopy(snapshot)
        replay.request_id = str(uuid.uuid4())
        replay.harness = None
        return await self.run(replay)

    def get_recent_tool_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self._recent_tool_traces:
            return []
        limit = max(1, min(int(limit or 20), len(self._recent_tool_traces)))
        return list(reversed(list(self._recent_tool_traces)[-limit:]))

                                                                             

    async def run(self, req: Request) -> OrchestratorResult:
\
\
\
           
        t0 = time.monotonic()
        req.harness = RunHarness(req.request_id, req.message)
        req.harness.record(
            "run_started", tenant_hash=hashlib.sha256(req.tenant_id.encode()).hexdigest()[:12],
            rollout_variant=req.rollout_variant,
            model=req.model_override or os.getenv("ANTHROPIC_MODEL", "configured-default"),
            decision_source=req.decision_source,
        )
        if req.request_id not in self._trace_inputs and len(self._trace_inputs) >= self._recent_tool_traces.maxlen:
            self._trace_inputs.pop(next(iter(self._trace_inputs)))
        self._trace_inputs[req.request_id] = redact_text(req.message)
        if self._replay_enabled and req.request_id not in self._replay_requests:
            if len(self._replay_order) == self._replay_order.maxlen:
                oldest = self._replay_order.popleft()
                self._replay_requests.pop(oldest, None)
            self._replay_order.append(req.request_id)
            self._replay_requests[req.request_id] = copy.deepcopy(req)

                              
        if req.intent is None:
            intent_result = await self.recognize_intent(req.message, history=req.history)
            req.intent  = intent_result.intent
            req.intent_group = intent_result.intent_group
            req.urgency = intent_result.urgency
            req.intent_confidence = intent_result.confidence
            req.decision_source = intent_result.decision_source
            req.decision_intent = intent_result.decision_intent
            req.decision_confidence = intent_result.decision_confidence
            req.decision_latency_ms = intent_result.decision_latency_ms
            req.decision_primary_agent = intent_result.decision_primary_agent
            req.decision_supporting_agents = list(intent_result.decision_supporting_agents)
            req.decision_needs_human = intent_result.decision_needs_human
            req.decision_needs_clarification = intent_result.decision_needs_clarification
            req.decision_fallback_reason = intent_result.decision_fallback_reason

        if self._needs_clarification(req):
            result = OrchestratorResult(
                request_id=req.request_id,
                response="我还不能确定您要处理的是哪类问题。请补充一下是订单物流、退款账单、账户资料，还是技术故障？",
                agent_type=AgentType.GENERAL,
                intent=req.intent,
                escalated=False,
                latency_ms=(time.monotonic() - t0) * 1000,
                agent_types=[AgentType.GENERAL],
                primary_agent=AgentType.GENERAL,
                routing_reason="低置信度 OTHER 意图，先澄清用户需求",
                routing_confidence=req.intent_confidence,
            )
            req.harness.record("clarification_requested")
            req.harness.record("run_finished", success=True)
            result.harness_trace = req.harness.summary()
            self._record_tool_trace(result)
            return result

                                         
        decision = self._route_decision(req)
        req.harness.record(
            "routing_decided", intent=req.intent.value if req.intent else None,
            primary_agent=decision.primary_agent.value,
            supporting_agents=[agent.value for agent in decision.supporting_agents],
            reason=decision.reason, confidence=round(decision.confidence, 4),
        )
        if decision.multi_agent:
            result = await self.run_parallel(req, decision)
            req.harness.record("run_finished", success=not bool(result.agent_failures))
            result.harness_trace = req.harness.summary()
            self._record_tool_trace(result)
            return result

                           
        response = await self._execute(req, decision.primary_agent)

                 
        escalated = False
        if response.escalate or req.urgency == UrgencyLevel.CRITICAL or req.intent in (
            IntentCategory.ESCALATION,
            IntentCategory.HUMAN_HANDOFF,
        ):
            escalated = True
            logger.warning(f"请求 {req.request_id} 触发升级: urgency={req.urgency}")
                                

        result = OrchestratorResult(
            request_id=req.request_id,
            response=response.content,
            agent_type=response.agent_type,
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            agent_types=[response.agent_type],
            primary_agent=decision.primary_agent,
            supporting_agents=[],
            tools_used=list(response.tools_used),
            tool_traces=list(response.tool_traces),
            routing_reason=decision.reason,
            routing_confidence=decision.confidence,
            agent_failures=[] if response.success else [{
                "agent_type": response.agent_type.value,
                "error_code": response.error_code or "AGENT_EXECUTION_FAILED",
                "message": response.content,
            }],
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        req.harness.record("run_finished", success=not bool(result.agent_failures))
        result.harness_trace = req.harness.summary()
        self._record_tool_trace(result)
        return result

    async def run_parallel(self, req: Request, decision: RoutingDecision) -> OrchestratorResult:
\
\
\
           
        t0 = time.monotonic()
        agent_types = decision.agent_types
        if req.harness is not None:
            agent_types = agent_types[:req.harness.max_agent_calls]
            req.harness.record("agents_started", agents=[agent.value for agent in agent_types])
        tasks = [self._execute(req, at) for at in agent_types]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        valid_responses = [r for r in responses if isinstance(r, AgentResponse)]
        agent_failures = [
            {
                "agent_type": agent_types[index].value,
                "error_code": (
                    response.error_code
                    if isinstance(response, AgentResponse)
                    else "AGENT_TASK_EXCEPTION"
                ),
                "message": (
                    response.content
                    if isinstance(response, AgentResponse)
                    else str(response)
                ),
            }
            for index, response in enumerate(responses)
            if not isinstance(response, AgentResponse) or not response.success
        ]
        combined = await self._composer.compose(req, valid_responses)
        escalated = any(isinstance(r, AgentResponse) and r.escalate for r in responses)
        tools_used = list(dict.fromkeys(
            tool_name
            for response in valid_responses
            for tool_name in response.tools_used
        ))
        tool_traces = [
            trace
            for response in valid_responses
            for trace in response.tool_traces
        ]
        input_tokens = sum(response.input_tokens for response in valid_responses)
        output_tokens = sum(response.output_tokens for response in valid_responses)
        result = OrchestratorResult(
            request_id=req.request_id,
            response=combined,
            agent_type=decision.primary_agent,
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            agent_types=[
                r.agent_type for r in responses
                if isinstance(r, AgentResponse) and r.success
            ] or agent_types,
            primary_agent=decision.primary_agent,
            supporting_agents=[agent for agent in decision.supporting_agents if agent in agent_types],
            tools_used=tools_used,
            tool_traces=tool_traces,
            routing_reason=decision.reason,
            routing_confidence=decision.confidence,
            agent_failures=agent_failures,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        if req.harness is not None:
            req.harness.record("run_finished", success=not bool(result.agent_failures))
            result.harness_trace = req.harness.summary()
        self._record_tool_trace(result)
        return result

                                                                            

    def _route(self, intent: Optional[IntentCategory], urgency: Optional[UrgencyLevel]) -> AgentType:
\
\
\
\
\
           
        if urgency == UrgencyLevel.CRITICAL:
            return AgentType.ESCALATION

        if intent and intent in self._INTENT_ROUTING:
            target = self._INTENT_ROUTING[intent]
                                 
            if target in self._pool and self._pool[target]:
                return target

        return AgentType.GENERAL

    def _route_decision(self, req: Request) -> RoutingDecision:
\
\
\
\
\
           
        if req.urgency == UrgencyLevel.CRITICAL:
            return RoutingDecision(
                primary_agent=AgentType.ESCALATION,
                reason="紧急度为 CRITICAL，触发升级路由",
                confidence=1.0,
            )

        if req.intent in (IntentCategory.ESCALATION, IntentCategory.HUMAN_HANDOFF):
            return RoutingDecision(
                primary_agent=AgentType.ESCALATION,
                reason=f"意图为 {req.intent.value if req.intent else 'unknown'}，触发升级路由",
                confidence=max(req.intent_confidence, 0.8),
            )

        if req.decision_needs_human:
            return RoutingDecision(
                primary_agent=AgentType.ESCALATION,
                reason="Version 2 local policy marked the request for human review",
                confidence=max(req.decision_confidence, req.intent_confidence),
            )

        if req.decision_source.endswith("_active") and req.decision_primary_agent:
            allowed = {agent.value for agent in self._pool}
            policy_decision = DecisionResult(
                intent=req.decision_intent or req.intent or IntentCategory.OTHER,
                confidence=req.decision_confidence,
                urgency=req.urgency or UrgencyLevel.LOW,
                primary_agent=req.decision_primary_agent,
                supporting_agents=list(req.decision_supporting_agents),
            )
            if validate_decision(policy_decision, allowed, self._decision_min_confidence):
                primary = self._agent_type_from_name(policy_decision.primary_agent)
                supporting = [
                    self._agent_type_from_name(name)
                    for name in policy_decision.supporting_agents
                    if self._agent_type_from_name(name) != primary
                ]
                return RoutingDecision(
                    primary_agent=primary,
                    supporting_agents=supporting,
                    reason=f"{req.decision_source} policy accepted",
                    confidence=policy_decision.confidence,
                )

        scores = self._domain_scores(req)
        available_scores = {
            agent_type: score
            for agent_type, score in scores.items()
            if agent_type == AgentType.GENERAL or self._pool.get(agent_type)
        }
        if not available_scores:
            return RoutingDecision(
                primary_agent=AgentType.GENERAL,
                reason="无可用专属 Agent，降级到 GeneralAgent",
                confidence=0.1,
            )

        ordered = sorted(available_scores.items(), key=lambda item: item[1], reverse=True)
        primary_agent, primary_score = ordered[0]

        collaboration_targets = self._collaboration_targets(req)
        supporting_agents = [
            agent_type
            for agent_type in collaboration_targets
            if agent_type != primary_agent and agent_type in available_scores
        ]

        if not supporting_agents:
            supporting_agents = [
                agent_type
                for agent_type, score in ordered[1:]
                if agent_type != AgentType.GENERAL
                and score >= 0.45
                and score >= primary_score * 0.55
            ]

        reason = self._routing_reason(req, available_scores, primary_agent, supporting_agents)
        return RoutingDecision(
            primary_agent=primary_agent,
            supporting_agents=supporting_agents,
            reason=reason,
            confidence=round(min(primary_score, 1.0), 3),
        )

    @staticmethod
    def _agent_type_from_name(name: Optional[str]) -> AgentType:
        normalized = str(name or "general").strip().lower().removesuffix("_0")
        return AgentType(normalized) if normalized in {agent.value for agent in AgentType} else AgentType.GENERAL

    def _domain_scores(self, req: Request) -> Dict[AgentType, float]:
                                      
        msg = req.message.lower()
        scores = {
            AgentType.GENERAL: 0.1,
            AgentType.TECHNICAL: 0.0,
            AgentType.BILLING: 0.0,
        }

        if req.intent in (
            IntentCategory.QUERY,
            IntentCategory.ORDER_STATUS,
            IntentCategory.LOGISTICS,
            IntentCategory.REQUEST,
            IntentCategory.COMPLAINT,
            IntentCategory.GREETING,
            IntentCategory.FEEDBACK,
            IntentCategory.OTHER,
        ):
            scores[AgentType.GENERAL] += 0.55

        if req.intent in (
            IntentCategory.TECHNICAL,
            IntentCategory.TECHNICAL_LOGIN,
            IntentCategory.TECHNICAL_CRASH,
        ):
            scores[AgentType.TECHNICAL] += 0.75

        if req.intent in (
            IntentCategory.BILLING,
            IntentCategory.ACCOUNT,
            IntentCategory.ACCOUNT_SECURITY,
            IntentCategory.REFUND,
            IntentCategory.INVOICE,
            IntentCategory.PAYMENT_ISSUE,
        ):
            scores[AgentType.BILLING] += 0.75

        technical_kws = ["崩溃", "报错", "error", "crash", "无法登录", "登录失败", "500", "401", "验证码"]
        billing_kws = ["退款", "退货", "扣款", "发票", "账单", "支付", "订阅", "refund", "invoice", "多扣"]
        general_kws = ["订单", "物流", "快递", "配送", "会员", "积分", "咨询", "帮助"]

        technical_hits = sum(1 for kw in technical_kws if kw in msg)
        billing_hits = sum(1 for kw in billing_kws if kw in msg)
        general_hits = sum(1 for kw in general_kws if kw in msg)

        scores[AgentType.TECHNICAL] += min(0.45, technical_hits * 0.18)
        scores[AgentType.BILLING] += min(0.45, billing_hits * 0.18)
        scores[AgentType.GENERAL] += min(0.35, general_hits * 0.12)

        entities = req.entities or {}
        if entities.get("error_code"):
            scores[AgentType.TECHNICAL] += 0.2
        if entities.get("amount"):
            scores[AgentType.BILLING] += 0.15
        if entities.get("order_id"):
            scores[AgentType.GENERAL] += 0.1

        return {agent_type: round(score, 3) for agent_type, score in scores.items()}

    @staticmethod
    def _routing_reason(
        req: Request,
        scores: Dict[AgentType, float],
        primary_agent: AgentType,
        supporting_agents: List[AgentType],
    ) -> str:
        score_text = ", ".join(
            f"{agent_type.value}={score:.2f}"
            for agent_type, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        )
        support_text = ", ".join(agent.value for agent in supporting_agents) or "none"
        intent = req.intent.value if req.intent else "unknown"
        return (
            f"intent={intent}, group={req.intent_group or 'unknown'}, "
            f"primary={primary_agent.value}, supporting={support_text}, scores=[{score_text}]"
        )

    def _collaboration_targets(self, req: Request) -> List[AgentType]:
\
\
\
\
\
           
        msg = req.message.lower()
        targets: List[AgentType] = []

        technical_kws = ["崩溃", "报错", "error", "crash", "无法登录", "登录失败", "500", "401"]
        billing_kws = ["退款", "扣款", "发票", "账单", "支付", "订阅", "refund", "invoice"]

        if req.intent in (
            IntentCategory.TECHNICAL,
            IntentCategory.TECHNICAL_LOGIN,
            IntentCategory.TECHNICAL_CRASH,
        ) or any(kw in msg for kw in technical_kws):
            targets.append(AgentType.TECHNICAL)
        if req.intent in (
            IntentCategory.BILLING,
            IntentCategory.ACCOUNT,
            IntentCategory.ACCOUNT_SECURITY,
            IntentCategory.REFUND,
            IntentCategory.INVOICE,
            IntentCategory.PAYMENT_ISSUE,
        ) or any(kw in msg for kw in billing_kws):
            targets.append(AgentType.BILLING)

                                     
        deduped = list(dict.fromkeys(targets))
        return [agent_type for agent_type in deduped if self._pool.get(agent_type)]

    @staticmethod
    def _needs_clarification(req: Request) -> bool:
                                    
        if req.intent != IntentCategory.OTHER:
            return False
        text = (req.message or "").strip()
        if len(text) <= 2:
            return False
        return req.intent_confidence < 0.5

    def _best_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
\
\
\
           
        agents = self._pool.get(agent_type, [])
        if not agents:
            return None
        return max(agents, key=lambda a: a.stats.routing_score())

    async def _execute(self, req: Request, agent_type: AgentType) -> AgentResponse:
                                           
        if req.harness is not None and not req.harness.allow_agent():
            return AgentResponse(agent_type=agent_type, content="本次请求的 Agent 调用预算已用完。",
                                 success=False, error_code="AGENT_BUDGET_EXHAUSTED")
        agent = self._best_agent(agent_type)
        if agent is None:
            agent = self._best_agent(AgentType.GENERAL)
        if agent is None:
            return AgentResponse(
                agent_type=AgentType.GENERAL,
                content="服务暂时不可用，请稍后重试。",
                success=False,
            )

        response = await agent.handle(req)
        if req.harness is not None:
            req.harness.record("agent_finished", agent=agent_type.value, success=response.success,
                               latency_ms=round(response.latency_ms, 1),
                               tools=list(response.tools_used))

                                      
        if not response.success and agent_type not in (AgentType.GENERAL, AgentType.ESCALATION):
            logger.warning(f"{agent_type.value} 失败，降级到 GeneralAgent")
            fallback = self._best_agent(AgentType.GENERAL)
            if fallback and (req.harness is None or req.harness.allow_agent()):
                response = await fallback.handle(req)

        return response

                                                                         

    def get_stats(self) -> Dict[str, Any]:
        result = {}
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                result[key] = {
                    "total":        agent.stats.total,
                    "success_rate": round(agent.stats.success_rate, 3),
                    "avg_ms":       round(agent.stats.avg_ms, 1),
                    "monitor_penalty": round(agent.stats.monitor_penalty, 3),
                    "routing_score": round(agent.stats.routing_score(), 3),
                    "role": agent.profile.role,
                    "workflow": list(agent.profile.workflow),
                    "tool_scope": list(agent.profile.tool_scope),
                    "available_tools": list(agent.get_tools()),
                    "model": agent._model,
                }
        return result

    def update_routing_penalties(self, penalties: Dict[str, float]) -> None:
\
\
\
\
           
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                penalty = penalties.get(key, 0.0)
                agent.stats.monitor_penalty = min(max(penalty, 0.0), 0.9)
