import asyncio
import logging
import os
import pathlib
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional


_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, UploadFile, File
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.errors import (
    ApiError,
    api_error_handler,
    http_exception_handler,
    request_context_middleware,
    unhandled_exception_handler,
    validation_exception_handler,
)
from api.runtime_controls import IdempotencyStore, RateLimiter
from api.security import request_principal, require_admin
from api.tenant import get_tenant_id
from core.decision_provider import LocalPolicyDecisionProvider
from core.version3_decision import Version3LocalDecisionProvider
from core.rollout import RolloutManager
from monitor.telemetry import TelemetryRegistry
from evaluation.feedback_store import FeedbackStore
from runtime.harness import replay_events

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BANNER = r"""
   
   ╔══════════════════════╗
    ║   多 Agent     ║
   ║   智能客服 AI 系统    ║
   ╚══════════════════════╝
   
"""

                                                                     
_orchestrator = None
_memory       = None
_tool_manager = None
_monitor      = None
_evaluator    = None
_skill_manager = None
_knowledge_base = None
_rate_limiter = RateLimiter()
_idempotency_store = IdempotencyStore()
_rollout_manager = RolloutManager(stable_model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"))
_telemetry = TelemetryRegistry()
_feedback_store = None

def _anthropic_cfg() -> Dict[str, Any]:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("未设置 ANTHROPIC_API_KEY")
    cfg: Dict[str, Any] = {
        "api_key":  key,
        "model":    os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip(),
    }
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    if base_url:
        cfg["base_url"] = base_url
    return cfg


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _memory, _tool_manager, _monitor, _evaluator, _skill_manager, _knowledge_base
    global _rate_limiter, _idempotency_store
    global _rollout_manager, _feedback_store

    print(BANNER, flush=True)

    from agents.agent_orchestrator import AgentOrchestrator, Request, build_shared_rag_tools
    from core.intent_recognizer import IntentRecognizer
    from evaluation.evaluator import EndToEndEvaluator
    from knowledge.base import KnowledgeBase
    from tools.manager import MCPToolManager, Tool
    from memory.conversation_memory import MemoryManager
    from monitor.performance_monitor import PerformanceMonitor
    from core.skill_loader import SkillManager

    cfg = _anthropic_cfg()
    logger.info(f"模型: {cfg['model']}  base_url: {cfg.get('base_url', '(官方)')}")

    _rollout_manager = RolloutManager(
        stable_model=cfg["model"],
        canary_model=os.getenv("ECHOMIND_CANARY_MODEL"),
    )

    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    _rate_limiter = RateLimiter(
        redis_url=redis_url,
        window_seconds=int(os.getenv("ECHOMIND_RATE_WINDOW_SECONDS", "60")),
    )
    _idempotency_store = IdempotencyStore(
        redis_url=redis_url,
        ttl_seconds=int(os.getenv("ECHOMIND_IDEMPOTENCY_TTL_SECONDS", "900")),
    )

                                                  
    recognizer = IntentRecognizer(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

                                                  
    skills_dir = os.getenv("ECHOMIND_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills"))
    _skill_manager = SkillManager(
        root_dir=skills_dir,
        max_prompt_chars=int(os.getenv("ECHOMIND_SKILLS_MAX_PROMPT_CHARS", "5000")),
    )
    _skill_manager.load()

    v2_mode = os.getenv("ECHOMIND_V2_MODE", "off").strip().lower()
    v3_mode = os.getenv("ECHOMIND_V3_MODE", "off").strip().lower()
    if v2_mode not in {"off", "shadow", "active"}:
        v2_mode = "off"
    if v3_mode not in {"off", "shadow", "active"}:
        v3_mode = "off"
    decision_mode = v3_mode if v3_mode != "off" else v2_mode
    decision_provider = (
        Version3LocalDecisionProvider()
        if v3_mode != "off"
        else LocalPolicyDecisionProvider() if v2_mode != "off" else None
    )

               
    _orchestrator = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=_skill_manager,
        decision_provider=decision_provider,
        decision_mode=decision_mode,
    )

                                            
    _memory = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

                                            
    _tool_manager = MCPToolManager(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )
    kb = KnowledgeBase(
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
    )
    _knowledge_base = kb
    logger.info(f"知识库已加载: {await kb.doc_count_async()} 个文档片段")

    def knowledge_fallback(params: Dict[str, Any], context: Optional[Dict[str, Any]], error: str):
        query = params.get("query", "")
        return [{
            "title": "知识库降级结果",
            "content": f"知识库暂时不可用，未能完成对“{query}”的语义检索。请稍后重试，或转人工客服确认。",
            "score": 0.0,
            "fallback": True,
            "error": error,
        }]

    _tool_manager.register(Tool(
        name="knowledge_search",
        description="搜索知识库（基于 ChromaDB 向量检索）",
        handler=kb.search_handler,
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 2000},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        cache_ttl=300.0,
        supports_rerank=True,
        fallback=knowledge_fallback,
    ))
    if _orchestrator is not None:
        _orchestrator.set_shared_tools(build_shared_rag_tools(_tool_manager))

                           
    prom_port = int(os.getenv("PROMETHEUS_PORT", "0")) or None
    _monitor = PerformanceMonitor(
        orchestrator=_orchestrator,
        tool_manager=_tool_manager,
        interval_s=float(os.getenv("MONITOR_INTERVAL", "10")),
        webhook_url=os.getenv("ALERT_WEBHOOK_URL") or None,
        prometheus_port=prom_port,
    )
    await _monitor.start()

         
    _evaluator = EndToEndEvaluator(
        orchestrator=_orchestrator,
        recognizer=recognizer,
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        baseline_path=os.getenv("EVAL_BASELINE_PATH", "/app/data/eval/baseline.json"),
    )
    _feedback_store = FeedbackStore(
        os.getenv("ECHOMIND_FEEDBACK_PENDING_PATH", str(pathlib.Path(_ROOT) / "data/eval/feedback-pending.jsonl")),
        os.getenv("ECHOMIND_FEEDBACK_APPROVED_PATH", str(pathlib.Path(_ROOT) / "data/eval/feedback-approved.jsonl")),
    )

    logger.info("EchoMind 已就绪")
    yield

    await _monitor.stop()
    if _memory is not None:
        await _memory.close()
    await _rate_limiter.close()
    await _idempotency_store.close()
    logger.info("EchoMind 已关闭")


                                                                                
app = FastAPI(
    title="EchoMind 智能客服",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

app.middleware("http")(request_context_middleware)
app.add_exception_handler(ApiError, api_error_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)


def _cors_origins() -> List[str]:
    raw = os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:5176,http://localhost:5173,http://127.0.0.1:5176,http://127.0.0.1:5173",
    )
    return [item.strip() for item in raw.split(",") if item.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=[
        "Content-Type", "Authorization", "X-API-Key", "X-Request-ID",
        "Idempotency-Key", "X-Tenant-ID",
    ],
)


                                                                          
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    user_id: str = Field(default="anonymous", min_length=1, max_length=128)
    conv_id: Optional[str] = Field(default=None, min_length=1, max_length=128)


class ChatResponse(BaseModel):
    conv_id:     str
    request_id:  str = ""
    response:    str
    intent:      str
    intent_group: str = "other"
    agent_type:  str
    agent_types: List[str] = Field(default_factory=list)
    primary_agent: str = ""
    supporting_agents: List[str] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    routing_reason: str = ""
    routing_confidence: float = 0.0
    escalated:   bool
    latency_ms:  float
    knowledge_used: bool = False
    entities: Dict[str, List[str]] = Field(default_factory=dict)
    intent_confidence: float = 0.0
    intent_source_scores: Dict[str, float] = Field(default_factory=dict)
    decision_source: str = "local"
    decision_confidence: float = 0.0
    decision_fallback_reason: str = ""
    decision_needs_clarification: bool = False
    decision_schema_version: str = "2.0"
    agent_failures: List[Dict[str, Any]] = Field(default_factory=list)
    rollout_variant: str = "stable"
    harness_trace: Dict[str, Any] = Field(default_factory=dict)


class ToolTraceResponse(BaseModel):
    request_id: str
    found: bool
    trace: Dict[str, Any] = Field(default_factory=dict)


class RecentToolTracesResponse(BaseModel):
    items: List[Dict[str, Any]] = Field(default_factory=list)


class FeedbackInput(BaseModel):
    request_id: str = Field(min_length=1, max_length=128)
    rating: int = Field(ge=1, le=5)
    issue: str = Field(min_length=1, max_length=80)
    expected_intent: Optional[str] = Field(default=None, max_length=64)
    expected_agent: Optional[str] = Field(default=None, max_length=32)
    expected_needs_human: Optional[bool] = None


class FeedbackApproval(BaseModel):
    feedback_id: str = Field(min_length=1, max_length=64)


class RolloutPromotionRequest(BaseModel):
    candidate_model: str = Field(min_length=1, max_length=128)
    canary_percent: int = Field(ge=1, le=100)
    metrics: Dict[str, float]
    baseline: Dict[str, float]
    candidate_dataset_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    baseline_dataset_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")


async def _rate_limit(request: Request, scope: str) -> None:
    identity = request_principal(request)
    await _rate_limiter.check(scope, identity)


async def _chat_rate_limit(request: Request) -> None:
    await _rate_limit(request, "chat")


async def _search_rate_limit(request: Request) -> None:
    await _rate_limit(request, "search")


async def _write_rate_limit(request: Request) -> None:
    await _rate_limit(request, "knowledge_write")


async def _admin_rate_limit(request: Request) -> None:
    await _rate_limit(request, "admin")


                                                                              
@app.get("/health")
async def health():
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return {"status": "ok", "agents": _orchestrator.get_stats()}


@app.get("/ready")
async def ready():
                                       
    checks: Dict[str, Any] = {
        "orchestrator": {"ready": _orchestrator is not None},
        "memory": await _memory.readiness() if _memory is not None else {"ready": False},
        "knowledge_base": (
            await _knowledge_base.readiness()
            if _knowledge_base is not None
            else {"ready": False}
        ),
        "tool_manager": {"ready": _tool_manager is not None},
        "monitor": {"ready": _monitor is not None},
        "evaluator": {"ready": _evaluator is not None},
        "decision_layer": {
            "ready": True,
            "mode": _orchestrator.get_decision_stats().get("mode", "off") if _orchestrator else "off",
            "version": _orchestrator.get_decision_stats().get("version", "1") if _orchestrator else "1",
        },
    }
    ready_state = all(bool(check.get("ready")) for check in checks.values())
    payload = {"status": "ready" if ready_state else "not_ready", "checks": checks}
    if not ready_state:
        raise HTTPException(status_code=503, detail=payload)
    return payload


@app.get("/skills", tags=["Skills"], dependencies=[Depends(_admin_rate_limit)])
async def skills_summary(_: str = Depends(require_admin)):
                                           
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    return _skill_manager.summary()


@app.post("/skills/reload", tags=["Skills"], dependencies=[Depends(_admin_rate_limit)])
async def reload_skills(_: str = Depends(require_admin)):
                                   
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    _skill_manager.reload()
    if _orchestrator is not None:
        _orchestrator.set_skill_manager(_skill_manager)
    return _skill_manager.summary()


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(_chat_rate_limit)])
async def chat(
    req: ChatRequest,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    tenant_id: str = Depends(get_tenant_id),
):
                       
    request_t0 = time.monotonic()
    rollout = _rollout_manager.select(tenant_id)
    if not idempotency_key:
        try:
            return await _chat_impl(req, tenant_id)
        except Exception:
            _telemetry.record(
                route="/chat", tenant_id=tenant_id, variant=rollout.variant,
                status="error", latency_ms=(time.monotonic() - request_t0) * 1000,
                input_text=req.message,
            )
            raise
    principal = f"{tenant_id}:{req.user_id}"
    replay = await _idempotency_store.begin("chat", principal, idempotency_key)
    if replay is not None:
        return ChatResponse.model_validate(replay)
    try:
        payload = await _chat_impl(req, tenant_id)
        await _idempotency_store.complete("chat", principal, idempotency_key, payload.model_dump())
        return payload
    except Exception:
        _telemetry.record(
            route="/chat", tenant_id=tenant_id, variant=rollout.variant,
            status="error", latency_ms=(time.monotonic() - request_t0) * 1000,
            input_text=req.message,
        )
        await _idempotency_store.fail("chat", principal, idempotency_key)
        raise


async def _chat_impl(req: ChatRequest, tenant_id: str):
\
\
\
       
    if _orchestrator is None or _memory is None:
        raise HTTPException(503, "服务未就绪")

    from agents.agent_orchestrator import Request as OrcReq
    from memory.conversation_memory import MsgRole

    rollout = _rollout_manager.select(tenant_id)
    conv_id = req.conv_id or str(uuid.uuid4())
    scoped_user_id = f"{tenant_id}:{req.user_id}"
    scoped_conv_id = f"{tenant_id}:{conv_id}"

                
    mem_ctx = await _memory.get_context(scoped_user_id, scoped_conv_id, query=req.message)

                                
    history = [
        {"role": m.role.value, "content": m.content}
        for m in mem_ctx.recent_messages[-5:]
    ] if mem_ctx.recent_messages else None

    intent_result = await _orchestrator.recognize_intent(req.message, history=history)
    full_context = mem_ctx.to_prompt_text()

    orch_req = OrcReq(
        message=req.message,
        user_id=scoped_user_id,
        conv_id=scoped_conv_id,
        context=full_context,
        history=history,
        entities=intent_result.entities,
        intent=intent_result.intent,
        intent_group=intent_result.intent_group,
        urgency=intent_result.urgency,
        intent_confidence=intent_result.confidence,
        decision_source=intent_result.decision_source,
        decision_intent=intent_result.decision_intent,
        decision_confidence=intent_result.decision_confidence,
        decision_latency_ms=intent_result.decision_latency_ms,
        decision_primary_agent=intent_result.decision_primary_agent,
        decision_supporting_agents=list(intent_result.decision_supporting_agents),
        decision_needs_human=intent_result.decision_needs_human,
        decision_needs_clarification=intent_result.decision_needs_clarification,
        decision_fallback_reason=intent_result.decision_fallback_reason,
        decision_schema_version=getattr(intent_result, "decision_schema_version", "2.0"),
        tenant_id=tenant_id,
        rollout_variant=rollout.variant,
        model_override=rollout.model if rollout.variant == "canary" else None,
    )

           
    result = await _orchestrator.run(orch_req)

             
    await _memory.add_message(scoped_user_id, scoped_conv_id, MsgRole.USER, req.message)
    await _memory.add_message(scoped_user_id, scoped_conv_id, MsgRole.ASSISTANT, result.response)

                        
    asyncio.create_task(_memory.update_profile(scoped_user_id, scoped_conv_id))

    _telemetry.record(
        route="/chat",
        tenant_id=tenant_id,
        variant=rollout.variant,
        status="success",
        latency_ms=result.latency_ms,
        input_text=req.message,
        output_text=result.response,
        input_tokens=result.input_tokens or None,
        output_tokens=result.output_tokens or None,
        usage_exact=bool(result.input_tokens or result.output_tokens),
    )

    return ChatResponse(
        conv_id=conv_id,
        request_id=result.request_id,
        response=result.response,
        intent=result.intent.value if result.intent else "other",
        intent_group=intent_result.intent_group,
        agent_type=result.agent_type.value,
        agent_types=[agent_type.value for agent_type in result.agent_types],
        primary_agent=result.primary_agent.value if result.primary_agent else result.agent_type.value,
        supporting_agents=[agent_type.value for agent_type in result.supporting_agents],
        tools_used=result.tools_used,
        routing_reason=result.routing_reason,
        routing_confidence=result.routing_confidence,
        escalated=result.escalated,
        latency_ms=round(result.latency_ms, 1),
        knowledge_used="search_knowledge_base" in result.tools_used,
        entities=intent_result.entities,
        intent_confidence=round(intent_result.confidence, 4),
        intent_source_scores=intent_result.source_scores,
        decision_source=intent_result.decision_source,
        decision_confidence=round(intent_result.decision_confidence, 4),
        decision_fallback_reason=intent_result.decision_fallback_reason,
        agent_failures=result.agent_failures,
        rollout_variant=rollout.variant,
        harness_trace=result.harness_trace,
    )


async def _build_knowledge_context(
    message: str, intent=None, top_k: int = 3, tenant_id: str = "default"
) -> tuple[str, bool]:
\
\
\
\
       
    if _tool_manager is None:
        return "", False
    if not _should_use_knowledge(message, intent=intent):
        return "", False
    try:
        result = await _tool_manager.search_with_rewrite(
            "knowledge_search", message, top_k=top_k,
            context={"tenant_id": tenant_id},
        )
        if not result.success or not isinstance(result.data, list) or not result.data:
            return "", False

        parts = ["[知识库检索结果]"]
        used = False
        for i, item in enumerate(result.data[:top_k], start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "未命名文档"))
            content = str(item.get("content", "")).strip()
            score = item.get("score", "")
            if not content:
                continue
            used = True
            parts.append(f"{i}. 标题: {title}\n   相关度: {score}\n   内容: {content[:600]}")

        if not used:
            return "", False
        parts.append("请优先依据以上知识库内容回答；如果知识库内容不足，再结合通用客服能力说明。")
        return "\n".join(parts), True
    except Exception as ex:
        logger.warning(f"构建知识库上下文失败: {ex}")
        return "", False


def _should_use_knowledge(message: str, intent=None) -> bool:
                                          
    msg = (message or "").strip().lower()
    if not msg:
        return False
    intent_value = getattr(intent, "value", intent)
    if intent_value in {"greeting", "feedback", "escalation", "human_handoff", "other"}:
        return False
    if intent_value in {
        "query", "request", "technical", "billing", "account", "complaint",
        "order_status", "logistics", "refund", "invoice", "payment_issue",
        "account_security", "technical_login", "technical_crash",
    }:
        return True
    greetings = {"你好", "您好", "嗨", "hi", "hello", "hey", "早上好", "晚上好"}
    if msg in greetings:
        return False
    business_keywords = [
        "退款", "订单", "物流", "配送", "发票", "扣款", "支付", "账单", "订阅",
        "登录", "报错", "错误", "崩溃", "会员", "积分", "账户", "密码", "地址",
        "refund", "order", "invoice", "payment", "error", "login",
    ]
    return len(msg) >= 4 or any(kw in msg for kw in business_keywords)


@app.get("/monitor", dependencies=[Depends(_admin_rate_limit)])
async def monitor_summary(_: str = Depends(require_admin)):
                                        
    if _monitor is None:
        raise HTTPException(503, "服务未就绪")
    payload = _monitor.summary()
    payload["telemetry"] = _telemetry.summary()
    return payload


@app.get("/telemetry", dependencies=[Depends(_admin_rate_limit)])
async def telemetry_summary(_: str = Depends(require_admin)):
                                 
    payload = _telemetry.summary()
    payload["decision"] = _orchestrator.get_decision_stats() if _orchestrator else {
        "enabled": False,
        "mode": "off",
    }
    return payload


@app.get("/rollout", dependencies=[Depends(_admin_rate_limit)])
async def rollout_summary(_: str = Depends(require_admin)):
                             
    return _rollout_manager.summary()


@app.post("/rollout/promote", dependencies=[Depends(_admin_rate_limit)])
async def promote_rollout(body: RolloutPromotionRequest, _: str = Depends(require_admin)):
    try:
        return _rollout_manager.promote_candidate(
            body.candidate_model, body.canary_percent, body.metrics, body.baseline,
            body.candidate_dataset_fingerprint, body.baseline_dataset_fingerprint,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/rollout/assess", dependencies=[Depends(_admin_rate_limit)])
async def assess_rollout(_: str = Depends(require_admin)):
    if _feedback_store is None:
        raise HTTPException(503, "反馈存储未就绪")
    summary = _feedback_store.quality_summary()
    return _rollout_manager.assess_online_quality(summary)


@app.get("/trace/tool/{request_id}", response_model=ToolTraceResponse, dependencies=[Depends(_admin_rate_limit)])
async def get_tool_trace(request_id: str, _: str = Depends(require_admin)):
                        
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    trace = _orchestrator.get_tool_trace(request_id)
    return ToolTraceResponse(
        request_id=request_id,
        found=trace is not None,
        trace=trace or {},
    )


@app.post("/trace/debug-replay/{request_id}", dependencies=[Depends(_admin_rate_limit)])
async def debug_replay_trace(request_id: str, _: str = Depends(require_admin)):
    """Replay only the recorded event timeline; never calls a model or tool."""
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    trace = _orchestrator.get_tool_trace(request_id)
    if trace is None:
        raise HTTPException(404, "Trace 不存在")
    return replay_events(trace)


@app.post("/trace/rerun/{request_id}", dependencies=[Depends(_admin_rate_limit)])
async def rerun_trace(request_id: str, _: str = Depends(require_admin)):
    """Re-execute model/tool work; disabled by default and separately named."""
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    try:
        result = await _orchestrator.replay_request(request_id)
    except PermissionError as exc:
        raise HTTPException(409, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, "可重跑的请求快照不存在") from exc
    return {
        "source_request_id": request_id,
        "replay_request_id": result.request_id,
        "response": result.response,
        "intent": result.intent.value if result.intent else None,
        "primary_agent": result.primary_agent.value if result.primary_agent else None,
        "agent_types": [agent.value for agent in result.agent_types],
        "tools_used": result.tools_used,
        "latency_ms": round(result.latency_ms, 1),
        "harness_trace": result.harness_trace,
    }


@app.post("/feedback", dependencies=[Depends(_write_rate_limit)])
async def submit_feedback(body: FeedbackInput):
    if _feedback_store is None or _orchestrator is None:
        raise HTTPException(503, "反馈存储未就绪")
    trace = _orchestrator.get_tool_trace(body.request_id)
    message = _orchestrator.get_feedback_input(body.request_id)
    if trace is None or message is None:
        raise HTTPException(404, "请求 Trace 已过期或不存在")
    try:
        return _feedback_store.submit(
            body.request_id, trace, message, rating=body.rating, issue=body.issue,
            expected_intent=body.expected_intent, expected_agent=body.expected_agent,
            expected_needs_human=body.expected_needs_human,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/feedback/pending", dependencies=[Depends(_admin_rate_limit)])
async def pending_feedback(_: str = Depends(require_admin)):
    if _feedback_store is None:
        raise HTTPException(503, "反馈存储未就绪")
    return {"items": _feedback_store.pending()}


@app.get("/feedback/metrics", dependencies=[Depends(_admin_rate_limit)])
async def feedback_metrics(_: str = Depends(require_admin)):
    if _feedback_store is None:
        raise HTTPException(503, "反馈存储未就绪")
    return _feedback_store.quality_summary()


@app.post("/feedback/approve", dependencies=[Depends(_admin_rate_limit)])
async def approve_feedback(body: FeedbackApproval, _: str = Depends(require_admin)):
    if _feedback_store is None:
        raise HTTPException(503, "反馈存储未就绪")
    try:
        return _feedback_store.approve(body.feedback_id)
    except KeyError as exc:
        raise HTTPException(404, "待审核反馈不存在") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/trace/tools", response_model=RecentToolTracesResponse, dependencies=[Depends(_admin_rate_limit)])
async def list_recent_tool_traces(limit: int = 20, _: str = Depends(require_admin)):
                            
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return RecentToolTracesResponse(items=_orchestrator.get_recent_tool_traces(limit=limit))


@app.get("/metrics")
async def prometheus_metrics():
                          
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/search", dependencies=[Depends(_search_rate_limit)])
async def search(
    query: str = Query(..., min_length=1, max_length=2000),
    top_k: int = Query(5, ge=1, le=50),
    tenant_id: str = Depends(get_tenant_id),
):
\
\
\
       
    if _tool_manager is None:
        raise HTTPException(503, "服务未就绪")
    result = await _tool_manager.search_with_rewrite(
        "knowledge_search", query, top_k=top_k,
        context={"tenant_id": tenant_id},
    )
    return {"query": query, "results": result.data, "reranked": result.reranked}


class DocInput(BaseModel):
                 
    title:   str
    content: str


class BatchDocInput(BaseModel):
                    
    documents: List[DocInput]


class EvalIntentInput(BaseModel):
                   
    message: str
    expected_intent: str
    context: Optional[Dict[str, Any]] = None


class EvalDialogInput(BaseModel):
                                        
    question: Optional[str] = None
    turns: Optional[List[str]] = None
    user_id: Optional[str] = None
    conv_id: Optional[str] = None


class EvalRunInput(BaseModel):
                           
    intent_cases: Optional[List[EvalIntentInput]] = None
    dialog_cases: Optional[List[EvalDialogInput]] = None


@app.post("/knowledge/add", tags=["知识库"], dependencies=[Depends(_admin_rate_limit), Depends(_write_rate_limit)])
async def add_knowledge(
    body: BatchDocInput,
    tenant_id: str = Depends(get_tenant_id),
    _: str = Depends(require_admin),
):
       
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    count = await kb.add_documents_async(
        [{"title": d.title, "content": d.content} for d in body.documents],
        tenant_id=tenant_id,
    )
    total = await kb.doc_count_async(tenant_id)
    return {
        "message": f"成功导入 {count} 个文档片段",
        "tenant_id": tenant_id,
        "added_chunks": count,
        "total_chunks": total,
    }


@app.post("/knowledge/upload", tags=["知识库"], dependencies=[Depends(_admin_rate_limit), Depends(_write_rate_limit)])
async def upload_knowledge(
    file: UploadFile = File(...),
    tenant_id: str = Depends(get_tenant_id),
    _: str = Depends(require_admin),
):
    
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "文件大小超过 10MB 限制")

    text = content.decode("utf-8", errors="ignore")
    filename = file.filename or "unknown"

    if filename.endswith(".json"):
        import json as _json
        try:
            docs = _json.loads(text)
            if not isinstance(docs, list):
                raise HTTPException(400, "JSON 文件应为数组格式: [{title, content}, ...]")
        except _json.JSONDecodeError as e:
            raise HTTPException(400, f"JSON 解析失败: {e}")
    else:
                             
        title = filename.rsplit(".", 1)[0] if "." in filename else filename
        docs = [{"title": title, "content": text}]

    count = await kb.add_documents_async(docs, tenant_id=tenant_id)
    total = await kb.doc_count_async(tenant_id)
    return {
        "message": f"文件 {filename} 导入成功",
        "tenant_id": tenant_id,
        "added_chunks": count,
        "total_chunks": total,
    }


@app.get("/knowledge/stats", tags=["知识库"], dependencies=[Depends(_admin_rate_limit)])
async def knowledge_stats(
    tenant_id: str = Depends(get_tenant_id),
    _: str = Depends(require_admin),
):
                            
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    return {"tenant_id": tenant_id, "total_chunks": await kb.doc_count_async(tenant_id)}


@app.post("/eval/run", dependencies=[Depends(_admin_rate_limit), Depends(_write_rate_limit)])
async def run_eval(body: Optional[EvalRunInput] = None, _: str = Depends(require_admin)):
                          
    if _evaluator is None:
        raise HTTPException(503, "服务未就绪")
    from evaluation.evaluator import DEFAULT_DIALOG_CASES, DEFAULT_INTENT_CASES, IntentTestCase

    if body and body.intent_cases is not None:
        intent_cases = [
            IntentTestCase(
                message=c.message,
                expected_intent=c.expected_intent,
                context=c.context,
            )
            for c in body.intent_cases
        ]
    else:
        intent_cases = DEFAULT_INTENT_CASES

    if body and body.dialog_cases is not None:
        dialog_cases = [
            c.model_dump(exclude_none=True)
            for c in body.dialog_cases
        ]
    else:
        dialog_cases = DEFAULT_DIALOG_CASES

    report = await _evaluator.run(
        intent_cases=intent_cases,
        dialog_cases=dialog_cases,
    )
    return {
        "pass_rate":       report.pass_rate,
        "total":           report.total,
        "passed":          report.passed,
        "avg_scores":      report.avg_scores,
        "regressions":     report.regressions,
        "recommendations": report.recommendations,
        "results": [
            {
                "test_id": r.test_id,
                "passed": r.passed,
                "scores": r.scores,
                "detail": r.detail,
                "metadata": r.metadata,
            }
            for r in report.results
        ],
    }


                                                                             
async def _cli():
    print(BANNER)
    print("EchoMind CLI — 输入 quit 退出\n")

    from agents.agent_orchestrator import AgentOrchestrator, Request
    from memory.conversation_memory import MemoryManager, MsgRole
    from core.skill_loader import SkillManager

    cfg = _anthropic_cfg()
    skill_manager = SkillManager(
        root_dir=os.getenv("ECHOMIND_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills")),
        max_prompt_chars=int(os.getenv("ECHOMIND_SKILLS_MAX_PROMPT_CHARS", "5000")),
    )
    skill_manager.load()
    orch = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=skill_manager,
    )
    mem  = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "localhost"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/tmp/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    user_id, conv_id = "cli_user", str(uuid.uuid4())

    while True:
        try:
            msg = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见 ʕ•ᴥ•ʔ")
            break
        if not msg or msg.lower() in ("quit", "exit", "退出"):
            print("再见 ʕ•ᴥ•ʔ")
            break

        ctx = await mem.get_context(user_id, conv_id, query=msg)
        history = [
            {"role": m.role.value, "content": m.content}
            for m in ctx.recent_messages[-5:]
        ] if ctx.recent_messages else None
        req = Request(message=msg, user_id=user_id, conv_id=conv_id, context=ctx.to_prompt_text(), history=history)
        result = await orch.run(req)

        await mem.add_message(user_id, conv_id, MsgRole.USER, msg)
        await mem.add_message(user_id, conv_id, MsgRole.ASSISTANT, result.response)

        print(f"\nEchoMind [{result.agent_type.value}]: {result.response}\n")

    await mem.close()


if __name__ == "__main__":
    if "--cli" in sys.argv:
        asyncio.run(_cli())
    else:
        uvicorn.run(
            "api.main:app",
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            reload=os.getenv("APP_ENV") == "development",
        )
