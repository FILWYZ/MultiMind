
import asyncio
import hashlib
import inspect
import json
import logging
import re
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic

from core.llm_utils import extract_text_content

logger = logging.getLogger(__name__)

__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "MCPToolManager",
    "Tool",
    "ToolResult",
    "ToolStats",
]


                                                                            

class CircuitState(Enum):
    CLOSED    = "closed"         
    OPEN      = "open"                
    HALF_OPEN = "half_open"        


@dataclass
class ToolResult:
    success:        bool
    data:           Any
    tool_name:      str
    error:          Optional[str] = None
    cached:         bool = False
    latency_ms:     float = 0.0
    reranked:       bool = False           


@dataclass
class ToolStats:
                               
    total:              int = 0
    success:            int = 0
    failed:             int = 0
    total_latency_ms:   float = 0.0
    consecutive_fails:  int = 0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total if self.total else 0.0


                                                                             

class CircuitBreaker:   

    def __init__(self, failure_threshold: int = 5, recovery_s: float = 60.0):
        self.threshold   = failure_threshold
        self.recovery_s  = recovery_s
        self.state       = CircuitState.CLOSED
        self.fail_count  = 0
        self.opened_at:  Optional[float] = None
        self._lock = threading.Lock()
        self._probe_in_flight = False

    def allow(self) -> bool:
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            if self.state == CircuitState.OPEN:
                if time.monotonic() - self.opened_at < self.recovery_s:                
                    return False
                self.state = CircuitState.HALF_OPEN
            if self.state == CircuitState.HALF_OPEN:
                if self._probe_in_flight:
                    return False
                self._probe_in_flight = True
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self.fail_count = 0
            self._probe_in_flight = False
            self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self.fail_count += 1
            self._probe_in_flight = False
            if self.fail_count >= self.threshold:
                self.state     = CircuitState.OPEN
                self.opened_at = time.monotonic()
                logger.warning(f"熔断器打开（连续失败 {self.fail_count} 次）")


                                                                            

@dataclass
class Tool:
    name:        str
    description: str
    handler:     Callable                                                    
    schema:      Dict[str, Any]                           
    cache_ttl:   float = 0.0                          
    timeout_s:   float = 30.0
    supports_rerank: bool = False                      
    fallback:    Optional[Callable] = None                                                

                  
    stats:   ToolStats    = field(default_factory=ToolStats, init=False)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker, init=False)


                                                                           

class MCPToolManager:     

    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self._model  = model
        self._tools: Dict[str, Tool] = {}
        self._cache: Dict[str, tuple] = {}                                        

                                                                            

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.info(f"注册工具: {tool.name}")

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

                                                                            

    async def call(
        self,
        name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        *,
        use_cache: bool = True,
        rerank_top_k: int = 0,                             
    ) -> ToolResult:

        tool = self._tools.get(name)
        if not tool:
            return ToolResult(success=False, data=None, tool_name=name, error=f"工具不存在: {name}")

        cache_rerank_top_k = rerank_top_k if rerank_top_k > 0 and tool.supports_rerank else 0

              
        if use_cache and tool.cache_ttl > 0:
            cached = self._get_cache(name, params, cache_rerank_top_k, context)
            if cached is not None:
                cached_data, cached_reranked = cached
                tool.stats.total += 1
                tool.stats.success += 1
                return ToolResult(
                    success=True,
                    data=cached_data,
                    tool_name=name,
                    cached=True,
                    reranked=cached_reranked,
                )

              
        if not tool.breaker.allow():
            error = f"工具熔断中: {name}，请稍后重试"
            return await self._fallback_result(tool, params, context, error)

        t0 = time.monotonic()
        tool.stats.total += 1
        try:
                                                               
            self._validate_params(tool, params)

            data = await asyncio.wait_for(self._run_handler(tool, params, context), timeout=tool.timeout_s)
            latency = (time.monotonic() - t0) * 1000

            tool.stats.success += 1
            tool.stats.consecutive_fails = 0
            tool.stats.total_latency_ms += latency
            tool.breaker.record_success()

                             
            reranked = False
            if rerank_top_k > 0 and tool.supports_rerank and isinstance(data, list):
                query = params.get("query", "")
                data, reranked = await self._rerank(query, data, rerank_top_k), True

                                          
            if tool.cache_ttl > 0:
                self._set_cache(name, params, data, tool.cache_ttl, cache_rerank_top_k, reranked, context)

            return ToolResult(success=True, data=data, tool_name=name,
                              latency_ms=latency, reranked=reranked)

        except asyncio.TimeoutError:
            tool.stats.failed += 1
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure()
            logger.error(f"工具超时: {name} ({tool.timeout_s}s)")
            return await self._fallback_result(tool, params, context, "执行超时")

        except Exception as ex:
            tool.stats.failed += 1
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure()
            logger.error(f"工具异常: {name} — {ex}")
            return await self._fallback_result(tool, params, context, str(ex))

    async def _fallback_result(
        self,
        tool: Tool,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        error: str,
    ) -> ToolResult:
                                           
        if tool.fallback is None:
            return ToolResult(success=False, data=None, tool_name=tool.name, error=error)
        try:
            data = tool.fallback(params, context, error)
            if asyncio.iscoroutine(data):
                data = await data
            return ToolResult(
                success=True,
                data=data,
                tool_name=tool.name,
                error=error,
            )
        except Exception as ex:
            logger.error(f"工具降级失败: {tool.name} — {ex}")
            return ToolResult(success=False, data=None, tool_name=tool.name, error=f"{error}; fallback失败: {ex}")

    async def _run_handler(
        self,
        tool: Tool,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> Any:
           
        if inspect.iscoroutinefunction(tool.handler):
            return await tool.handler(params, context)
        result = await asyncio.to_thread(tool.handler, params, context)
        if inspect.isawaitable(result):
            return await result
        return result

                                                                     

    async def rewrite_query(self, query: str, n: int = 3) -> List[str]:
           
        prompt = f"""将以下用户查询改写为 {n} 个不同角度的搜索子查询，用于检索知识库。
要求：每个子查询角度不同，覆盖原始问题的不同方面。
原始查询: "{query}"
返回 JSON 数组，例如: ["子查询1", "子查询2", "子查询3"]"""
        prompt = self._clean_text(prompt)
        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=256, temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("["), raw.rfind("]") + 1
            queries = json.loads(raw[s:e])
                        
            return list(dict.fromkeys([query] + queries))
        except Exception as ex:
            logger.warning(f"查询改写失败，使用原始查询: {ex}")
            return [query]

    async def search_with_rewrite(
        self,
        tool_name: str,
        query: str,
        top_k: int = 5,
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:       
                          
        sub_queries = await self.rewrite_query(query, n=3)
        logger.info(f"查询改写: {query!r} → {sub_queries}")

                           
        recall_k = max(top_k, 5)
        tasks = [
            self.call(tool_name, {"query": q, "top_k": recall_k}, context, use_cache=True)
            for q in sub_queries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

                          
        seen, merged = set(), []
        for r in results:
            if isinstance(r, ToolResult) and r.success and isinstance(r.data, list):
                for item in r.data:
                    key = hashlib.md5(str(item).encode()).hexdigest()
                    if key not in seen:
                        seen.add(key)
                        merged.append(item)

        if not merged:
            return ToolResult(success=False, data=[], tool_name=tool_name, error="所有子查询均无结果")

                                         
        reranked = await self._rerank(query, merged, top_k)
        return ToolResult(success=True, data=reranked, tool_name=tool_name, reranked=True)

                                                                   

    async def _rerank(self, query: str, items: List[Any], top_k: int) -> List[Any]:
           
        if len(items) <= top_k:
            return items

                           
        items_text = "\n".join(f"{i}. {json.dumps(item, ensure_ascii=False)[:200]}"
                               for i, item in enumerate(items))
        prompt = f"""根据用户查询，对以下检索结果按相关性打分（0-10），返回 JSON 数组。
用户查询: "{query}"
检索结果:
{items_text}

返回格式（按相关性降序排列的索引列表）: [最相关的索引, ..., 最不相关的索引]
只返回 JSON 数组，不要其他文字。"""
        prompt = self._clean_text(prompt)

        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=256, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("["), raw.rfind("]") + 1
            order: List[int] = json.loads(raw[s:e])
            reranked = [items[i] for i in order if 0 <= i < len(items)]
            return reranked[:top_k]
        except Exception as ex:
            logger.warning(f"重排失败，返回原始顺序: {ex}")
            return items[:top_k]

                                                                              

    def _cache_key(
        self,
        name: str,
        params: Dict,
        rerank_top_k: int = 0,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        tenant_id = context.get("tenant_id", "default") if isinstance(context, dict) else "default"
        payload = {"params": params, "rerank_top_k": rerank_top_k, "tenant_id": tenant_id}
        return f"{name}:{hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()}"

    def _get_cache(
        self,
        name: str,
        params: Dict,
        rerank_top_k: int = 0,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[Any, bool]]:
        key = self._cache_key(name, params, rerank_top_k, context)
        if key in self._cache:
            data, expire_at, reranked = self._cache[key]
            if time.monotonic() < expire_at:
                return data, reranked
            del self._cache[key]
        return None

    def _set_cache(
        self,
        name: str,
        params: Dict,
        data: Any,
        ttl: float,
        rerank_top_k: int = 0,
        reranked: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if len(self._cache) >= 5000:
                       
            for k in list(self._cache)[:1250]:
                del self._cache[k]
        self._cache[self._cache_key(name, params, rerank_top_k, context)] = (
            data, time.monotonic() + ttl, reranked
        )

                                                                            

    _TYPE_MAP = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "array": list, "object": dict}

    def _validate_params(self, tool: Tool, params: Dict[str, Any]) -> None:
                                                 
        self._validate_schema_value(params, tool.schema, tool.name)

    @classmethod
    def _validate_schema_value(cls, value: Any, schema: Dict[str, Any], path: str) -> None:
        expected_type = schema.get("type")
        if expected_type:
            type_map = cls._TYPE_MAP
            expected = type_map.get(expected_type)
            if expected is not None:
                valid = isinstance(value, expected)
                if expected_type in {"integer", "number"} and isinstance(value, bool):
                    valid = False
                if not valid:
                    raise ValueError(
                        f"参数 {path} 类型错误: 期望 {expected_type}，实际 {type(value).__name__}"
                    )

        if "enum" in schema and value not in schema["enum"]:
            raise ValueError(f"参数 {path} 不在允许值范围内")

        if isinstance(value, str):
            if "minLength" in schema and len(value) < schema["minLength"]:
                raise ValueError(f"参数 {path} 长度不能少于 {schema['minLength']}")
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                raise ValueError(f"参数 {path} 长度不能超过 {schema['maxLength']}")
            if "pattern" in schema and re.search(schema["pattern"], value) is None:
                raise ValueError(f"参数 {path} 格式不合法")

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in schema and value < schema["minimum"]:
                raise ValueError(f"参数 {path} 不能小于 {schema['minimum']}")
            if "maximum" in schema and value > schema["maximum"]:
                raise ValueError(f"参数 {path} 不能大于 {schema['maximum']}")

        if isinstance(value, dict):
            properties = schema.get("properties", {})
            for field_name in schema.get("required", []):
                if field_name not in value:
                    raise ValueError(f"工具 {path} 缺少必需参数: {field_name}")
            if schema.get("additionalProperties") is False:
                unknown = set(value) - set(properties)
                if unknown:
                    raise ValueError(f"工具 {path} 不允许的参数: {', '.join(sorted(unknown))}")
            for key, child in value.items():
                if key in properties:
                    cls._validate_schema_value(child, properties[key], f"{path}.{key}")

        if isinstance(value, list):
            if "minItems" in schema and len(value) < schema["minItems"]:
                raise ValueError(f"参数 {path} 至少需要 {schema['minItems']} 项")
            if "maxItems" in schema and len(value) > schema["maxItems"]:
                raise ValueError(f"参数 {path} 不能超过 {schema['maxItems']} 项")
            item_schema = schema.get("items")
            if isinstance(item_schema, dict):
                for index, item in enumerate(value):
                    cls._validate_schema_value(item, item_schema, f"{path}[{index}]")

    @staticmethod
    def _clean_text(value: Any) -> str:
                                            
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

                                                                              

    def get_stats(self) -> Dict[str, Any]:
        return {
            name: {
                "total": t.stats.total,
                "success_rate": round(t.stats.success_rate, 3),
                "avg_latency_ms": round(t.stats.avg_latency_ms, 1),
                "consecutive_fails": t.stats.consecutive_fails,
                "circuit_state": t.breaker.state.value,
            }
            for name, t in self._tools.items()
        }
