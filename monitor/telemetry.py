                            

from __future__ import annotations

import hashlib
import os
import statistics
import threading
from collections import Counter as Count, deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Dict, Optional

from prometheus_client import Counter, Histogram

REQUESTS = Counter(
    "echomind_requests_total",
    "EchoMind request count",
    ["route", "variant", "status"],
)
LATENCY = Histogram(
    "echomind_request_latency_seconds",
    "EchoMind request latency",
    ["route", "variant"],
)
TOKENS = Counter(
    "echomind_tokens_total",
    "EchoMind estimated token count",
    ["direction", "variant"],
)
COST = Counter(
    "echomind_estimated_cost_usd_total",
    "EchoMind estimated model cost",
    ["variant"],
)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text.encode("utf-8", errors="ignore")) + 3) // 4)


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    try:
        input_rate = float(os.getenv("ECHOMIND_INPUT_COST_PER_1K_USD", "0.0008"))
        output_rate = float(os.getenv("ECHOMIND_OUTPUT_COST_PER_1K_USD", "0.0024"))
    except ValueError:
        input_rate, output_rate = 0.0008, 0.0024
    return (input_tokens / 1000) * input_rate + (output_tokens / 1000) * output_rate


@dataclass(frozen=True)
class TelemetryEvent:
    route: str
    tenant_hash: str
    variant: str
    status: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    token_source: str


class TelemetryRegistry:
    def __init__(self, max_events: int = 10000) -> None:
        self._events: Deque[TelemetryEvent] = deque(maxlen=max_events)
        self._lock = threading.RLock()

    def record(
        self,
        route: str,
        tenant_id: str,
        variant: str,
        status: str,
        latency_ms: float,
        input_text: str = "",
        output_text: str = "",
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        usage_exact: bool = False,
    ) -> TelemetryEvent:
        input_tokens = estimate_tokens(input_text) if input_tokens is None else max(0, int(input_tokens))
        output_tokens = estimate_tokens(output_text) if output_tokens is None else max(0, int(output_tokens))
        event = TelemetryEvent(
            route=route,
            tenant_hash=hashlib.sha256(str(tenant_id).encode()).hexdigest()[:12],
            variant=variant,
            status=status,
            latency_ms=round(float(latency_ms), 3),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=round(estimate_cost_usd(input_tokens, output_tokens), 8),
            token_source="provider_usage" if usage_exact else "estimate",
        )
        with self._lock:
            self._events.append(event)
        REQUESTS.labels(route=route, variant=variant, status=status).inc()
        LATENCY.labels(route=route, variant=variant).observe(event.latency_ms / 1000)
        TOKENS.labels(direction="input", variant=variant).inc(input_tokens)
        TOKENS.labels(direction="output", variant=variant).inc(output_tokens)
        COST.labels(variant=variant).inc(event.estimated_cost_usd)
        return event

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            events = list(self._events)
        if not events:
            return {
                "total": 0,
                "success_rate": 1.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost_usd": 0.0,
                "by_variant": {},
                "by_tenant_hash": {},
                "token_source": {},
            }
        latencies = sorted(event.latency_ms for event in events)
        success = sum(event.status == "success" for event in events)
        variants = Count(event.variant for event in events)
        tenants = Count(event.tenant_hash for event in events)
        token_sources = Count(event.token_source for event in events)
        return {
            "total": len(events),
            "success_rate": round(success / len(events), 4),
            "p50_latency_ms": round(latencies[len(latencies) // 2], 3),
            "p95_latency_ms": round(latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))], 3),
            "input_tokens": sum(event.input_tokens for event in events),
            "output_tokens": sum(event.output_tokens for event in events),
            "estimated_cost_usd": round(sum(event.estimated_cost_usd for event in events), 8),
            "by_variant": dict(variants),
            "by_tenant_hash": dict(tenants),
            "token_source": dict(token_sources),
        }

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
