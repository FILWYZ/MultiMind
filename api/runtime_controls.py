                                 

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Optional, Tuple

from api.errors import ApiError


class RateLimiter:
    def __init__(self, redis_url: str | None = None, window_seconds: int = 60) -> None:
        self.window_seconds = max(1, int(window_seconds))
        self.redis = None
        self.backend = "memory"
        self._redis_disabled_until = 0.0
        self._lock = asyncio.Lock()
        self._events: Dict[str, Deque[float]] = defaultdict(deque)
        self.policies: Dict[str, Tuple[int, int]] = {
            "chat": (30, self.window_seconds),
            "search": (60, self.window_seconds),
            "knowledge_write": (20, self.window_seconds),
            "admin": (60, self.window_seconds),
        }
        if redis_url:
            try:
                from redis.asyncio import from_url
                self.redis = from_url(redis_url, decode_responses=True)
            except Exception:
                self.redis = None

    async def check(self, scope: str, identity: str, limit: int | None = None) -> Dict[str, Any]:
        configured_limit, window = self.policies.get(scope, (60, self.window_seconds))
        limit = configured_limit if limit is None else int(limit)
        now = time.time()
        if self.redis is not None and now >= self._redis_disabled_until:
            try:
                bucket = int(now // window)
                key = f"echomind:rate:v1:{scope}:{self._safe(identity)}:{bucket}"
                count = int(await self.redis.incr(key))
                if count == 1:
                    await self.redis.expire(key, window + 1)
                self.backend = "redis"
                if count > limit:
                    retry_after = max(1, int((bucket + 1) * window - now))
                    raise ApiError(
                        429,
                        "RATE_LIMITED",
                        "请求频率超过限制",
                        {"scope": scope, "limit": limit, "window_seconds": window},
                        headers={"Retry-After": str(retry_after)},
                    )
                return {"allowed": True, "count": count, "limit": limit, "backend": self.backend}
            except ApiError:
                raise
            except Exception:
                self._redis_disabled_until = now + 10.0
                self.backend = "memory"

        async with self._lock:
            events = self._events[f"{scope}:{identity}"]
            cutoff = now - window
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= limit:
                retry_after = max(1, int(events[0] + window - now))
                raise ApiError(
                    429,
                    "RATE_LIMITED",
                    "请求频率超过限制",
                    {"scope": scope, "limit": limit, "window_seconds": window},
                    headers={"Retry-After": str(retry_after)},
                )
            events.append(now)
            return {"allowed": True, "count": len(events), "limit": limit, "backend": self.backend}

    @staticmethod
    def _safe(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:32]

    async def close(self) -> None:
        if self.redis is not None:
            await self.redis.aclose()


class IdempotencyStore:
    def __init__(self, redis_url: str | None = None, ttl_seconds: int = 900) -> None:
        self.ttl_seconds = max(30, int(ttl_seconds))
        self.redis = None
        self.backend = "memory"
        self._lock = asyncio.Lock()
        self._items: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        if redis_url:
            try:
                from redis.asyncio import from_url
                self.redis = from_url(redis_url, decode_responses=True)
            except Exception:
                self.redis = None

    def _key(self, scope: str, principal: str, key: str) -> str:
        raw = f"{scope}:{principal}:{key}".encode("utf-8", errors="ignore")
        return "echomind:idempotency:v1:" + hashlib.sha256(raw).hexdigest()

    async def begin(self, scope: str, principal: str, key: str) -> Optional[Dict[str, Any]]:
        key = self._validate(key)
        storage_key = self._key(scope, principal, key)
        marker = {"state": "processing", "created_at": time.time()}
        if self.redis is not None:
            try:
                claimed = await self.redis.set(storage_key, json.dumps(marker), nx=True, ex=self.ttl_seconds)
                self.backend = "redis"
                if claimed:
                    return None
                raw = await self.redis.get(storage_key)
                if raw:
                    item = json.loads(raw)
                    if item.get("state") == "completed":
                        return item.get("response")
                    raise ApiError(409, "IDEMPOTENCY_IN_PROGRESS", "相同 Idempotency-Key 的请求正在处理中")
            except ApiError:
                raise
            except Exception:
                self.backend = "memory"
                self.redis = None

        async with self._lock:
            now = time.time()
            expired = [k for k, (deadline, _) in self._items.items() if deadline <= now]
            for item_key in expired:
                self._items.pop(item_key, None)
            current = self._items.get(storage_key)
            if current:
                item = current[1]
                if item.get("state") == "completed":
                    return item.get("response")
                raise ApiError(409, "IDEMPOTENCY_IN_PROGRESS", "相同 Idempotency-Key 的请求正在处理中")
            self._items[storage_key] = (now + self.ttl_seconds, marker)
            return None

    async def complete(self, scope: str, principal: str, key: str, response: Dict[str, Any]) -> None:
        key = self._validate(key)
        storage_key = self._key(scope, principal, key)
        item = {"state": "completed", "response": response, "completed_at": time.time()}
        if self.redis is not None:
            try:
                await self.redis.set(storage_key, json.dumps(item, ensure_ascii=False), ex=self.ttl_seconds)
                return
            except Exception:
                self.redis = None
        async with self._lock:
            self._items[storage_key] = (time.time() + self.ttl_seconds, item)

    async def fail(self, scope: str, principal: str, key: str) -> None:
        key = self._validate(key)
        storage_key = self._key(scope, principal, key)
        if self.redis is not None:
            try:
                await self.redis.delete(storage_key)
                return
            except Exception:
                self.redis = None
        async with self._lock:
            self._items.pop(storage_key, None)

    @staticmethod
    def _validate(key: str) -> str:
        value = (key or "").strip()
        if not value or len(value) > 128:
            raise ApiError(422, "INVALID_IDEMPOTENCY_KEY", "Idempotency-Key 必须为 1-128 个字符")
        return value

    async def close(self) -> None:
        if self.redis is not None:
            await self.redis.aclose()
