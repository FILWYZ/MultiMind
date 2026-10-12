                       

from __future__ import annotations

import hmac
import os

from fastapi import Request

from api.errors import ApiError


def _configured_key() -> str:
    return os.getenv("ECHOMIND_ADMIN_API_KEY", "").strip()


def _provided_key(request: Request) -> str:
    value = request.headers.get("X-API-Key", "").strip()
    if value:
        return value
    authorization = request.headers.get("Authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer":
        return token.strip()
    return ""


async def require_admin(request: Request) -> str:
                                    
    expected = _configured_key()
    if not expected:
        raise ApiError(503, "AUTH_NOT_CONFIGURED", "管理接口未配置 ECHOMIND_ADMIN_API_KEY")
    actual = _provided_key(request)
    if not actual:
        raise ApiError(
            401,
            "UNAUTHORIZED",
            "需要提供 X-API-Key 或 Bearer API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not hmac.compare_digest(actual, expected):
        raise ApiError(403, "FORBIDDEN", "API Key 无效")
    return actual


def request_principal(request: Request) -> str:
                            
    api_key = _provided_key(request)
    tenant_id = request.headers.get("X-Tenant-ID", "default").strip() or "default"
    if api_key:
        return f"tenant:{tenant_id}:key:" + api_key[:12]
    client = request.client.host if request.client else "unknown"
    return f"tenant:{tenant_id}:ip:" + client
