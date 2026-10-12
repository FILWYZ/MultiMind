                      

from __future__ import annotations

import os

from fastapi import Request

from api.errors import ApiError
from core.tenant import DEFAULT_TENANT_ID, normalize_tenant_id


async def get_tenant_id(request: Request) -> str:
    raw = request.headers.get("X-Tenant-ID") or os.getenv(
        "ECHOMIND_DEFAULT_TENANT_ID", DEFAULT_TENANT_ID
    )
    try:
        return normalize_tenant_id(raw)
    except ValueError as exc:
        raise ApiError(400, "INVALID_TENANT_ID", str(exc)) from exc
