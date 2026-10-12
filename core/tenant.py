                               

from __future__ import annotations

import re

DEFAULT_TENANT_ID = "default"
_TENANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def normalize_tenant_id(value: object, default: str = DEFAULT_TENANT_ID) -> str:
    tenant_id = str(value or default).strip()
    if not _TENANT_RE.fullmatch(tenant_id):
        raise ValueError("tenant_id 必须是 1-64 位字母、数字、下划线或短横线")
    return tenant_id
