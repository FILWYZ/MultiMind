                            

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Dict

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,96}$")


class ApiError(HTTPException):
                            

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Any = None,
        headers: Dict[str, str] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.details = details
        super().__init__(status_code=status_code, detail=message, headers=headers)


def request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", "")
    return value if value else str(uuid.uuid4())


def error_payload(request: Request, code: str, message: str, details: Any = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "code": code,
        "message": message,
        "request_id": request_id(request),
    }
    if details is not None:
        payload["details"] = details
    return payload


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(request, exc.code, exc.message, exc.details),
        headers=exc.headers,
    )


def _status_code_name(status_code: int) -> str:
    return {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        409: "CONFLICT",
        413: "PAYLOAD_TOO_LARGE",
        422: "VALIDATION_ERROR",
        429: "RATE_LIMITED",
        500: "INTERNAL_ERROR",
        503: "SERVICE_UNAVAILABLE",
    }.get(status_code, f"HTTP_{status_code}")


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    details = None
    message = exc.detail
    code = _status_code_name(exc.status_code)
    if isinstance(message, dict):
        details = message.get("details", message)
        code = str(message.get("code", code))
        message = str(message.get("message", "请求失败"))
    else:
        message = str(message)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(request, code, message, details),
        headers=exc.headers,
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=error_payload(request, "VALIDATION_ERROR", "请求参数校验失败", exc.errors()),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("未处理的 API 异常 request_id=%s", request_id(request), exc_info=exc)
    return JSONResponse(
        status_code=500,
        content=error_payload(request, "INTERNAL_ERROR", "服务内部错误"),
    )


async def request_context_middleware(request: Request, call_next):
                                        
    incoming = request.headers.get("X-Request-ID", "")
    rid = incoming if _REQUEST_ID_RE.fullmatch(incoming) else str(uuid.uuid4())
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response
