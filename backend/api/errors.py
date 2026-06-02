from fastapi import Request
from fastapi.responses import JSONResponse

from backend.domain.exceptions import RAGException


class APIError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400, detail: dict | None = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.detail = detail or {}


async def rag_exception_handler(request: Request, exc: RAGException) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"code": exc.code, "message": exc.message, "detail": exc.detail},
    )


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.code, "message": exc.message, "detail": exc.detail},
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"code": "INTERNAL_ERROR", "message": "Internal server error", "detail": {}},
    )


def success_response(data: object = None, message: str = "ok") -> dict:
    return {"code": 0, "data": data, "message": message}


def error_response(code: str, message: str, detail: dict | None = None) -> dict:
    return {"code": code, "message": message, "detail": detail or {}}
