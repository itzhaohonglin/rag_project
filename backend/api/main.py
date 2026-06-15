import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.errors import (
    APIError,
    api_error_handler,
    general_exception_handler,
    rag_exception_handler,
)
from backend.api.middleware.auth import APIKeyMiddleware
from backend.api.middleware.logging import RequestLoggingMiddleware
from backend.api.routes import conversation, documents, health, retrieval
from backend.core.config import settings
from backend.domain.exceptions import RAGException


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化日志，关闭时清理资源。"""
    logging.basicConfig(level=getattr(logging, settings.app.debug and "DEBUG" or "INFO"))
    logging.getLogger("rag.api").info("Application starting...")
    yield
    logging.getLogger("rag.api").info("Application shutting down...")


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用实例。"""
    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        debug=settings.app.debug,
        lifespan=lifespan,
    )

    # ── 中间件 ──────────────────────────────────────────────
    # CORS：开发阶段允许所有来源，生产环境应收紧
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 非调试模式启用 API Key 鉴权；请求日志中间件始终启用
    if not settings.app.debug:
        app.add_middleware(APIKeyMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    # ── 异常处理器 ──────────────────────────────────────────
    # 按优先级从具体到通用注册
    app.add_exception_handler(RAGException, rag_exception_handler)
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    # ── 路由注册 ────────────────────────────────────────────
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
    app.include_router(retrieval.router, prefix="/api/v1", tags=["retrieval"])
    app.include_router(conversation.router, prefix="/api/v1/conversations", tags=["conversations"])

    return app


# 全局应用实例，供 uvicorn 导入
app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "backend.api.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.debug,
    )
