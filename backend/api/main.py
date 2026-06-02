import logging

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


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        debug=settings.app.debug,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if not settings.app.debug:
        app.add_middleware(APIKeyMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    app.add_exception_handler(RAGException, rag_exception_handler)
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
    app.include_router(retrieval.router, prefix="/api/v1", tags=["retrieval"])
    app.include_router(conversation.router, prefix="/api/v1/conversations", tags=["conversations"])

    @app.on_event("startup")
    async def startup():
        logging.basicConfig(level=getattr(logging, settings.app.debug and "DEBUG" or "INFO"))
        logging.getLogger("rag.api").info("Application starting...")

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "backend.api.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.debug,
    )
