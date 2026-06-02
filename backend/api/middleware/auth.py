import hmac

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware


class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str = ""):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):
        if self.api_key:
            key = request.headers.get("X-API-Key", "")
            if not hmac.compare_digest(key, self.api_key):
                raise HTTPException(status_code=401, detail="Invalid API key")
        return await call_next(request)
