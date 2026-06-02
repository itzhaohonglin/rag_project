from fastapi import APIRouter

from backend.api.errors import success_response

router = APIRouter()


@router.get("/health")
async def health_check():
    return success_response({
        "status": "ok",
        "version": "0.1.0",
    })
