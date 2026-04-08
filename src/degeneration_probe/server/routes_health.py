"""Health and metadata endpoints."""

from fastapi import APIRouter

from degeneration_probe.worker.steering import STRATEGY_REGISTRY

router = APIRouter()


@router.get("/api/health")
async def health():
    return {"status": "ok"}


@router.get("/api/strategies")
async def strategies():
    return {
        name: {"description": cls.__doc__ or name}
        for name, cls in STRATEGY_REGISTRY.items()
    }
