"""Generation history endpoints."""

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("/api/generations")
async def list_generations(request: Request, limit: int = 20, offset: int = 0):
    db = request.app.state.db
    return await db.list_generations(limit=limit, offset=offset)


@router.get("/api/generations/{gen_id}")
async def get_generation(gen_id: int, request: Request):
    db = request.app.state.db
    gen = await db.get_generation(gen_id)
    if gen is None:
        raise HTTPException(status_code=404, detail="Generation not found")
    return gen
