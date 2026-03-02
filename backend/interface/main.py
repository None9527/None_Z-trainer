# -*- coding: utf-8 -*-
"""
FastAPI Application Entry Point

Assembles all DDD layers:
- Interface routers (training, dataset, generation, system, websocket)
- Infrastructure DI container
- CORS configuration
- Static file serving for Vue frontend
"""

import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure trainer_core/ is on sys.path for zimage_trainer imports (v2-local copy)
TRAINER_CORE_PATH = Path(__file__).resolve().parents[1] / "trainer_core"  # interface/ → backend/trainer_core
if str(TRAINER_CORE_PATH) not in sys.path:
    sys.path.insert(0, str(TRAINER_CORE_PATH))

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Z-Image Trainer starting...")
    yield
    logger.info("Z-Image Trainer shutting down...")


app = FastAPI(
    title="Z-Image Trainer",
    version="2.0.0",
    description="DDD-based training management API",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Register Routers ---
from .training_router import router as training_router
from .dataset_router import router as dataset_router
from .generation_router import router as generation_router
from .system_router import router as system_router
from .cache_router import router as cache_router
from .websocket_manager import router as ws_router
from .skill_router import router as skill_router

app.include_router(training_router)
app.include_router(dataset_router)
app.include_router(generation_router)
app.include_router(system_router)
app.include_router(cache_router)
app.include_router(ws_router)
app.include_router(skill_router)


# --- Health Check ---
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "2.0.0"}


# --- Static Files (Vue Frontend) ---
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"  # v2/frontend/dist
if FRONTEND_DIST.exists():
    # SPA catch-all: serve index.html for any non-API path
    from fastapi.responses import FileResponse

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        # If the requested file exists in dist, serve it
        file_path = FRONTEND_DIST / full_path
        if full_path and file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        # Otherwise serve index.html for SPA routing
        return FileResponse(FRONTEND_DIST / "index.html")

    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")


# --- CLI Entry ---
if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    uvicorn.run(
        "v2.backend.interface.main:app",
        host="0.0.0.0",
        port=28000,
        reload=True,
    )
