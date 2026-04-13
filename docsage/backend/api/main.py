"""
api/main.py — FastAPI application entry point.

Mounts:
  /api/v1/documents  — upload, list, delete documents
  /api/v1/qa         — ask questions, get answers
  /api/v1/sessions   — session management, history
  /api/v1/health     — health check
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from core.config import settings
from core.logging import configure_logging, get_logger
from api.routers import documents, qa, sessions

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info("docsage_starting", version=settings.app_version, env=settings.environment)
    settings.create_dirs()
    # Warm up pipeline (lazy-load models on first request to avoid blocking startup)
    from core.pipeline import DocSagePipeline
    DocSagePipeline.get()
    logger.info("docsage_ready")
    yield
    logger.info("docsage_shutdown")


app = FastAPI(
    title="DocSage API",
    description="Smart Document Question Answering System",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(documents.router, prefix=f"{settings.api_prefix}/documents", tags=["Documents"])
app.include_router(qa.router, prefix=f"{settings.api_prefix}/qa", tags=["QA"])
app.include_router(sessions.router, prefix=f"{settings.api_prefix}/sessions", tags=["Sessions"])


@app.get(f"{settings.api_prefix}/health", tags=["Health"])
async def health():
    return {"status": "ok", "version": settings.app_version}