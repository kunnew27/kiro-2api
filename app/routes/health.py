# -*- coding: utf-8 -*-
"""Health check routes."""

from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.config import APP_VERSION
from app.libs.auth import KiroAuthManager
from app.libs.cache import ModelInfoCache

router = APIRouter()


@router.get("/")
async def root():
    """Root endpoint - API info."""
    return {
        "status": "ok",
        "message": "Kiro-2API Gateway is running",
        "version": APP_VERSION
    }


@router.get("/api")
async def api_root():
    """API health check endpoint (JSON)."""
    return {
        "status": "ok",
        "message": "Kiro-2API Gateway is running",
        "version": APP_VERSION
    }


@router.get("/health")
async def health(request: Request):
    """
    Detailed health check.

    Returns:
        Status, timestamp, version and runtime info
    """
    auth_manager: KiroAuthManager = request.app.state.auth_manager
    model_cache: ModelInfoCache = request.app.state.model_cache

    token_valid = False
    try:
        if auth_manager._access_token and not auth_manager.is_token_expiring_soon():
            token_valid = True
    except Exception:
        token_valid = False

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": APP_VERSION,
        "token_valid": token_valid,
        "cache_size": model_cache.size,
        "cache_last_update": model_cache.last_update_time
    }
