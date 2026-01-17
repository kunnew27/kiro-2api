# -*- coding: utf-8 -*-
"""Model listing routes."""

import secrets

from fastapi import APIRouter, Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from loguru import logger

from app.core.config import settings, AVAILABLE_MODELS
from app.libs.auth import KiroAuthManager
from app.libs.cache import ModelInfoCache
from app.models.schemas import OpenAIModel, ModelList
from app.middleware.tracking import get_timestamp

router = APIRouter()

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


def _mask_token(token: str) -> str:
    """Mask token for logging."""
    if len(token) <= 8:
        return "***"
    return f"{token[:4]}...{token[-4:]}"


async def verify_api_key(
    request: Request,
    auth_header: str = Security(api_key_header)
) -> KiroAuthManager:
    """
    Verify API key in Authorization header.

    Supports:
    1. Traditional: "Bearer {PROXY_API_KEY}" - uses global AuthManager
    2. Multi-tenant: "Bearer {PROXY_API_KEY}:{REFRESH_TOKEN}" - creates per-user AuthManager

    Args:
        request: FastAPI Request
        auth_header: Authorization header value

    Returns:
        KiroAuthManager instance

    Raises:
        HTTPException: 401 if key is invalid or missing
    """
    if not auth_header:
        logger.warning(f"[{get_timestamp()}] Missing Authorization header")
        raise HTTPException(status_code=401, detail="API Key invalid or missing")

    # Support both "Bearer {token}" format and raw token (for Swagger UI)
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
    else:
        token = auth_header
    proxy_api_key = settings.proxy_api_key

    if ':' in token:
        parts = token.split(':', 1)
        proxy_key = parts[0]
        refresh_token = parts[1]

        if not secrets.compare_digest(proxy_key, proxy_api_key):
            logger.warning(f"[{get_timestamp()}] Invalid Proxy Key in multi-tenant mode: {_mask_token(proxy_key)}")
            raise HTTPException(status_code=401, detail="API Key invalid or missing")

        logger.debug(f"[{get_timestamp()}] Multi-tenant mode: using custom Refresh Token {_mask_token(refresh_token)}")
        auth_manager = KiroAuthManager(
            refresh_token=refresh_token,
            region=settings.region,
            profile_arn=settings.profile_arn
        )
        return auth_manager

    if secrets.compare_digest(token, proxy_api_key):
        logger.debug(f"[{get_timestamp()}] Traditional mode: using global AuthManager")
        return request.app.state.auth_manager

    logger.warning(f"[{get_timestamp()}] Invalid API Key in traditional mode")
    raise HTTPException(status_code=401, detail="API Key invalid or missing")


@router.get("/v1/models", response_model=ModelList)
async def get_models(
    request: Request,
    auth_manager: KiroAuthManager = Depends(verify_api_key)
):
    """
    Return available models list.

    Uses static model list with optional dynamic updates from API.
    Results are cached to reduce API load.

    Args:
        request: FastAPI Request
        auth_manager: KiroAuthManager instance

    Returns:
        ModelList containing available models
    """
    logger.info(f"[{get_timestamp()}] Received /v1/models request")

    model_cache: ModelInfoCache = request.app.state.model_cache

    if model_cache.is_empty() or model_cache.is_stale():
        try:
            import asyncio
            asyncio.create_task(model_cache.refresh())
        except Exception as e:
            logger.warning(f"[{get_timestamp()}] Failed to trigger model cache refresh: {e}")

    openai_models = [
        OpenAIModel(
            id=model_id,
            owned_by="anthropic",
            description="Claude model via Kiro API"
        )
        for model_id in AVAILABLE_MODELS
    ]

    return ModelList(data=openai_models)
