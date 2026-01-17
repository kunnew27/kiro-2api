# -*- coding: utf-8 -*-
"""
Model metadata cache.

Thread-safe storage with TTL, lazy loading and background refresh.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from app.core.config import settings
from app.libs.http_client import global_http_client_manager


class ModelInfoCache:
    """
    Thread-safe model metadata cache.

    Uses lazy loading - data is loaded only on first access or when cache expires.
    Supports background auto-refresh mechanism.
    """

    def __init__(self, cache_ttl: int = None):
        """Initialize model cache."""
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._last_update: Optional[float] = None
        self._cache_ttl = cache_ttl or settings.model_cache_ttl
        self._refresh_task: Optional[asyncio.Task] = None
        self._auth_manager = None

    def set_auth_manager(self, auth_manager) -> None:
        """Set authentication manager (for background refresh)."""
        self._auth_manager = auth_manager

    async def update(self, models_data: List[Dict[str, Any]]) -> None:
        """Update model cache."""
        async with self._lock:
            logger.info(f"Updating model cache. Found {len(models_data)} models.")
            self._cache = {model["modelId"]: model for model in models_data}
            self._last_update = time.time()

    async def refresh(self) -> bool:
        """Refresh cache from API using global connection pool."""
        if not self._auth_manager:
            logger.warning("No auth manager set, cannot refresh cache")
            return False

        try:
            token = await self._auth_manager.get_access_token()
            from app.utils.helpers import get_kiro_headers
            headers = get_kiro_headers(self._auth_manager, token)

            client = await global_http_client_manager.get_client()
            response = await client.get(
                f"{self._auth_manager.q_host}/ListAvailableModels",
                headers=headers,
                params={
                    "origin": "AI_EDITOR",
                    "profileArn": self._auth_manager.profile_arn or ""
                },
                timeout=30.0
            )

            if response.status_code == 200:
                data = response.json()
                models_list = data.get("models", [])
                await self.update(models_list)
                logger.info(f"Successfully refreshed model cache with {len(models_list)} models")
                return True
            else:
                logger.error(f"Failed to refresh models: HTTP {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error refreshing model cache: {e}")
            return False

    async def start_background_refresh(self) -> None:
        """Start background refresh task."""
        if self._refresh_task and not self._refresh_task.done():
            logger.warning("Background refresh task is already running")
            return

        self._refresh_task = asyncio.create_task(self._background_refresh_loop())
        logger.info("Started background model cache refresh task")

    async def stop_background_refresh(self) -> None:
        """Stop background refresh task."""
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                logger.info("Stopped background model cache refresh task")
            except Exception as e:
                logger.error(f"Error stopping refresh task: {e}")

    async def _background_refresh_loop(self) -> None:
        """Background refresh loop."""
        refresh_interval = self._cache_ttl / 2
        logger.info(f"Background refresh will run every {refresh_interval} seconds")

        while True:
            try:
                await asyncio.sleep(refresh_interval)
                logger.debug("Running scheduled model cache refresh")
                await self.refresh()
            except asyncio.CancelledError:
                logger.info("Background refresh task cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in background refresh: {e}")

    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model info."""
        return self._cache.get(model_id)

    def get_max_input_tokens(self, model_id: str) -> int:
        """Get model's maxInputTokens."""
        model = self._cache.get(model_id)
        if model and model.get("tokenLimits"):
            return model["tokenLimits"].get("maxInputTokens") or settings.default_max_input_tokens
        return settings.default_max_input_tokens

    def is_empty(self) -> bool:
        """Check if cache is empty."""
        return not self._cache

    def is_stale(self) -> bool:
        """Check if cache is stale."""
        if not self._last_update:
            return True
        return time.time() - self._last_update > self._cache_ttl

    def get_all_model_ids(self) -> List[str]:
        """Return all model IDs in cache."""
        return list(self._cache.keys())

    @property
    def size(self) -> int:
        """Number of models in cache."""
        return len(self._cache)

    @property
    def last_update_time(self) -> Optional[float]:
        """Last update timestamp (seconds) or None."""
        return self._last_update

    @property
    def is_background_refresh_running(self) -> bool:
        """Check if background refresh is running."""
        return self._refresh_task is not None and not self._refresh_task.done()
