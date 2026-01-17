# -*- coding: utf-8 -*-
"""
HTTP Client Manager.

Global HTTP client connection pool management for improved performance.
Supports adaptive timeout for slow models (like Opus).
"""

import asyncio
from typing import Optional

import httpx
from fastapi import HTTPException
from loguru import logger

from app.libs.auth import KiroAuthManager
from app.core.config import settings, get_adaptive_timeout
from app.utils.helpers import get_kiro_headers


class GlobalHTTPClientManager:
    """
    Global HTTP client manager.

    Maintains a global connection pool to avoid creating new clients for each request.
    """

    def __init__(self):
        """Initialize global client manager."""
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()

    async def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        async with self._lock:
            if self._client is None or self._client.is_closed:
                limits = httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                    keepalive_expiry=60.0
                )

                self._client = httpx.AsyncClient(
                    timeout=None,
                    follow_redirects=True,
                    limits=limits,
                    http2=False
                )
                logger.debug("Created new global HTTP client with connection pool")

            return self._client

    async def close(self) -> None:
        """Close global HTTP client."""
        async with self._lock:
            if self._client and not self._client.is_closed:
                await self._client.aclose()
                logger.debug("Closed global HTTP client")


global_http_client_manager = GlobalHTTPClientManager()


class KiroHttpClient:
    """
    Kiro API HTTP client with retry logic.

    Uses global connection pool for better performance.
    Automatically handles various error types:
    - 403: Auto-refresh token and retry
    - 429: Exponential backoff retry
    - 5xx: Exponential backoff retry
    - Timeout: Exponential backoff retry
    """

    def __init__(self, auth_manager: KiroAuthManager):
        """Initialize HTTP client."""
        self.auth_manager = auth_manager
        self.client = None

    def _extract_model_from_payload(self, json_data: Optional[dict]) -> str:
        """Extract model name from common payload locations."""
        if not json_data:
            return ""
        model = json_data.get("modelId") or json_data.get("model")
        if model:
            return model
        conversation = json_data.get("conversationState") or {}
        current = conversation.get("currentMessage") or {}
        user_input = current.get("userInputMessage") or {}
        model = user_input.get("modelId") or user_input.get("model")
        if model:
            return model
        history = conversation.get("history") or []
        for entry in reversed(history):
            user_input = entry.get("userInputMessage") if isinstance(entry, dict) else None
            if user_input and user_input.get("modelId"):
                return user_input.get("modelId")
        return ""

    async def _get_client(self) -> httpx.AsyncClient:
        """Get HTTP client (uses global connection pool)."""
        return await global_http_client_manager.get_client()

    async def close(self) -> None:
        """Close client (does not actually close global client)."""
        pass

    async def request_with_retry(
        self,
        method: str,
        url: str,
        json_data: dict,
        stream: bool = False,
        first_token_timeout: float = None,
        model: str = None
    ) -> httpx.Response:
        """
        Execute HTTP request with retry logic.

        Args:
            method: HTTP method
            url: Request URL
            json_data: JSON request body
            stream: Whether to use streaming response
            first_token_timeout: First token timeout (streaming only)
            model: Model name (for adaptive timeout)

        Returns:
            HTTP response

        Raises:
            HTTPException: After retry failure
        """
        if model is None:
            model = self._extract_model_from_payload(json_data)

        if stream:
            base_timeout = first_token_timeout or settings.first_token_timeout
            timeout = get_adaptive_timeout(model, base_timeout)
            max_retries = settings.first_token_max_retries
        else:
            base_timeout = settings.non_stream_timeout
            timeout = get_adaptive_timeout(model, base_timeout)
            max_retries = settings.max_retries

        client = await self._get_client()
        last_error = None

        for attempt in range(max_retries):
            try:
                token = await self.auth_manager.get_access_token()
                headers = self._get_headers(token)

                request_timeout = httpx.Timeout(timeout)

                if stream:
                    req = client.build_request(
                        method, url, json=json_data, headers=headers, timeout=request_timeout
                    )
                    response = await client.send(req, stream=True)
                else:
                    response = await client.request(
                        method, url, json=json_data, headers=headers, timeout=request_timeout
                    )

                if response.status_code == 200:
                    return response

                if response.status_code == 403:
                    logger.warning(f"Received 403, refreshing token (attempt {attempt + 1}/{max_retries})")
                    await response.aclose()
                    await self.auth_manager.force_refresh()
                    continue

                if response.status_code == 429:
                    delay = settings.base_retry_delay * (2 ** attempt)
                    logger.warning(f"Received 429, waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                    await response.aclose()
                    await asyncio.sleep(delay)
                    continue

                if 500 <= response.status_code < 600:
                    delay = settings.base_retry_delay * (2 ** attempt)
                    logger.warning(f"Received {response.status_code}, waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                    await response.aclose()
                    await asyncio.sleep(delay)
                    continue

                return response

            except httpx.TimeoutException as e:
                last_error = e
                if stream:
                    logger.warning(f"First token timeout after {timeout}s for model {model} (attempt {attempt + 1}/{max_retries})")
                else:
                    delay = settings.base_retry_delay * (2 ** attempt)
                    logger.warning(f"Timeout after {timeout}s for model {model}, waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)

            except httpx.RequestError as e:
                last_error = e
                delay = settings.base_retry_delay * (2 ** attempt)
                logger.warning(f"Request error: {e}, waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)

        if stream:
            raise HTTPException(
                status_code=504,
                detail=f"Model did not respond within {timeout}s after {max_retries} attempts."
            )
        else:
            raise HTTPException(
                status_code=502,
                detail=f"Request failed after {max_retries} attempts: {last_error}"
            )

    def _get_headers(self, token: str) -> dict:
        """Build request headers."""
        return get_kiro_headers(self.auth_manager, token)

    async def __aenter__(self) -> "KiroHttpClient":
        """Support async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context without closing global client."""
        pass


async def close_global_http_client():
    """Close global HTTP client (called on app shutdown)."""
    await global_http_client_manager.close()
