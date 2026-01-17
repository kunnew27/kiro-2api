# -*- coding: utf-8 -*-
"""Chat completions routes."""

import json
import secrets
import time
from typing import Union

from fastapi import APIRouter, Depends, HTTPException, Request, Security
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from loguru import logger

from app.core.config import settings
from app.libs.auth import KiroAuthManager
from app.libs.cache import ModelInfoCache
from app.libs.http_client import KiroHttpClient
from app.libs.converters import build_kiro_payload
from app.libs.streaming import stream_kiro_to_openai, collect_stream_response
from app.models.schemas import ChatCompletionRequest
from app.utils.helpers import generate_conversation_id
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


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    request_data: ChatCompletionRequest,
    auth_manager: KiroAuthManager = Depends(verify_api_key)
):
    """
    Chat completions endpoint - OpenAI API compatible.

    Accepts OpenAI format requests and converts to Kiro API.
    Supports streaming and non-streaming modes.

    Args:
        request: FastAPI Request
        request_data: OpenAI ChatCompletionRequest format
        auth_manager: KiroAuthManager instance

    Returns:
        StreamingResponse for streaming mode
        JSONResponse for non-streaming mode

    Raises:
        HTTPException: On validation or API errors
    """
    logger.info(f"[{get_timestamp()}] Received /v1/chat/completions request (model={request_data.model}, stream={request_data.stream})")

    request.state.auth_manager = auth_manager
    request.state.model = request_data.model

    model_cache: ModelInfoCache = request.app.state.model_cache

    conversation_id = generate_conversation_id()

    try:
        kiro_payload = build_kiro_payload(
            request_data,
            conversation_id,
            auth_manager.profile_arn or ""
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    http_client = KiroHttpClient(auth_manager)
    url = f"{auth_manager.api_host}/generateAssistantResponse"

    try:
        response = await http_client.request_with_retry(
            "POST",
            url,
            kiro_payload,
            stream=True,
            model=request_data.model
        )

        if response.status_code != 200:
            try:
                error_content = await response.aread()
            except Exception:
                error_content = b"Unknown error"
            finally:
                try:
                    await response.aclose()
                except Exception:
                    pass

            await http_client.close()
            error_text = error_content.decode('utf-8', errors='replace')
            logger.error(f"Error from Kiro API: {response.status_code} - {error_text}")

            error_message = error_text
            try:
                error_json = json.loads(error_text)
                if isinstance(error_json, dict):
                    if "message" in error_json:
                        error_message = error_json["message"]
                    elif "error" in error_json and isinstance(error_json["error"], dict):
                        if "message" in error_json["error"]:
                            error_message = error_json["error"]["message"]
            except (json.JSONDecodeError, KeyError):
                pass

            return JSONResponse(
                status_code=response.status_code,
                content={
                    "error": {
                        "message": error_message,
                        "type": "kiro_api_error",
                        "code": response.status_code
                    }
                }
            )

        messages_for_tokenizer = [msg.model_dump() for msg in request_data.messages]
        tools_for_tokenizer = [tool.model_dump() for tool in request_data.tools] if request_data.tools else None

        if request_data.stream:
            async def stream_wrapper():
                try:
                    async for chunk in stream_kiro_to_openai(
                        http_client.client,
                        response,
                        request_data.model,
                        model_cache,
                        auth_manager,
                        request_messages=messages_for_tokenizer,
                        request_tools=tools_for_tokenizer
                    ):
                        yield chunk
                finally:
                    await http_client.close()
                    logger.info(f"[{get_timestamp()}] HTTP 200 - POST /v1/chat/completions (streaming) - completed")

            return StreamingResponse(stream_wrapper(), media_type="text/event-stream")
        else:
            collected_response = await collect_stream_response(
                http_client.client,
                response,
                request_data.model,
                model_cache,
                auth_manager,
                request_messages=messages_for_tokenizer,
                request_tools=tools_for_tokenizer
            )

            await http_client.close()
            logger.info(f"[{get_timestamp()}] HTTP 200 - POST /v1/chat/completions (non-streaming) - completed")

            return JSONResponse(content=collected_response)

    except HTTPException:
        await http_client.close()
        raise
    except Exception as e:
        await http_client.close()
        error_msg = str(e) if str(e) else f"{type(e).__name__}: {repr(e)}"
        logger.error(f"Internal error: {error_msg}", exc_info=True)
        if settings.debug_mode == "off":
            detail = "Internal server error"
        else:
            detail = f"Internal server error: {error_msg}"
        raise HTTPException(status_code=500, detail=detail)
