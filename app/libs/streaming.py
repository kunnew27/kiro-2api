# -*- coding: utf-8 -*-
"""
Streaming response processing logic.

Converts Kiro stream to OpenAI format.
Contains generators for:
- Converting AWS SSE to OpenAI SSE
- Forming streaming chunks
- Processing tool calls in stream
- Adaptive timeout handling for slow models
"""

import asyncio
import json
import time
from typing import TYPE_CHECKING, AsyncGenerator, Optional, Dict, Any, List

import httpx
from loguru import logger

from app.libs.parsers import AwsEventStreamParser, parse_bracket_tool_calls, deduplicate_tool_calls
from app.libs.tokenizer import count_tokens, count_message_tokens, count_tools_tokens
from app.utils.helpers import generate_completion_id
from app.core.config import settings, get_adaptive_timeout

if TYPE_CHECKING:
    from app.libs.auth import KiroAuthManager
    from app.libs.cache import ModelInfoCache


class FirstTokenTimeoutError(Exception):
    """Exception raised when first token timeout occurs."""
    pass


class StreamReadTimeoutError(Exception):
    """Exception raised when stream read timeout occurs."""
    pass


async def _read_chunk_with_timeout(byte_iterator, timeout: float) -> bytes:
    """Read a chunk from byte iterator with timeout."""
    try:
        return await asyncio.wait_for(
            byte_iterator.__anext__(),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        raise StreamReadTimeoutError(f"Stream read timeout after {timeout}s")


def _calculate_usage_tokens(
    full_content: str,
    context_usage_percentage: Optional[float],
    model_cache: "ModelInfoCache",
    model: str,
    request_messages: Optional[list],
    request_tools: Optional[list]
) -> Dict[str, Any]:
    """Calculate token usage from response."""
    completion_tokens = count_tokens(full_content)

    total_tokens_from_api = 0
    if context_usage_percentage is not None and context_usage_percentage > 0:
        max_input_tokens = model_cache.get_max_input_tokens(model)
        total_tokens_from_api = int((context_usage_percentage / 100) * max_input_tokens)

    if total_tokens_from_api > 0:
        prompt_tokens = max(0, total_tokens_from_api - completion_tokens)
        total_tokens = total_tokens_from_api
        prompt_source = "subtraction"
        total_source = "API Kiro"
    else:
        prompt_tokens = 0
        if request_messages:
            prompt_tokens += count_message_tokens(request_messages, apply_claude_correction=False)
        if request_tools:
            prompt_tokens += count_tools_tokens(request_tools, apply_claude_correction=False)
        total_tokens = prompt_tokens + completion_tokens
        prompt_source = "tiktoken"
        total_source = "tiktoken"

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_source": prompt_source,
        "total_source": total_source
    }


def _format_tool_calls_for_streaming(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format tool calls for streaming response with required index field."""
    indexed_tool_calls = []
    valid_idx = 0
    for idx, tc in enumerate(tool_calls):
        func = tc.get("function") or {}
        tool_name = func.get("name") or ""
        tool_args = func.get("arguments") or "{}"

        if tool_args == "{}" or tool_args.strip() == "":
            logger.warning(f"Tool call '{tool_name}' has empty arguments")

        if not tool_name:
            logger.warning(f"Dropping tool call with no name at index {idx}")
            continue

        logger.debug(f"Tool call [{valid_idx}] '{tool_name}': id={tc.get('id')}, args_length={len(tool_args)}")

        indexed_tc = {
            "index": valid_idx,
            "id": tc.get("id"),
            "type": tc.get("type", "function"),
            "function": {
                "name": tool_name,
                "arguments": tool_args
            }
        }
        indexed_tool_calls.append(indexed_tc)
        valid_idx += 1

    return indexed_tool_calls


def _format_tool_calls_for_non_streaming(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format tool calls for non-streaming response (without index field)."""
    cleaned_tool_calls = []
    for tc in tool_calls:
        func = tc.get("function") or {}
        tool_name = func.get("name", "")
        tool_args = func.get("arguments", "{}")

        if tool_args == "{}" or tool_args.strip() == "":
            logger.warning(f"Tool call '{tool_name}' has empty arguments")

        if not tool_name:
            logger.warning(f"Dropping tool call with no name")
            continue

        cleaned_tc = {
            "id": tc.get("id"),
            "type": tc.get("type", "function"),
            "function": {
                "name": tool_name,
                "arguments": tool_args
            }
        }
        cleaned_tool_calls.append(cleaned_tc)

    return cleaned_tool_calls


async def stream_kiro_to_openai(
    client: httpx.AsyncClient,
    response: httpx.Response,
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None
) -> AsyncGenerator[str, None]:
    """
    Generator for converting Kiro stream to OpenAI format.

    Parses AWS SSE stream and converts events to OpenAI chat.completion.chunk.
    Supports tool calls and usage calculation.

    Args:
        client: HTTP client (for connection management)
        response: HTTP response with data stream
        model: Model name to include in response
        model_cache: Model cache for token limits
        auth_manager: Authentication manager
        request_messages: Original request messages (for fallback token counting)
        request_tools: Original request tools (for fallback token counting)

    Yields:
        Strings in SSE format: "data: {...}\\n\\n" or "data: [DONE]\\n\\n"
    """
    completion_id = generate_completion_id()
    created_time = int(time.time())
    first_chunk = True

    parser = AwsEventStreamParser()
    metering_data = None
    context_usage_percentage = None
    content_parts: list[str] = []

    adaptive_first_token_timeout = get_adaptive_timeout(model, settings.first_token_timeout)
    adaptive_stream_read_timeout = get_adaptive_timeout(model, settings.stream_read_timeout)

    try:
        byte_iterator = response.aiter_bytes()

        try:
            first_byte_chunk = await asyncio.wait_for(
                byte_iterator.__anext__(),
                timeout=adaptive_first_token_timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"First token timeout after {adaptive_first_token_timeout}s (model: {model})")
            raise FirstTokenTimeoutError(f"No response within {adaptive_first_token_timeout}s")
        except StopAsyncIteration:
            logger.debug("Empty response from Kiro API")
            yield "data: [DONE]\n\n"
            return

        events = parser.feed(first_byte_chunk)
        for event in events:
            if event["type"] == "content":
                content = event["data"]
                content_parts.append(content)

                delta = {"content": content}
                if first_chunk:
                    delta["role"] = "assistant"
                    first_chunk = False

                openai_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}]
                }

                yield f"data: {json.dumps(openai_chunk, ensure_ascii=False)}\n\n"

            elif event["type"] == "usage":
                metering_data = event["data"]

            elif event["type"] == "context_usage":
                context_usage_percentage = event["data"]

        consecutive_timeouts = 0
        max_consecutive_timeouts = 3
        while True:
            try:
                chunk = await _read_chunk_with_timeout(byte_iterator, adaptive_stream_read_timeout)
                consecutive_timeouts = 0
            except StopAsyncIteration:
                break
            except StreamReadTimeoutError as e:
                consecutive_timeouts += 1
                if consecutive_timeouts <= max_consecutive_timeouts:
                    logger.warning(
                        f"Stream read timeout {consecutive_timeouts}/{max_consecutive_timeouts} "
                        f"after {adaptive_stream_read_timeout}s (model: {model}). "
                        f"Model may be processing large content - continuing to wait..."
                    )
                    continue
                else:
                    logger.error(f"Stream read timeout after {max_consecutive_timeouts} consecutive timeouts (model: {model}): {e}")
                    raise

            events = parser.feed(chunk)

            for event in events:
                if event["type"] == "content":
                    content = event["data"]
                    content_parts.append(content)

                    delta = {"content": content}
                    if first_chunk:
                        delta["role"] = "assistant"
                        first_chunk = False

                    openai_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}]
                    }

                    yield f"data: {json.dumps(openai_chunk, ensure_ascii=False)}\n\n"

                elif event["type"] == "usage":
                    metering_data = event["data"]

                elif event["type"] == "context_usage":
                    context_usage_percentage = event["data"]

        full_content = ''.join(content_parts)

        bracket_tool_calls = parse_bracket_tool_calls(full_content)
        all_tool_calls = parser.get_tool_calls() + bracket_tool_calls
        all_tool_calls = deduplicate_tool_calls(all_tool_calls)

        finish_reason = "tool_calls" if all_tool_calls else "stop"

        usage_info = _calculate_usage_tokens(
            full_content, context_usage_percentage, model_cache, model,
            request_messages, request_tools
        )

        if all_tool_calls:
            logger.debug(f"Processing {len(all_tool_calls)} tool calls for streaming response")
            indexed_tool_calls = _format_tool_calls_for_streaming(all_tool_calls)

            tool_calls_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"tool_calls": indexed_tool_calls},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(tool_calls_chunk, ensure_ascii=False)}\n\n"

        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": usage_info["prompt_tokens"],
                "completion_tokens": usage_info["completion_tokens"],
                "total_tokens": usage_info["total_tokens"],
            }
        }

        if metering_data:
            final_chunk["usage"]["credits_used"] = metering_data

        logger.debug(
            f"[Usage] {model}: "
            f"prompt_tokens={usage_info['prompt_tokens']} ({usage_info['prompt_source']}), "
            f"completion_tokens={usage_info['completion_tokens']}, "
            f"total_tokens={usage_info['total_tokens']} ({usage_info['total_source']})"
        )

        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    except FirstTokenTimeoutError:
        raise
    except StreamReadTimeoutError:
        raise
    except Exception as e:
        logger.error(f"Error during streaming: {e}", exc_info=True)
    finally:
        await response.aclose()
        logger.debug("Streaming completed")


async def collect_stream_response(
    client: httpx.AsyncClient,
    response: httpx.Response,
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None
) -> Dict[str, Any]:
    """
    Collect streaming response into non-streaming format.

    Args:
        client: HTTP client
        response: HTTP response with data stream
        model: Model name
        model_cache: Model cache for token limits
        auth_manager: Authentication manager
        request_messages: Original request messages
        request_tools: Original request tools

    Returns:
        Complete OpenAI chat completion response
    """
    completion_id = generate_completion_id()
    created_time = int(time.time())

    parser = AwsEventStreamParser()
    metering_data = None
    context_usage_percentage = None
    content_parts: list[str] = []

    adaptive_stream_read_timeout = get_adaptive_timeout(model, settings.stream_read_timeout)

    try:
        async for chunk in response.aiter_bytes():
            events = parser.feed(chunk)

            for event in events:
                if event["type"] == "content":
                    content_parts.append(event["data"])
                elif event["type"] == "usage":
                    metering_data = event["data"]
                elif event["type"] == "context_usage":
                    context_usage_percentage = event["data"]

    finally:
        await response.aclose()

    full_content = ''.join(content_parts)

    bracket_tool_calls = parse_bracket_tool_calls(full_content)
    all_tool_calls = parser.get_tool_calls() + bracket_tool_calls
    all_tool_calls = deduplicate_tool_calls(all_tool_calls)

    finish_reason = "tool_calls" if all_tool_calls else "stop"

    usage_info = _calculate_usage_tokens(
        full_content, context_usage_percentage, model_cache, model,
        request_messages, request_tools
    )

    message = {
        "role": "assistant",
        "content": full_content if not all_tool_calls else None,
    }

    if all_tool_calls:
        message["tool_calls"] = _format_tool_calls_for_non_streaming(all_tool_calls)

    response_obj = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_time,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": usage_info["prompt_tokens"],
            "completion_tokens": usage_info["completion_tokens"],
            "total_tokens": usage_info["total_tokens"],
        }
    }

    if metering_data:
        response_obj["usage"]["credits_used"] = metering_data

    return response_obj
