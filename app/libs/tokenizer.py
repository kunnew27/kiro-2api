# -*- coding: utf-8 -*-
"""
Fast token counting module.

Uses tiktoken (OpenAI's Rust-based library) for approximate token counting.
The cl100k_base encoding is close to Claude tokenization.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

_encoding = None

CLAUDE_CORRECTION_FACTOR = 1.15


def _get_encoding():
    """
    Lazy initialization of tokenizer.

    Uses cl100k_base - encoding for GPT-4/ChatGPT,
    which is close to Claude tokenization.

    Returns:
        tiktoken.Encoding or None if tiktoken unavailable
    """
    global _encoding
    if _encoding is None:
        try:
            import tiktoken
            _encoding = tiktoken.get_encoding("cl100k_base")
            logger.debug("[Tokenizer] Initialized tiktoken with cl100k_base encoding")
        except ImportError:
            logger.warning(
                "[Tokenizer] tiktoken not installed. "
                "Token counting will use fallback estimation. "
                "Install with: pip install tiktoken"
            )
            _encoding = False
        except Exception as e:
            logger.error(f"[Tokenizer] Failed to initialize tiktoken: {e}")
            _encoding = False
    return _encoding if _encoding else None


def count_tokens(text: str, apply_claude_correction: bool = True) -> int:
    """
    Count number of tokens in text.

    Args:
        text: Text to count tokens
        apply_claude_correction: Apply correction factor for Claude (default True)

    Returns:
        Number of tokens (approximate, with Claude correction)
    """
    if not text:
        return 0

    encoding = _get_encoding()
    if encoding:
        try:
            base_tokens = len(encoding.encode(text))
            if apply_claude_correction:
                return int(base_tokens * CLAUDE_CORRECTION_FACTOR)
            return base_tokens
        except Exception as e:
            logger.warning(f"[Tokenizer] Error encoding text: {e}")

    base_estimate = len(text) // 4 + 1
    if apply_claude_correction:
        return int(base_estimate * CLAUDE_CORRECTION_FACTOR)
    return base_estimate


def count_message_tokens(messages: List[Dict[str, Any]], apply_claude_correction: bool = True) -> int:
    """
    Count tokens in chat message list.

    Args:
        messages: List of messages in OpenAI format
        apply_claude_correction: Apply correction factor for Claude

    Returns:
        Approximate number of tokens (with Claude correction)
    """
    if not messages:
        return 0

    total_tokens = 0

    for message in messages:
        total_tokens += 4

        role = message.get("role", "")
        total_tokens += count_tokens(role, apply_claude_correction=False)

        content = message.get("content")
        if content:
            if isinstance(content, str):
                total_tokens += count_tokens(content, apply_claude_correction=False)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            total_tokens += count_tokens(item.get("text", ""), apply_claude_correction=False)
                        elif item.get("type") == "image_url":
                            total_tokens += 100

        tool_calls = message.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                total_tokens += 4
                func = tc.get("function", {})
                total_tokens += count_tokens(func.get("name", ""), apply_claude_correction=False)
                total_tokens += count_tokens(func.get("arguments", ""), apply_claude_correction=False)

        if message.get("tool_call_id"):
            total_tokens += count_tokens(message["tool_call_id"], apply_claude_correction=False)

    total_tokens += 3

    if apply_claude_correction:
        return int(total_tokens * CLAUDE_CORRECTION_FACTOR)
    return total_tokens


def count_tools_tokens(tools: Optional[List[Dict[str, Any]]], apply_claude_correction: bool = True) -> int:
    """
    Count tokens in tool definitions.

    Args:
        tools: List of tools in OpenAI format
        apply_claude_correction: Apply correction factor for Claude

    Returns:
        Approximate number of tokens (with Claude correction)
    """
    if not tools:
        return 0

    total_tokens = 0

    for tool in tools:
        total_tokens += 4

        if tool.get("type") == "function":
            func = tool.get("function", {})

            total_tokens += count_tokens(func.get("name", ""), apply_claude_correction=False)
            total_tokens += count_tokens(func.get("description", ""), apply_claude_correction=False)

            params = func.get("parameters")
            if params:
                import json
                params_str = json.dumps(params, ensure_ascii=False)
                total_tokens += count_tokens(params_str, apply_claude_correction=False)

    if apply_claude_correction:
        return int(total_tokens * CLAUDE_CORRECTION_FACTOR)
    return total_tokens


def estimate_request_tokens(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    system_prompt: Optional[str] = None
) -> Dict[str, int]:
    """
    Estimate total tokens in request.

    Args:
        messages: List of messages
        tools: List of tools (optional)
        system_prompt: System prompt (optional)

    Returns:
        Dictionary with token breakdown
    """
    messages_tokens = count_message_tokens(messages)
    tools_tokens = count_tools_tokens(tools)
    system_tokens = count_tokens(system_prompt) if system_prompt else 0

    return {
        "messages_tokens": messages_tokens,
        "tools_tokens": tools_tokens,
        "system_tokens": system_tokens,
        "total_tokens": messages_tokens + tools_tokens + system_tokens
    }
