# -*- coding: utf-8 -*-
"""
OpenAI <-> Kiro format converters.

Contains functions for:
- Extracting text content from various formats
- Merging adjacent messages
- Building conversation history for Kiro API
- Assembling complete request payload
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from app.core.config import get_internal_model_id, settings
from app.models.schemas import (
    ChatMessage,
    ChatCompletionRequest,
    Tool,
    ToolFunction,
)


def extract_text_content(content: Any) -> str:
    """
    Extract text content from various formats.

    OpenAI API supports multiple content formats:
    - String: "Hello, world!"
    - List: [{"type": "text", "text": "Hello"}]
    - None: empty message

    Args:
        content: Content in any supported format

    Returns:
        Extracted text or empty string
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif "text" in item:
                    text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        return "".join(text_parts)
    return str(content)


def extract_images_from_content(content: Any) -> List[Dict[str, Any]]:
    """
    Extract images from message content in unified format.

    Supports multiple image formats:
    - OpenAI format: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    - Anthropic format: {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}}

    Args:
        content: Content (usually a list of content blocks)

    Returns:
        List of images in unified format: [{"media_type": "image/jpeg", "data": "base64..."}]
    """
    images: List[Dict[str, Any]] = []

    if not isinstance(content, list):
        return images

    for item in content:
        if isinstance(item, dict):
            item_type = item.get("type")
        elif hasattr(item, "type"):
            item_type = item.type
        else:
            continue

        # OpenAI format: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        if item_type == "image_url":
            if isinstance(item, dict):
                image_url_obj = item.get("image_url", {})
            else:
                image_url_obj = getattr(item, "image_url", {})

            if isinstance(image_url_obj, dict):
                url = image_url_obj.get("url", "")
            elif hasattr(image_url_obj, "url"):
                url = image_url_obj.url
            else:
                url = ""

            if url.startswith("data:"):
                try:
                    header, data = url.split(",", 1)
                    media_part = header.split(";")[0]
                    media_type = media_part.replace("data:", "")

                    if data:
                        images.append({
                            "media_type": media_type,
                            "data": data
                        })
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse image data URL: {e}")
            elif url.startswith("http"):
                logger.warning(f"URL-based images are not supported, skipping: {url[:80]}...")

        # Anthropic format: {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
        elif item_type == "image":
            source = item.get("source", {}) if isinstance(item, dict) else getattr(item, "source", None)

            if source is None:
                continue

            if isinstance(source, dict):
                source_type = source.get("type")

                if source_type == "base64":
                    media_type = source.get("media_type", "image/jpeg")
                    data = source.get("data", "")

                    if data:
                        images.append({
                            "media_type": media_type,
                            "data": data
                        })
                elif source_type == "url":
                    url = source.get("url", "")
                    logger.warning(f"URL-based images are not supported, skipping: {url[:80]}...")

            elif hasattr(source, "type"):
                if source.type == "base64":
                    media_type = getattr(source, "media_type", "image/jpeg")
                    data = getattr(source, "data", "")

                    if data:
                        images.append({
                            "media_type": media_type,
                            "data": data
                        })
                elif source.type == "url":
                    url = getattr(source, "url", "")
                    logger.warning(f"URL-based images are not supported, skipping: {url[:80]}...")

    if images:
        logger.debug(f"Extracted {len(images)} image(s) from content")

    return images


def convert_images_to_kiro_format(images: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Convert unified images to Kiro API format.

    Unified format: [{"media_type": "image/jpeg", "data": "base64..."}]
    Kiro format: [{"format": "jpeg", "source": {"bytes": "base64..."}}]

    Args:
        images: List of images in unified format

    Returns:
        List of images in Kiro format
    """
    if not images:
        return []

    kiro_images = []
    for img in images:
        media_type = img.get("media_type", "image/jpeg")
        data = img.get("data", "")

        if not data:
            logger.warning("Skipping image with empty data")
            continue

        if data.startswith("data:"):
            try:
                header, actual_data = data.split(",", 1)
                media_part = header.split(";")[0]
                extracted_media_type = media_part.replace("data:", "")
                if extracted_media_type:
                    media_type = extracted_media_type
                data = actual_data
                logger.debug(f"Stripped data URL prefix, extracted media_type: {media_type}")
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse data URL prefix: {e}")

        format_str = media_type.split("/")[-1] if "/" in media_type else media_type

        kiro_images.append({
            "format": format_str,
            "source": {
                "bytes": data
            }
        })

    if kiro_images:
        logger.debug(f"Converted {len(kiro_images)} image(s) to Kiro format")

    return kiro_images


def merge_adjacent_messages(messages: List[ChatMessage]) -> List[ChatMessage]:
    """
    Merge adjacent messages with same role and process tool messages.

    Kiro API doesn't accept multiple consecutive messages from same role.
    This function merges such messages into one.

    Tool messages (role="tool") are converted to user messages with tool_results.

    Args:
        messages: List of messages

    Returns:
        List of messages with merged adjacent messages
    """
    if not messages:
        return []

    processed = []
    pending_tool_results = []

    for msg in messages:
        if msg.role == "tool":
            tool_result = {
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id or "",
                "content": extract_text_content(msg.content) or "(empty result)"
            }
            pending_tool_results.append(tool_result)
            logger.debug(f"Collected tool result for tool_call_id={msg.tool_call_id}")
        else:
            if pending_tool_results:
                tool_results_msg = ChatMessage(
                    role="user",
                    content=pending_tool_results.copy()
                )
                processed.append(tool_results_msg)
                pending_tool_results.clear()
                logger.debug(f"Created user message with {len(tool_results_msg.content)} tool results")

            processed.append(msg)

    if pending_tool_results:
        tool_results_msg = ChatMessage(
            role="user",
            content=pending_tool_results.copy()
        )
        processed.append(tool_results_msg)
        logger.debug(f"Created final user message with {len(pending_tool_results)} tool results")

    merged = []
    for msg in processed:
        if not merged:
            merged.append(msg)
            continue

        last = merged[-1]
        if msg.role == last.role:
            if isinstance(last.content, list) and isinstance(msg.content, list):
                last.content = last.content + msg.content
            elif isinstance(last.content, list):
                last.content = last.content + [{"type": "text", "text": extract_text_content(msg.content)}]
            elif isinstance(msg.content, list):
                last.content = [{"type": "text", "text": extract_text_content(last.content)}] + msg.content
            else:
                last_text = extract_text_content(last.content)
                current_text = extract_text_content(msg.content)
                last.content = f"{last_text}\n{current_text}"

            if msg.role == "assistant" and msg.tool_calls:
                if last.tool_calls is None:
                    last.tool_calls = []
                last.tool_calls = list(last.tool_calls) + list(msg.tool_calls)
                logger.debug(f"Merged tool_calls: added {len(msg.tool_calls)} tool calls, total now: {len(last.tool_calls)}")

            logger.debug(f"Merged adjacent messages with role {msg.role}")
        else:
            merged.append(msg)

    return merged


def build_kiro_history(messages: List[ChatMessage], model_id: str) -> List[Dict[str, Any]]:
    """
    Build history array for Kiro API from OpenAI messages.

    Kiro API expects alternating userInputMessage and assistantResponseMessage.
    This function converts OpenAI format to Kiro format, including image support.

    Args:
        messages: List of messages in OpenAI format
        model_id: Internal Kiro model ID

    Returns:
        List of dictionaries for history field in Kiro API
    """
    history = []

    for msg in messages:
        if msg.role == "user":
            content = extract_text_content(msg.content)

            user_input = {
                "content": content,
                "modelId": model_id,
                "origin": "AI_EDITOR",
            }

            images = extract_images_from_content(msg.content)
            if images:
                kiro_images = convert_images_to_kiro_format(images)
                if kiro_images:
                    user_input["images"] = kiro_images
                    logger.debug(f"Added {len(kiro_images)} image(s) to user message in history")

            tool_results = _extract_tool_results(msg.content)
            if tool_results:
                user_input["userInputMessageContext"] = {"toolResults": tool_results}

            history.append({"userInputMessage": user_input})

        elif msg.role == "assistant":
            content = extract_text_content(msg.content)

            assistant_response = {"content": content}

            tool_uses = _extract_tool_uses(msg)
            if tool_uses:
                assistant_response["toolUses"] = tool_uses

            history.append({"assistantResponseMessage": assistant_response})

        elif msg.role == "system":
            pass

    return history


def _extract_tool_results(content: Any) -> List[Dict[str, Any]]:
    """Extract tool results from message content."""
    tool_results = []

    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "tool_result":
                tool_results.append({
                    "content": [{"text": extract_text_content(item.get("content", ""))}],
                    "status": "success",
                    "toolUseId": item.get("tool_use_id", "")
                })

    return tool_results


def _extract_tool_uses(msg: ChatMessage) -> List[Dict[str, Any]]:
    """Extract tool uses from assistant message."""
    tool_uses = []

    if msg.tool_calls:
        for tc in msg.tool_calls:
            if isinstance(tc, dict):
                tool_uses.append({
                    "name": tc.get("function", {}).get("name", ""),
                    "input": json.loads(tc.get("function", {}).get("arguments", "{}")),
                    "toolUseId": tc.get("id", "")
                })

    if isinstance(msg.content, list):
        for item in msg.content:
            if isinstance(item, dict) and item.get("type") == "tool_use":
                tool_uses.append({
                    "name": item.get("name", ""),
                    "input": item.get("input", {}),
                    "toolUseId": item.get("id", "")
                })

    return tool_uses


def process_tools_with_long_descriptions(
    tools: Optional[List[Tool]]
) -> Tuple[Optional[List[Tool]], str]:
    """
    Process tools with long descriptions.

    Kiro API has a limit on description length in toolSpecification.
    If description exceeds limit, full description is moved to system prompt,
    and tool keeps a reference to documentation.

    Args:
        tools: List of tools from OpenAI request

    Returns:
        Tuple of:
        - List of tools with processed descriptions (or None if tools is empty)
        - String with documentation to add to system prompt
    """
    if not tools:
        return None, ""

    if settings.tool_description_max_length <= 0:
        return tools, ""

    tool_documentation_parts = []
    processed_tools = []

    for tool in tools:
        if tool.type != "function":
            processed_tools.append(tool)
            continue

        description = tool.function.description or ""

        if len(description) <= settings.tool_description_max_length:
            processed_tools.append(tool)
        else:
            tool_name = tool.function.name

            logger.debug(
                f"Tool '{tool_name}' has long description ({len(description)} chars > {settings.tool_description_max_length}), "
                f"moving to system prompt"
            )

            tool_documentation_parts.append(f"## Tool: {tool_name}\n\n{description}")

            reference_description = f"[Full documentation in system prompt under '## Tool: {tool_name}']"

            processed_tool = Tool(
                type=tool.type,
                function=ToolFunction(
                    name=tool.function.name,
                    description=reference_description,
                    parameters=tool.function.parameters
                )
            )
            processed_tools.append(processed_tool)

    tool_documentation = ""
    if tool_documentation_parts:
        tool_documentation = (
            "\n\n---\n"
            "# Tool Documentation\n"
            "The following tools have detailed documentation that couldn't fit in the tool definition.\n\n"
            + "\n\n---\n\n".join(tool_documentation_parts)
        )

    return processed_tools if processed_tools else None, tool_documentation


def _extract_system_and_tool_docs(
    messages: List[ChatMessage],
    tools: Optional[List[Tool]]
) -> Tuple[str, List[ChatMessage], Optional[List[Tool]]]:
    """Extract system prompt and tool documentation."""
    processed_tools, tool_documentation = process_tools_with_long_descriptions(tools)

    system_prompt = ""
    non_system_messages = []
    for msg in messages:
        if msg.role == "system":
            system_prompt += extract_text_content(msg.content) + "\n"
        else:
            non_system_messages.append(msg)
    system_prompt = system_prompt.strip()

    if tool_documentation:
        system_prompt = system_prompt + tool_documentation if system_prompt else tool_documentation.strip()

    return system_prompt, non_system_messages, processed_tools


def build_kiro_payload(
    request_data: ChatCompletionRequest,
    conversation_id: str,
    profile_arn: str
) -> dict:
    """
    Build complete payload for Kiro API.

    Includes:
    - Complete message history
    - System prompt (added to first user message)
    - Tools definitions (with long description processing)
    - Current message

    Args:
        request_data: Request in OpenAI format
        conversation_id: Unique conversation ID
        profile_arn: AWS CodeWhisperer profile ARN

    Returns:
        Dictionary payload for POST request to Kiro API

    Raises:
        ValueError: If no messages to send
    """
    messages = list(request_data.messages)

    system_prompt, non_system_messages, processed_tools = _extract_system_and_tool_docs(
        messages, request_data.tools
    )

    merged_messages = merge_adjacent_messages(non_system_messages)

    if not merged_messages:
        raise ValueError("No messages to send")

    model_id = get_internal_model_id(request_data.model)

    history_messages = merged_messages[:-1] if len(merged_messages) > 1 else []

    if system_prompt and history_messages:
        first_msg = history_messages[0]
        if first_msg.role == "user":
            original_content = extract_text_content(first_msg.content)
            first_msg.content = f"{system_prompt}\n\n{original_content}"

    history = build_kiro_history(history_messages, model_id)

    current_message = merged_messages[-1]
    current_content = extract_text_content(current_message.content)

    if system_prompt and not history:
        current_content = f"{system_prompt}\n\n{current_content}"

    if current_message.role == "assistant":
        history.append({
            "assistantResponseMessage": {
                "content": current_content
            }
        })
        current_content = "Continue"

    if not current_content:
        current_content = "Continue"

    user_input_message = {
        "content": current_content,
        "modelId": model_id,
        "origin": "AI_EDITOR",
    }

    if current_message.role != "assistant":
        current_images = extract_images_from_content(current_message.content)
        if current_images:
            kiro_images = convert_images_to_kiro_format(current_images)
            if kiro_images:
                user_input_message["images"] = kiro_images
                logger.debug(f"Added {len(kiro_images)} image(s) to current message")

    user_input_context = _build_user_input_context(request_data, current_message, processed_tools)
    if user_input_context:
        user_input_message["userInputMessageContext"] = user_input_context

    payload = {
        "conversationState": {
            "chatTriggerType": "MANUAL",
            "conversationId": conversation_id,
            "currentMessage": {
                "userInputMessage": user_input_message
            }
        }
    }

    if history:
        payload["conversationState"]["history"] = history

    if profile_arn:
        payload["profileArn"] = profile_arn

    return payload


def _build_user_input_context(
    request_data: ChatCompletionRequest,
    current_message: ChatMessage,
    processed_tools: Optional[List[Tool]] = None
) -> Dict[str, Any]:
    """Build userInputMessageContext for current message."""
    context = {}

    tools_to_use = processed_tools if processed_tools is not None else request_data.tools

    if tools_to_use:
        tools_list = []
        for tool in tools_to_use:
            if tool.type == "function":
                tools_list.append({
                    "toolSpecification": {
                        "name": tool.function.name,
                        "description": tool.function.description or "",
                        "inputSchema": {"json": tool.function.parameters or {}}
                    }
                })
        if tools_list:
            context["tools"] = tools_list

    tool_results = _extract_tool_results(current_message.content)
    if tool_results:
        context["toolResults"] = tool_results

    return context
