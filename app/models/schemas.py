# -*- coding: utf-8 -*-
"""
OpenAI Compatible API Pydantic Models.

Defines request and response data schemas with validation and serialization.
"""

import time
from typing import Any, Dict, List, Optional, Union
from typing_extensions import Annotated
from pydantic import BaseModel, Field, field_validator


# ==================================================================================================
# /v1/models endpoint models
# ==================================================================================================

class OpenAIModel(BaseModel):
    """OpenAI format AI model description for /v1/models endpoint."""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "anthropic"
    description: Optional[str] = None


class ModelList(BaseModel):
    """OpenAI format model list for GET /v1/models endpoint."""
    object: str = "list"
    data: List[OpenAIModel]


# ==================================================================================================
# /v1/chat/completions endpoint models
# ==================================================================================================

class ChatMessage(BaseModel):
    """
    OpenAI format chat message.

    Supports multiple roles (user, assistant, system, tool) and
    multiple content formats (string, list, object).

    Attributes:
        role: Sender role (user, assistant, system, tool)
        content: Message content (can be string, list, or None)
        name: Optional sender name
        tool_calls: Tool calls list (for assistant)
        tool_call_id: Tool call ID (for tool)
    """
    role: str
    content: Optional[Union[str, List[Any], Any]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Any]] = None
    tool_call_id: Optional[str] = None

    model_config = {"extra": "allow"}


class ToolFunction(BaseModel):
    """
    Tool function description.

    Attributes:
        name: Function name
        description: Function description
        parameters: Function parameters JSON Schema
    """
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    """
    OpenAI format tool.

    Attributes:
        type: Tool type (usually "function")
        function: Function description
    """
    type: str = "function"
    function: ToolFunction


class ChatCompletionRequest(BaseModel):
    """
    OpenAI Chat Completions API format request.

    Supports all standard OpenAI API fields including:
    - Basic parameters (model, messages, stream)
    - Generation parameters (temperature, top_p, max_tokens)
    - Tool calling (function calling)
    - Compatibility parameters (accepted but ignored)

    Attributes:
        model: Generation model ID
        messages: Chat message list
        stream: Whether to use streaming response (default False)
        temperature: Generation temperature (0-2)
        top_p: Top-p sampling
        n: Number of response variants
        max_tokens: Maximum response tokens
        max_completion_tokens: Alternative field for max_tokens
        stop: Stop sequences
        presence_penalty: Topic repetition penalty
        frequency_penalty: Word repetition penalty
        tools: Available tools list
        tool_choice: Tool selection strategy
    """
    model: str
    messages: Annotated[List[ChatMessage], Field(min_length=1)]
    stream: bool = False

    # Generation parameters
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None

    # Tool calling
    tools: Optional[List[Union[Tool, Dict[str, Any]]]] = None
    tool_choice: Optional[Union[str, Dict]] = None

    # Compatibility fields (ignored)
    stream_options: Optional[Dict[str, Any]] = None
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    user: Optional[str] = None
    seed: Optional[int] = None
    parallel_tool_calls: Optional[bool] = None

    model_config = {"extra": "allow"}

    @field_validator('tools', mode='before')
    @classmethod
    def convert_anthropic_tools(cls, v):
        """
        Auto-convert Anthropic format tools to OpenAI format.

        If Anthropic format detected (has input_schema field),
        auto-convert to OpenAI format (function.parameters).
        """
        if v is None:
            return v

        converted_tools = []
        for tool in v:
            if isinstance(tool, Tool):
                converted_tools.append(tool)
                continue

            if isinstance(tool, dict):
                if 'input_schema' in tool and 'name' in tool:
                    converted_tool = Tool(
                        type='function',
                        function=ToolFunction(
                            name=tool['name'],
                            description=tool.get('description'),
                            parameters=tool['input_schema']
                        )
                    )
                    converted_tools.append(converted_tool)
                elif 'function' in tool:
                    converted_tools.append(tool)
                else:
                    converted_tools.append(tool)
            else:
                converted_tools.append(tool)

        return converted_tools


# ==================================================================================================
# Response models
# ==================================================================================================

class ChatCompletionChoice(BaseModel):
    """
    Chat Completion single response choice.

    Attributes:
        index: Choice index
        message: Response message
        finish_reason: Completion reason (stop, tool_calls, length)
    """
    index: int = 0
    message: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionUsage(BaseModel):
    """
    Token usage information.

    Attributes:
        prompt_tokens: Request token count
        completion_tokens: Response token count
        total_tokens: Total token count
        credits_used: Credits used (Kiro specific)
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    credits_used: Optional[float] = None


class ChatCompletionResponse(BaseModel):
    """
    Chat Completion complete response (non-streaming).

    Attributes:
        id: Response unique ID
        object: Object type ("chat.completion")
        created: Creation timestamp
        model: Model used
        choices: Response choices list
        usage: Token usage info
    """
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ChatCompletionChunkDelta(BaseModel):
    """
    Streaming chunk delta change.

    Attributes:
        role: Role (only in first chunk)
        content: New content
        tool_calls: New tool calls
    """
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionChunkChoice(BaseModel):
    """
    Single choice in streaming chunk.

    Attributes:
        index: Choice index
        delta: Delta change
        finish_reason: Completion reason (only in last chunk)
    """
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """
    OpenAI format streaming chunk.

    Attributes:
        id: Response unique ID
        object: Object type ("chat.completion.chunk")
        created: Creation timestamp
        model: Model used
        choices: Choices list
        usage: Usage info (only in last chunk)
    """
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChunkChoice]
    usage: Optional[ChatCompletionUsage] = None
