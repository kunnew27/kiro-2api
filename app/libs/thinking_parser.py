# -*- coding: utf-8 -*-

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional

from loguru import logger

from app.core.config import (
    FAKE_REASONING_HANDLING,
    FAKE_REASONING_OPEN_TAGS,
    FAKE_REASONING_INITIAL_BUFFER_SIZE,
)


class ParserState(IntEnum):
    PRE_CONTENT = 0
    IN_THINKING = 1
    STREAMING = 2


@dataclass
class ThinkingParseResult:
    thinking_content: Optional[str] = None
    regular_content: Optional[str] = None
    is_first_thinking_chunk: bool = False
    is_last_thinking_chunk: bool = False
    state_changed: bool = False


class ThinkingParser:
    def __init__(
        self,
        handling_mode: Optional[str] = None,
        open_tags: Optional[List[str]] = None,
        initial_buffer_size: int = FAKE_REASONING_INITIAL_BUFFER_SIZE,
    ):
        self.handling_mode = handling_mode or FAKE_REASONING_HANDLING
        self.open_tags = open_tags or FAKE_REASONING_OPEN_TAGS
        self.initial_buffer_size = initial_buffer_size

        self.max_tag_length = max(len(tag) for tag in self.open_tags) * 2

        self.state = ParserState.PRE_CONTENT
        self.initial_buffer = ""
        self.thinking_buffer = ""
        self.open_tag: Optional[str] = None
        self.close_tag: Optional[str] = None
        self.is_first_thinking_chunk = True
        self._thinking_block_found = False

    def feed(self, content: str) -> ThinkingParseResult:
        result = ThinkingParseResult()
        if not content:
            return result

        if self.state == ParserState.PRE_CONTENT:
            result = self._handle_pre_content(content)

        if self.state == ParserState.IN_THINKING and result.state_changed:
            pass
        elif self.state == ParserState.IN_THINKING and not result.state_changed:
            result = self._handle_in_thinking(content)

        if self.state == ParserState.STREAMING and not result.state_changed:
            result.regular_content = content

        return result

    def _handle_pre_content(self, content: str) -> ThinkingParseResult:
        result = ThinkingParseResult()
        self.initial_buffer += content

        stripped = self.initial_buffer.lstrip()

        for tag in self.open_tags:
            if stripped.startswith(tag):
                self.state = ParserState.IN_THINKING
                self.open_tag = tag
                self.close_tag = f"</{tag[1:]}"
                self._thinking_block_found = True
                result.state_changed = True

                logger.debug(f"Thinking tag '{tag}' detected. Transitioning to IN_THINKING.")

                content_after_tag = stripped[len(tag):]
                self.thinking_buffer = content_after_tag
                self.initial_buffer = ""

                thinking_result = self._process_thinking_buffer()
                if thinking_result.thinking_content:
                    result.thinking_content = thinking_result.thinking_content
                    result.is_first_thinking_chunk = thinking_result.is_first_thinking_chunk
                if thinking_result.is_last_thinking_chunk:
                    result.is_last_thinking_chunk = True
                if thinking_result.regular_content:
                    result.regular_content = thinking_result.regular_content

                return result

        for tag in self.open_tags:
            if tag.startswith(stripped) and len(stripped) < len(tag):
                return result

        if len(self.initial_buffer) > self.initial_buffer_size or not self._could_be_tag_prefix(stripped):
            self.state = ParserState.STREAMING
            result.state_changed = True
            result.regular_content = self.initial_buffer
            self.initial_buffer = ""
            logger.debug("No thinking tag detected. Transitioning to STREAMING.")

        return result

    def _could_be_tag_prefix(self, text: str) -> bool:
        if not text:
            return True
        for tag in self.open_tags:
            if tag.startswith(text):
                return True
        return False

    def _handle_in_thinking(self, content: str) -> ThinkingParseResult:
        self.thinking_buffer += content
        return self._process_thinking_buffer()

    def _process_thinking_buffer(self) -> ThinkingParseResult:
        result = ThinkingParseResult()
        if not self.close_tag:
            return result

        if self.close_tag in self.thinking_buffer:
            idx = self.thinking_buffer.find(self.close_tag)
            thinking_content = self.thinking_buffer[:idx]
            after_tag = self.thinking_buffer[idx + len(self.close_tag):]

            if thinking_content:
                result.thinking_content = thinking_content
                result.is_first_thinking_chunk = self.is_first_thinking_chunk
                self.is_first_thinking_chunk = False

            result.is_last_thinking_chunk = True

            self.state = ParserState.STREAMING
            result.state_changed = True
            self.thinking_buffer = ""

            logger.debug(f"Closing tag '{self.close_tag}' found. Transitioning to STREAMING.")

            if after_tag:
                stripped_after = after_tag.lstrip()
                if stripped_after:
                    result.regular_content = stripped_after

            return result

        if len(self.thinking_buffer) > self.max_tag_length:
            send_part = self.thinking_buffer[:-self.max_tag_length]
            self.thinking_buffer = self.thinking_buffer[-self.max_tag_length:]

            result.thinking_content = send_part
            result.is_first_thinking_chunk = self.is_first_thinking_chunk
            self.is_first_thinking_chunk = False

        return result

    def finalize(self) -> ThinkingParseResult:
        result = ThinkingParseResult()

        if self.thinking_buffer:
            if self.state == ParserState.IN_THINKING:
                result.thinking_content = self.thinking_buffer
                result.is_first_thinking_chunk = self.is_first_thinking_chunk
                result.is_last_thinking_chunk = True
                logger.warning("Stream ended while still in thinking block. Flushing remaining content.")
            else:
                result.regular_content = self.thinking_buffer
            self.thinking_buffer = ""

        if self.initial_buffer:
            result.regular_content = (result.regular_content or "") + self.initial_buffer
            self.initial_buffer = ""

        return result

    def reset(self) -> None:
        self.state = ParserState.PRE_CONTENT
        self.initial_buffer = ""
        self.thinking_buffer = ""
        self.open_tag = None
        self.close_tag = None
        self.is_first_thinking_chunk = True
        self._thinking_block_found = False

    @property
    def found_thinking_block(self) -> bool:
        return self._thinking_block_found

    def process_for_output(
        self,
        thinking_content: Optional[str],
        is_first: bool,
        is_last: bool,
    ) -> Optional[str]:
        if not thinking_content:
            return None

        if self.handling_mode == "remove":
            return None

        if self.handling_mode == "pass":
            prefix = self.open_tag if is_first and self.open_tag else ""
            suffix = self.close_tag if is_last and self.close_tag else ""
            return f"{prefix}{thinking_content}{suffix}"

        if self.handling_mode == "strip_tags":
            return thinking_content

        return thinking_content
