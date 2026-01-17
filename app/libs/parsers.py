# -*- coding: utf-8 -*-
"""
AWS Event Stream format parser.

Contains classes and functions for:
- Parsing binary AWS SSE stream
- Extracting JSON events
- Processing tool calls
- Content deduplication
"""

import json
import re
from typing import Any, Dict, List, Optional

from loguru import logger

from app.utils.helpers import generate_tool_call_id


def find_matching_brace(text: str, start_pos: int) -> int:
    """
    Find position of closing brace considering nesting and strings.

    Args:
        text: Text to search
        start_pos: Position of opening brace '{'

    Returns:
        Position of closing brace or -1 if not found
    """
    if start_pos >= len(text) or text[start_pos] != '{':
        return -1

    brace_count = 0
    in_string = False
    escape_next = False

    for i in range(start_pos, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == '\\' and in_string:
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return i

    return -1


def parse_bracket_tool_calls(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse tool calls in format [Called func_name with args: {...}].

    Some models return tool calls in text format instead of
    structured JSON. This function extracts them.

    Args:
        response_text: Model response text

    Returns:
        List of tool calls in OpenAI format
    """
    if not response_text or "[Called" not in response_text:
        return []

    tool_calls = []
    pattern = r'\[Called\s+(\w+)\s+with\s+args:\s*'

    for match in re.finditer(pattern, response_text, re.IGNORECASE):
        func_name = match.group(1)
        args_start = match.end()

        json_start = response_text.find('{', args_start)
        if json_start == -1:
            continue

        json_end = find_matching_brace(response_text, json_start)
        if json_end == -1:
            continue

        json_str = response_text[json_start:json_end + 1]

        try:
            args = json.loads(json_str)
            tool_call_id = generate_tool_call_id()
            tool_calls.append({
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(args)
                }
            })
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tool call arguments: {json_str[:100]}")

    return tool_calls


def deduplicate_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate tool calls.

    Deduplication by two criteria:
    1. By id - keep the one with more arguments
    2. By name+arguments - remove full duplicates

    Args:
        tool_calls: List of tool calls

    Returns:
        List of unique tool calls
    """
    by_id: Dict[str, Dict[str, Any]] = {}
    for tc in tool_calls:
        tc_id = tc.get("id", "")
        if not tc_id:
            continue

        existing = by_id.get(tc_id)
        if existing is None:
            by_id[tc_id] = tc
        else:
            existing_args = existing.get("function", {}).get("arguments", "{}")
            current_args = tc.get("function", {}).get("arguments", "{}")

            if current_args != "{}" and (existing_args == "{}" or len(current_args) > len(existing_args)):
                logger.debug(f"Replacing tool call {tc_id} with better arguments: {len(existing_args)} -> {len(current_args)}")
                by_id[tc_id] = tc

    result_with_id = list(by_id.values())
    result_without_id = [tc for tc in tool_calls if not tc.get("id")]

    seen = set()
    unique = []

    for tc in result_with_id + result_without_id:
        func = tc.get("function") or {}
        func_name = func.get("name") or ""
        func_args = func.get("arguments") or "{}"
        key = f"{func_name}-{func_args}"
        if key not in seen:
            seen.add(key)
            unique.append(tc)

    if len(tool_calls) != len(unique):
        logger.debug(f"Deduplicated tool calls: {len(tool_calls)} -> {len(unique)}")

    return unique


class AwsEventStreamParser:
    """
    Parser for AWS Event Stream format.

    Extracts JSON events from stream and converts to convenient format.

    Supported event types:
    - content: Text content response
    - tool_start: Tool call start (name, toolUseId)
    - tool_input: Tool call input continuation
    - tool_stop: Tool call end
    - usage: Credits consumption info
    - context_usage: Context usage percentage
    """

    _PATTERN_TYPE_MAP = {
        '{"content":': 'content',
        '{"name":': 'tool_start',
        '{"input":': 'tool_input',
        '{"stop":': 'tool_stop',
        '{"followupPrompt":': 'followup',
        '{"usage":': 'usage',
        '{"contextUsagePercentage":': 'context_usage',
    }

    _PATTERN_REGEX = re.compile(
        r'\{"(?:content|name|input|stop|followupPrompt|usage|contextUsagePercentage)":'
    )

    def __init__(self):
        """Initialize parser."""
        self.buffer = ""
        self.last_content: Optional[str] = None
        self.current_tool_call: Optional[Dict[str, Any]] = None
        self.tool_calls: List[Dict[str, Any]] = []

    def feed(self, chunk: bytes) -> List[Dict[str, Any]]:
        """
        Add chunk to buffer and return parsed events.

        Args:
            chunk: Bytes data from stream

        Returns:
            List of events in format {"type": str, "data": Any}
        """
        try:
            self.buffer += chunk.decode('utf-8', errors='ignore')
        except Exception:
            return []

        events = []

        while True:
            match = self._PATTERN_REGEX.search(self.buffer)
            if not match:
                break

            earliest_pos = match.start()
            colon_pos = self.buffer.find(':', earliest_pos)
            if colon_pos == -1:
                break
            pattern_prefix = self.buffer[earliest_pos:colon_pos + 1]
            earliest_type = self._PATTERN_TYPE_MAP.get(pattern_prefix)

            if earliest_type is None:
                self.buffer = self.buffer[earliest_pos + 1:]
                continue

            json_end = find_matching_brace(self.buffer, earliest_pos)
            if json_end == -1:
                break

            json_str = self.buffer[earliest_pos:json_end + 1]
            self.buffer = self.buffer[json_end + 1:]

            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                continue

            event = self._process_event(earliest_type, data)
            if event:
                events.append(event)

        return events

    def _process_event(self, event_type: str, data: dict) -> Optional[Dict[str, Any]]:
        """Process single event based on type."""
        if event_type == 'content':
            content = data.get('content', '')
            if data.get('followupPrompt'):
                return None
            if content and content != self.last_content:
                self.last_content = content
                return {"type": "content", "data": content}

        elif event_type == 'tool_start':
            if self.current_tool_call:
                self._finalize_tool_call()

            input_data = data.get('input', '')
            if isinstance(input_data, dict):
                input_str = json.dumps(input_data)
            else:
                input_str = str(input_data) if input_data else ''

            tool_name = data.get('name', '')
            logger.debug(f"[TOOL_START] name='{tool_name}', initial_input={repr(input_str[:100]) if input_str else 'empty'}")

            self.current_tool_call = {
                "id": data.get('toolUseId', generate_tool_call_id()),
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": input_str
                }
            }

            if data.get('stop'):
                self._finalize_tool_call()

        elif event_type == 'tool_input':
            if self.current_tool_call:
                input_data = data.get('input', '')
                if isinstance(input_data, dict):
                    input_str = json.dumps(input_data)
                else:
                    input_str = str(input_data) if input_data else ''

                current_args = self.current_tool_call['function']['arguments']
                if current_args.strip():
                    try:
                        json.loads(current_args)
                        logger.warning(f"[TOOL_INPUT] IGNORING extra input after valid JSON")
                        return None
                    except json.JSONDecodeError:
                        pass

                self.current_tool_call["function"]["arguments"] += input_str

        elif event_type == 'tool_stop':
            if self.current_tool_call and data.get('stop'):
                self._finalize_tool_call()

        elif event_type == 'usage':
            return {"type": "usage", "data": data.get('usage')}

        elif event_type == 'context_usage':
            return {"type": "context_usage", "data": data.get('contextUsagePercentage')}

        return None

    def _finalize_tool_call(self) -> None:
        """Finalize current tool call and add to list."""
        if not self.current_tool_call:
            return

        args = self.current_tool_call['function']['arguments']
        tool_name = self.current_tool_call['function'].get('name', 'unknown')

        logger.debug(f"Finalizing tool call '{tool_name}' with raw arguments: {repr(args)[:200]}")

        if isinstance(args, str):
            if args.strip():
                try:
                    parsed = json.loads(args)
                    self.current_tool_call['function']['arguments'] = json.dumps(parsed)
                    logger.debug(f"Tool '{tool_name}' arguments parsed successfully")
                except json.JSONDecodeError as e:
                    fixed_args = args.strip()
                    open_braces = fixed_args.count('{')
                    close_braces = fixed_args.count('}')
                    if open_braces > close_braces:
                        fixed_args += '}' * (open_braces - close_braces)
                        logger.info(f"Auto-completing JSON for '{tool_name}': added {open_braces - close_braces} closing brace(s)")
                        try:
                            parsed = json.loads(fixed_args)
                            self.current_tool_call['function']['arguments'] = json.dumps(parsed)
                            logger.info(f"Successfully auto-completed JSON for '{tool_name}'")
                        except json.JSONDecodeError:
                            logger.warning(f"Auto-complete failed for '{tool_name}'")
                            self.current_tool_call['function']['arguments'] = "{}"
                    else:
                        logger.warning(f"Failed to parse tool '{tool_name}' arguments: {e}")
                        self.current_tool_call['function']['arguments'] = "{}"
            else:
                self.current_tool_call['function']['arguments'] = "{}"
        elif isinstance(args, dict):
            self.current_tool_call['function']['arguments'] = json.dumps(args)
        else:
            self.current_tool_call['function']['arguments'] = "{}"

        self.tool_calls.append(self.current_tool_call)
        self.current_tool_call = None

    def get_tool_calls(self) -> List[Dict[str, Any]]:
        """Return collected tool calls, finalizing any pending one."""
        if self.current_tool_call:
            self._finalize_tool_call()
        return deduplicate_tool_calls(self.tool_calls)

    def reset(self) -> None:
        """Reset parser state."""
        self.buffer = ""
        self.last_content = None
        self.current_tool_call = None
        self.tool_calls = []
