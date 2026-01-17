# -*- coding: utf-8 -*-
"""
Utility helper functions.

Contains functions for fingerprint generation, header formatting,
and other common utilities.
"""

import hashlib
import uuid
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from app.libs.auth import KiroAuthManager


def get_machine_fingerprint() -> str:
    """
    Generate unique machine fingerprint based on hostname and username.

    Used for User-Agent formatting to identify specific gateway installation.

    Returns:
        SHA256 hash of string "{hostname}-{username}-kiro-2api"
    """
    try:
        import socket
        import getpass

        hostname = socket.gethostname()
        username = getpass.getuser()
        unique_string = f"{hostname}-{username}-kiro-2api"

        return hashlib.sha256(unique_string.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Failed to get machine fingerprint: {e}")
        return hashlib.sha256(b"default-kiro-2api").hexdigest()


def get_kiro_headers(auth_manager: "KiroAuthManager", token: str) -> dict:
    """
    Build headers for Kiro API requests.

    Includes all necessary headers for authentication and identification:
    - Authorization with Bearer token
    - User-Agent with fingerprint
    - AWS CodeWhisperer specific headers

    Args:
        auth_manager: Authentication manager for fingerprint
        token: Access token for authorization

    Returns:
        Dictionary with HTTP request headers
    """
    fingerprint = auth_manager.fingerprint

    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": f"aws-sdk-js/1.0.27 ua/2.1 os/win32#10.0.19044 lang/js md/nodejs#22.21.1 api/codewhispererstreaming#1.0.27 m/E Kiro2API-{fingerprint[:32]}",
        "x-amz-user-agent": f"aws-sdk-js/1.0.27 Kiro2API-{fingerprint[:32]}",
        "x-amzn-codewhisperer-optout": "true",
        "x-amzn-kiro-agent-mode": "vibe",
        "amz-sdk-invocation-id": str(uuid.uuid4()),
        "amz-sdk-request": "attempt=1; max=3",
    }


def generate_completion_id() -> str:
    """
    Generate unique ID for chat completion.

    Returns:
        ID in format "chatcmpl-{uuid_hex}"
    """
    return f"chatcmpl-{uuid.uuid4().hex}"


def generate_conversation_id() -> str:
    """
    Generate unique ID for conversation.

    Returns:
        UUID in string format
    """
    return str(uuid.uuid4())


def generate_tool_call_id() -> str:
    """
    Generate unique ID for tool call.

    Returns:
        ID in format "call_{uuid_hex[:8]}"
    """
    return f"call_{uuid.uuid4().hex[:8]}"
