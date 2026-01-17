# -*- coding: utf-8 -*-
"""
Kiro-2API Configuration Module.

Centralized configuration management using Pydantic Settings.
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_raw_env_value(var_name: str, env_file: str = ".env") -> Optional[str]:
    """Read raw variable value from .env file without processing escape sequences."""
    env_path = Path(env_file)
    if not env_path.exists():
        return None

    try:
        content = env_path.read_text(encoding="utf-8")
        pattern = rf'^{re.escape(var_name)}=(["\']?)(.+?)\1\s*$'

        for line in content.splitlines():
            line = line.strip()
            if line.startswith("#") or not line:
                continue

            match = re.match(pattern, line)
            if match:
                return match.group(2)
    except (FileNotFoundError, PermissionError, OSError):
        pass
    except (re.error, ValueError):
        pass

    return None


class Settings(BaseSettings):
    """Application configuration using Pydantic Settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Proxy Server Settings
    proxy_api_key: str = Field(default="changeme_proxy_secret", alias="PROXY_API_KEY")
    port: int = Field(default=8000, alias="PORT")

    # Kiro API Credentials
    refresh_token: str = Field(default="", alias="REFRESH_TOKEN")
    profile_arn: str = Field(default="", alias="PROFILE_ARN")
    region: str = Field(default="us-east-1", alias="KIRO_REGION")
    kiro_creds_file: str = Field(default="", alias="KIRO_CREDS_FILE")

    # Token Settings
    token_refresh_threshold: int = Field(default=600)

    # Retry Configuration
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    base_retry_delay: float = Field(default=1.0, alias="BASE_RETRY_DELAY")

    # Model Cache Settings
    model_cache_ttl: int = Field(default=3600, alias="MODEL_CACHE_TTL")
    default_max_input_tokens: int = Field(default=200000)

    # Tool Description Processing
    tool_description_max_length: int = Field(default=10000, alias="TOOL_DESCRIPTION_MAX_LENGTH")

    # Logging Settings
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Timeout Settings
    first_token_timeout: float = Field(default=120.0, alias="FIRST_TOKEN_TIMEOUT")
    first_token_max_retries: int = Field(default=3, alias="FIRST_TOKEN_MAX_RETRIES")
    stream_read_timeout: float = Field(default=300.0, alias="STREAM_READ_TIMEOUT")
    non_stream_timeout: float = Field(default=900.0, alias="NON_STREAM_TIMEOUT")

    # Debug Settings
    debug_mode: str = Field(default="off", alias="DEBUG_MODE")
    debug_dir: str = Field(default="debug_logs", alias="DEBUG_DIR")

    # Rate Limiting
    rate_limit_per_minute: int = Field(default=0, alias="RATE_LIMIT_PER_MINUTE")

    # Slow Model Configuration
    slow_model_timeout_multiplier: float = Field(default=3.0, alias="SLOW_MODEL_TIMEOUT_MULTIPLIER")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in valid_levels:
            return "INFO"
        return v

    @field_validator("debug_mode")
    @classmethod
    def validate_debug_mode(cls, v: str) -> str:
        valid_modes = {"off", "errors", "all"}
        v = v.lower()
        if v not in valid_modes:
            return "off"
        return v


# Global settings instance
settings = Settings()

# Handle KIRO_CREDS_FILE Windows path issue
_raw_creds_file = _get_raw_env_value("KIRO_CREDS_FILE") or settings.kiro_creds_file
if _raw_creds_file:
    settings.kiro_creds_file = str(Path(_raw_creds_file))


# Slow Model Configuration
SLOW_MODELS: frozenset = frozenset({
    "claude-opus-4-5",
    "claude-opus-4-5-20251101",
    "claude-3-opus",
    "claude-3-opus-20240229",
})


# Kiro API URL Templates
KIRO_REFRESH_URL_TEMPLATE: str = "https://prod.{region}.auth.desktop.kiro.dev/refreshToken"
AWS_SSO_OIDC_URL_TEMPLATE: str = "https://oidc.{region}.amazonaws.com/token"
KIRO_API_HOST_TEMPLATE: str = "https://codewhisperer.{region}.amazonaws.com"
KIRO_Q_HOST_TEMPLATE: str = "https://q.{region}.amazonaws.com"


# Model Mapping - External model names (OpenAI compatible) -> Kiro internal ID
MODEL_MAPPING: Dict[str, str] = {
    "claude-opus-4-5": "claude-opus-4.5",
    "claude-opus-4-5-20251101": "claude-opus-4.5",
    "claude-haiku-4-5": "claude-haiku-4.5",
    "claude-haiku-4-5-20251001": "claude-haiku-4.5",
    "claude-haiku-4.5": "claude-haiku-4.5",
    "claude-sonnet-4-5": "CLAUDE_SONNET_4_5_20250929_V1_0",
    "claude-sonnet-4-5-20250929": "CLAUDE_SONNET_4_5_20250929_V1_0",
    "claude-sonnet-4": "CLAUDE_SONNET_4_20250514_V1_0",
    "claude-sonnet-4-20250514": "CLAUDE_SONNET_4_20250514_V1_0",
    "claude-3-7-sonnet-20250219": "CLAUDE_3_7_SONNET_20250219_V1_0",
    "auto": "claude-sonnet-4.5",
}

# Available models list for /v1/models endpoint
AVAILABLE_MODELS: List[str] = [
    "claude-opus-4-5",
    "claude-opus-4-5-20251101",
    "claude-haiku-4-5",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
]


_FAKE_REASONING_RAW: str = os.getenv("FAKE_REASONING", "").lower()
FAKE_REASONING_ENABLED: bool = _FAKE_REASONING_RAW not in ("false", "0", "no", "disabled", "off")
FAKE_REASONING_MAX_TOKENS: int = int(os.getenv("FAKE_REASONING_MAX_TOKENS", "4000"))

_FAKE_REASONING_HANDLING_RAW: str = os.getenv("FAKE_REASONING_HANDLING", "as_reasoning_content").lower()
if _FAKE_REASONING_HANDLING_RAW in ("as_reasoning_content", "remove", "pass", "strip_tags"):
    FAKE_REASONING_HANDLING: str = _FAKE_REASONING_HANDLING_RAW
else:
    FAKE_REASONING_HANDLING: str = "as_reasoning_content"

FAKE_REASONING_OPEN_TAGS: List[str] = ["<thinking>", "<think>", "<reasoning>", "<thought>"]
FAKE_REASONING_INITIAL_BUFFER_SIZE: int = int(os.getenv("FAKE_REASONING_INITIAL_BUFFER_SIZE", "20"))


# Version Info
APP_VERSION: str = "1.0.0"
APP_TITLE: str = "Kiro-2API"
APP_DESCRIPTION: str = "OpenAI compatible Kiro API gateway"


def get_kiro_refresh_url(region: str) -> str:
    """Return token refresh URL for specified region."""
    return KIRO_REFRESH_URL_TEMPLATE.format(region=region)


def get_aws_sso_oidc_url(region: str) -> str:
    """Return AWS SSO OIDC token URL for specified region."""
    return AWS_SSO_OIDC_URL_TEMPLATE.format(region=region)


def get_kiro_api_host(region: str) -> str:
    """Return API host for specified region."""
    return KIRO_API_HOST_TEMPLATE.format(region=region)


def get_kiro_q_host(region: str) -> str:
    """Return Q API host for specified region."""
    return KIRO_Q_HOST_TEMPLATE.format(region=region)


def get_internal_model_id(external_model: str) -> str:
    """
    Convert external model name to Kiro internal ID.

    Args:
        external_model: External model name (e.g. "claude-sonnet-4-5")

    Returns:
        Kiro API internal model ID

    Raises:
        ValueError: If model is not supported
    """
    if external_model in MODEL_MAPPING:
        return MODEL_MAPPING[external_model]

    valid_internal_ids = set(MODEL_MAPPING.values())
    if external_model in valid_internal_ids:
        return external_model

    available = ", ".join(sorted(AVAILABLE_MODELS))
    raise ValueError(f"Unsupported model: {external_model}. Available: {available}")


def get_adaptive_timeout(model: str, base_timeout: float) -> float:
    """
    Get adaptive timeout based on model type.

    For slow models (like Opus), automatically increase timeout.

    Args:
        model: Model name
        base_timeout: Base timeout in seconds

    Returns:
        Adjusted timeout in seconds
    """
    if not model:
        return base_timeout

    model_lower = model.lower()
    for slow_model in SLOW_MODELS:
        if slow_model.lower() in model_lower:
            return base_timeout * settings.slow_model_timeout_multiplier

    return base_timeout
