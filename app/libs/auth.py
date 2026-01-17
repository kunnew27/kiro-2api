# -*- coding: utf-8 -*-
"""
Kiro API Authentication Manager.

Manages access token lifecycle:
- Load credentials from .env or JSON file
- Auto-refresh token on expiration
- Thread-safe refresh using asyncio.Lock
- Support for both Social (Kiro Desktop) and IDC (AWS SSO OIDC) authentication
"""

import asyncio
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger

from app.core.config import (
    settings,
    get_kiro_refresh_url,
    get_kiro_api_host,
    get_kiro_q_host,
    get_aws_sso_oidc_url,
)
from app.utils.helpers import get_machine_fingerprint


class AuthType(Enum):
    """Authentication type enumeration."""
    SOCIAL = "social"
    IDC = "idc"


class KiroAuthManager:
    """
    Manages token lifecycle for Kiro API access.

    Supports:
    - Loading credentials from .env or JSON file
    - Auto-refresh token on expiration
    - Checking expiration time (expiresAt)
    - Saving updated tokens to file
    - Both Social (Kiro Desktop) and IDC (AWS SSO OIDC) authentication
    """

    def __init__(
        self,
        refresh_token: Optional[str] = None,
        profile_arn: Optional[str] = None,
        region: str = "us-east-1",
        creds_file: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ):
        """
        Initialize authentication manager.

        Args:
            refresh_token: Refresh token for obtaining access token
            profile_arn: AWS CodeWhisperer profile ARN
            region: AWS region (default us-east-1)
            creds_file: Path to JSON credentials file (optional)
            client_id: OAuth client ID (for IDC mode, optional)
            client_secret: OAuth client secret (for IDC mode, optional)
        """
        self._refresh_token = refresh_token
        self._profile_arn = profile_arn
        self._region = region
        self._creds_file = creds_file

        self._client_id: Optional[str] = client_id
        self._client_secret: Optional[str] = client_secret
        self._scopes: list = [
            "codewhisperer:completions",
            "codewhisperer:analysis",
            "codewhisperer:conversations",
            "codewhisperer:transformations",
            "codewhisperer:taskassist",
        ]

        self._access_token: Optional[str] = None
        self._expires_at: Optional[datetime] = None
        self._lock = asyncio.Lock()

        self._auth_type: AuthType = AuthType.SOCIAL

        self._refresh_url = get_kiro_refresh_url(region)
        self._api_host = get_kiro_api_host(region)
        self._q_host = get_kiro_q_host(region)

        self._fingerprint = get_machine_fingerprint()

        if creds_file:
            self._load_credentials_from_file(creds_file)

        self._detect_auth_type()

    def _detect_auth_type(self) -> None:
        """Detect auth type based on credentials."""
        if self._client_id and self._client_secret:
            self._auth_type = AuthType.IDC
            logger.info("Detected auth type: IDC (AWS SSO OIDC)")
        else:
            self._auth_type = AuthType.SOCIAL
            logger.debug("Using auth type: Social (Kiro Desktop)")

    @staticmethod
    def _is_url(path: str) -> bool:
        """Check if path is a URL."""
        return path.startswith(('http://', 'https://'))

    def _load_credentials_from_file(self, file_path: str) -> None:
        """Load credentials from JSON file or remote URL."""
        try:
            if self._is_url(file_path):
                response = httpx.get(file_path, timeout=10.0, follow_redirects=True)
                response.raise_for_status()
                data = response.json()
                logger.info(f"Credentials loaded from URL: {file_path}")
            else:
                path = Path(file_path).expanduser()
                if not path.exists():
                    logger.warning(f"Credentials file not found: {file_path}")
                    return

                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Credentials loaded from file: {file_path}")

            if 'refreshToken' in data:
                self._refresh_token = data['refreshToken']
            if 'accessToken' in data:
                self._access_token = data['accessToken']
            if 'profileArn' in data:
                self._profile_arn = data['profileArn']
            if 'region' in data:
                self._region = data['region']
                self._refresh_url = get_kiro_refresh_url(self._region)
                self._api_host = get_kiro_api_host(self._region)
                self._q_host = get_kiro_q_host(self._region)

            if 'clientId' in data:
                self._client_id = data['clientId']
            if 'clientSecret' in data:
                self._client_secret = data['clientSecret']

            if 'expiresAt' in data:
                try:
                    expires_str = data['expiresAt']
                    if expires_str.endswith('Z'):
                        self._expires_at = datetime.fromisoformat(expires_str.replace('Z', '+00:00'))
                    else:
                        self._expires_at = datetime.fromisoformat(expires_str)
                except Exception as e:
                    logger.warning(f"Failed to parse expiresAt: {e}")

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error loading credentials from URL: {e}")
        except httpx.RequestError as e:
            logger.error(f"Request error loading credentials from URL: {e}")
        except Exception as e:
            logger.error(f"Error loading credentials: {e}")

    def _save_credentials_to_file(
        self,
        access_token: Optional[str] = None,
        refresh_token: Optional[str] = None,
        profile_arn: Optional[str] = None
    ) -> None:
        """Save updated credentials to JSON file."""
        if not self._creds_file:
            return

        try:
            path = Path(self._creds_file).expanduser()

            existing_data = {}
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)

            existing_data['accessToken'] = access_token if access_token is not None else self._access_token
            existing_data['refreshToken'] = refresh_token if refresh_token is not None else self._refresh_token
            if self._expires_at:
                existing_data['expiresAt'] = self._expires_at.isoformat()
            if profile_arn is not None:
                existing_data['profileArn'] = profile_arn
            elif self._profile_arn:
                existing_data['profileArn'] = self._profile_arn

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Credentials saved to {self._creds_file}")

        except Exception as e:
            logger.error(f"Error saving credentials: {e}")

    def is_token_expiring_soon(self) -> bool:
        """Check if token is expiring soon."""
        if not self._expires_at:
            return True

        now = datetime.now(timezone.utc)
        threshold = now.timestamp() + settings.token_refresh_threshold

        return self._expires_at.timestamp() <= threshold

    async def _refresh_token_request(self) -> None:
        """Execute token refresh request."""
        if self._auth_type == AuthType.IDC:
            await self._refresh_token_idc()
        else:
            await self._refresh_token_social()

    async def _refresh_token_social(self) -> None:
        """Refresh token using Social (Kiro Desktop Auth) endpoint."""
        if not self._refresh_token:
            raise ValueError("Refresh token is not set")

        logger.info("Refreshing token via Social (Kiro Desktop Auth)...")

        payload = {'refreshToken': self._refresh_token}
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"Kiro2API-{self._fingerprint[:16]}",
        }

        data = await self._execute_refresh_request(self._refresh_url, json_data=payload, headers=headers)
        self._process_refresh_response(data)

    async def _refresh_token_idc(self) -> None:
        """Refresh token using IDC (AWS SSO OIDC) endpoint."""
        if not self._refresh_token:
            raise ValueError("Refresh token is not set")
        if not self._client_id:
            raise ValueError("Client ID is not set (required for IDC mode)")
        if not self._client_secret:
            raise ValueError("Client secret is not set (required for IDC mode)")

        logger.info("Refreshing token via IDC (AWS SSO OIDC)...")

        url = get_aws_sso_oidc_url(self._region)
        json_data = {
            "clientId": self._client_id,
            "clientSecret": self._client_secret,
            "grantType": "refresh_token",
            "refreshToken": self._refresh_token,
        }

        headers = {
            "Content-Type": "application/json",
        }

        data = await self._execute_refresh_request(url, json_data=json_data, headers=headers)
        self._process_refresh_response(data)

    async def _execute_refresh_request(
        self,
        url: str,
        json_data: Optional[dict] = None,
        form_data: Optional[dict] = None,
        headers: Optional[dict] = None
    ) -> dict:
        """Execute refresh request with exponential backoff retry."""
        max_retries = 3
        base_delay = 1.0
        last_error = None

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    if json_data:
                        response = await client.post(url, json=json_data, headers=headers)
                    else:
                        response = await client.post(url, data=form_data, headers=headers)
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code in (429, 500, 502, 503, 504):
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Token refresh failed (attempt {attempt + 1}/{max_retries}): "
                        f"HTTP {e.response.status_code}, retrying in {delay}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"Token refresh failed (attempt {attempt + 1}/{max_retries}): "
                    f"{type(e).__name__}, retrying in {delay}s"
                )
                await asyncio.sleep(delay)

        logger.error(f"Token refresh failed after {max_retries} attempts")
        raise last_error

    def _process_refresh_response(self, data: dict) -> None:
        """Process refresh response and update internal state."""
        new_access_token = data.get("accessToken")
        new_refresh_token = data.get("refreshToken")
        expires_in = data.get("expiresIn", 3600)
        new_profile_arn = data.get("profileArn")

        if not new_access_token:
            raise ValueError(f"No accessToken in response: {data}")

        now = datetime.now(timezone.utc).replace(microsecond=0)
        new_expires_at = datetime.fromtimestamp(
            now.timestamp() + expires_in - 60,
            tz=timezone.utc
        )

        self._save_credentials_to_file(new_access_token, new_refresh_token, new_profile_arn)

        self._access_token = new_access_token
        if new_refresh_token:
            self._refresh_token = new_refresh_token
        if new_profile_arn:
            self._profile_arn = new_profile_arn
        self._expires_at = new_expires_at

        logger.info(f"Token refreshed successfully, expires at: {self._expires_at.isoformat()}")

    async def get_access_token(self) -> str:
        """Return valid access_token, refreshing if necessary."""
        async with self._lock:
            if not self._access_token or self.is_token_expiring_soon():
                await self._refresh_token_request()

            if not self._access_token:
                raise ValueError("Failed to obtain access token")

            return self._access_token

    async def force_refresh(self) -> str:
        """Force token refresh."""
        async with self._lock:
            await self._refresh_token_request()
            return self._access_token

    @property
    def profile_arn(self) -> Optional[str]:
        """AWS CodeWhisperer profile ARN."""
        return self._profile_arn

    @property
    def region(self) -> str:
        """AWS region."""
        return self._region

    @property
    def api_host(self) -> str:
        """API host for current region."""
        return self._api_host

    @property
    def q_host(self) -> str:
        """Q API host for current region."""
        return self._q_host

    @property
    def fingerprint(self) -> str:
        """Unique machine fingerprint."""
        return self._fingerprint

    @property
    def auth_type(self) -> AuthType:
        """Authentication type (SOCIAL or IDC)."""
        return self._auth_type
