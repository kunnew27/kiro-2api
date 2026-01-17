# -*- coding: utf-8 -*-
"""
Request tracking middleware.

Adds unique ID to each request for log correlation and debugging.
"""

import time
import uuid
from datetime import datetime
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger


def get_timestamp() -> str:
    """Get formatted timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_client_ip(request: Request) -> str:
    """Extract client IP from request, supporting X-Forwarded-For."""
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """
    Request tracking middleware.

    For each request:
    - Generates unique request ID
    - Records request start and end time
    - Calculates request processing time
    - Adds request ID context to logs
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and add tracking info.

        Args:
            request: HTTP request
            call_next: Next middleware or route handler

        Returns:
            HTTP response
        """
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())

        start_time = time.time()
        request.state.request_id = request_id

        with logger.contextualize(request_id=request_id):
            client_ip = get_client_ip(request)
            logger.info(
                f"[{get_timestamp()}] [IP: {client_ip}] Request start: {request.method} {request.url.path}"
                + (f" Params: {request.url.query}" if request.url.query else "")
            )

            try:
                response = await call_next(request)

                process_time = time.time() - start_time

                response.headers["X-Request-ID"] = request_id
                response.headers["X-Process-Time"] = str(round(process_time, 4))

                status_text = "success" if 200 <= response.status_code < 400 else "failed"
                logger.info(
                    f"[{get_timestamp()}] [IP: {client_ip}] "
                    f"Request {status_text}: {request.method} {request.url.path} "
                    f"status={response.status_code} time={process_time:.4f}s"
                )

                return response

            except Exception as e:
                process_time = time.time() - start_time
                logger.error(
                    f"[{get_timestamp()}] [IP: {client_ip}] "
                    f"Request error: {request.method} {request.url.path} "
                    f"error={str(e)} time={process_time:.4f}s"
                )
                raise
