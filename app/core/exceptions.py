# -*- coding: utf-8 -*-
"""
Exception handlers for Kiro-2API.

Contains functions for handling validation errors and other exceptions
in a JSON-serializable format.
"""

from typing import Any, List, Dict

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger


def sanitize_validation_errors(errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transform validation errors to JSON-serializable format.

    Pydantic may include bytes objects in 'input' field which
    cannot be serialized to JSON. This function converts them to strings.

    Args:
        errors: List of validation errors from Pydantic

    Returns:
        List of errors with bytes converted to strings
    """
    sanitized = []
    for error in errors:
        sanitized_error = {}
        for key, value in error.items():
            if isinstance(value, bytes):
                sanitized_error[key] = value.decode("utf-8", errors="replace")
            elif isinstance(value, (list, tuple)):
                sanitized_error[key] = [
                    v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v
                    for v in value
                ]
            else:
                sanitized_error[key] = value
        sanitized.append(sanitized_error)
    return sanitized


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handler for Pydantic validation errors.

    Logs error details and returns informative response.
    Correctly handles bytes objects in errors by converting them to strings.

    Args:
        request: FastAPI Request object
        exc: Validation exception from Pydantic

    Returns:
        JSONResponse with error details and status 422
    """
    body = await request.body()
    body_str = body.decode("utf-8", errors="replace")

    sanitized_errors = sanitize_validation_errors(exc.errors())

    logger.error(f"Validation error (422): {sanitized_errors}")
    logger.error(f"Request body: {body_str[:500]}...")

    return JSONResponse(
        status_code=422,
        content={"detail": sanitized_errors, "body": body_str[:500]},
    )
