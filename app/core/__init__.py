# -*- coding: utf-8 -*-
"""Core configuration and exceptions."""

from app.core.config import settings
from app.core.exceptions import validation_exception_handler

__all__ = ["settings", "validation_exception_handler"]
