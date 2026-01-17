# -*- coding: utf-8 -*-
"""API Routes."""

from fastapi import APIRouter

from app.routes.health import router as health_router
from app.routes.models import router as models_router
from app.routes.chat import router as chat_router

router = APIRouter()

router.include_router(health_router)
router.include_router(models_router)
router.include_router(chat_router)

__all__ = ["router"]
