# -*- coding: utf-8 -*-
"""
Kiro-2API - OpenAI Compatible Kiro API Gateway.

Application entry point. Creates FastAPI app and connects routes.

Usage:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
    or run directly:
    python -m app.main
"""

import logging
import sys
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from loguru import logger

from app.core.config import (
    APP_TITLE,
    APP_DESCRIPTION,
    APP_VERSION,
    settings,
)
from app.core.exceptions import validation_exception_handler
from app.libs.auth import KiroAuthManager
from app.libs.cache import ModelInfoCache
from app.libs.http_client import close_global_http_client
from app.middleware.tracking import RequestTrackingMiddleware
from app.routes import router


# --- Loguru Configuration ---
logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)


class InterceptHandler(logging.Handler):
    """Intercept standard logging and redirect to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging_intercept():
    """Configure logging intercept from standard logging to loguru."""
    loggers_to_intercept = [
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "fastapi",
    ]

    for logger_name in loggers_to_intercept:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False


setup_logging_intercept()


# --- Startup Banner ---
def _print_startup_banner():
    """Print startup ASCII art logo and project info."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     _  ___           ____    _    ____ ___                    ║
║    | |/ (_)_ __ ___ |___ \\  / \\  |  _ \\_ _|                   ║
║    | ' /| | '__/ _ \\  __) |/ _ \\ | |_) | |                    ║
║    | . \\| | | | (_) |/ __// ___ \\|  __/| |                    ║
║    |_|\\_\\_|_|  \\___/|____/_/   \\_\\_|  |___|                   ║
║                                                               ║
║                  OpenAI Compatible Gateway                    ║
║                       Version 1.0.0                           ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""

    print(banner)

    logger.info("=" * 60)
    logger.info("🚀 Kiro-2API started successfully!")
    logger.info("=" * 60)
    logger.info("📍 Endpoints:")
    logger.info(f"   • Local: http://127.0.0.1:{settings.port}")
    logger.info(f"   • Network: http://0.0.0.0:{settings.port}")
    logger.info("📖 API Documentation:")
    logger.info(f"   • Swagger UI: http://127.0.0.1:{settings.port}/docs")
    logger.info("=" * 60)


# --- Configuration Validation ---
def validate_configuration() -> None:
    """Validate required configuration exists."""
    errors = []

    if not settings.proxy_api_key:
        errors.append(
            "PROXY_API_KEY is required!\n"
            "\n"
            "Set PROXY_API_KEY in environment variable or .env file.\n"
            "This is the password used to authenticate API requests."
        )

    has_refresh_token = bool(settings.refresh_token)
    has_creds_file = bool(settings.kiro_creds_file)

    if settings.kiro_creds_file:
        is_url = settings.kiro_creds_file.startswith(('http://', 'https://'))
        if not is_url:
            creds_path = Path(settings.kiro_creds_file).expanduser()
            if not creds_path.exists():
                has_creds_file = False
                logger.warning(f"KIRO_CREDS_FILE not found: {settings.kiro_creds_file}")

    if errors:
        logger.error("")
        logger.error("=" * 60)
        logger.error("  CONFIGURATION ERROR")
        logger.error("=" * 60)
        for error in errors:
            for line in error.split('\n'):
                logger.error(f"  {line}")
        logger.error("=" * 60)
        logger.error("")
        sys.exit(1)

    config_source = "environment variables" if not Path(".env").exists() else ".env file"

    if has_refresh_token or has_creds_file:
        if settings.kiro_creds_file:
            if settings.kiro_creds_file.startswith(('http://', 'https://')):
                logger.info(f"Using credentials from URL: {settings.kiro_creds_file} (via {config_source})")
            else:
                logger.info(f"Using credentials file: {settings.kiro_creds_file} (via {config_source})")
        elif settings.refresh_token:
            logger.info(f"Using refresh token (via {config_source})")
        logger.info("Auth mode: Simple mode (server-configured REFRESH_TOKEN) + Multi-tenant mode supported")
    else:
        logger.info("No REFRESH_TOKEN configured - running in multi-tenant only mode")
        logger.info("Auth mode: Multi-tenant only (users must provide PROXY_API_KEY:REFRESH_TOKEN)")
        logger.info("Tip: Configure REFRESH_TOKEN to enable simple mode authentication")


validate_configuration()


# --- Lifecycle Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    logger.info("Starting application... Creating state managers.")

    debug_dir = Path(settings.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    has_global_credentials = bool(settings.refresh_token) or bool(settings.kiro_creds_file)

    auth_manager = KiroAuthManager(
        refresh_token=settings.refresh_token,
        profile_arn=settings.profile_arn,
        region=settings.region,
        creds_file=settings.kiro_creds_file if settings.kiro_creds_file else None
    )
    app.state.auth_manager = auth_manager

    model_cache = ModelInfoCache()
    model_cache.set_auth_manager(auth_manager)
    app.state.model_cache = model_cache

    if has_global_credentials:
        await model_cache.start_background_refresh()

        if model_cache.is_empty():
            logger.info("Performing initial model cache population...")
            await model_cache.refresh()
    else:
        logger.warning("No global credentials configured - model cache refresh disabled")
        logger.warning("Simple mode authentication will not work, only multi-tenant mode available")

    logger.info("Application startup complete.")

    _print_startup_banner()

    yield

    logger.info("Shutting down application...")

    if has_global_credentials:
        await model_cache.stop_background_refresh()

    await close_global_http_client()

    logger.info("Application shutdown complete.")


# --- FastAPI Application ---
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(RequestTrackingMiddleware)

app.add_exception_handler(RequestValidationError, validation_exception_handler)

app.include_router(router)


# --- Uvicorn Log Configuration ---
UVICORN_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "default": {
            "class": "app.main.InterceptHandler",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
    },
}


# --- Entry Point ---
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Uvicorn server on port {settings.port}...")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=settings.port,
        log_config=UVICORN_LOG_CONFIG,
    )
