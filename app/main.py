"""
Main FastAPI application
"""

from contextlib import asynccontextmanager
import asyncio
import logging

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.tts_model import initialize_model
from app.api.router import api_router
from app.config import Config
from app.core.observability import configure_logging, get_logger, log_event
from app.core.version import get_version


ascii_art = r"""
  ____ _           _   _            _               
 / ___| |__   __ _| |_| |_ ___ _ __| |__   _____  __
| |   | '_ \ / _` | __| __/ _ \ '__| '_ \ / _ \ \/ /
| |___| | | | (_| | |_| ||  __/ |  | |_) | (_) >  < 
 \____|_| |_|\__,_|\__|\__\___|_|  |_.__/ \___/_/\_\
                                                    
"""

SERVICE_NAME = "chatterbox-tts-api"
APP_VERSION = get_version()

configure_logging(service=SERVICE_NAME, version=APP_VERSION)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(ascii_art)
    log_event(
        logger,
        level=logging.INFO,
        event="app_starting",
        service=SERVICE_NAME,
        configured_pool_size=Config.MODEL_INSTANCE_COUNT,
    )
    model_init_task = asyncio.create_task(initialize_model())

    yield

    log_event(
        logger,
        level=logging.INFO,
        event="app_shutting_down",
        service=SERVICE_NAME,
    )

    if not model_init_task.done():
        model_init_task.cancel()
        try:
            await model_init_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="Chatterbox TTS API",
    description="REST API for Chatterbox TTS with OpenAI-compatible endpoints",
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

cors_origins = Config.CORS_ORIGINS
if cors_origins == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [origin.strip() for origin in cors_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content=exc.detail)


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    log_event(
        logger,
        level=logging.ERROR,
        event="unhandled_exception",
        route=str(request.url.path),
        method=request.method,
        error_type=type(exc).__name__,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": f"Internal server error: {str(exc)}",
                "type": "internal_error",
            }
        },
    )
