"""
Configuration management for Chatterbox TTS API
"""

import json
import os

import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration class"""

    # Server settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 4123))

    # TTS Model settings
    EXAGGERATION = float(os.getenv("EXAGGERATION", 0.5))
    CFG_WEIGHT = float(os.getenv("CFG_WEIGHT", 0.5))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.8))

    # Text processing
    MAX_CHUNK_LENGTH = int(os.getenv("MAX_CHUNK_LENGTH", 280))
    MAX_TOTAL_LENGTH = int(os.getenv("MAX_TOTAL_LENGTH", 3000))

    # Voice and model settings
    VOICE_SAMPLE_PATH = os.getenv("VOICE_SAMPLE_PATH", "./voice-sample.mp3")
    DEVICE_OVERRIDE = os.getenv("DEVICE", "auto")
    MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./models")
    MODEL_SOURCE = os.getenv("MODEL_SOURCE", "default").strip().lower()
    MODEL_CLASS = (os.getenv("MODEL_CLASS") or "").strip().lower()
    MODEL_REPO_ID = (os.getenv("MODEL_REPO_ID") or "").strip()
    MODEL_REVISION = (os.getenv("MODEL_REVISION") or "").strip() or None
    MODEL_LOCAL_PATH = (os.getenv("MODEL_LOCAL_PATH") or "").strip() or None
    MODEL_SUPPORTED_LANGUAGES_RAW = (
        os.getenv("MODEL_SUPPORTED_LANGUAGES") or ""
    ).strip()
    DEFAULT_LANGUAGE = (os.getenv("DEFAULT_LANGUAGE") or "").strip().lower() or None
    HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip() or None
    HF_ALLOW_PATTERNS_RAW = (os.getenv("HF_ALLOW_PATTERNS") or "").strip()

    # Voice library settings
    VOICE_LIBRARY_DIR = os.getenv("VOICE_LIBRARY_DIR", "./voices")

    # Long text processing settings
    LONG_TEXT_DATA_DIR = os.getenv("LONG_TEXT_DATA_DIR", "./data/long_text_jobs")
    LONG_TEXT_MAX_LENGTH = int(os.getenv("LONG_TEXT_MAX_LENGTH", 100000))
    LONG_TEXT_CHUNK_SIZE = int(os.getenv("LONG_TEXT_CHUNK_SIZE", 2500))
    LONG_TEXT_SILENCE_PADDING_MS = int(os.getenv("LONG_TEXT_SILENCE_PADDING_MS", 200))
    LONG_TEXT_JOB_RETENTION_DAYS = int(os.getenv("LONG_TEXT_JOB_RETENTION_DAYS", 7))
    LONG_TEXT_MAX_CONCURRENT_JOBS = int(os.getenv("LONG_TEXT_MAX_CONCURRENT_JOBS", 3))

    # Multilingual model settings
    USE_MULTILINGUAL_MODEL = (
        os.getenv("USE_MULTILINGUAL_MODEL", "true").lower() == "true"
    )

    # Memory management settings
    MEMORY_CLEANUP_INTERVAL = int(os.getenv("MEMORY_CLEANUP_INTERVAL", 5))
    CUDA_CACHE_CLEAR_INTERVAL = int(os.getenv("CUDA_CACHE_CLEAR_INTERVAL", 3))
    ENABLE_MEMORY_MONITORING = (
        os.getenv("ENABLE_MEMORY_MONITORING", "true").lower() == "true"
    )

    # CORS settings
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

    @classmethod
    def _parse_supported_languages(cls):
        raw_value = cls.MODEL_SUPPORTED_LANGUAGES_RAW
        if not raw_value:
            return {}

        try:
            parsed = json.loads(raw_value)
            if isinstance(parsed, dict):
                return {
                    str(code).strip().lower(): str(name).strip()
                    for code, name in parsed.items()
                    if str(code).strip() and str(name).strip()
                }
            if isinstance(parsed, list):
                languages = {}
                for item in parsed:
                    if isinstance(item, str) and item.strip():
                        code = item.strip().lower()
                        languages[code] = code
                    elif isinstance(item, dict):
                        code = str(item.get("code", "")).strip().lower()
                        name = str(item.get("name", "")).strip()
                        if code and name:
                            languages[code] = name
                return languages
        except json.JSONDecodeError:
            pass

        languages = {}
        for item in raw_value.split(","):
            value = item.strip()
            if not value:
                continue
            if ":" in value:
                code, name = value.split(":", 1)
                code = code.strip().lower()
                name = name.strip()
                if code and name:
                    languages[code] = name
            else:
                code = value.lower()
                languages[code] = code
        return languages

    @classmethod
    def _parse_hf_allow_patterns(cls):
        raw_value = cls.HF_ALLOW_PATTERNS_RAW
        if not raw_value:
            return ["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]

        try:
            parsed = json.loads(raw_value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass

        return [item.strip() for item in raw_value.split(",") if item.strip()]

    @classmethod
    def get_model_source(cls):
        return cls.MODEL_SOURCE or "default"

    @classmethod
    def get_model_class(cls):
        if cls.get_model_source() == "default":
            return "multilingual" if cls.USE_MULTILINGUAL_MODEL else "standard"
        return cls.MODEL_CLASS

    @classmethod
    def get_configured_supported_languages(cls):
        return cls._parse_supported_languages()

    @classmethod
    def get_default_language(cls):
        if cls.DEFAULT_LANGUAGE:
            return cls.DEFAULT_LANGUAGE

        model_class = cls.get_model_class()
        configured_languages = cls.get_configured_supported_languages()
        if model_class == "multilingual" and configured_languages:
            return next(iter(configured_languages.keys()))
        return "en"

    @classmethod
    def get_hf_allow_patterns(cls):
        return cls._parse_hf_allow_patterns()

    @classmethod
    def validate(cls):
        """Validate configuration values"""
        if not (0.25 <= cls.EXAGGERATION <= 2.0):
            raise ValueError(
                f"EXAGGERATION must be between 0.25 and 2.0, got {cls.EXAGGERATION}"
            )
        if not (0.0 <= cls.CFG_WEIGHT <= 1.0):
            raise ValueError(
                f"CFG_WEIGHT must be between 0.0 and 1.0, got {cls.CFG_WEIGHT}"
            )
        if not (0.05 <= cls.TEMPERATURE <= 5.0):
            raise ValueError(
                f"TEMPERATURE must be between 0.05 and 5.0, got {cls.TEMPERATURE}"
            )
        if cls.MAX_CHUNK_LENGTH <= 0:
            raise ValueError(
                f"MAX_CHUNK_LENGTH must be positive, got {cls.MAX_CHUNK_LENGTH}"
            )
        if cls.MAX_TOTAL_LENGTH <= 0:
            raise ValueError(
                f"MAX_TOTAL_LENGTH must be positive, got {cls.MAX_TOTAL_LENGTH}"
            )
        if cls.MEMORY_CLEANUP_INTERVAL <= 0:
            raise ValueError(
                f"MEMORY_CLEANUP_INTERVAL must be positive, got {cls.MEMORY_CLEANUP_INTERVAL}"
            )
        if cls.CUDA_CACHE_CLEAR_INTERVAL <= 0:
            raise ValueError(
                f"CUDA_CACHE_CLEAR_INTERVAL must be positive, got {cls.CUDA_CACHE_CLEAR_INTERVAL}"
            )
        if cls.LONG_TEXT_MAX_LENGTH <= cls.MAX_TOTAL_LENGTH:
            raise ValueError(
                f"LONG_TEXT_MAX_LENGTH ({cls.LONG_TEXT_MAX_LENGTH}) must be greater than MAX_TOTAL_LENGTH ({cls.MAX_TOTAL_LENGTH})"
            )
        if cls.LONG_TEXT_CHUNK_SIZE <= 0:
            raise ValueError(
                f"LONG_TEXT_CHUNK_SIZE must be positive, got {cls.LONG_TEXT_CHUNK_SIZE}"
            )
        if cls.LONG_TEXT_CHUNK_SIZE >= cls.MAX_TOTAL_LENGTH:
            raise ValueError(
                f"LONG_TEXT_CHUNK_SIZE ({cls.LONG_TEXT_CHUNK_SIZE}) must be less than MAX_TOTAL_LENGTH ({cls.MAX_TOTAL_LENGTH})"
            )
        if cls.LONG_TEXT_SILENCE_PADDING_MS < 0:
            raise ValueError(
                f"LONG_TEXT_SILENCE_PADDING_MS must be non-negative, got {cls.LONG_TEXT_SILENCE_PADDING_MS}"
            )
        if cls.LONG_TEXT_JOB_RETENTION_DAYS <= 0:
            raise ValueError(
                f"LONG_TEXT_JOB_RETENTION_DAYS must be positive, got {cls.LONG_TEXT_JOB_RETENTION_DAYS}"
            )
        if cls.LONG_TEXT_MAX_CONCURRENT_JOBS <= 0:
            raise ValueError(
                f"LONG_TEXT_MAX_CONCURRENT_JOBS must be positive, got {cls.LONG_TEXT_MAX_CONCURRENT_JOBS}"
            )
        model_source = cls.get_model_source()
        if model_source not in {"default", "hf_repo", "local_dir"}:
            raise ValueError(
                f"MODEL_SOURCE must be one of: default, hf_repo, local_dir. Got {model_source}"
            )

        model_class = cls.get_model_class()
        if model_class not in {"standard", "multilingual"}:
            raise ValueError(
                f"MODEL_CLASS must resolve to standard or multilingual. Got {model_class!r}"
            )

        if model_source == "hf_repo" and not cls.MODEL_REPO_ID:
            raise ValueError("MODEL_REPO_ID is required when MODEL_SOURCE=hf_repo")

        if model_source == "local_dir" and not cls.MODEL_LOCAL_PATH:
            raise ValueError("MODEL_LOCAL_PATH is required when MODEL_SOURCE=local_dir")

        configured_languages = cls.get_configured_supported_languages()
        if (
            model_source in {"hf_repo", "local_dir"}
            and model_class == "multilingual"
            and not configured_languages
        ):
            raise ValueError(
                "MODEL_SUPPORTED_LANGUAGES is required for multilingual hf_repo/local_dir models"
            )

        default_language = cls.get_default_language()
        if configured_languages and default_language not in configured_languages:
            raise ValueError(
                f"DEFAULT_LANGUAGE ({default_language}) must be included in MODEL_SUPPORTED_LANGUAGES ({', '.join(configured_languages.keys())})"
            )


def detect_device():
    """Detect the best available device"""
    if Config.DEVICE_OVERRIDE.lower() != "auto":
        requested_device = Config.DEVICE_OVERRIDE.lower()
        if requested_device == "cuda":
            if torch.version.cuda is None:
                raise RuntimeError(
                    "DEVICE=cuda was requested, but the installed torch build does not include CUDA support"
                )
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "DEVICE=cuda was requested, but CUDA is not available inside this runtime"
                )
        if requested_device == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise RuntimeError(
                "DEVICE=mps was requested, but MPS is not available on this machine"
            )
        return requested_device

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
