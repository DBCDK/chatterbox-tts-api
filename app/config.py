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
    MIN_TEXT_LENGTH = int(os.getenv("MIN_TEXT_LENGTH", 2))
    MAX_TOTAL_LENGTH = int(os.getenv("MAX_TOTAL_LENGTH", 3000))
    MODEL_INSTANCE_COUNT = int(os.getenv("MODEL_INSTANCE_COUNT", 4))
    MAX_QUEUE_WAIT_SECONDS = float(os.getenv("MAX_QUEUE_WAIT_SECONDS", 60))
    REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", 120))

    # Voice and model settings
    VOICE_SAMPLE_PATH = os.getenv("VOICE_SAMPLE_PATH", "./voice-sample.mp3")
    DEFAULT_VOICE_NAME = (os.getenv("DEFAULT_VOICE_NAME") or "mic").strip().lower()
    VOICE_LIBRARY_RAW = (os.getenv("VOICE_LIBRARY") or "").strip()
    DEVICE_OVERRIDE = os.getenv("DEVICE", "auto")
    MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./models")
    MODEL_SOURCE = os.getenv("MODEL_SOURCE", "default").strip().lower()
    MODEL_TYPE = (os.getenv("MODEL_TYPE") or "").strip().lower()
    MODEL_REPO_ID = (os.getenv("MODEL_REPO_ID") or "").strip()
    MODEL_REVISION = (os.getenv("MODEL_REVISION") or "").strip() or None
    MODEL_LOCAL_PATH = (os.getenv("MODEL_LOCAL_PATH") or "").strip() or None
    MODEL_SUPPORTED_LANGUAGES_RAW = (
        os.getenv("MODEL_SUPPORTED_LANGUAGES") or ""
    ).strip()
    DEFAULT_LANGUAGE = (os.getenv("DEFAULT_LANGUAGE") or "").strip().lower() or None
    HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip() or None
    HF_ALLOW_PATTERNS_RAW = (os.getenv("HF_ALLOW_PATTERNS") or "").strip()
    NORMALIZE_TEXT = os.getenv("NORMALIZE_TEXT", "true").lower() == "true"

    # Generation parameters
    TOP_P = float(os.getenv("TOP_P", 1.0))
    MIN_P = float(os.getenv("MIN_P", 0.05))
    REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", 2.0))

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
    def _parse_voice_library(cls):
        raw_value = cls.VOICE_LIBRARY_RAW
        if not raw_value:
            return {}

        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return {}

        if not isinstance(parsed, dict):
            return {}

        return {
            str(name).strip().lower(): str(path).strip()
            for name, path in parsed.items()
            if str(name).strip() and str(path).strip()
        }

    @classmethod
    def get_voice_library(cls):
        """Map of voice name -> reference audio path, always including the default voice."""
        library = {cls.DEFAULT_VOICE_NAME: cls.VOICE_SAMPLE_PATH}
        library.update(cls._parse_voice_library())
        return library

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
    def get_model_type(cls):
        return cls.MODEL_TYPE or "multilingual"

    @classmethod
    def get_configured_supported_languages(cls):
        return cls._parse_supported_languages()

    @classmethod
    def get_default_language(cls):
        if cls.DEFAULT_LANGUAGE:
            return cls.DEFAULT_LANGUAGE

        model_type = cls.get_model_type()
        configured_languages = cls.get_configured_supported_languages()
        if model_type == "multilingual" and configured_languages:
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
        if cls.MIN_TEXT_LENGTH < 1:
            raise ValueError(
                f"MIN_TEXT_LENGTH must be at least 1, got {cls.MIN_TEXT_LENGTH}"
            )
        if cls.MIN_TEXT_LENGTH >= cls.MAX_TOTAL_LENGTH:
            raise ValueError(
                f"MIN_TEXT_LENGTH ({cls.MIN_TEXT_LENGTH}) must be less than "
                f"MAX_TOTAL_LENGTH ({cls.MAX_TOTAL_LENGTH})"
            )
        if cls.MAX_TOTAL_LENGTH <= 0:
            raise ValueError(
                f"MAX_TOTAL_LENGTH must be positive, got {cls.MAX_TOTAL_LENGTH}"
            )
        if cls.MODEL_INSTANCE_COUNT <= 0:
            raise ValueError(
                f"MODEL_INSTANCE_COUNT must be positive, got {cls.MODEL_INSTANCE_COUNT}"
            )
        if cls.MAX_QUEUE_WAIT_SECONDS < 0:
            raise ValueError(
                "MAX_QUEUE_WAIT_SECONDS must be non-negative, "
                f"got {cls.MAX_QUEUE_WAIT_SECONDS}"
            )
        if cls.REQUEST_TIMEOUT_SECONDS <= 0:
            raise ValueError(
                "REQUEST_TIMEOUT_SECONDS must be positive, "
                f"got {cls.REQUEST_TIMEOUT_SECONDS}"
            )
        if not (0.0 < cls.TOP_P <= 1.0):
            raise ValueError(f"TOP_P must be in (0.0, 1.0], got {cls.TOP_P}")
        if not (0.0 <= cls.MIN_P < 1.0):
            raise ValueError(f"MIN_P must be in [0.0, 1.0), got {cls.MIN_P}")
        if cls.REPETITION_PENALTY < 1.0:
            raise ValueError(f"REPETITION_PENALTY must be >= 1.0, got {cls.REPETITION_PENALTY}")

        model_source = cls.get_model_source()
        if model_source not in {"default", "hf_repo", "local_dir"}:
            raise ValueError(
                f"MODEL_SOURCE must be one of: default, hf_repo, local_dir. Got {model_source}"
            )

        model_type = cls.get_model_type()
        if model_type not in {"base", "multilingual", "turbo"}:
            raise ValueError(
                f"MODEL_TYPE must be one of: base, multilingual, turbo. Got {model_type!r}"
            )

        if model_source == "hf_repo" and not cls.MODEL_REPO_ID:
            raise ValueError("MODEL_REPO_ID is required when MODEL_SOURCE=hf_repo")

        if model_source == "local_dir" and not cls.MODEL_LOCAL_PATH:
            raise ValueError("MODEL_LOCAL_PATH is required when MODEL_SOURCE=local_dir")

        configured_languages = cls.get_configured_supported_languages()
        if (
            model_source in {"hf_repo", "local_dir"}
            and model_type == "multilingual"
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
