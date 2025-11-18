"""
Configuration settings for the Robinson Crusoe API.

Uses environment variables with sensible defaults.
"""

import os
from typing import List


class Settings:
    """API configuration settings."""

    # Server settings
    HOST: str = os.getenv("API_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("API_PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # CORS settings
    ALLOWED_ORIGINS: List[str] = os.getenv(
        "ALLOWED_ORIGINS",
        "*"
    ).split(",")

    # Model settings
    MODEL_PATH: str = os.getenv(
        "MODEL_PATH",
        "/home/user/An-Adaptive-Methodology/models/final_model.keras"
    )
    USE_MODEL_URL: str = "https://tfhub.dev/google/universal-sentence-encoder/4"

    # Text processing settings
    MIN_TEXT_LENGTH: int = int(os.getenv("MIN_TEXT_LENGTH", "50"))
    MAX_TEXT_LENGTH: int = int(os.getenv("MAX_TEXT_LENGTH", "1000000"))
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "100"))

    # Cache settings
    CACHE_EMBEDDINGS: bool = os.getenv("CACHE_EMBEDDINGS", "True").lower() == "true"
    EMBEDDINGS_CACHE_PATH: str = os.getenv(
        "EMBEDDINGS_CACHE_PATH",
        "/home/user/An-Adaptive-Methodology/data/cache"
    )

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
