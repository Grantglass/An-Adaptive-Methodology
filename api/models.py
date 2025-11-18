"""
Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request model for single prediction."""

    text: str = Field(
        ...,
        description="Text to analyze for Robinson Crusoe adaptation",
        min_length=50,
        max_length=1000000
    )

    @validator('text')
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "text": "A young man finds himself shipwrecked on a deserted island. With only his wits and determination, he builds shelter, hunts for food, and learns to survive in complete isolation. Years pass as he adapts to his solitary existence, keeping a journal to maintain his sanity and marking the passage of time."
            }
        }


class PredictionResponse(BaseModel):
    """Response model for single prediction."""

    is_adaptation: bool = Field(..., description="Whether the text is classified as a Robinson Crusoe adaptation")
    confidence: float = Field(..., description="Model confidence (0-1)", ge=0, le=1)
    probability_adaptation: float = Field(..., description="Probability of being an adaptation (0-1)", ge=0, le=1)
    probability_random: float = Field(..., description="Probability of being random text (0-1)", ge=0, le=1)
    predicted_class: int = Field(..., description="Predicted class (0=random, 1=adaptation)", ge=0, le=1)
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    text_length: int = Field(..., description="Length of input text in characters")
    model_version: str = Field(..., description="Model version used for prediction")

    class Config:
        schema_extra = {
            "example": {
                "is_adaptation": True,
                "confidence": 0.98,
                "probability_adaptation": 0.98,
                "probability_random": 0.02,
                "predicted_class": 1,
                "processing_time_ms": 234.5,
                "text_length": 512,
                "model_version": "2.0.0"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""

    texts: List[str] = Field(
        ...,
        description="List of texts to analyze",
        min_items=1,
        max_items=100
    )

    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('At least one text is required')
        return v

    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    "A sailor survives a shipwreck and must learn to survive alone on an island...",
                    "The weather today is quite pleasant with clear skies...",
                    "Robinson built a fortress to protect himself from wild beasts..."
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""

    predictions: List[PredictionResponse] = Field(..., description="List of prediction results")
    total_texts: int = Field(..., description="Total number of texts submitted")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    adaptations_found: int = Field(..., description="Number of adaptations detected")
    total_processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    average_time_per_text_ms: float = Field(..., description="Average time per text in milliseconds")


class SimilarityRequest(BaseModel):
    """Request model for similarity scoring."""

    text: str = Field(
        ...,
        description="Text to score for similarity to Robinson Crusoe",
        min_length=50,
        max_length=1000000
    )

    @validator('text')
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "text": "A man is stranded on a tropical island after his ship sinks in a storm. He salvages what he can from the wreckage and begins building a new life in isolation."
            }
        }


class SimilarityResponse(BaseModel):
    """Response model for similarity scoring."""

    similarity_score: float = Field(..., description="Similarity score on 0-100 scale", ge=0, le=100)
    cosine_similarity: float = Field(..., description="Raw cosine similarity (-1 to 1)", ge=-1, le=1)
    interpretation: str = Field(..., description="Human-readable interpretation of the score")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_version: str = Field(..., description="Model version used for scoring")

    class Config:
        schema_extra = {
            "example": {
                "similarity_score": 87.5,
                "cosine_similarity": 0.875,
                "interpretation": "Very High Similarity - Strong adaptation candidate",
                "processing_time_ms": 156.3,
                "model_version": "2.0.0"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Overall health status")
    model_loaded: bool = Field(..., description="Whether the classification model is loaded")
    embedder_loaded: bool = Field(..., description="Whether the USE embedder is loaded")
    uptime_seconds: float = Field(..., description="API uptime in seconds")
    version: str = Field(..., description="API version")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "embedder_loaded": True,
                "uptime_seconds": 3600.5,
                "version": "2.0.0"
            }
        }


class StatsResponse(BaseModel):
    """Response model for usage statistics."""

    total_requests: int = Field(..., description="Total number of requests processed")
    total_predictions: int = Field(..., description="Total number of predictions made")
    total_adaptations_found: int = Field(..., description="Total number of adaptations detected")
    uptime_seconds: float = Field(..., description="API uptime in seconds")
    requests_per_second: float = Field(..., description="Average requests per second")
    start_time: str = Field(..., description="API start time (ISO format)")

    class Config:
        schema_extra = {
            "example": {
                "total_requests": 1523,
                "total_predictions": 1845,
                "total_adaptations_found": 234,
                "uptime_seconds": 86400.0,
                "requests_per_second": 0.018,
                "start_time": "2025-01-15T10:30:00"
            }
        }
