"""
Robinson Crusoe Adaptation Detection API

A production-ready REST API for detecting Robinson Crusoe adaptations using
deep learning models. Supports single predictions, batch processing, and
similarity scoring.

Author: An-Adaptive-Methodology
License: MIT
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging
import time
from datetime import datetime

from .models import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    SimilarityRequest,
    SimilarityResponse,
    HealthResponse,
    StatsResponse
)
from .model_loader import ModelLoader
from .utils import process_text, validate_text
from .config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Robinson Crusoe Adaptation Detector API",
    description="Detect Robinson Crusoe adaptations using deep learning",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model loader (lazy loading)
model_loader = ModelLoader()

# Statistics tracking
stats = {
    "total_requests": 0,
    "total_predictions": 0,
    "total_adaptations_found": 0,
    "start_time": datetime.now()
}


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    logger.info("Starting Robinson Crusoe Adaptation Detector API...")
    logger.info(f"Loading models from: {settings.MODEL_PATH}")

    try:
        # Preload default model
        model_loader.load_model()
        logger.info("✓ Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.warning("API will attempt to load models on first request")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API...")
    logger.info(f"Total requests processed: {stats['total_requests']}")
    logger.info(f"Total adaptations found: {stats['total_adaptations_found']}")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Robinson Crusoe Adaptation Detector API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch",
            "similarity": "/similarity",
            "health": "/health",
            "stats": "/stats",
            "docs": "/docs"
        },
        "description": "Detect Robinson Crusoe adaptations using deep learning"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check if model is loaded
        model_status = "loaded" if model_loader.is_loaded else "not_loaded"

        # Check if USE embedder is loaded
        embedder_status = "loaded" if model_loader.embedder is not None else "not_loaded"

        return HealthResponse(
            status="healthy",
            model_loaded=model_loader.is_loaded,
            embedder_loaded=model_loader.embedder is not None,
            uptime_seconds=(datetime.now() - stats['start_time']).total_seconds(),
            version="2.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            embedder_loaded=False,
            uptime_seconds=0,
            version="2.0.0"
        )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get API usage statistics."""
    uptime = (datetime.now() - stats['start_time']).total_seconds()

    return StatsResponse(
        total_requests=stats['total_requests'],
        total_predictions=stats['total_predictions'],
        total_adaptations_found=stats['total_adaptations_found'],
        uptime_seconds=uptime,
        requests_per_second=stats['total_requests'] / uptime if uptime > 0 else 0,
        start_time=stats['start_time'].isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict whether a text is a Robinson Crusoe adaptation.

    Args:
        request: PredictionRequest containing text to analyze

    Returns:
        PredictionResponse with prediction results
    """
    start_time = time.time()
    stats['total_requests'] += 1
    stats['total_predictions'] += 1

    try:
        # Validate text
        if not validate_text(request.text):
            raise HTTPException(
                status_code=400,
                detail="Text is too short or invalid. Minimum 50 characters required."
            )

        # Preprocess text
        processed_text = process_text(request.text)

        # Load model if not already loaded
        if not model_loader.is_loaded:
            logger.info("Loading model for first prediction...")
            model_loader.load_model()

        # Make prediction
        prediction_result = model_loader.predict(processed_text)

        # Update stats
        if prediction_result['predicted_class'] == 1:
            stats['total_adaptations_found'] += 1

        # Build response
        response = PredictionResponse(
            is_adaptation=(prediction_result['predicted_class'] == 1),
            confidence=float(prediction_result['confidence']),
            probability_adaptation=float(prediction_result['probabilities'][1]),
            probability_random=float(prediction_result['probabilities'][0]),
            predicted_class=int(prediction_result['predicted_class']),
            processing_time_ms=(time.time() - start_time) * 1000,
            text_length=len(request.text),
            model_version=model_loader.model_version
        )

        logger.info(
            f"Prediction: {'ADAPTATION' if response.is_adaptation else 'RANDOM'} "
            f"(confidence: {response.confidence:.2%}, time: {response.processing_time_ms:.0f}ms)"
        )

        return response

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """
    Predict multiple texts in batch.

    Args:
        request: BatchPredictionRequest containing multiple texts

    Returns:
        BatchPredictionResponse with results for all texts
    """
    start_time = time.time()
    stats['total_requests'] += 1

    try:
        if len(request.texts) > settings.MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size {len(request.texts)} exceeds maximum {settings.MAX_BATCH_SIZE}"
            )

        # Load model if needed
        if not model_loader.is_loaded:
            model_loader.load_model()

        # Process all texts
        results = []
        adaptations_count = 0

        for i, text in enumerate(request.texts):
            # Validate and preprocess
            if not validate_text(text):
                logger.warning(f"Skipping invalid text at index {i}")
                continue

            processed_text = process_text(text)

            # Predict
            pred_result = model_loader.predict(processed_text)

            results.append(PredictionResponse(
                is_adaptation=(pred_result['predicted_class'] == 1),
                confidence=float(pred_result['confidence']),
                probability_adaptation=float(pred_result['probabilities'][1]),
                probability_random=float(pred_result['probabilities'][0]),
                predicted_class=int(pred_result['predicted_class']),
                processing_time_ms=0,  # Individual timing not tracked in batch
                text_length=len(text),
                model_version=model_loader.model_version
            ))

            if pred_result['predicted_class'] == 1:
                adaptations_count += 1

        # Update stats
        stats['total_predictions'] += len(results)
        stats['total_adaptations_found'] += adaptations_count

        total_time = (time.time() - start_time) * 1000

        response = BatchPredictionResponse(
            predictions=results,
            total_texts=len(request.texts),
            successful_predictions=len(results),
            adaptations_found=adaptations_count,
            total_processing_time_ms=total_time,
            average_time_per_text_ms=total_time / len(results) if results else 0
        )

        logger.info(
            f"Batch prediction: {len(results)}/{len(request.texts)} texts, "
            f"{adaptations_found} adaptations, {total_time:.0f}ms total"
        )

        return response

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/similarity", response_model=SimilarityResponse)
async def calculate_similarity(request: SimilarityRequest):
    """
    Calculate similarity score between text and Robinson Crusoe.

    Args:
        request: SimilarityRequest containing text to score

    Returns:
        SimilarityResponse with similarity score (0-100)
    """
    start_time = time.time()
    stats['total_requests'] += 1

    try:
        # Validate text
        if not validate_text(request.text):
            raise HTTPException(
                status_code=400,
                detail="Text is too short or invalid"
            )

        # Preprocess
        processed_text = process_text(request.text)

        # Load model if needed
        if not model_loader.is_loaded:
            model_loader.load_model()

        # Calculate similarity
        similarity_result = model_loader.calculate_similarity(processed_text)

        response = SimilarityResponse(
            similarity_score=float(similarity_result['similarity_score']),
            cosine_similarity=float(similarity_result['cosine_similarity']),
            interpretation=similarity_result['interpretation'],
            processing_time_ms=(time.time() - start_time) * 1000,
            model_version=model_loader.model_version
        )

        logger.info(
            f"Similarity: {response.similarity_score:.1f}/100 "
            f"({response.interpretation})"
        )

        return response

    except Exception as e:
        logger.error(f"Similarity calculation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Similarity calculation failed: {str(e)}"
        )


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a text file for prediction.

    Args:
        file: Text file to analyze

    Returns:
        Prediction result
    """
    try:
        # Read file content
        content = await file.read()
        text = content.decode('utf-8')

        # Create prediction request
        request = PredictionRequest(text=text)

        # Use existing prediction endpoint
        return await predict(request)

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File must be UTF-8 encoded text"
        )
    except Exception as e:
        logger.error(f"File upload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"File processing failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
