"""
Model loading and prediction logic.

Handles lazy loading of TensorFlow models and embedders.
"""

import os
import logging
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from typing import Dict, Optional
from .config import settings

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Manages loading and inference for the Robinson Crusoe adaptation detector.

    Implements lazy loading to avoid loading models until needed.
    """

    def __init__(self):
        """Initialize model loader."""
        self.model: Optional[tf.keras.Model] = None
        self.embedder = None
        self.is_loaded = False
        self.model_version = "2.0.0"

        # Reference embeddings for similarity (loaded on demand)
        self.reference_embedding: Optional[np.ndarray] = None

    def load_model(self, model_path: Optional[str] = None):
        """
        Load the classification model and embedder.

        Args:
            model_path: Path to model file (uses settings.MODEL_PATH if not specified)
        """
        if self.is_loaded:
            logger.info("Model already loaded, skipping reload")
            return

        try:
            # Determine model path
            path = model_path or settings.MODEL_PATH

            logger.info(f"Loading classification model from: {path}")

            # Check if model file exists
            if not os.path.exists(path):
                # Try alternate locations
                alternate_path = path.replace('final_model.keras', 'best_model.keras')
                if os.path.exists(alternate_path):
                    path = alternate_path
                    logger.info(f"Using alternate model: {path}")
                else:
                    raise FileNotFoundError(f"Model not found at {path}")

            # Load the trained model
            self.model = tf.keras.models.load_model(
                path,
                custom_objects={'KerasLayer': hub.KerasLayer}
            )

            logger.info("Classification model loaded successfully")

            # Load Universal Sentence Encoder for embeddings
            logger.info(f"Loading USE embedder from: {settings.USE_MODEL_URL}")
            self.embedder = hub.load(settings.USE_MODEL_URL)
            logger.info("USE embedder loaded successfully")

            self.is_loaded = True

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def predict(self, text: str) -> Dict:
        """
        Predict whether text is a Robinson Crusoe adaptation.

        Args:
            text: Preprocessed text to classify

        Returns:
            Dictionary with prediction results:
                - predicted_class: 0 (random) or 1 (adaptation)
                - confidence: Model confidence (0-1)
                - probabilities: [prob_random, prob_adaptation]
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Get embeddings
            embeddings = self.embedder([text])
            embeddings_np = np.array(embeddings)

            # Make prediction
            predictions = self.model.predict(embeddings_np, verbose=0)

            # Extract results
            prob_adaptation = float(predictions[0][0])
            prob_random = 1.0 - prob_adaptation

            predicted_class = 1 if prob_adaptation >= 0.5 else 0
            confidence = max(prob_adaptation, prob_random)

            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': [prob_random, prob_adaptation]
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def calculate_similarity(self, text: str, reference_text: Optional[str] = None) -> Dict:
        """
        Calculate similarity between text and Robinson Crusoe.

        Uses cosine similarity between USE embeddings.

        Args:
            text: Text to score
            reference_text: Optional reference text (uses default RC text if not provided)

        Returns:
            Dictionary with similarity results:
                - similarity_score: Score on 0-100 scale
                - cosine_similarity: Raw cosine similarity (-1 to 1)
                - interpretation: Human-readable interpretation
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Get embedding for input text
            text_embedding = self.embedder([text])
            text_embedding_np = np.array(text_embedding)[0]

            # Get or create reference embedding
            if reference_text:
                ref_embedding = self.embedder([reference_text])
                ref_embedding_np = np.array(ref_embedding)[0]
            else:
                # Use default Robinson Crusoe reference
                if self.reference_embedding is None:
                    self._load_reference_embedding()
                ref_embedding_np = self.reference_embedding

            # Calculate cosine similarity
            cosine_sim = self._cosine_similarity(text_embedding_np, ref_embedding_np)

            # Convert to 0-100 scale (shift from -1,1 to 0,1 then scale)
            similarity_score = ((cosine_sim + 1) / 2) * 100

            # Interpret the score
            interpretation = self._interpret_similarity(similarity_score)

            return {
                'similarity_score': similarity_score,
                'cosine_similarity': cosine_sim,
                'interpretation': interpretation
            }

        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}", exc_info=True)
            raise RuntimeError(f"Similarity calculation failed: {str(e)}")

    def _load_reference_embedding(self):
        """Load or create reference embedding for Robinson Crusoe."""
        # Default Robinson Crusoe reference text (from original novel)
        reference_text = """
        I was born in the year 1632, in the city of York, of a good family, though not of that country,
        my father being a foreigner of Bremen, who settled first at Hull. He got a good estate by merchandise,
        and leaving off his trade, lived afterwards at York, from whence he had married my mother,
        whose relations were named Robinson, a very good family in that country, and from whom I was called Robinson Kreutznaer;
        but, by the usual corruption of words in England, we are now called—nay we call ourselves and write our name—Crusoe;
        and so my companions always called me. I had two elder brothers, one of whom was lieutenant-colonel to an English regiment
        of foot in Flanders, formerly commanded by the famous Colonel Lockhart, and was killed at the battle near Dunkirk against the Spaniards.
        What became of my second brother I never knew, any more than my father or mother knew what became of me.
        Being the third son of the family, and not bred to any trade, my head began to be filled very early with rambling thoughts.
        """

        self.reference_embedding = np.array(self.embedder([reference_text]))[0]
        logger.info("Loaded default Robinson Crusoe reference embedding")

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    @staticmethod
    def _interpret_similarity(score: float) -> str:
        """Interpret similarity score with human-readable label."""
        if score >= 90:
            return "Extremely High Similarity - Almost certainly an adaptation"
        elif score >= 80:
            return "Very High Similarity - Strong adaptation candidate"
        elif score >= 70:
            return "High Similarity - Likely adaptation"
        elif score >= 60:
            return "Moderate Similarity - Possible adaptation"
        elif score >= 50:
            return "Medium Similarity - Some shared themes"
        elif score >= 40:
            return "Low Similarity - Few shared elements"
        else:
            return "Very Low Similarity - Unlikely to be an adaptation"
