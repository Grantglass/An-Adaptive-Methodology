"""
Unit tests for the Robinson Crusoe Adaptation Detection API.

Run with: pytest api/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import numpy as np

# Import the app
from .main import app, model_loader, stats

# Create test client
client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_stats():
    """Reset stats before each test."""
    stats['total_requests'] = 0
    stats['total_predictions'] = 0
    stats['total_adaptations_found'] = 0
    yield


@pytest.fixture
def mock_model_loader():
    """Mock the model loader to avoid loading actual models in tests."""
    with patch.object(model_loader, 'is_loaded', True), \
         patch.object(model_loader, 'model_version', '2.0.0'), \
         patch.object(model_loader, 'embedder', Mock()):
        yield model_loader


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_api_info(self):
        """Test that root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data['name'] == "Robinson Crusoe Adaptation Detector API"
        assert data['version'] == "2.0.0"
        assert data['status'] == "running"
        assert 'endpoints' in data


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check_success(self, mock_model_loader):
        """Test health check with loaded model."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data['status'] == "healthy"
        assert data['version'] == "2.0.0"
        assert 'uptime_seconds' in data

    def test_health_check_model_not_loaded(self):
        """Test health check when model is not loaded."""
        with patch.object(model_loader, 'is_loaded', False):
            response = client.get("/health")
            assert response.status_code == 200

            data = response.json()
            assert data['model_loaded'] is False


class TestStatsEndpoint:
    """Tests for statistics endpoint."""

    def test_stats_initial_state(self):
        """Test stats endpoint returns initial state."""
        response = client.get("/stats")
        assert response.status_code == 200

        data = response.json()
        assert data['total_requests'] == 0
        assert data['total_predictions'] == 0
        assert data['total_adaptations_found'] == 0
        assert 'uptime_seconds' in data
        assert 'start_time' in data


class TestPredictionEndpoint:
    """Tests for prediction endpoint."""

    def test_predict_adaptation_text(self, mock_model_loader):
        """Test prediction with adaptation text."""
        # Mock the predict method
        mock_model_loader.predict = Mock(return_value={
            'predicted_class': 1,
            'confidence': 0.98,
            'probabilities': [0.02, 0.98]
        })

        test_text = "A young sailor finds himself shipwrecked on a deserted island. " \
                   "With nothing but his wits, he must learn to survive alone, " \
                   "building shelter and hunting for food in complete isolation."

        response = client.post("/predict", json={"text": test_text})
        assert response.status_code == 200

        data = response.json()
        assert data['is_adaptation'] is True
        assert data['confidence'] == 0.98
        assert data['predicted_class'] == 1
        assert data['probability_adaptation'] == 0.98
        assert data['model_version'] == "2.0.0"

    def test_predict_random_text(self, mock_model_loader):
        """Test prediction with random text."""
        mock_model_loader.predict = Mock(return_value={
            'predicted_class': 0,
            'confidence': 0.95,
            'probabilities': [0.95, 0.05]
        })

        test_text = "The weather today is quite pleasant with clear blue skies. " \
                   "I think I will go for a walk in the park and enjoy the sunshine. " \
                   "Perhaps I'll bring a book to read on the bench by the pond."

        response = client.post("/predict", json={"text": test_text})
        assert response.status_code == 200

        data = response.json()
        assert data['is_adaptation'] is False
        assert data['predicted_class'] == 0

    def test_predict_text_too_short(self):
        """Test prediction with text that is too short."""
        response = client.post("/predict", json={"text": "Too short"})
        assert response.status_code == 400
        assert "too short" in response.json()['detail'].lower()

    def test_predict_empty_text(self):
        """Test prediction with empty text."""
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422  # Validation error

    def test_predict_whitespace_only(self):
        """Test prediction with whitespace-only text."""
        response = client.post("/predict", json={"text": "   " * 100})
        assert response.status_code == 422  # Validation error


class TestBatchPredictionEndpoint:
    """Tests for batch prediction endpoint."""

    def test_batch_predict_success(self, mock_model_loader):
        """Test batch prediction with multiple texts."""
        mock_model_loader.predict = Mock(side_effect=[
            {'predicted_class': 1, 'confidence': 0.98, 'probabilities': [0.02, 0.98]},
            {'predicted_class': 0, 'confidence': 0.95, 'probabilities': [0.95, 0.05]},
            {'predicted_class': 1, 'confidence': 0.92, 'probabilities': [0.08, 0.92]},
        ])

        texts = [
            "A sailor survives a shipwreck and must learn to survive on a deserted island alone.",
            "The weather today is sunny and bright with clear skies overhead.",
            "Robinson built a fortress to protect himself from the wild beasts on the island."
        ]

        response = client.post("/batch", json={"texts": texts})
        assert response.status_code == 200

        data = response.json()
        assert data['total_texts'] == 3
        assert data['successful_predictions'] == 3
        assert data['adaptations_found'] == 2
        assert len(data['predictions']) == 3

    def test_batch_predict_exceeds_limit(self, mock_model_loader):
        """Test batch prediction with too many texts."""
        # Create 101 texts (exceeds default MAX_BATCH_SIZE of 100)
        texts = ["Test text number " + str(i) + " " * 50 for i in range(101)]

        response = client.post("/batch", json={"texts": texts})
        assert response.status_code == 400
        assert "exceeds maximum" in response.json()['detail'].lower()

    def test_batch_predict_empty_list(self):
        """Test batch prediction with empty list."""
        response = client.post("/batch", json={"texts": []})
        assert response.status_code == 422  # Validation error


class TestSimilarityEndpoint:
    """Tests for similarity scoring endpoint."""

    def test_similarity_high_score(self, mock_model_loader):
        """Test similarity with high similarity text."""
        mock_model_loader.calculate_similarity = Mock(return_value={
            'similarity_score': 87.5,
            'cosine_similarity': 0.875,
            'interpretation': "Very High Similarity - Strong adaptation candidate"
        })

        test_text = "A man is stranded on a tropical island after his ship sinks. " \
                   "He salvages what he can and begins building a new life in isolation."

        response = client.post("/similarity", json={"text": test_text})
        assert response.status_code == 200

        data = response.json()
        assert data['similarity_score'] == 87.5
        assert data['cosine_similarity'] == 0.875
        assert "Very High" in data['interpretation']

    def test_similarity_low_score(self, mock_model_loader):
        """Test similarity with low similarity text."""
        mock_model_loader.calculate_similarity = Mock(return_value={
            'similarity_score': 25.3,
            'cosine_similarity': 0.253,
            'interpretation': "Very Low Similarity - Unlikely to be an adaptation"
        })

        test_text = "The stock market today showed significant gains across all sectors. " \
                   "Investors are optimistic about the upcoming quarterly earnings reports."

        response = client.post("/similarity", json={"text": test_text})
        assert response.status_code == 200

        data = response.json()
        assert data['similarity_score'] == 25.3
        assert "Very Low" in data['interpretation']

    def test_similarity_text_too_short(self):
        """Test similarity with text that is too short."""
        response = client.post("/similarity", json={"text": "Short"})
        assert response.status_code == 400


class TestFileUploadEndpoint:
    """Tests for file upload endpoint."""

    def test_upload_text_file(self, mock_model_loader):
        """Test uploading a text file for prediction."""
        mock_model_loader.predict = Mock(return_value={
            'predicted_class': 1,
            'confidence': 0.96,
            'probabilities': [0.04, 0.96]
        })

        # Create a mock file
        file_content = b"A sailor is shipwrecked on a remote island and must survive alone. " * 5
        files = {'file': ('test.txt', file_content, 'text/plain')}

        response = client.post("/upload", files=files)
        assert response.status_code == 200

        data = response.json()
        assert data['is_adaptation'] is True
        assert data['confidence'] == 0.96

    def test_upload_invalid_encoding(self):
        """Test uploading a file with invalid encoding."""
        # Create a file with invalid UTF-8
        files = {'file': ('test.txt', b'\x80\x81\x82', 'text/plain')}

        response = client.post("/upload", files=files)
        assert response.status_code == 400
        assert "UTF-8" in response.json()['detail']


class TestErrorHandling:
    """Tests for error handling."""

    def test_model_not_loaded_error(self):
        """Test prediction when model is not loaded."""
        with patch.object(model_loader, 'is_loaded', False), \
             patch.object(model_loader, 'load_model', side_effect=RuntimeError("Model load failed")):

            test_text = "A" * 100  # Valid length text

            response = client.post("/predict", json={"text": test_text})
            assert response.status_code == 500

    def test_prediction_error(self, mock_model_loader):
        """Test handling of prediction errors."""
        mock_model_loader.predict = Mock(side_effect=RuntimeError("Prediction failed"))

        test_text = "A" * 100

        response = client.post("/predict", json={"text": test_text})
        assert response.status_code == 500
        assert "Prediction failed" in response.json()['detail']


# Integration test markers
@pytest.mark.integration
class TestIntegration:
    """Integration tests (require actual model)."""

    def test_full_prediction_workflow(self):
        """
        Full prediction workflow test.

        NOTE: This test requires the actual model to be present.
        Skip if running in CI/CD without model files.
        """
        pytest.skip("Requires actual model files - run manually")

        # Real prediction test would go here
        test_text = "Robinson Crusoe was shipwrecked on a desert island."
        response = client.post("/predict", json={"text": test_text})
        assert response.status_code == 200


# Performance test markers
@pytest.mark.performance
class TestPerformance:
    """Performance tests."""

    def test_prediction_performance(self, mock_model_loader):
        """Test prediction performance."""
        mock_model_loader.predict = Mock(return_value={
            'predicted_class': 1,
            'confidence': 0.98,
            'probabilities': [0.02, 0.98]
        })

        test_text = "A" * 500

        response = client.post("/predict", json={"text": test_text})
        data = response.json()

        # Check that processing time is reasonable (< 1 second for mock)
        assert data['processing_time_ms'] < 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
