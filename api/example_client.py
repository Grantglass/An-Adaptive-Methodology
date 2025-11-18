"""
Example client for the Robinson Crusoe Adaptation Detection API.

This script demonstrates how to interact with the API using Python.
"""

import requests
import json
from typing import List, Dict


class RobinsonCrusoeAPIClient:
    """Client for interacting with the Robinson Crusoe API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API client.

        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url.rstrip('/')

    def health_check(self) -> Dict:
        """Check API health status."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def get_stats(self) -> Dict:
        """Get API usage statistics."""
        response = requests.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()

    def predict(self, text: str) -> Dict:
        """
        Predict whether a text is a Robinson Crusoe adaptation.

        Args:
            text: Text to analyze

        Returns:
            Prediction result dictionary
        """
        response = requests.post(
            f"{self.base_url}/predict",
            json={"text": text}
        )
        response.raise_for_status()
        return response.json()

    def predict_batch(self, texts: List[str]) -> Dict:
        """
        Predict multiple texts in batch.

        Args:
            texts: List of texts to analyze

        Returns:
            Batch prediction results
        """
        response = requests.post(
            f"{self.base_url}/batch",
            json={"texts": texts}
        )
        response.raise_for_status()
        return response.json()

    def calculate_similarity(self, text: str) -> Dict:
        """
        Calculate similarity to Robinson Crusoe.

        Args:
            text: Text to score

        Returns:
            Similarity score and interpretation
        """
        response = requests.post(
            f"{self.base_url}/similarity",
            json={"text": text}
        )
        response.raise_for_status()
        return response.json()

    def upload_file(self, file_path: str) -> Dict:
        """
        Upload a text file for prediction.

        Args:
            file_path: Path to text file

        Returns:
            Prediction result
        """
        with open(file_path, 'rb') as f:
            files = {'file': (file_path, f, 'text/plain')}
            response = requests.post(
                f"{self.base_url}/upload",
                files=files
            )
        response.raise_for_status()
        return response.json()


def main():
    """Example usage of the API client."""
    # Initialize client
    client = RobinsonCrusoeAPIClient("http://localhost:8000")

    print("=" * 80)
    print("Robinson Crusoe Adaptation Detection API - Example Client")
    print("=" * 80)

    # Check health
    print("\n1. Health Check")
    print("-" * 40)
    health = client.health_check()
    print(f"Status: {health['status']}")
    print(f"Model loaded: {health['model_loaded']}")
    print(f"Uptime: {health['uptime_seconds']:.1f} seconds")

    # Single prediction - Adaptation
    print("\n2. Single Prediction (Adaptation)")
    print("-" * 40)
    adaptation_text = """
    I was born in the year 1632, in the city of York. After many adventures at sea,
    I found myself shipwrecked on a desolate island. With nothing but my wits and
    the few items I could salvage from the wreck, I began my solitary existence.
    I built a shelter, hunted for food, and learned to survive in complete isolation.
    Years passed as I adapted to my lonely life, keeping a journal and marking the
    passage of time with notches on a post.
    """

    result = client.predict(adaptation_text)
    print(f"Text: {adaptation_text[:100]}...")
    print(f"Is adaptation: {result['is_adaptation']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Processing time: {result['processing_time_ms']:.1f}ms")

    # Single prediction - Random
    print("\n3. Single Prediction (Random Text)")
    print("-" * 40)
    random_text = """
    The annual technology conference was held last week in San Francisco.
    Attendees from around the world gathered to discuss the latest innovations
    in artificial intelligence, cloud computing, and blockchain technology.
    The keynote speaker emphasized the importance of ethical AI development
    and the need for responsible innovation in the tech industry.
    """

    result = client.predict(random_text)
    print(f"Text: {random_text[:100]}...")
    print(f"Is adaptation: {result['is_adaptation']}")
    print(f"Confidence: {result['confidence']:.2%}")

    # Batch prediction
    print("\n4. Batch Prediction")
    print("-" * 40)
    texts = [
        "A sailor survives a terrible shipwreck and finds himself alone on a remote island.",
        "The recipe for chocolate cake requires flour, sugar, eggs, and cocoa powder.",
        "Robinson constructed a fortress to protect himself from wild beasts and savages.",
        "Climate change is affecting weather patterns across the globe.",
        "Alone on the island, he learned to make pottery, bake bread, and tend crops."
    ]

    batch_result = client.predict_batch(texts)
    print(f"Total texts: {batch_result['total_texts']}")
    print(f"Successful predictions: {batch_result['successful_predictions']}")
    print(f"Adaptations found: {batch_result['adaptations_found']}")
    print(f"Total processing time: {batch_result['total_processing_time_ms']:.1f}ms")

    print("\nDetailed results:")
    for i, pred in enumerate(batch_result['predictions'], 1):
        print(f"  {i}. {'ADAPTATION' if pred['is_adaptation'] else 'RANDOM'} "
              f"(confidence: {pred['confidence']:.2%})")

    # Similarity scoring
    print("\n5. Similarity Scoring")
    print("-" * 40)
    similarity_text = """
    After the storm destroyed our vessel, I managed to swim to shore. The island
    was uninhabited, and I was completely alone. I spent my days building shelter,
    finding food, and trying to maintain hope that I would someday be rescued.
    """

    sim_result = client.calculate_similarity(similarity_text)
    print(f"Text: {similarity_text[:100]}...")
    print(f"Similarity score: {sim_result['similarity_score']:.1f}/100")
    print(f"Interpretation: {sim_result['interpretation']}")

    # Statistics
    print("\n6. API Statistics")
    print("-" * 40)
    stats = client.get_stats()
    print(f"Total requests: {stats['total_requests']}")
    print(f"Total predictions: {stats['total_predictions']}")
    print(f"Adaptations found: {stats['total_adaptations_found']}")
    print(f"Requests per second: {stats['requests_per_second']:.3f}")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API at http://localhost:8000")
        print("Please ensure the API is running:")
        print("  uvicorn api.main:app --reload")
        print("or")
        print("  docker-compose up")
    except Exception as e:
        print(f"Error: {e}")
