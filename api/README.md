# Robinson Crusoe Adaptation Detection API

A production-ready REST API for detecting Robinson Crusoe adaptations using deep learning models built with FastAPI.

## Features

- **Single Prediction**: Analyze individual texts for Robinson Crusoe adaptation detection
- **Batch Processing**: Process multiple texts in a single request (up to 100 texts)
- **Similarity Scoring**: Calculate semantic similarity to Robinson Crusoe (0-100 scale)
- **File Upload**: Upload text files for analysis
- **Health Monitoring**: Built-in health checks and usage statistics
- **OpenAPI Documentation**: Interactive API docs at `/docs` and `/redoc`
- **Docker Support**: Containerized deployment with Docker Compose

## Quick Start

### Local Development

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set environment variables** (optional):
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Run the API**:
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

4. **Access the API**:
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Docker Deployment

1. **Build and run with Docker Compose**:
```bash
docker-compose up -d
```

2. **View logs**:
```bash
docker-compose logs -f api
```

3. **Stop the API**:
```bash
docker-compose down
```

## API Endpoints

### Root
```
GET /
```
Returns API information and available endpoints.

**Example Response**:
```json
{
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
  }
}
```

### Health Check
```
GET /health
```
Check API health and model status.

**Example Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "embedder_loaded": true,
  "uptime_seconds": 3600.5,
  "version": "2.0.0"
}
```

### Statistics
```
GET /stats
```
Get API usage statistics.

**Example Response**:
```json
{
  "total_requests": 1523,
  "total_predictions": 1845,
  "total_adaptations_found": 234,
  "uptime_seconds": 86400.0,
  "requests_per_second": 0.018,
  "start_time": "2025-01-15T10:30:00"
}
```

### Predict (Single Text)
```
POST /predict
Content-Type: application/json
```

Predict whether a text is a Robinson Crusoe adaptation.

**Request Body**:
```json
{
  "text": "A young man finds himself shipwrecked on a deserted island. With only his wits and determination, he builds shelter, hunts for food, and learns to survive in complete isolation."
}
```

**Response**:
```json
{
  "is_adaptation": true,
  "confidence": 0.98,
  "probability_adaptation": 0.98,
  "probability_random": 0.02,
  "predicted_class": 1,
  "processing_time_ms": 234.5,
  "text_length": 182,
  "model_version": "2.0.0"
}
```

### Batch Predict
```
POST /batch
Content-Type: application/json
```

Process multiple texts in a single request.

**Request Body**:
```json
{
  "texts": [
    "A sailor survives a shipwreck and must learn to survive alone on an island...",
    "The weather today is quite pleasant with clear skies...",
    "Robinson built a fortress to protect himself from wild beasts..."
  ]
}
```

**Response**:
```json
{
  "predictions": [
    {
      "is_adaptation": true,
      "confidence": 0.96,
      ...
    },
    ...
  ],
  "total_texts": 3,
  "successful_predictions": 3,
  "adaptations_found": 2,
  "total_processing_time_ms": 456.7,
  "average_time_per_text_ms": 152.2
}
```

### Similarity Score
```
POST /similarity
Content-Type: application/json
```

Calculate similarity to Robinson Crusoe on a 0-100 scale.

**Request Body**:
```json
{
  "text": "A man is stranded on a tropical island after his ship sinks in a storm. He salvages what he can from the wreckage and begins building a new life in isolation."
}
```

**Response**:
```json
{
  "similarity_score": 87.5,
  "cosine_similarity": 0.875,
  "interpretation": "Very High Similarity - Strong adaptation candidate",
  "processing_time_ms": 156.3,
  "model_version": "2.0.0"
}
```

### Upload File
```
POST /upload
Content-Type: multipart/form-data
```

Upload a text file for prediction.

**Example (curl)**:
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -F "file=@yourfile.txt"
```

## Usage Examples

### Python (requests)

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "A sailor is shipwrecked on a deserted island..."}
)
result = response.json()
print(f"Is adaptation: {result['is_adaptation']}")
print(f"Confidence: {result['confidence']:.2%}")

# Batch prediction
texts = [
    "Text about shipwreck and survival...",
    "Random text about weather...",
    "Another Robinson Crusoe adaptation..."
]
response = requests.post(
    "http://localhost:8000/batch",
    json={"texts": texts}
)
batch_result = response.json()
print(f"Found {batch_result['adaptations_found']} adaptations")

# Similarity scoring
response = requests.post(
    "http://localhost:8000/similarity",
    json={"text": "A man alone on an island..."}
)
sim_result = response.json()
print(f"Similarity: {sim_result['similarity_score']:.1f}/100")
print(f"Interpretation: {sim_result['interpretation']}")
```

### JavaScript (fetch)

```javascript
// Single prediction
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'A sailor is shipwrecked on a deserted island...'
  })
});

const result = await response.json();
console.log(`Is adaptation: ${result.is_adaptation}`);
console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);
```

### cURL

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "A sailor is shipwrecked on a deserted island and must survive alone..."}'

# Batch prediction
curl -X POST "http://localhost:8000/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Text 1...", "Text 2...", "Text 3..."]}'

# Similarity scoring
curl -X POST "http://localhost:8000/similarity" \
  -H "Content-Type: application/json" \
  -d '{"text": "A man alone on an island..."}'

# Health check
curl "http://localhost:8000/health"

# Statistics
curl "http://localhost:8000/stats"
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Host to bind to |
| `API_PORT` | `8000` | Port to bind to |
| `DEBUG` | `false` | Enable debug mode |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins (comma-separated) |
| `MODEL_PATH` | `models/final_model.keras` | Path to trained model |
| `MIN_TEXT_LENGTH` | `50` | Minimum text length (characters) |
| `MAX_TEXT_LENGTH` | `1000000` | Maximum text length (characters) |
| `MAX_BATCH_SIZE` | `100` | Maximum batch size |
| `CACHE_EMBEDDINGS` | `true` | Enable embedding caching |
| `EMBEDDINGS_CACHE_PATH` | `data/cache` | Cache directory path |
| `LOG_LEVEL` | `INFO` | Logging level |

### Configuration File

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

## Testing

### Run all tests
```bash
pytest api/test_api.py -v
```

### Run with coverage
```bash
pytest api/test_api.py --cov=api --cov-report=html
```

### Run specific test class
```bash
pytest api/test_api.py::TestPredictionEndpoint -v
```

### Run integration tests
```bash
pytest api/test_api.py -v -m integration
```

## Model Requirements

The API requires:
1. **Trained Keras model**: `models/final_model.keras` or `models/best_model.keras`
2. **Universal Sentence Encoder**: Downloaded automatically from TensorFlow Hub on first use

### Training the Model

If you don't have a trained model:

```bash
# Run the training notebook
jupyter notebook Notebooks/train.ipynb

# Or use the command line version
python -c "import nbformat; from nbconvert import PythonExporter; ..."
```

## Performance

- **Single prediction**: ~200-300ms (including embedding generation)
- **Batch prediction**: ~150ms per text (parallel processing)
- **Similarity scoring**: ~150-200ms
- **Model loading**: ~10-15 seconds (one-time on startup)

### Optimization Tips

1. **Use batch endpoints** for multiple texts
2. **Enable embedding caching** for repeated texts
3. **Adjust Docker resource limits** based on load
4. **Use GPU** for faster embedding generation (requires GPU-enabled TensorFlow)

## Deployment

### Production Checklist

- [ ] Set `DEBUG=false` in environment
- [ ] Configure specific `ALLOWED_ORIGINS` (not `*`)
- [ ] Set up HTTPS with reverse proxy (nginx)
- [ ] Configure resource limits in docker-compose.yml
- [ ] Set up log rotation
- [ ] Configure monitoring and alerts
- [ ] Backup model files
- [ ] Test health check endpoint
- [ ] Load test with expected traffic

### Scaling

For high-traffic deployments:

1. **Horizontal scaling**: Run multiple API containers behind a load balancer
2. **Caching**: Use Redis for embedding cache
3. **Queue system**: Add Celery for async processing
4. **CDN**: Cache static documentation
5. **Database**: Store predictions for analytics

## Troubleshooting

### Model not loading

**Error**: `Model not found at /path/to/model.keras`

**Solution**:
- Check that model file exists: `ls -lh models/`
- Verify `MODEL_PATH` environment variable
- Ensure model was trained successfully

### Out of memory

**Error**: `OOM when allocating tensor`

**Solution**:
- Reduce `MAX_BATCH_SIZE`
- Increase Docker memory limits
- Use smaller model or quantization
- Enable swap memory

### Slow predictions

**Issue**: Predictions take >1 second

**Solution**:
- Use batch endpoint instead of multiple single requests
- Enable embedding caching
- Check CPU/GPU utilization
- Consider using GPU-enabled TensorFlow

### CORS errors

**Error**: `Access blocked by CORS policy`

**Solution**:
- Add your frontend domain to `ALLOWED_ORIGINS`
- Use comma-separated list for multiple origins
- Restart API after changing environment variables

## API Versioning

Current version: **2.0.0**

The API follows semantic versioning:
- **Major**: Breaking changes to API contract
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible

## License

MIT License - see LICENSE file for details

## Support

- **Documentation**: See `/docs` endpoint
- **Issues**: GitHub Issues
- **Questions**: See main project README

## Related

- [Main Project README](../README.md)
- [Modernization Documentation](../MODERNIZATION.md)
- [Training Notebooks](../Notebooks/)
- [Model Architecture](../Notebooks/train.ipynb)
