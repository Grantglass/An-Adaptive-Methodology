# Web Interface Documentation

An interactive Streamlit dashboard for Robinson Crusoe adaptation detection without writing code.

## Features

### 🔍 Single Prediction
- Analyze individual texts with one click
- Example texts provided for quick testing
- Real-time confidence visualization with gauge charts
- Probability breakdown for both classes
- Character count validation

### 📑 Batch Analysis
- Process up to 100 texts at once
- Paste texts directly or upload a file
- Comprehensive results table
- Export results to CSV
- Performance metrics (avg time per text)

### 📊 Similarity Scoring
- Calculate 0-100 similarity score to Robinson Crusoe
- Visual similarity chart with color gradient
- Detailed interpretation of scores
- Raw cosine similarity values

### 📜 History Tracking
- View all previous predictions
- Downloadable prediction history
- Timestamps for each analysis

### ⚙️ Live Monitoring
- Real-time API health status
- Usage statistics display
- Requests per second tracking
- Total predictions counter

## Quick Start

### Local Development

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Start the API** (in one terminal):
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

3. **Start the web interface** (in another terminal):
```bash
streamlit run app.py
```

4. **Open your browser** to:
```
http://localhost:8501
```

### Docker Deployment

Use the provided docker-compose with Streamlit:

1. **Update docker-compose.yml** to include Streamlit service:
```yaml
services:
  streamlit:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api
```

2. **Create Dockerfile.streamlit**:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install streamlit plotly requests pandas

COPY app.py .
COPY .streamlit/ .streamlit/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

3. **Run**:
```bash
docker-compose up -d
```

### Streamlit Cloud Deployment

1. **Push your repository to GitHub**

2. **Visit [streamlit.io/cloud](https://streamlit.io/cloud)**

3. **Click "New app"**

4. **Configure**:
   - Repository: Your GitHub repo
   - Branch: main
   - Main file path: `app.py`

5. **Add secrets** (Settings → Secrets):
```toml
API_URL = "https://your-api-url.com"
```

6. **Deploy!**

The app will be available at: `https://your-app.streamlit.app`

## Configuration

### API Connection

The web interface connects to the API using the URL specified in `.streamlit/secrets.toml`:

1. **Copy the example**:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

2. **Edit `.streamlit/secrets.toml`**:
```toml
API_URL = "http://localhost:8000"
```

For production, use your deployed API URL:
```toml
API_URL = "https://your-api.example.com"
```

### Streamlit Configuration

Additional configuration in `.streamlit/config.toml` (optional):

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

## Usage Examples

### Single Text Analysis

1. Navigate to "🔍 Single Prediction" tab
2. Choose an example or enter custom text (min 50 characters)
3. Click "🔍 Analyze Text"
4. View results:
   - Classification (Adaptation or Random)
   - Confidence gauge
   - Probability breakdown
   - Processing time

### Batch Processing

1. Navigate to "📑 Batch Analysis" tab
2. Either:
   - Paste texts (one per line)
   - Upload a .txt file (one text per line)
3. Click "📊 Analyze Batch"
4. View:
   - Summary statistics
   - Detailed results table
   - Download CSV option

### Similarity Analysis

1. Navigate to "📊 Similarity Scoring" tab
2. Enter your text
3. Click "📏 Calculate Similarity"
4. View:
   - Similarity score (0-100)
   - Cosine similarity
   - Interpretation
   - Visual chart

## Features in Detail

### Visualizations

**Confidence Gauge**:
- Color-coded based on classification
- Green for adaptations
- Orange for random texts
- Threshold markers at 50%, 75%, 90%

**Probability Chart**:
- Side-by-side bar comparison
- Percentage labels on bars
- Color-coded (green for adaptation, orange for random)

**Similarity Chart**:
- Horizontal bar with color gradient
- Red → Yellow → Green scale
- Score displayed inline

### Error Handling

The app handles:
- API connection failures (shows error message)
- Invalid text length (minimum 50 characters)
- Batch size limits (maximum 100 texts)
- File upload errors (encoding issues)
- Timeout errors (shows spinner during processing)

### Session State

The app maintains:
- Prediction history across page interactions
- Batch results for re-display
- Settings persistence

## Customization

### Adding New Features

1. **Add a new tab** in `app.py`:
```python
tab5 = st.tabs([..., "🆕 New Feature"])

with tab5:
    st.header("New Feature")
    # Your code here
```

2. **Add new visualizations**:
```python
def create_my_chart(data):
    fig = go.Figure(...)
    return fig
```

3. **Add new API calls**:
```python
def my_new_endpoint(data):
    response = requests.post(f"{API_BASE_URL}/my-endpoint", json=data)
    return response.json()
```

### Styling

Modify the CSS in the `st.markdown()` section at the top of `app.py`:

```python
st.markdown("""
<style>
    .main-header {
        /* Your custom styles */
    }
</style>
""", unsafe_allow_html=True)
```

### Example Texts

Modify the `example_texts` dictionary in Tab 1:

```python
example_texts = {
    "Your Example Name": "Your example text here...",
    # Add more examples
}
```

## Troubleshooting

### API Connection Failed

**Error**: "❌ API is unhealthy"

**Solutions**:
1. Ensure API is running: `uvicorn api.main:app --reload`
2. Check API URL in `.streamlit/secrets.toml`
3. Verify API health directly: `curl http://localhost:8000/health`

### Streamlit Won't Start

**Error**: `ModuleNotFoundError: No module named 'streamlit'`

**Solution**:
```bash
pip install streamlit plotly requests pandas
```

### Slow Performance

**Issue**: App feels slow

**Solutions**:
1. Enable caching in API (check `CACHE_EMBEDDINGS=true`)
2. Use batch processing instead of multiple single predictions
3. Reduce visualization complexity
4. Check API response times in developer tools

### CORS Errors

**Error**: CORS policy blocking requests

**Solution**: Ensure API CORS settings allow your Streamlit URL:
```python
# In api/config.py
ALLOWED_ORIGINS = "http://localhost:8501,https://your-app.streamlit.app"
```

## Performance

- **Page load**: < 2 seconds
- **Single prediction**: 200-400ms (depends on API)
- **Batch processing**: ~150ms per text
- **Visualization rendering**: < 100ms

## Security

### Best Practices

1. **Never commit secrets**:
   - Add `.streamlit/secrets.toml` to `.gitignore`
   - Use Streamlit Cloud secrets for production

2. **Input validation**:
   - Text length checks enforced
   - Batch size limits enforced
   - File upload validation

3. **API security**:
   - Use HTTPS for production API
   - Implement rate limiting on API
   - Set appropriate CORS policies

## Browser Support

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

## Mobile Support

The interface is responsive and works on:
- 📱 iOS Safari
- 📱 Android Chrome
- 📱 Tablets

## Keyboard Shortcuts

When using the app:
- `Ctrl/Cmd + Enter` - Submit text (in text areas)
- `Ctrl/Cmd + S` - Save (triggers Streamlit save)
- `R` - Rerun the app

## Analytics

Track usage with Streamlit's built-in analytics:

1. **Enable in config.toml**:
```toml
[browser]
gatherUsageStats = true
```

2. **View in Streamlit Cloud dashboard**

## Accessibility

The interface includes:
- Proper heading hierarchy
- Alt text for visualizations
- Keyboard navigation support
- Screen reader compatible
- High contrast mode support

## Contributing

To add features to the web interface:

1. Fork the repository
2. Create a feature branch
3. Add your feature to `app.py`
4. Test locally
5. Submit a pull request

## License

Same as main project (MIT License)

## Support

- **Documentation**: This file
- **Issues**: GitHub Issues
- **Streamlit Docs**: https://docs.streamlit.io

## Related

- [Main README](README.md)
- [API Documentation](api/README.md)
- [Deployment Guide](MODERNIZATION.md)
