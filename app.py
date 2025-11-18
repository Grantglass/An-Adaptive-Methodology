"""
Streamlit Web Interface for Robinson Crusoe Adaptation Detection

An interactive dashboard for analyzing texts without writing code.
"""

import streamlit as st
import requests
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict

# Page configuration
st.set_page_config(
    page_title="Robinson Crusoe Adaptation Detector",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-top: 0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = st.secrets.get("API_URL", "http://localhost:8000")

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None


def check_api_health() -> Dict:
    """Check if API is healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def predict_single(text: str) -> Dict:
    """Make a single prediction."""
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json={"text": text},
        timeout=30
    )
    response.raise_for_status()
    return response.json()


def predict_batch(texts: List[str]) -> Dict:
    """Make batch predictions."""
    response = requests.post(
        f"{API_BASE_URL}/batch",
        json={"texts": texts},
        timeout=60
    )
    response.raise_for_status()
    return response.json()


def calculate_similarity(text: str) -> Dict:
    """Calculate similarity score."""
    response = requests.post(
        f"{API_BASE_URL}/similarity",
        json={"text": text},
        timeout=30
    )
    response.raise_for_status()
    return response.json()


def get_api_stats() -> Dict:
    """Get API statistics."""
    response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
    response.raise_for_status()
    return response.json()


def create_confidence_gauge(confidence: float, is_adaptation: bool) -> go.Figure:
    """Create a gauge chart for confidence visualization."""
    color = "green" if is_adaptation else "orange"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffcccc'},
                {'range': [50, 75], 'color': '#fff4cc'},
                {'range': [75, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_probability_chart(prob_random: float, prob_adaptation: float) -> go.Figure:
    """Create a bar chart for class probabilities."""
    fig = go.Figure(data=[
        go.Bar(
            x=['Random Text', 'Adaptation'],
            y=[prob_random * 100, prob_adaptation * 100],
            marker_color=['#ff7f0e', '#2ca02c'],
            text=[f'{prob_random*100:.1f}%', f'{prob_adaptation*100:.1f}%'],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title="Class Probabilities",
        yaxis_title="Probability (%)",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


def create_similarity_chart(similarity_score: float) -> go.Figure:
    """Create a horizontal bar chart for similarity score."""
    fig = go.Figure(go.Bar(
        y=['Similarity to Robinson Crusoe'],
        x=[similarity_score],
        orientation='h',
        marker=dict(
            color=similarity_score,
            colorscale='RdYlGn',
            cmin=0,
            cmax=100,
            showscale=True
        ),
        text=f'{similarity_score:.1f}/100',
        textposition='inside'
    ))

    fig.update_layout(
        xaxis=dict(range=[0, 100], title="Similarity Score"),
        height=150,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    return fig


# Main App
def main():
    # Header
    st.markdown('<p class="main-header">📚 Robinson Crusoe Adaptation Detector</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Detect literary adaptations using deep learning and semantic embeddings</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # API Health Check
        st.subheader("API Status")
        health = check_api_health()

        if health.get('status') == 'healthy':
            st.success(f"✅ API is healthy")
            st.caption(f"Uptime: {health.get('uptime_seconds', 0):.0f}s")
        else:
            st.error(f"❌ API is unhealthy")
            st.caption(f"Error: {health.get('error', 'Unknown')}")

        st.divider()

        # API Statistics
        st.subheader("📊 Usage Statistics")
        try:
            stats = get_api_stats()
            st.metric("Total Predictions", stats.get('total_predictions', 0))
            st.metric("Adaptations Found", stats.get('total_adaptations_found', 0))
            st.metric("Requests/sec", f"{stats.get('requests_per_second', 0):.3f}")
        except:
            st.caption("Statistics unavailable")

        st.divider()

        # About
        st.subheader("ℹ️ About")
        st.caption("""
        This tool uses a neural network trained on Robinson Crusoe adaptations
        to detect plot-level similarities in texts.

        **Model**: Universal Sentence Encoder + Deep Neural Network

        **Accuracy**: 99%+ on test set
        """)

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Prediction", "📑 Batch Analysis", "📊 Similarity Scoring", "📜 History"])

    # Tab 1: Single Prediction
    with tab1:
        st.header("Analyze a Single Text")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Text input
            example_texts = {
                "Example: Strong Adaptation": "I was born in the year 1632, in the city of York. After many adventures at sea, I found myself shipwrecked on a desolate island. With nothing but my wits and the few items I could salvage from the wreck, I began my solitary existence. I built a shelter, hunted for food, and learned to survive in complete isolation.",
                "Example: Weak Adaptation": "After the plane crash, Sarah found herself alone in the wilderness. She had to learn survival skills quickly.",
                "Example: Random Text": "The annual technology conference was held in San Francisco. Attendees discussed innovations in AI and blockchain.",
                "Custom Text": ""
            }

            selected_example = st.selectbox("Choose an example or enter custom text:", list(example_texts.keys()))

            if selected_example == "Custom Text":
                text_input = st.text_area(
                    "Enter your text (minimum 50 characters):",
                    height=200,
                    placeholder="Enter the text you want to analyze..."
                )
            else:
                text_input = st.text_area(
                    "Text to analyze:",
                    value=example_texts[selected_example],
                    height=200
                )

            st.caption(f"Character count: {len(text_input)}")

            analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)

        with col2:
            st.info("""
            **How it works:**

            1. Your text is converted to a semantic embedding using Universal Sentence Encoder
            2. A neural network classifies it as an adaptation or random text
            3. Results include confidence score and class probabilities
            """)

        # Process prediction
        if analyze_button:
            if len(text_input.strip()) < 50:
                st.error("⚠️ Text must be at least 50 characters long")
            else:
                with st.spinner("Analyzing text..."):
                    try:
                        result = predict_single(text_input)

                        # Add to history
                        st.session_state.prediction_history.append({
                            'text': text_input[:100] + "..." if len(text_input) > 100 else text_input,
                            'is_adaptation': result['is_adaptation'],
                            'confidence': result['confidence'],
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                        })

                        # Display results
                        st.success("✅ Analysis Complete!")

                        # Main result
                        if result['is_adaptation']:
                            st.markdown('<div class="success-box"><h3>✅ ADAPTATION DETECTED</h3></div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="error-box"><h3>❌ NOT AN ADAPTATION</h3></div>', unsafe_allow_html=True)

                        # Metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Confidence", f"{result['confidence']:.1%}")
                        with col2:
                            st.metric("Processing Time", f"{result['processing_time_ms']:.0f}ms")
                        with col3:
                            st.metric("Text Length", f"{result['text_length']} chars")

                        # Visualizations
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(
                                create_confidence_gauge(result['confidence'], result['is_adaptation']),
                                use_container_width=True
                            )
                        with col2:
                            st.plotly_chart(
                                create_probability_chart(
                                    result['probability_random'],
                                    result['probability_adaptation']
                                ),
                                use_container_width=True
                            )

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

    # Tab 2: Batch Analysis
    with tab2:
        st.header("Batch Text Analysis")

        st.info("Analyze multiple texts at once (up to 100 texts)")

        # Input method selection
        input_method = st.radio("Input method:", ["Paste texts (one per line)", "Upload file"])

        if input_method == "Paste texts (one per line)":
            batch_input = st.text_area(
                "Enter texts (one per line):",
                height=300,
                placeholder="Text 1...\nText 2...\nText 3..."
            )
            texts = [line.strip() for line in batch_input.split('\n') if line.strip() and len(line.strip()) >= 50]
        else:
            uploaded_file = st.file_uploader("Upload a text file (one text per line)", type=['txt'])
            if uploaded_file:
                texts = [line.decode('utf-8').strip() for line in uploaded_file if len(line.decode('utf-8').strip()) >= 50]
            else:
                texts = []

        st.caption(f"Valid texts: {len(texts)}")

        if st.button("📊 Analyze Batch", type="primary", disabled=len(texts) == 0):
            if len(texts) > 100:
                st.error("⚠️ Maximum 100 texts allowed per batch")
            else:
                with st.spinner(f"Analyzing {len(texts)} texts..."):
                    try:
                        result = predict_batch(texts)
                        st.session_state.batch_results = result

                        # Summary metrics
                        st.success("✅ Batch Analysis Complete!")

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Texts", result['total_texts'])
                        with col2:
                            st.metric("Adaptations Found", result['adaptations_found'])
                        with col3:
                            st.metric("Success Rate", f"{result['successful_predictions']/result['total_texts']*100:.0f}%")
                        with col4:
                            st.metric("Avg Time/Text", f"{result['average_time_per_text_ms']:.0f}ms")

                        # Results table
                        st.subheader("Detailed Results")
                        df = pd.DataFrame([
                            {
                                'Text': texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                                'Classification': '✅ Adaptation' if pred['is_adaptation'] else '❌ Random',
                                'Confidence': f"{pred['confidence']:.1%}",
                                'Time (ms)': f"{pred['processing_time_ms']:.0f}"
                            }
                            for i, pred in enumerate(result['predictions'])
                        ])

                        st.dataframe(df, use_container_width=True)

                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results (CSV)",
                            csv,
                            "batch_results.csv",
                            "text/csv"
                        )

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

    # Tab 3: Similarity Scoring
    with tab3:
        st.header("Calculate Similarity to Robinson Crusoe")

        st.info("Get a numerical similarity score (0-100) showing how similar your text is to Robinson Crusoe")

        similarity_input = st.text_area(
            "Enter your text:",
            height=200,
            placeholder="Enter text to calculate similarity..."
        )

        if st.button("📏 Calculate Similarity", type="primary"):
            if len(similarity_input.strip()) < 50:
                st.error("⚠️ Text must be at least 50 characters long")
            else:
                with st.spinner("Calculating similarity..."):
                    try:
                        result = calculate_similarity(similarity_input)

                        st.success("✅ Similarity Calculated!")

                        # Display score
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.metric(
                                "Similarity Score",
                                f"{result['similarity_score']:.1f}/100",
                                help="Higher scores indicate greater similarity to Robinson Crusoe"
                            )
                            st.metric(
                                "Cosine Similarity",
                                f"{result['cosine_similarity']:.3f}",
                                help="Raw cosine similarity between embeddings"
                            )

                        with col2:
                            st.plotly_chart(
                                create_similarity_chart(result['similarity_score']),
                                use_container_width=True
                            )

                        # Interpretation
                        st.info(f"**Interpretation:** {result['interpretation']}")

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

    # Tab 4: History
    with tab4:
        st.header("Prediction History")

        if st.session_state.prediction_history:
            st.caption(f"Total predictions: {len(st.session_state.prediction_history)}")

            # Display as dataframe
            history_df = pd.DataFrame(st.session_state.prediction_history)
            history_df['Classification'] = history_df['is_adaptation'].apply(
                lambda x: '✅ Adaptation' if x else '❌ Random'
            )
            history_df['Confidence'] = history_df['confidence'].apply(lambda x: f"{x:.1%}")

            st.dataframe(
                history_df[['timestamp', 'text', 'Classification', 'Confidence']],
                use_container_width=True
            )

            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.rerun()
        else:
            st.info("No predictions yet. Analyze some texts to see your history!")


if __name__ == "__main__":
    main()
