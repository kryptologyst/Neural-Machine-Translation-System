"""
Streamlit Web Interface for Neural Machine Translation

This module provides a user-friendly web interface for the translation system
with real-time translation, batch processing, and evaluation features.
"""

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import logging
from typing import List, Dict, Optional
import time

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from translation_engine import create_translation_engine, TranslationEngine
from dataset_generator import SyntheticDatasetGenerator
from evaluation import TranslationEvaluator, TranslationVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Neural Machine Translation",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .translation-box {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'translation_engine' not in st.session_state:
    st.session_state.translation_engine = None
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = TranslationEvaluator()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = TranslationVisualizer()

def load_translation_engine(language_pair: str) -> TranslationEngine:
    """Load translation engine for the specified language pair."""
    try:
        return create_translation_engine(language_pair)
    except Exception as e:
        st.error(f"Failed to load translation engine: {e}")
        return None

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🌐 Neural Machine Translation</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Language pair selection
        language_pairs = {
            "English → French": "en-fr",
            "English → German": "en-de", 
            "English → Spanish": "en-es",
            "English → Italian": "en-it",
            "English → Portuguese": "en-pt",
            "French → English": "fr-en",
            "German → English": "de-en",
            "Spanish → English": "es-en",
            "Italian → English": "it-en",
            "Portuguese → English": "pt-en"
        }
        
        selected_pair = st.selectbox(
            "Select Language Pair:",
            options=list(language_pairs.keys()),
            index=0
        )
        
        language_code = language_pairs[selected_pair]
        
        # Load engine button
        if st.button("🔄 Load Translation Engine"):
            with st.spinner("Loading translation engine..."):
                st.session_state.translation_engine = load_translation_engine(language_code)
                if st.session_state.translation_engine:
                    st.success(f"✅ Engine loaded for {selected_pair}")
                else:
                    st.error("❌ Failed to load engine")
        
        # Engine status
        if st.session_state.translation_engine:
            st.success("✅ Engine Ready")
        else:
            st.warning("⚠️ No engine loaded")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔤 Single Translation", 
        "📝 Batch Translation", 
        "📊 Evaluation", 
        "📈 Analytics", 
        "ℹ️ About"
    ])
    
    with tab1:
        single_translation_tab()
    
    with tab2:
        batch_translation_tab()
    
    with tab3:
        evaluation_tab()
    
    with tab4:
        analytics_tab()
    
    with tab5:
        about_tab()

def single_translation_tab():
    """Single translation interface."""
    st.header("🔤 Single Text Translation")
    
    if not st.session_state.translation_engine:
        st.warning("Please load a translation engine from the sidebar first.")
        return
    
    # Input text
    st.subheader("Input Text")
    input_text = st.text_area(
        "Enter text to translate:",
        value="Hello, how are you today?",
        height=100,
        help="Enter the text you want to translate"
    )
    
    # Translation options
    col1, col2 = st.columns(2)
    
    with col1:
        max_length = st.slider("Max Length", 50, 512, 256)
    
    with col2:
        num_beams = st.slider("Number of Beams", 1, 8, 4)
    
    # Translate button
    if st.button("🚀 Translate", type="primary"):
        if input_text.strip():
            with st.spinner("Translating..."):
                start_time = time.time()
                
                # Update engine configuration
                st.session_state.translation_engine.config.max_length = max_length
                st.session_state.translation_engine.config.num_beams = num_beams
                
                # Translate
                translated_text = st.session_state.translation_engine.translate(input_text)
                
                translation_time = time.time() - start_time
            
            # Display results
            st.subheader("Translation Result")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="translation-box">', unsafe_allow_html=True)
                st.write("**Original Text:**")
                st.write(input_text)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="translation-box">', unsafe_allow_html=True)
                st.write("**Translated Text:**")
                st.write(translated_text)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Translation metrics
            st.subheader("Translation Info")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Translation Time", f"{translation_time:.3f}s")
            
            with col2:
                st.metric("Input Length", f"{len(input_text.split())} words")
            
            with col3:
                st.metric("Output Length", f"{len(translated_text.split())} words")
        else:
            st.warning("Please enter some text to translate.")

def batch_translation_tab():
    """Batch translation interface."""
    st.header("📝 Batch Translation")
    
    if not st.session_state.translation_engine:
        st.warning("Please load a translation engine from the sidebar first.")
        return
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["Manual Input", "Upload File", "Generate Sample Data"]
    )
    
    texts_to_translate = []
    
    if input_method == "Manual Input":
        st.subheader("Manual Text Input")
        batch_text = st.text_area(
            "Enter texts to translate (one per line):",
            value="Hello, how are you?\nThe weather is beautiful today.\nThank you for your help.",
            height=150
        )
        texts_to_translate = [line.strip() for line in batch_text.split('\n') if line.strip()]
    
    elif input_method == "Upload File":
        st.subheader("Upload File")
        uploaded_file = st.file_uploader(
            "Upload a text file:",
            type=['txt', 'json'],
            help="Upload a text file with one sentence per line, or a JSON file with 'text' fields"
        )
        
        if uploaded_file:
            if uploaded_file.name.endswith('.json'):
                data = json.load(uploaded_file)
                if isinstance(data, list):
                    texts_to_translate = [item.get('text', str(item)) for item in data]
                else:
                    texts_to_translate = [data.get('text', str(data))]
            else:
                content = uploaded_file.read().decode('utf-8')
                texts_to_translate = [line.strip() for line in content.split('\n') if line.strip()]
    
    else:  # Generate Sample Data
        st.subheader("Generate Sample Data")
        col1, col2 = st.columns(2)
        
        with col1:
            num_samples = st.number_input("Number of samples:", min_value=1, max_value=50, value=10)
        
        with col2:
            domain = st.selectbox("Domain:", ["greetings", "business", "technology", "travel"])
        
        if st.button("Generate Sample Texts"):
            generator = SyntheticDatasetGenerator()
            # Generate sample texts (simplified for demo)
            sample_texts = [
                "Hello, how are you today?",
                "The project deadline has been extended.",
                "Machine learning algorithms are becoming more sophisticated.",
                "The hotel is located in the city center."
            ]
            texts_to_translate = sample_texts[:num_samples]
    
    # Display input texts
    if texts_to_translate:
        st.subheader(f"Texts to Translate ({len(texts_to_translate)} items)")
        
        with st.expander("View Input Texts"):
            for i, text in enumerate(texts_to_translate):
                st.write(f"{i+1}. {text}")
        
        # Translation options
        col1, col2 = st.columns(2)
        
        with col1:
            max_length = st.slider("Max Length", 50, 512, 256, key="batch_max_length")
        
        with col2:
            num_beams = st.slider("Number of Beams", 1, 8, 4, key="batch_num_beams")
        
        # Translate button
        if st.button("🚀 Translate All", type="primary"):
            if texts_to_translate:
                with st.spinner("Translating batch..."):
                    start_time = time.time()
                    
                    # Update engine configuration
                    st.session_state.translation_engine.config.max_length = max_length
                    st.session_state.translation_engine.config.num_beams = num_beams
                    
                    # Translate batch
                    translations = st.session_state.translation_engine.translate(texts_to_translate)
                    
                    translation_time = time.time() - start_time
                
                # Display results
                st.subheader("Translation Results")
                
                # Create results dataframe
                results_df = pd.DataFrame({
                    'Original': texts_to_translate,
                    'Translation': translations
                })
                
                st.dataframe(results_df, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="translation_results.csv",
                    mime="text/csv"
                )
                
                # Summary metrics
                st.subheader("Batch Translation Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Translations", len(translations))
                
                with col2:
                    st.metric("Average Time per Translation", f"{translation_time/len(translations):.3f}s")
                
                with col3:
                    avg_input_length = sum(len(text.split()) for text in texts_to_translate) / len(texts_to_translate)
                    st.metric("Average Input Length", f"{avg_input_length:.1f} words")

def evaluation_tab():
    """Evaluation interface."""
    st.header("📊 Translation Evaluation")
    
    st.write("Evaluate translation quality using various metrics including BLEU, CHRF, and ROUGE scores.")
    
    # Evaluation options
    eval_method = st.radio(
        "Choose evaluation method:",
        ["Upload Test Data", "Generate Test Data", "Manual Evaluation"]
    )
    
    if eval_method == "Upload Test Data":
        st.subheader("Upload Test Data")
        uploaded_file = st.file_uploader(
            "Upload test data (JSON format):",
            type=['json'],
            help="Upload a JSON file with 'source', 'target', and optional 'domain' fields"
        )
        
        if uploaded_file:
            test_data = json.load(uploaded_file)
            st.success(f"Loaded {len(test_data)} test samples")
    
    elif eval_method == "Generate Test Data":
        st.subheader("Generate Test Data")
        col1, col2 = st.columns(2)
        
        with col1:
            num_samples = st.number_input("Number of samples:", min_value=1, max_value=100, value=20)
        
        with col2:
            domain = st.selectbox("Domain:", ["greetings", "business", "technology", "travel"])
        
        if st.button("Generate Test Data"):
            generator = SyntheticDatasetGenerator()
            test_data = generator.generate_test_set(("en", "fr"), num_samples)
            st.success(f"Generated {len(test_data)} test samples")
    
    else:  # Manual Evaluation
        st.subheader("Manual Evaluation")
        st.write("Enter reference and hypothesis translations manually:")
        
        reference = st.text_input("Reference Translation:")
        hypothesis = st.text_input("Hypothesis Translation:")
        
        if reference and hypothesis:
            test_data = [{"target": reference}]
            predictions = [hypothesis]
        else:
            test_data = []
            predictions = []
    
    # Run evaluation
    if 'test_data' in locals() and test_data:
        if st.button("🔍 Run Evaluation"):
            if not st.session_state.translation_engine:
                st.warning("Please load a translation engine first.")
                return
            
            with st.spinner("Running evaluation..."):
                # Generate predictions
                sources = [item["source"] for item in test_data]
                predictions = st.session_state.translation_engine.translate(sources)
                
                # Evaluate
                evaluator = st.session_state.evaluator
                avg_metrics = evaluator.evaluate_batch(
                    [item["target"] for item in test_data],
                    predictions
                )
                
                # Domain analysis (if available)
                domain_results = evaluator.evaluate_by_domain(test_data, predictions)
            
            # Display results
            st.subheader("Evaluation Results")
            
            # Overall metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("BLEU Score", f"{avg_metrics['bleu_score']:.4f}")
            
            with col2:
                st.metric("CHRF Score", f"{avg_metrics['chrf_score']:.4f}")
            
            with col3:
                st.metric("ROUGE-L Score", f"{avg_metrics['rouge_l_score']:.4f}")
            
            with col4:
                if 'bert_score_f1' in avg_metrics:
                    st.metric("BERT Score F1", f"{avg_metrics['bert_score_f1']:.4f}")
                else:
                    st.metric("BERT Score F1", "N/A")
            
            # Domain analysis
            if len(domain_results) > 1:
                st.subheader("Domain Analysis")
                
                domain_df = pd.DataFrame([
                    {
                        "Domain": domain,
                        "BLEU Score": metrics["bleu_score"],
                        "CHRF Score": metrics["chrf_score"],
                        "ROUGE-L Score": metrics["rouge_l_score"],
                        "Sample Count": metrics["sample_count"]
                    }
                    for domain, metrics in domain_results.items()
                ])
                
                st.dataframe(domain_df, use_container_width=True)
                
                # Domain comparison chart
                fig = px.bar(
                    domain_df, 
                    x="Domain", 
                    y=["BLEU Score", "CHRF Score", "ROUGE-L Score"],
                    title="Translation Quality by Domain",
                    barmode="group"
                )
                st.plotly_chart(fig, use_container_width=True)

def analytics_tab():
    """Analytics and visualization tab."""
    st.header("📈 Translation Analytics")
    
    st.write("Visualize translation patterns and performance metrics.")
    
    # Sample analytics data
    st.subheader("Performance Metrics Over Time")
    
    # Generate sample data
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    bleu_scores = np.random.normal(0.75, 0.05, 30)
    chrf_scores = np.random.normal(0.80, 0.04, 30)
    
    analytics_df = pd.DataFrame({
        'Date': dates,
        'BLEU Score': bleu_scores,
        'CHRF Score': chrf_scores
    })
    
    # Line chart
    fig = px.line(
        analytics_df, 
        x='Date', 
        y=['BLEU Score', 'CHRF Score'],
        title='Translation Quality Metrics Over Time'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Language pair distribution
    st.subheader("Language Pair Usage")
    
    language_pairs = ['en-fr', 'en-de', 'en-es', 'fr-en', 'de-en', 'es-en']
    usage_counts = [45, 32, 28, 25, 20, 18]
    
    fig = px.pie(
        values=usage_counts,
        names=language_pairs,
        title='Language Pair Usage Distribution'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Text length analysis
    st.subheader("Text Length Distribution")
    
    text_lengths = np.random.normal(15, 8, 1000)
    text_lengths = np.clip(text_lengths, 1, 50)  # Clip to reasonable range
    
    fig = px.histogram(
        x=text_lengths,
        nbins=20,
        title='Distribution of Input Text Lengths',
        labels={'x': 'Number of Words', 'y': 'Frequency'}
    )
    st.plotly_chart(fig, use_container_width=True)

def about_tab():
    """About tab with project information."""
    st.header("ℹ️ About Neural Machine Translation")
    
    st.markdown("""
    ## Project Overview
    
    This Neural Machine Translation system uses state-of-the-art transformer models
    from Hugging Face to provide high-quality translations between multiple language pairs.
    
    ## Features
    
    - **Multiple Language Pairs**: Support for 10+ language combinations
    - **Real-time Translation**: Instant translation with configurable parameters
    - **Batch Processing**: Translate multiple texts efficiently
    - **Comprehensive Evaluation**: BLEU, CHRF, ROUGE, and BERT scores
    - **Domain Analysis**: Evaluate performance across different text domains
    - **Visualization**: Interactive charts and analytics
    
    ## Supported Languages
    
    - English (en)
    - French (fr)
    - German (de)
    - Spanish (es)
    - Italian (it)
    - Portuguese (pt)
    
    ## Technology Stack
    
    - **Backend**: Python, PyTorch, Transformers
    - **Frontend**: Streamlit
    - **Models**: MarianMT (Helsinki-NLP)
    - **Evaluation**: SacreBLEU, ROUGE, BERT-Score
    
    ## Usage Tips
    
    1. **Load Engine**: Select your desired language pair and click "Load Translation Engine"
    2. **Single Translation**: Use the first tab for quick translations
    3. **Batch Processing**: Upload files or generate sample data for bulk translation
    4. **Evaluation**: Test your translations with comprehensive metrics
    5. **Analytics**: View performance trends and usage patterns
    
    ## Performance Notes
    
    - First-time model loading may take a few minutes
    - Translation speed depends on text length and hardware
    - GPU acceleration is automatically used when available
    - Models are cached for faster subsequent loads
    """)
    
    # Technical specifications
    st.subheader("Technical Specifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Model Details:**
        - Architecture: MarianMT (Transformer-based)
        - Parameters: ~100M per language pair
        - Training Data: OPUS parallel corpora
        - Max Sequence Length: 512 tokens
        """)
    
    with col2:
        st.markdown("""
        **System Requirements:**
        - Python 3.8+
        - PyTorch 1.9+
        - 4GB+ RAM recommended
        - GPU support optional
        """)

if __name__ == "__main__":
    main()
