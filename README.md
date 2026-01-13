# Neural Machine Translation System

A comprehensive neural machine translation system built with state-of-the-art transformer models from Hugging Face. This project provides high-quality translations between multiple language pairs with advanced evaluation metrics, visualization tools, and an intuitive web interface.

## Features

- **Multiple Language Pairs**: Support for 10+ language combinations (EN↔FR, EN↔DE, EN↔ES, EN↔IT, EN↔PT)
- **Real-time Translation**: Instant translation with configurable parameters
- **Batch Processing**: Efficient translation of multiple texts
- **Comprehensive Evaluation**: BLEU, CHRF, ROUGE, and BERT scores
- **Domain Analysis**: Performance evaluation across different text domains
- **Interactive Visualization**: Charts and analytics for translation patterns
- **Web Interface**: User-friendly Streamlit application
- **Modern Architecture**: Type hints, logging, configuration management
- **Easy Setup**: Simple installation and configuration

## Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Neural-Machine-Translation-System.git
   cd Neural-Machine-Translation-System
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web application**:
   ```bash
   streamlit run web_app/app.py
   ```

4. **Access the interface**:
   Open your browser to `http://localhost:8501`

### Command Line Usage

```python
from src.translation_engine import create_translation_engine

# Create translation engine
engine = create_translation_engine("en-fr")

# Translate text
result = engine.translate("Hello, how are you today?")
print(f"Translation: {result}")
```

## 📁 Project Structure

```
neural-machine-translation/
├── src/                          # Source code
│   ├── translation_engine.py     # Core translation engine
│   ├── dataset_generator.py      # Synthetic dataset generation
│   └── evaluation.py            # Evaluation metrics and visualization
├── web_app/                      # Streamlit web application
│   └── app.py                   # Main web interface
├── data/                         # Data storage
│   ├── synthetic_dataset.json   # Generated datasets
│   └── test_set.json           # Test data
├── models/                       # Model storage and results
│   └── visualizations/         # Generated charts and plots
├── config/                       # Configuration files
│   └── config.yaml             # Main configuration
├── tests/                        # Unit tests
├── logs/                         # Log files
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🔧 Configuration

The system uses YAML configuration files for easy customization. Key configuration options:

### Model Settings
```yaml
model:
  default_language_pair: "en-fr"
  max_length: 512
  num_beams: 4
  early_stopping: true
  device: "auto"  # auto, cpu, cuda, mps
```

### Language Pairs
```yaml
language_pairs:
  en-fr: "Helsinki-NLP/opus-mt-en-fr"
  en-de: "Helsinki-NLP/opus-mt-en-de"
  en-es: "Helsinki-NLP/opus-mt-en-es"
  # ... more pairs
```

## Supported Languages

| Language | Code | Status |
|----------|------|--------|
| English  | en   | ✅     |
| French   | fr   | ✅     |
| German   | de   | ✅     |
| Spanish  | es   | ✅     |
| Italian  | it   | ✅     |
| Portuguese | pt | ✅     |

## Evaluation Metrics

The system provides comprehensive evaluation using multiple metrics:

- **BLEU Score**: Measures n-gram precision with brevity penalty
- **CHRF Score**: Character-level F-score for better handling of morphology
- **ROUGE-L Score**: Longest common subsequence-based evaluation
- **BERT Score**: Contextual embedding-based semantic similarity

### Example Evaluation

```python
from src.evaluation import TranslationEvaluator

evaluator = TranslationEvaluator()
metrics = evaluator.evaluate_single(
    reference="Bonjour, comment allez-vous ?",
    hypothesis="Hello, how are you?"
)
print(f"BLEU Score: {metrics.bleu_score:.4f}")
```

## Visualization Features

- **Performance Trends**: Track translation quality over time
- **Domain Analysis**: Compare performance across different text domains
- **Language Pair Usage**: Visualize usage patterns
- **Text Length Analysis**: Understand performance by input length

## Web Interface

The Streamlit web application provides:

1. **Single Translation**: Quick translations with real-time results
2. **Batch Processing**: Upload files or generate sample data
3. **Evaluation Dashboard**: Comprehensive metrics and analysis
4. **Analytics**: Interactive charts and performance insights
5. **Configuration**: Easy model and parameter adjustment

### Web Interface Features

- **Real-time Translation**: Instant results with configurable parameters
- **File Upload**: Support for TXT and JSON files
- **Batch Processing**: Efficient handling of multiple texts
- **Interactive Charts**: Plotly-powered visualizations
- **Export Results**: Download translations as CSV files
- **Responsive Design**: Works on desktop and mobile devices

## Testing and Evaluation

### Generate Test Data

```python
from src.dataset_generator import SyntheticDatasetGenerator

generator = SyntheticDatasetGenerator()
dataset_path = generator.generate_dataset(
    language_pairs=[("en", "fr"), ("en", "de")],
    domains=["greetings", "business", "technology"],
    samples_per_domain=50
)
```

### Run Evaluation

```python
from src.evaluation import TranslationEvaluator, load_test_data

# Load test data
test_data = load_test_data("data/test_set.json")

# Create evaluator
evaluator = TranslationEvaluator()

# Evaluate translations
results = evaluator.evaluate_by_domain(test_data, predictions)
```

## 🔧 Advanced Usage

### Custom Model Configuration

```python
from src.translation_engine import TranslationEngine, TranslationConfig

config = TranslationConfig(
    model_name="Helsinki-NLP/opus-mt-en-fr",
    max_length=256,
    num_beams=6,
    temperature=0.8,
    do_sample=True
)

engine = TranslationEngine(config)
```

### Pipeline-based Translation

```python
from src.translation_engine import ModernTranslationPipeline

pipeline = ModernTranslationPipeline("Helsinki-NLP/opus-mt-en-fr")
result = pipeline.translate("Hello, world!")
```

## Performance Optimization

- **GPU Acceleration**: Automatic CUDA/MPS detection and usage
- **Model Caching**: Efficient model loading and caching
- **Batch Processing**: Optimized for multiple translations
- **Memory Management**: Efficient memory usage for large texts

## 🛠️ Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Quality

The project follows modern Python best practices:

- **Type Hints**: Full type annotation support
- **PEP 8**: Consistent code formatting
- **Docstrings**: Comprehensive documentation
- **Logging**: Structured logging throughout
- **Error Handling**: Robust error management

### Adding New Language Pairs

1. Add the model to `config/config.yaml`
2. Update `SUPPORTED_PAIRS` in `translation_engine.py`
3. Test with the new language pair
4. Update documentation

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.20+
- Streamlit 1.20+
- 4GB+ RAM recommended
- GPU support optional but recommended

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Hugging Face**: For the excellent transformers library and pre-trained models
- **Helsinki-NLP**: For the MarianMT models
- **SacreBLEU**: For evaluation metrics
- **Streamlit**: For the web interface framework

## Support

For questions, issues, or contributions:

- Create an issue on GitHub
- Check the documentation
- Review the configuration examples

## Future Enhancements

- [ ] Support for more language pairs
- [ ] Fine-tuning capabilities
- [ ] Real-time translation API
- [ ] Mobile application
- [ ] Advanced visualization features
- [ ] Integration with external translation services
- [ ] Multi-modal translation (text + images)
# Neural-Machine-Translation-System
