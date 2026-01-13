"""
Neural Machine Translation Engine

This module provides a modern, robust neural machine translation system using
Hugging Face transformers with support for multiple language pairs and
state-of-the-art techniques including zero-shot translation and advanced
generation strategies.
"""

import logging
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import torch
from transformers import (
    MarianMTModel, 
    MarianTokenizer, 
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
    GenerationConfig,
    BitsAndBytesConfig
)
import warnings
import time
from contextlib import contextmanager
import gc

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TranslationConfig:
    """Configuration for translation models."""
    model_name: str
    max_length: int = 512
    num_beams: int = 4
    early_stopping: bool = True
    temperature: float = 1.0
    do_sample: bool = False
    device: str = "auto"
    batch_size: int = 1
    use_cache: bool = True
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    use_quantization: bool = False
    quantization_config: Optional[Dict[str, Any]] = None
    trust_remote_code: bool = False
    low_cpu_mem_usage: bool = True


class TranslationEngine:
    """
    Modern Neural Machine Translation Engine with support for multiple models and techniques.
    
    Features:
    - Support for multiple language pairs
    - Advanced generation strategies
    - Memory-efficient loading
    - Quantization support
    - Batch processing
    - Performance monitoring
    """
    
    # Supported language pairs with their model names
    SUPPORTED_PAIRS = {
        "en-fr": "Helsinki-NLP/opus-mt-en-fr",
        "en-de": "Helsinki-NLP/opus-mt-en-de", 
        "en-es": "Helsinki-NLP/opus-mt-en-es",
        "en-it": "Helsinki-NLP/opus-mt-en-it",
        "en-pt": "Helsinki-NLP/opus-mt-en-pt",
        "fr-en": "Helsinki-NLP/opus-mt-fr-en",
        "de-en": "Helsinki-NLP/opus-mt-de-en",
        "es-en": "Helsinki-NLP/opus-mt-es-en",
        "it-en": "Helsinki-NLP/opus-mt-it-en",
        "pt-en": "Helsinki-NLP/opus-mt-pt-en",
        # Additional modern models
        "en-ru": "Helsinki-NLP/opus-mt-en-ru",
        "en-zh": "Helsinki-NLP/opus-mt-en-zh",
        "en-ja": "Helsinki-NLP/opus-mt-en-ja",
        "en-ar": "Helsinki-NLP/opus-mt-en-ar",
    }
    
    # Zero-shot translation models
    ZERO_SHOT_MODELS = {
        "mbart": "facebook/mbart-large-50-many-to-many-mmt",
        "m2m100": "facebook/m2m100_418M",
    }
    
    def __init__(self, config: Optional[TranslationConfig] = None):
        """
        Initialize the translation engine.
        
        Args:
            config: Translation configuration. If None, uses default settings.
        """
        self.config = config or TranslationConfig(model_name="Helsinki-NLP/opus-mt-en-fr")
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        self.generation_config = None
        self.performance_stats = {
            "total_translations": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        self._load_model()
    
    def _get_device(self) -> str:
        """Determine the best available device with fallback options."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                device = "mps"
                logger.info("Using MPS device (Apple Silicon)")
            else:
                device = "cpu"
                logger.info("Using CPU device")
        else:
            device = self.config.device
            logger.info(f"Using specified device: {device}")
        
        return device
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration if enabled."""
        if not self.config.use_quantization:
            return None
        
        if self.config.quantization_config:
            return BitsAndBytesConfig(**self.config.quantization_config)
        
        # Default 8-bit quantization
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    
    def _load_model(self) -> None:
        """Load the translation model and tokenizer with modern optimizations."""
        try:
            logger.info(f"Loading model: {self.config.model_name}")
            logger.info(f"Using device: {self.device}")
            
            # Get quantization config
            quantization_config = self._get_quantization_config()
            
            # Load tokenizer first
            self.tokenizer = MarianTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # Load model with optimizations
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "trust_remote_code": self.config.trust_remote_code,
                "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                logger.info("Using quantization for memory efficiency")
            
            self.model = MarianMTModel.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            
            # Move model to device
            if not quantization_config:  # Quantized models are already on device
                self.model = self.model.to(self.device)
            
            # Create generation config
            self.generation_config = GenerationConfig(
                max_length=self.config.max_length,
                num_beams=self.config.num_beams,
                early_stopping=self.config.early_stopping,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                repetition_penalty=self.config.repetition_penalty,
                length_penalty=self.config.length_penalty,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    @contextmanager
    def _performance_monitor(self, operation: str):
        """Context manager for performance monitoring."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.performance_stats["total_time"] += duration
            self.performance_stats["total_translations"] += 1
            self.performance_stats["average_time"] = (
                self.performance_stats["total_time"] / 
                self.performance_stats["total_translations"]
            )
            logger.debug(f"{operation} completed in {duration:.3f}s")
    
    def translate(
        self, 
        text: Union[str, List[str]], 
        target_language: Optional[str] = None,
        **generation_kwargs
    ) -> Union[str, List[str]]:
        """
        Translate text from source to target language.
        
        Args:
            text: Input text(s) to translate
            target_language: Target language code (optional, inferred from model)
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Translated text(s)
        """
        if isinstance(text, str):
            return self._translate_single(text, **generation_kwargs)
        else:
            return self._translate_batch(text, **generation_kwargs)
    
    def _translate_single(self, text: str, **generation_kwargs) -> str:
        """Translate a single text with performance monitoring."""
        if not text.strip():
            return ""
        
        with self._performance_monitor("single_translation"):
            try:
                # Tokenize input
                inputs = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=self.config.max_length
                ).to(self.device)
                
                # Merge generation config with kwargs
                gen_config = self.generation_config
                if generation_kwargs:
                    gen_config = GenerationConfig(
                        **{k: v for k, v in gen_config.to_dict().items()},
                        **generation_kwargs
                    )
                
                # Generate translation
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=gen_config,
                        use_cache=self.config.use_cache
                    )
                
                # Decode output
                translated_text = self.tokenizer.decode(
                    outputs[0], 
                    skip_special_tokens=True
                )
                
                return translated_text.strip()
                
            except Exception as e:
                logger.error(f"Translation failed for text '{text[:50]}...': {e}")
                return f"Translation error: {str(e)}"
    
    def _translate_batch(self, texts: List[str], **generation_kwargs) -> List[str]:
        """Translate a batch of texts efficiently."""
        if not texts:
            return []
        
        with self._performance_monitor("batch_translation"):
            results = []
            
            # Process in chunks for memory efficiency
            batch_size = self.config.batch_size
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_results = []
                
                for text in batch:
                    result = self._translate_single(text, **generation_kwargs)
                    batch_results.append(result)
                
                results.extend(batch_results)
                
                # Clear cache periodically
                if i % (batch_size * 5) == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            return results
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported language pairs."""
        return self.SUPPORTED_PAIRS.copy()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self.performance_stats.copy()
    
    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        self.performance_stats = {
            "total_translations": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.model:
            return {"error": "No model loaded"}
        
        return {
            "model_name": self.config.model_name,
            "device": self.device,
            "model_type": type(self.model).__name__,
            "tokenizer_type": type(self.tokenizer).__name__,
            "vocab_size": self.tokenizer.vocab_size,
            "max_length": self.config.max_length,
            "num_beams": self.config.num_beams,
            "quantization_enabled": self.config.use_quantization,
        }
    
    def switch_language_pair(self, source_lang: str, target_lang: str) -> None:
        """
        Switch to a different language pair.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
        """
        pair_key = f"{source_lang}-{target_lang}"
        if pair_key not in self.SUPPORTED_PAIRS:
            raise ValueError(f"Unsupported language pair: {pair_key}")
        
        model_name = self.SUPPORTED_PAIRS[pair_key]
        if model_name != self.config.model_name:
            logger.info(f"Switching from {self.config.model_name} to {model_name}")
            
            # Clear current model from memory
            if self.model:
                del self.model
                del self.tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Update config and reload
            self.config.model_name = model_name
            self._load_model()
            logger.info(f"Successfully switched to language pair: {pair_key}")
    
    def translate_with_confidence(
        self, 
        text: str, 
        return_confidence: bool = True,
        **generation_kwargs
    ) -> Union[str, Tuple[str, float]]:
        """
        Translate text with confidence score.
        
        Args:
            text: Input text to translate
            return_confidence: Whether to return confidence score
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Translation result with optional confidence score
        """
        if not return_confidence:
            return self._translate_single(text, **generation_kwargs)
        
        try:
            # Generate multiple candidates for confidence estimation
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=self.config.max_length
            ).to(self.device)
            
            # Generate multiple outputs
            gen_config = GenerationConfig(
                **{k: v for k, v in self.generation_config.to_dict().items()},
                num_return_sequences=3,
                **generation_kwargs
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config,
                    use_cache=self.config.use_cache
                )
            
            # Decode outputs
            translations = []
            for output in outputs:
                translation = self.tokenizer.decode(output, skip_special_tokens=True)
                translations.append(translation.strip())
            
            # Use the first translation as primary result
            primary_translation = translations[0]
            
            # Calculate confidence based on consistency
            if len(set(translations)) == 1:
                confidence = 1.0  # All translations identical
            else:
                # Simple confidence based on agreement
                confidence = len(set(translations)) / len(translations)
            
            return primary_translation, confidence
            
        except Exception as e:
            logger.error(f"Confidence translation failed: {e}")
            return self._translate_single(text, **generation_kwargs), 0.0


class ModernTranslationPipeline:
    """
    Modern translation pipeline using Hugging Face pipelines for easier usage.
    
    This class provides a simplified interface using HF pipelines with
    automatic device detection and optimized settings.
    """
    
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-fr", **pipeline_kwargs):
        """
        Initialize the translation pipeline.
        
        Args:
            model_name: Name of the translation model to use
            **pipeline_kwargs: Additional pipeline configuration
        """
        self.model_name = model_name
        self.pipeline = None
        self.device = self._get_device()
        self._load_pipeline(**pipeline_kwargs)
    
    def _get_device(self) -> int:
        """Determine the best available device for pipeline."""
        if torch.cuda.is_available():
            return 0
        elif torch.backends.mps.is_available():
            return 0  # MPS uses device 0
        else:
            return -1  # CPU
    
    def _load_pipeline(self, **pipeline_kwargs) -> None:
        """Load the translation pipeline with optimizations."""
        try:
            logger.info(f"Loading translation pipeline: {self.model_name}")
            
            # Default pipeline configuration
            default_kwargs = {
                "device": self.device,
                "batch_size": 1,
                "max_length": 512,
                "num_beams": 4,
                "early_stopping": True,
            }
            
            # Merge with user-provided kwargs
            config = {**default_kwargs, **pipeline_kwargs}
            
            self.pipeline = pipeline(
                "translation",
                model=self.model_name,
                **config
            )
            logger.info("Pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise RuntimeError(f"Pipeline loading failed: {e}") from e
    
    def translate(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Translate text using the pipeline.
        
        Args:
            text: Input text(s) to translate
            
        Returns:
            Translated text(s)
        """
        try:
            result = self.pipeline(text)
            
            if isinstance(text, str):
                return result[0]["translation_text"]
            else:
                return [item["translation_text"] for item in result]
                
        except Exception as e:
            logger.error(f"Pipeline translation failed: {e}")
            return f"Translation error: {str(e)}"


class ZeroShotTranslationEngine:
    """
    Zero-shot translation engine using multilingual models.
    
    This engine can translate between language pairs not explicitly trained
    using models like mBART or M2M100.
    """
    
    def __init__(self, model_name: str = "facebook/mbart-large-50-many-to-many-mmt"):
        """
        Initialize the zero-shot translation engine.
        
        Args:
            model_name: Name of the multilingual model to use
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        self._load_model()
    
    def _get_device(self) -> str:
        """Determine the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self) -> None:
        """Load the multilingual model and tokenizer."""
        try:
            logger.info(f"Loading zero-shot model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.model = self.model.to(self.device)
            logger.info("Zero-shot model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load zero-shot model: {e}")
            raise RuntimeError(f"Zero-shot model loading failed: {e}") from e
    
    def translate(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str
    ) -> str:
        """
        Translate text using zero-shot approach.
        
        Args:
            text: Input text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        try:
            # Set source and target language tokens
            self.tokenizer.src_lang = source_lang
            
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang],
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode output
            translated_text = self.tokenizer.batch_decode(
                generated_tokens, 
                skip_special_tokens=True
            )[0]
            
            return translated_text.strip()
            
        except Exception as e:
            logger.error(f"Zero-shot translation failed: {e}")
            return f"Translation error: {str(e)}"
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return list(self.tokenizer.lang_code_to_id.keys())


def create_translation_engine(
    language_pair: str = "en-fr",
    use_pipeline: bool = False,
    use_zero_shot: bool = False,
    **kwargs
) -> Union[TranslationEngine, ModernTranslationPipeline, ZeroShotTranslationEngine]:
    """
    Factory function to create a translation engine.
    
    Args:
        language_pair: Language pair (e.g., "en-fr")
        use_pipeline: Whether to use the pipeline approach
        use_zero_shot: Whether to use zero-shot translation
        **kwargs: Additional configuration parameters
        
    Returns:
        Translation engine instance
    """
    if use_zero_shot:
        logger.info("Creating zero-shot translation engine")
        return ZeroShotTranslationEngine(**kwargs)
    
    if language_pair not in TranslationEngine.SUPPORTED_PAIRS:
        logger.warning(f"Unsupported language pair: {language_pair}, falling back to zero-shot")
        return ZeroShotTranslationEngine(**kwargs)
    
    model_name = TranslationEngine.SUPPORTED_PAIRS[language_pair]
    
    if use_pipeline:
        logger.info(f"Creating pipeline engine for {language_pair}")
        return ModernTranslationPipeline(model_name, **kwargs)
    else:
        logger.info(f"Creating standard engine for {language_pair}")
        config = TranslationConfig(model_name=model_name, **kwargs)
        return TranslationEngine(config)


def create_advanced_translation_engine(
    language_pair: str = "en-fr",
    quantization: bool = False,
    batch_size: int = 1,
    **kwargs
) -> TranslationEngine:
    """
    Create an advanced translation engine with modern features.
    
    Args:
        language_pair: Language pair (e.g., "en-fr")
        quantization: Whether to use quantization
        batch_size: Batch size for processing
        **kwargs: Additional configuration parameters
        
    Returns:
        Advanced translation engine instance
    """
    model_name = TranslationEngine.SUPPORTED_PAIRS.get(
        language_pair, 
        "Helsinki-NLP/opus-mt-en-fr"
    )
    
    config = TranslationConfig(
        model_name=model_name,
        use_quantization=quantization,
        batch_size=batch_size,
        **kwargs
    )
    
    return TranslationEngine(config)


if __name__ == "__main__":
    # Example usage
    print("🌐 Neural Machine Translation Engine Demo")
    print("=" * 60)
    
    # Test different engine types
    print("\n1. Standard Translation Engine")
    print("-" * 40)
    
    try:
        engine = create_translation_engine("en-fr")
        print(f"✅ Engine created: {engine.get_model_info()['model_name']}")
        
        # Test translation
        test_text = "Hello, how are you today?"
        translation = engine.translate(test_text)
        print(f"EN: {test_text}")
        print(f"FR: {translation}")
        
        # Test confidence translation
        translation_with_conf, confidence = engine.translate_with_confidence(test_text)
        print(f"Confidence: {confidence:.2f}")
        
        # Performance stats
        stats = engine.get_performance_stats()
        print(f"Performance: {stats['average_time']:.3f}s average")
        
    except Exception as e:
        print(f"❌ Standard engine failed: {e}")
    
    print("\n2. Pipeline Translation Engine")
    print("-" * 40)
    
    try:
        pipeline = create_translation_engine("en-de", use_pipeline=True)
        test_texts = ["Good morning!", "How are you?"]
        translations = pipeline.translate(test_texts)
        
        for orig, trans in zip(test_texts, translations):
            print(f"EN: {orig} → DE: {trans}")
            
    except Exception as e:
        print(f"❌ Pipeline engine failed: {e}")
    
    print("\n3. Zero-Shot Translation Engine")
    print("-" * 40)
    
    try:
        zero_shot = create_translation_engine(use_zero_shot=True)
        langs = zero_shot.get_supported_languages()
        print(f"Supported languages: {len(langs)}")
        print(f"Sample languages: {langs[:10]}")
        
        # Test zero-shot translation
        test_text = "Hello, world!"
        translation = zero_shot.translate(test_text, "en_XX", "fr_XX")
        print(f"Zero-shot EN→FR: {test_text} → {translation}")
        
    except Exception as e:
        print(f"❌ Zero-shot engine failed: {e}")
    
    print("\n4. Advanced Engine with Quantization")
    print("-" * 40)
    
    try:
        advanced = create_advanced_translation_engine(
            "en-es", 
            quantization=False,  # Set to True if you have the required dependencies
            batch_size=2
        )
        
        batch_texts = [
            "The weather is beautiful today.",
            "Machine learning is fascinating."
        ]
        
        batch_translations = advanced.translate(batch_texts)
        
        for orig, trans in zip(batch_texts, batch_translations):
            print(f"EN: {orig} → ES: {trans}")
            
    except Exception as e:
        print(f"❌ Advanced engine failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Demo completed!")
    print("\nNext steps:")
    print("1. Run the web interface: streamlit run web_app/app.py")
    print("2. Use the CLI: python cli.py --help")
    print("3. Check the configuration: config/config.yaml")
