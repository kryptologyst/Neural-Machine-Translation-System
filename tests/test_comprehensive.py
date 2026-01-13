"""
Comprehensive Test Suite for Neural Machine Translation System

This module contains extensive unit tests, integration tests, and performance tests
for all components of the translation system.
"""

import unittest
import tempfile
import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from translation_engine import TranslationEngine, TranslationConfig, create_translation_engine, ModernTranslationPipeline
from dataset_generator import SyntheticDatasetGenerator, TranslationSample
from evaluation import TranslationEvaluator, TranslationVisualizer
from config_manager import ConfigManager, AppConfig
from logging_config import TranslationLogger, PerformanceLogger, StructuredLogger


class TestTranslationEngine(unittest.TestCase):
    """Comprehensive test cases for TranslationEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TranslationConfig(
            model_name="Helsinki-NLP/opus-mt-en-fr",
            max_length=128,
            num_beams=2
        )
    
    @patch('src.translation_engine.MarianMTModel')
    @patch('src.translation_engine.MarianTokenizer')
    def test_engine_initialization(self, mock_tokenizer, mock_model):
        """Test engine initialization."""
        mock_model.from_pretrained.return_value = Mock()
        mock_tokenizer.from_pretrained.return_value = Mock()
        
        engine = TranslationEngine(self.config)
        
        self.assertIsNotNone(engine.model)
        self.assertIsNotNone(engine.tokenizer)
        mock_model.from_pretrained.assert_called_once()
        mock_tokenizer.from_pretrained.assert_called_once()
    
    def test_supported_languages(self):
        """Test supported languages dictionary."""
        languages = TranslationEngine.SUPPORTED_PAIRS
        self.assertIn("en-fr", languages)
        self.assertIn("en-de", languages)
        self.assertIn("en-es", languages)
        self.assertEqual(len(languages), 10)
    
    def test_create_translation_engine(self):
        """Test factory function."""
        with patch('src.translation_engine.TranslationEngine') as mock_engine:
            mock_engine.return_value = Mock()
            engine = create_translation_engine("en-fr")
            self.assertIsNotNone(engine)
    
    @patch('src.translation_engine.MarianMTModel')
    @patch('src.translation_engine.MarianTokenizer')
    def test_translate_single(self, mock_tokenizer, mock_model):
        """Test single text translation."""
        # Mock model and tokenizer
        mock_model_instance = Mock()
        mock_model_instance.generate.return_value = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {"input_ids": Mock()}
        mock_tokenizer_instance.decode.return_value = "Bonjour le monde"
        mock_tokenizer_instance.pad_token_id = 0
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        engine = TranslationEngine(self.config)
        result = engine.translate("Hello world")
        
        self.assertEqual(result, "Bonjour le monde")
    
    def test_switch_language_pair(self):
        """Test switching language pairs."""
        with patch('src.translation_engine.TranslationEngine._load_model'):
            engine = TranslationEngine(self.config)
            
            with patch.object(engine, '_load_model') as mock_load:
                engine.switch_language_pair("en", "de")
                mock_load.assert_called_once()
    
    def test_get_supported_languages(self):
        """Test getting supported languages."""
        with patch('src.translation_engine.TranslationEngine._load_model'):
            engine = TranslationEngine(self.config)
            languages = engine.get_supported_languages()
            
            self.assertIsInstance(languages, dict)
            self.assertIn("en-fr", languages)


class TestModernTranslationPipeline(unittest.TestCase):
    """Test cases for ModernTranslationPipeline."""
    
    @patch('src.translation_engine.pipeline')
    def test_pipeline_initialization(self, mock_pipeline):
        """Test pipeline initialization."""
        mock_pipeline.return_value = Mock()
        
        pipeline = ModernTranslationPipeline("Helsinki-NLP/opus-mt-en-fr")
        
        self.assertIsNotNone(pipeline.pipeline)
        mock_pipeline.assert_called_once()
    
    @patch('src.translation_engine.pipeline')
    def test_pipeline_translate(self, mock_pipeline):
        """Test pipeline translation."""
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [{"translation_text": "Bonjour"}]
        mock_pipeline.return_value = mock_pipeline_instance
        
        pipeline = ModernTranslationPipeline("Helsinki-NLP/opus-mt-en-fr")
        result = pipeline.translate("Hello")
        
        self.assertEqual(result, "Bonjour")


class TestDatasetGenerator(unittest.TestCase):
    """Comprehensive test cases for SyntheticDatasetGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = SyntheticDatasetGenerator(self.temp_dir)
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        self.assertEqual(self.generator.output_dir, Path(self.temp_dir))
        self.assertIn("greetings", self.generator.TEMPLATES)
        self.assertIn("business", self.generator.TEMPLATES)
    
    def test_determine_complexity(self):
        """Test complexity determination."""
        simple_text = "Hello world."
        medium_text = "This is a medium length sentence with some complexity."
        complex_text = "This is a very complex sentence with multiple clauses, punctuation, and extensive vocabulary that should be classified as complex."
        
        self.assertEqual(self.generator._determine_complexity(simple_text), "simple")
        self.assertEqual(self.generator._determine_complexity(medium_text), "medium")
        self.assertEqual(self.generator._determine_complexity(complex_text), "complex")
    
    def test_determine_length(self):
        """Test length determination."""
        short_text = "Hi"
        medium_text = "This is a medium length text."
        long_text = "This is a very long text that should be classified as long because it exceeds the character limit for medium length texts."
        
        self.assertEqual(self.generator._determine_length(short_text), "short")
        self.assertEqual(self.generator._determine_length(medium_text), "medium")
        self.assertEqual(self.generator._determine_length(long_text), "long")
    
    def test_generate_test_set(self):
        """Test test set generation."""
        test_set = self.generator.generate_test_set(("en", "fr"), 10)
        
        self.assertEqual(len(test_set), 10)
        self.assertIn("source", test_set[0])
        self.assertIn("target", test_set[0])
        self.assertIn("domain", test_set[0])
    
    def test_generate_dataset(self):
        """Test dataset generation."""
        dataset_path = self.generator.generate_dataset(
            language_pairs=[("en", "fr")],
            domains=["greetings"],
            samples_per_domain=5
        )
        
        self.assertTrue(Path(dataset_path).exists())
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)


class TestTranslationEvaluator(unittest.TestCase):
    """Comprehensive test cases for TranslationEvaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = TranslationEvaluator()
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        self.assertIsNotNone(self.evaluator)
    
    @patch('src.evaluation.SACREBLEU_AVAILABLE', True)
    @patch('src.evaluation.BLEU')
    @patch('src.evaluation.CHRF')
    def test_evaluate_single(self, mock_chrf, mock_bleu):
        """Test single evaluation."""
        mock_bleu_instance = Mock()
        mock_bleu_instance.sentence_score.return_value.score = 0.5
        mock_bleu.return_value = mock_bleu_instance
        
        mock_chrf_instance = Mock()
        mock_chrf_instance.sentence_score.return_value.score = 0.6
        mock_chrf.return_value = mock_chrf_instance
        
        with patch.object(self.evaluator, 'scorer') as mock_scorer:
            mock_scorer.score.return_value = {'rougeL': Mock(fmeasure=0.7)}
            
            metrics = self.evaluator.evaluate_single(
                "Bonjour, comment allez-vous ?",
                "Hello, how are you?"
            )
            
            self.assertEqual(metrics.bleu_score, 0.5)
            self.assertEqual(metrics.chrf_score, 0.6)
            self.assertEqual(metrics.rouge_l_score, 0.7)
    
    def test_evaluate_batch(self):
        """Test batch evaluation."""
        references = ["Bonjour", "Merci"]
        hypotheses = ["Hello", "Thank you"]
        
        with patch.object(self.evaluator, 'evaluate_single') as mock_evaluate:
            mock_evaluate.return_value = Mock(
                bleu_score=0.5,
                chrf_score=0.6,
                rouge_l_score=0.7
            )
            
            results = self.evaluator.evaluate_batch(references, hypotheses)
            
            self.assertIn("bleu_score", results)
            self.assertIn("chrf_score", results)
            self.assertIn("rouge_l_score", results)
            self.assertEqual(len(results), 3)
    
    def test_evaluate_by_domain(self):
        """Test domain-based evaluation."""
        test_data = [
            {"source": "Hello", "target": "Bonjour", "domain": "greetings"},
            {"source": "Thanks", "target": "Merci", "domain": "greetings"}
        ]
        predictions = ["Bonjour", "Merci"]
        
        with patch.object(self.evaluator, 'evaluate_single') as mock_evaluate:
            mock_evaluate.return_value = Mock(
                bleu_score=0.5,
                chrf_score=0.6,
                rouge_l_score=0.7
            )
            
            results = self.evaluator.evaluate_by_domain(test_data, predictions)
            
            self.assertIn("greetings", results)
            self.assertIn("bleu_score", results["greetings"])


class TestTranslationVisualizer(unittest.TestCase):
    """Test cases for TranslationVisualizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = TranslationVisualizer(self.temp_dir)
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        self.assertEqual(self.visualizer.output_dir, Path(self.temp_dir))
    
    def test_categorize_length(self):
        """Test length categorization."""
        self.assertEqual(self.visualizer._categorize_length(3), "Short (≤5 words)")
        self.assertEqual(self.visualizer._categorize_length(10), "Medium (6-15 words)")
        self.assertEqual(self.visualizer._categorize_length(20), "Long (>15 words)")
    
    def test_plot_metrics_comparison(self):
        """Test metrics comparison plotting."""
        results = {
            "Model A": {"bleu_score": 0.75, "chrf_score": 0.80},
            "Model B": {"bleu_score": 0.72, "chrf_score": 0.78}
        }
        
        # Test that the method runs without error
        try:
            self.visualizer.plot_metrics_comparison(results, interactive=False)
        except Exception as e:
            self.fail(f"plot_metrics_comparison raised {e}")


class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.yaml"
        
        # Create test config
        test_config = {
            "model": {
                "default_language_pair": "en-fr",
                "max_length": 256
            },
            "language_pairs": {
                "en-fr": "Helsinki-NLP/opus-mt-en-fr"
            }
        }
        
        with open(self.config_file, 'w') as f:
            import yaml
            yaml.dump(test_config, f)
    
    def test_config_loading(self):
        """Test configuration loading."""
        config_manager = ConfigManager(str(self.config_file))
        config = config_manager.load_config()
        
        self.assertEqual(config.model.default_language_pair, "en-fr")
        self.assertEqual(config.model.max_length, 256)
        self.assertIn("en-fr", config.language_pairs)
    
    def test_env_overrides(self):
        """Test environment variable overrides."""
        os.environ["NMT_MODEL_MAX_LENGTH"] = "512"
        
        config_manager = ConfigManager(str(self.config_file))
        config = config_manager.load_config()
        
        self.assertEqual(config.model.max_length, 512)
        
        # Clean up
        del os.environ["NMT_MODEL_MAX_LENGTH"]
    
    def test_config_validation(self):
        """Test configuration validation."""
        config_manager = ConfigManager(str(self.config_file))
        config = config_manager.load_config()
        
        errors = config_manager.validate_config()
        self.assertIsInstance(errors, list)


class TestLoggingConfig(unittest.TestCase):
    """Test cases for logging configuration."""
    
    def test_translation_logger(self):
        """Test translation logger."""
        logger = TranslationLogger("test")
        self.assertIsNotNone(logger.logger)
    
    def test_performance_logger(self):
        """Test performance logger."""
        logger = PerformanceLogger("test")
        self.assertIsNotNone(logger.logger)
    
    def test_structured_logger(self):
        """Test structured logger."""
        logger = StructuredLogger("test")
        self.assertIsNotNone(logger.logger)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from data generation to evaluation."""
        temp_dir = tempfile.mkdtemp()
        
        # Generate test data
        generator = SyntheticDatasetGenerator(temp_dir)
        test_data = generator.generate_test_set(("en", "fr"), 5)
        
        # Mock translation engine
        with patch('src.translation_engine.TranslationEngine') as mock_engine_class:
            mock_engine = Mock()
            mock_engine.translate.return_value = "Mock translation"
            mock_engine_class.return_value = mock_engine
            
            # Create engine and translate
            engine = create_translation_engine("en-fr")
            sources = [item["source"] for item in test_data]
            predictions = engine.translate(sources)
            
            # Evaluate
            evaluator = TranslationEvaluator()
            with patch.object(evaluator, 'evaluate_single') as mock_evaluate:
                mock_evaluate.return_value = Mock(
                    bleu_score=0.5,
                    chrf_score=0.6,
                    rouge_l_score=0.7
                )
                
                avg_metrics = evaluator.evaluate_batch(
                    [item["target"] for item in test_data],
                    predictions
                )
                
                self.assertIn("bleu_score", avg_metrics)
                self.assertIn("chrf_score", avg_metrics)
                self.assertIn("rouge_l_score", avg_metrics)


class TestPerformance(unittest.TestCase):
    """Performance tests."""
    
    def test_translation_performance(self):
        """Test translation performance."""
        with patch('src.translation_engine.TranslationEngine') as mock_engine_class:
            mock_engine = Mock()
            mock_engine.translate.return_value = "Mock translation"
            mock_engine_class.return_value = mock_engine
            
            engine = create_translation_engine("en-fr")
            
            # Test batch translation performance
            texts = ["Hello world"] * 100
            
            import time
            start_time = time.time()
            results = engine.translate(texts)
            duration = time.time() - start_time
            
            self.assertEqual(len(results), 100)
            self.assertLess(duration, 1.0)  # Should complete in less than 1 second


def run_tests():
    """Run all tests with detailed output."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestTranslationEngine,
        TestModernTranslationPipeline,
        TestDatasetGenerator,
        TestTranslationEvaluator,
        TestTranslationVisualizer,
        TestConfigManager,
        TestLoggingConfig,
        TestIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        descriptions=True,
        failfast=False
    )
    
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print(f"  Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*60}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
