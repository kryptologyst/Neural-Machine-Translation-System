"""
Unit Tests for Neural Machine Translation System

This module contains comprehensive unit tests for all components
of the translation system.
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from translation_engine import TranslationEngine, TranslationConfig, create_translation_engine
from dataset_generator import SyntheticDatasetGenerator, TranslationSample
from evaluation import TranslationEvaluator, TranslationVisualizer


class TestTranslationEngine(unittest.TestCase):
    """Test cases for TranslationEngine."""
    
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


class TestDatasetGenerator(unittest.TestCase):
    """Test cases for SyntheticDatasetGenerator."""
    
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


class TestTranslationEvaluator(unittest.TestCase):
    """Test cases for TranslationEvaluator."""
    
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


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow."""
        # This would be a more comprehensive integration test
        # that tests the entire pipeline from data generation
        # to evaluation
        pass


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestTranslationEngine,
        TestDatasetGenerator,
        TestTranslationEvaluator,
        TestTranslationVisualizer,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
