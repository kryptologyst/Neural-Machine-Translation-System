#!/usr/bin/env python3
"""
Demo Script for Neural Machine Translation

This script demonstrates the key features of the translation system
including single translation, batch processing, and evaluation.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from translation_engine import create_translation_engine
from dataset_generator import SyntheticDatasetGenerator
from evaluation import TranslationEvaluator, TranslationVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_single_translation():
    """Demonstrate single text translation."""
    print("\n" + "="*60)
    print("🔤 SINGLE TEXT TRANSLATION DEMO")
    print("="*60)
    
    # Create translation engine
    print("Loading translation engine (EN → FR)...")
    engine = create_translation_engine("en-fr")
    
    # Test texts
    test_texts = [
        "Hello, how are you today?",
        "The weather is beautiful today.",
        "Thank you for your help.",
        "Machine learning is fascinating.",
        "Have a wonderful day!"
    ]
    
    print(f"\nTranslating {len(test_texts)} texts:")
    print("-" * 40)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Original: {text}")
        translation = engine.translate(text)
        print(f"   Translation: {translation}")


def demo_batch_translation():
    """Demonstrate batch translation."""
    print("\n" + "="*60)
    print("📝 BATCH TRANSLATION DEMO")
    print("="*60)
    
    # Create translation engine
    print("Loading translation engine (EN → DE)...")
    engine = create_translation_engine("en-de")
    
    # Batch of texts
    batch_texts = [
        "Good morning!",
        "How was your weekend?",
        "The project is going well.",
        "See you tomorrow.",
        "Have a great day!"
    ]
    
    print(f"\nTranslating batch of {len(batch_texts)} texts:")
    print("-" * 40)
    
    # Translate batch
    translations = engine.translate(batch_texts)
    
    # Display results
    for original, translation in zip(batch_texts, translations):
        print(f"EN: {original}")
        print(f"DE: {translation}")
        print()


def demo_dataset_generation():
    """Demonstrate synthetic dataset generation."""
    print("\n" + "="*60)
    print("📊 DATASET GENERATION DEMO")
    print("="*60)
    
    # Create generator
    generator = SyntheticDatasetGenerator("data")
    
    print("Generating synthetic dataset...")
    dataset_path = generator.generate_dataset(
        language_pairs=[("en", "fr"), ("en", "es")],
        domains=["greetings", "business"],
        samples_per_domain=10
    )
    
    print(f"✅ Dataset generated: {dataset_path}")
    
    # Generate test set
    print("\nGenerating test set...")
    test_set = generator.generate_test_set(("en", "fr"), 5)
    
    print(f"✅ Test set generated with {len(test_set)} samples")
    print("\nSample test data:")
    for i, sample in enumerate(test_set[:3], 1):
        print(f"{i}. {sample['source']} → {sample['target']}")


def demo_evaluation():
    """Demonstrate evaluation capabilities."""
    print("\n" + "="*60)
    print("📈 EVALUATION DEMO")
    print("="*60)
    
    # Create evaluator
    evaluator = TranslationEvaluator()
    
    # Sample evaluation data
    references = [
        "Bonjour, comment allez-vous ?",
        "Merci pour votre aide.",
        "Le temps est magnifique aujourd'hui."
    ]
    
    hypotheses = [
        "Hello, how are you?",
        "Thank you for your help.",
        "The weather is beautiful today."
    ]
    
    print("Evaluating translation quality...")
    print("-" * 40)
    
    # Evaluate each pair
    for i, (ref, hyp) in enumerate(zip(references, hypotheses), 1):
        print(f"\n{i}. Reference: {ref}")
        print(f"   Hypothesis: {hyp}")
        
        metrics = evaluator.evaluate_single(ref, hyp)
        print(f"   BLEU: {metrics.bleu_score:.4f}")
        print(f"   CHRF: {metrics.chrf_score:.4f}")
        print(f"   ROUGE-L: {metrics.rouge_l_score:.4f}")
    
    # Batch evaluation
    print(f"\nBatch evaluation results:")
    avg_metrics = evaluator.evaluate_batch(references, hypotheses)
    
    for metric, score in avg_metrics.items():
        print(f"  {metric}: {score:.4f}")


def demo_language_pairs():
    """Demonstrate different language pairs."""
    print("\n" + "="*60)
    print("🌍 MULTIPLE LANGUAGE PAIRS DEMO")
    print("="*60)
    
    text = "Hello, how are you today?"
    language_pairs = ["en-fr", "en-de", "en-es", "en-it"]
    
    print(f"Original text: {text}")
    print("-" * 40)
    
    for pair in language_pairs:
        try:
            print(f"\n{pair.upper()}:")
            engine = create_translation_engine(pair)
            translation = engine.translate(text)
            print(f"  {translation}")
        except Exception as e:
            print(f"  Error: {e}")


def main():
    """Main demo function."""
    print("🌐 NEURAL MACHINE TRANSLATION DEMO")
    print("="*60)
    print("This demo showcases the key features of the translation system.")
    print("Note: First-time model loading may take a few minutes.")
    
    try:
        # Run demos
        demo_single_translation()
        demo_batch_translation()
        demo_dataset_generation()
        demo_evaluation()
        demo_language_pairs()
        
        print("\n" + "="*60)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("1. Run the web interface: streamlit run web_app/app.py")
        print("2. Use the CLI: python cli.py --help")
        print("3. Check the generated data in the 'data/' directory")
        print("4. Explore the configuration in 'config/config.yaml'")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        print("Please check the error message and try again.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
