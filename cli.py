"""
Command Line Interface for Neural Machine Translation

This module provides a command-line interface for the translation system
with batch processing, evaluation, and configuration options.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from translation_engine import create_translation_engine, TranslationEngine
from dataset_generator import SyntheticDatasetGenerator
from evaluation import TranslationEvaluator, load_test_data, save_evaluation_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def translate_text(
    text: str,
    language_pair: str = "en-fr",
    output_file: Optional[str] = None
) -> str:
    """
    Translate a single text.
    
    Args:
        text: Text to translate
        language_pair: Language pair code
        output_file: Optional output file path
        
    Returns:
        Translated text
    """
    try:
        engine = create_translation_engine(language_pair)
        result = engine.translate(text)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            logger.info(f"Translation saved to {output_file}")
        
        return result
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return f"Error: {str(e)}"


def translate_file(
    input_file: str,
    language_pair: str = "en-fr",
    output_file: Optional[str] = None
) -> List[str]:
    """
    Translate texts from a file.
    
    Args:
        input_file: Input file path
        language_pair: Language pair code
        output_file: Optional output file path
        
    Returns:
        List of translations
    """
    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            if input_file.endswith('.json'):
                data = json.load(f)
                if isinstance(data, list):
                    texts = [item.get('text', str(item)) for item in data]
                else:
                    texts = [data.get('text', str(data))]
            else:
                texts = [line.strip() for line in f if line.strip()]
        
        # Translate
        engine = create_translation_engine(language_pair)
        translations = engine.translate(texts)
        
        # Save results
        if output_file:
            results = []
            for text, translation in zip(texts, translations):
                results.append({
                    "source": text,
                    "translation": translation
                })
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Translations saved to {output_file}")
        
        return translations
        
    except Exception as e:
        logger.error(f"File translation failed: {e}")
        return [f"Error: {str(e)}"]


def generate_dataset(
    output_dir: str = "data",
    language_pairs: List[str] = None,
    domains: List[str] = None,
    samples_per_domain: int = 50
) -> str:
    """
    Generate synthetic dataset.
    
    Args:
        output_dir: Output directory
        language_pairs: List of language pairs
        domains: List of domains
        samples_per_domain: Samples per domain
        
    Returns:
        Path to generated dataset
    """
    if language_pairs is None:
        language_pairs = [("en", "fr"), ("en", "de"), ("en", "es")]
    
    if domains is None:
        domains = ["greetings", "business", "technology", "travel"]
    
    try:
        generator = SyntheticDatasetGenerator(output_dir)
        dataset_path = generator.generate_dataset(
            language_pairs=language_pairs,
            domains=domains,
            samples_per_domain=samples_per_domain
        )
        
        logger.info(f"Dataset generated: {dataset_path}")
        return dataset_path
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        return ""


def evaluate_translations(
    test_file: str,
    language_pair: str = "en-fr",
    output_file: Optional[str] = None
) -> dict:
    """
    Evaluate translations.
    
    Args:
        test_file: Test data file path
        language_pair: Language pair code
        output_file: Optional output file path
        
    Returns:
        Evaluation results
    """
    try:
        # Load test data
        test_data = load_test_data(test_file)
        
        # Create engine and generate predictions
        engine = create_translation_engine(language_pair)
        sources = [item["source"] for item in test_data]
        predictions = engine.translate(sources)
        
        # Evaluate
        evaluator = TranslationEvaluator()
        avg_metrics = evaluator.evaluate_batch(
            [item["target"] for item in test_data],
            predictions
        )
        
        # Domain analysis
        domain_results = evaluator.evaluate_by_domain(test_data, predictions)
        
        results = {
            "overall_metrics": avg_metrics,
            "domain_results": domain_results,
            "sample_count": len(test_data)
        }
        
        # Save results
        if output_file:
            save_evaluation_results(results, output_file)
            logger.info(f"Evaluation results saved to {output_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {"error": str(e)}


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Neural Machine Translation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Translate a single text
  python cli.py translate "Hello, world!" --language-pair en-fr
  
  # Translate from file
  python cli.py translate-file input.txt --output output.json
  
  # Generate dataset
  python cli.py generate-dataset --samples-per-domain 100
  
  # Evaluate translations
  python cli.py evaluate test_data.json --output results.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Translate command
    translate_parser = subparsers.add_parser('translate', help='Translate a single text')
    translate_parser.add_argument('text', help='Text to translate')
    translate_parser.add_argument('--language-pair', default='en-fr', help='Language pair (default: en-fr)')
    translate_parser.add_argument('--output', help='Output file path')
    
    # Translate file command
    translate_file_parser = subparsers.add_parser('translate-file', help='Translate texts from file')
    translate_file_parser.add_argument('input', help='Input file path')
    translate_file_parser.add_argument('--language-pair', default='en-fr', help='Language pair (default: en-fr)')
    translate_file_parser.add_argument('--output', help='Output file path')
    
    # Generate dataset command
    generate_parser = subparsers.add_parser('generate-dataset', help='Generate synthetic dataset')
    generate_parser.add_argument('--output-dir', default='data', help='Output directory (default: data)')
    generate_parser.add_argument('--language-pairs', nargs='+', default=['en-fr', 'en-de'], help='Language pairs')
    generate_parser.add_argument('--domains', nargs='+', default=['greetings', 'business'], help='Domains')
    generate_parser.add_argument('--samples-per-domain', type=int, default=50, help='Samples per domain')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate translations')
    evaluate_parser.add_argument('test_file', help='Test data file path')
    evaluate_parser.add_argument('--language-pair', default='en-fr', help='Language pair (default: en-fr)')
    evaluate_parser.add_argument('--output', help='Output file path')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute commands
    if args.command == 'translate':
        result = translate_text(args.text, args.language_pair, args.output)
        print(f"Translation: {result}")
    
    elif args.command == 'translate-file':
        results = translate_file(args.input, args.language_pair, args.output)
        print(f"Translated {len(results)} texts")
        if not args.output:
            for i, result in enumerate(results):
                print(f"{i+1}. {result}")
    
    elif args.command == 'generate-dataset':
        # Convert language pairs format
        lang_pairs = []
        for pair in args.language_pairs:
            if '-' in pair:
                source, target = pair.split('-')
                lang_pairs.append((source, target))
        
        dataset_path = generate_dataset(
            args.output_dir,
            lang_pairs,
            args.domains,
            args.samples_per_domain
        )
        print(f"Dataset generated: {dataset_path}")
    
    elif args.command == 'evaluate':
        results = evaluate_translations(args.test_file, args.language_pair, args.output)
        print("Evaluation Results:")
        print(f"Sample Count: {results.get('sample_count', 0)}")
        
        if 'overall_metrics' in results:
            print("\nOverall Metrics:")
            for metric, score in results['overall_metrics'].items():
                print(f"  {metric}: {score:.4f}")
        
        if 'domain_results' in results:
            print("\nDomain Results:")
            for domain, metrics in results['domain_results'].items():
                print(f"  {domain}: BLEU={metrics['bleu_score']:.4f}")


if __name__ == "__main__":
    main()
