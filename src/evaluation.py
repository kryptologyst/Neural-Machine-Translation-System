"""
Evaluation Module for Neural Machine Translation

This module provides comprehensive evaluation metrics and visualization
tools for assessing translation quality.
"""

import json
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Import evaluation metrics
try:
    from sacrebleu import BLEU, CHRF, TER
    from rouge_score import rouge_scorer
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    logging.warning("sacrebleu not available. Install with: pip install sacrebleu")

try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    logging.warning("bert-score not available. Install with: pip install bert-score")

try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    logging.warning("COMET not available. Install with: pip install unbabel-comet")

try:
    from evaluate import load as load_metric
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False
    logging.warning("evaluate not available. Install with: pip install evaluate")

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class EvaluationMetrics:
    """Container for comprehensive evaluation metrics."""
    bleu_score: float
    chrf_score: float
    rouge_l_score: float
    ter_score: Optional[float] = None
    bert_score_f1: Optional[float] = None
    bert_score_precision: Optional[float] = None
    bert_score_recall: Optional[float] = None
    comet_score: Optional[float] = None
    meteor_score: Optional[float] = None
    evaluation_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "bleu_score": self.bleu_score,
            "chrf_score": self.chrf_score,
            "rouge_l_score": self.rouge_l_score,
            "ter_score": self.ter_score,
            "bert_score_f1": self.bert_score_f1,
            "bert_score_precision": self.bert_score_precision,
            "bert_score_recall": self.bert_score_recall,
            "comet_score": self.comet_score,
            "meteor_score": self.meteor_score,
            "evaluation_time": self.evaluation_time
        }
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available (non-None) metrics."""
        return [k for k, v in self.to_dict().items() if v is not None]


class TranslationEvaluator:
    """Comprehensive evaluation system for translation models with modern metrics."""
    
    def __init__(self, enable_comet: bool = False, enable_meteor: bool = False):
        """
        Initialize the evaluator with optional advanced metrics.
        
        Args:
            enable_comet: Whether to enable COMET evaluation (requires model download)
            enable_meteor: Whether to enable METEOR evaluation
        """
        self.scorer = None
        self.comet_model = None
        self.meteor_metric = None
        
        if SACREBLEU_AVAILABLE:
            self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        if enable_comet and COMET_AVAILABLE:
            try:
                model_path = download_model("Unbabel/wmt22-comet-da")
                self.comet_model = load_from_checkpoint(model_path)
                logger.info("COMET model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load COMET model: {e}")
                self.comet_model = None
        
        if enable_meteor and EVALUATE_AVAILABLE:
            try:
                self.meteor_metric = load_metric("meteor")
                logger.info("METEOR metric loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load METEOR metric: {e}")
                self.meteor_metric = None
    
    def evaluate_single(
        self,
        reference: str,
        hypothesis: str,
        source: Optional[str] = None
    ) -> EvaluationMetrics:
        """
        Evaluate a single translation with comprehensive metrics.
        
        Args:
            reference: Reference translation
            hypothesis: Model translation
            source: Source text (optional, needed for COMET)
            
        Returns:
            Evaluation metrics
        """
        start_time = time.time()
        
        metrics = EvaluationMetrics(
            bleu_score=0.0,
            chrf_score=0.0,
            rouge_l_score=0.0
        )
        
        if SACREBLEU_AVAILABLE:
            try:
                # BLEU score
                bleu = BLEU()
                metrics.bleu_score = bleu.sentence_score(hypothesis, [reference]).score
                
                # CHRF score
                chrf = CHRF()
                metrics.chrf_score = chrf.sentence_score(hypothesis, [reference]).score
                
                # TER score
                ter = TER()
                metrics.ter_score = ter.sentence_score(hypothesis, [reference]).score
                
                # ROUGE-L score
                rouge_scores = self.scorer.score(reference, hypothesis)
                metrics.rouge_l_score = rouge_scores['rougeL'].fmeasure
                
            except Exception as e:
                logger.warning(f"SacreBLEU evaluation failed: {e}")
        
        # BERT Score (if available)
        if BERT_SCORE_AVAILABLE:
            try:
                P, R, F1 = bert_score([hypothesis], [reference], lang="en", verbose=False)
                metrics.bert_score_precision = P.item()
                metrics.bert_score_recall = R.item()
                metrics.bert_score_f1 = F1.item()
            except Exception as e:
                logger.warning(f"BERT Score calculation failed: {e}")
        
        # COMET Score (if available and source provided)
        if self.comet_model and source:
            try:
                data = [{"src": source, "mt": hypothesis, "ref": reference}]
                scores = self.comet_model.predict(data, batch_size=1)
                metrics.comet_score = scores[0]
            except Exception as e:
                logger.warning(f"COMET evaluation failed: {e}")
        
        # METEOR Score (if available)
        if self.meteor_metric:
            try:
                meteor_result = self.meteor_metric.compute(
                    predictions=[hypothesis], 
                    references=[reference]
                )
                metrics.meteor_score = meteor_result["meteor"]
            except Exception as e:
                logger.warning(f"METEOR evaluation failed: {e}")
        
        metrics.evaluation_time = time.time() - start_time
        return metrics
    
    def evaluate_batch(
        self,
        references: List[str],
        hypotheses: List[str],
        sources: Optional[List[str]] = None,
        parallel: bool = True,
        max_workers: int = 4
    ) -> Dict[str, float]:
        """
        Evaluate a batch of translations with optional parallel processing.
        
        Args:
            references: List of reference translations
            hypotheses: List of model translations
            sources: List of source texts (optional, needed for COMET)
            parallel: Whether to use parallel processing
            max_workers: Maximum number of worker threads
            
        Returns:
            Dictionary of average metrics
        """
        if len(references) != len(hypotheses):
            raise ValueError("References and hypotheses must have the same length")
        
        if sources and len(sources) != len(references):
            raise ValueError("Sources must have the same length as references")
        
        start_time = time.time()
        
        if parallel and len(references) > 10:
            # Use parallel processing for large batches
            metrics_list = self._evaluate_batch_parallel(
                references, hypotheses, sources, max_workers
            )
        else:
            # Sequential processing for small batches
            metrics_list = []
            for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
                source = sources[i] if sources else None
                metrics = self.evaluate_single(ref, hyp, source)
                metrics_list.append(metrics)
        
        # Calculate averages
        avg_metrics = self._calculate_average_metrics(metrics_list)
        avg_metrics["total_evaluation_time"] = time.time() - start_time
        avg_metrics["sample_count"] = len(references)
        
        return avg_metrics
    
    def _evaluate_batch_parallel(
        self,
        references: List[str],
        hypotheses: List[str],
        sources: Optional[List[str]],
        max_workers: int
    ) -> List[EvaluationMetrics]:
        """Evaluate batch using parallel processing."""
        metrics_list = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_index = {}
            for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
                source = sources[i] if sources else None
                future = executor.submit(self.evaluate_single, ref, hyp, source)
                future_to_index[future] = i
            
            # Collect results in order
            results = [None] * len(references)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Evaluation failed for sample {index}: {e}")
                    # Create default metrics for failed evaluation
                    results[index] = EvaluationMetrics(
                        bleu_score=0.0, chrf_score=0.0, rouge_l_score=0.0
                    )
            
            metrics_list = results
        
        return metrics_list
    
    def _calculate_average_metrics(self, metrics_list: List[EvaluationMetrics]) -> Dict[str, float]:
        """Calculate average metrics from a list of EvaluationMetrics."""
        avg_metrics = {}
        
        # Get all available metric names
        all_metrics = set()
        for metrics in metrics_list:
            all_metrics.update(metrics.get_available_metrics())
        
        # Calculate averages for each metric
        for metric_name in all_metrics:
            values = []
            for metrics in metrics_list:
                metric_dict = metrics.to_dict()
                if metric_name in metric_dict and metric_dict[metric_name] is not None:
                    values.append(metric_dict[metric_name])
            
            if values:
                avg_metrics[metric_name] = np.mean(values)
        
        return avg_metrics
    
    def evaluate_by_domain(
        self,
        test_data: List[Dict],
        predictions: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate translations grouped by domain.
        
        Args:
            test_data: Test dataset with domain information
            predictions: Model predictions
            
        Returns:
            Dictionary of metrics by domain
        """
        domain_metrics = defaultdict(list)
        
        for i, (sample, pred) in enumerate(zip(test_data, predictions)):
            domain = sample.get("domain", "unknown")
            reference = sample["target"]
            
            metrics = self.evaluate_single(reference, pred)
            domain_metrics[domain].append(metrics)
        
        # Calculate averages per domain
        domain_results = {}
        for domain, metrics_list in domain_metrics.items():
            domain_results[domain] = {
                "bleu_score": np.mean([m.bleu_score for m in metrics_list]),
                "chrf_score": np.mean([m.chrf_score for m in metrics_list]),
                "rouge_l_score": np.mean([m.rouge_l_score for m in metrics_list]),
                "sample_count": len(metrics_list)
            }
            
            # BERT Score averages (if available)
            bert_scores = [m for m in metrics_list if m.bert_score_f1 is not None]
            if bert_scores:
                domain_results[domain]["bert_score_f1"] = np.mean([m.bert_score_f1 for m in bert_scores])
        
        return domain_results


class TranslationVisualizer:
    """Visualization tools for translation evaluation results."""
    
    def __init__(self, output_dir: str = "models/visualizations"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_metrics_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot comparison of metrics across different models/domains.
        
        Args:
            results: Dictionary of results (model/domain -> metrics)
            metrics: List of metrics to plot
            save_path: Path to save the plot
        """
        if metrics is None:
            metrics = ["bleu_score", "chrf_score", "rouge_l_score"]
        
        # Prepare data
        data = []
        for name, result in results.items():
            for metric in metrics:
                if metric in result:
                    data.append({
                        "Model/Domain": name,
                        "Metric": metric.replace("_", " ").title(),
                        "Score": result[metric]
                    })
        
        df = pd.DataFrame(data)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=df, x="Model/Domain", y="Score", hue="Metric")
        plt.title("Translation Quality Metrics Comparison")
        plt.xlabel("Model/Domain")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.legend(title="Metrics")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_domain_analysis(
        self,
        domain_results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot analysis of translation quality by domain.
        
        Args:
            domain_results: Results grouped by domain
            save_path: Path to save the plot
        """
        domains = list(domain_results.keys())
        metrics = ["bleu_score", "chrf_score", "rouge_l_score"]
        
        # Prepare data
        data = {metric: [] for metric in metrics}
        for domain in domains:
            for metric in metrics:
                if metric in domain_results[domain]:
                    data[metric].append(domain_results[domain][metric])
                else:
                    data[metric].append(0)
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            axes[i].bar(domains, data[metric])
            axes[i].set_title(f"{metric.replace('_', ' ').title()} by Domain")
            axes[i].set_xlabel("Domain")
            axes[i].set_ylabel("Score")
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_length_analysis(
        self,
        test_data: List[Dict],
        predictions: List[str],
        evaluator: TranslationEvaluator,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot translation quality by text length.
        
        Args:
            test_data: Test dataset
            predictions: Model predictions
            evaluator: Evaluation instance
            save_path: Path to save the plot
        """
        length_groups = defaultdict(list)
        
        for sample, pred in zip(test_data, predictions):
            source_length = len(sample["source"].split())
            length_category = self._categorize_length(source_length)
            
            metrics = evaluator.evaluate_single(sample["target"], pred)
            length_groups[length_category].append(metrics.bleu_score)
        
        # Calculate averages
        length_metrics = {}
        for length_cat, scores in length_groups.items():
            length_metrics[length_cat] = np.mean(scores)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        lengths = list(length_metrics.keys())
        scores = list(length_metrics.values())
        
        plt.bar(lengths, scores)
        plt.title("Translation Quality by Text Length")
        plt.xlabel("Text Length Category")
        plt.ylabel("Average BLEU Score")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _categorize_length(self, word_count: int) -> str:
        """Categorize text length based on word count."""
        if word_count <= 5:
            return "Short (≤5 words)"
        elif word_count <= 15:
            return "Medium (6-15 words)"
        else:
            return "Long (>15 words)"


def load_test_data(file_path: str) -> List[Dict]:
    """Load test data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_evaluation_results(
    results: Dict,
    output_path: str
) -> None:
    """Save evaluation results to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Example usage
    print("Translation Evaluation Demo")
    print("=" * 40)
    
    # Initialize evaluator and visualizer
    evaluator = TranslationEvaluator()
    visualizer = TranslationVisualizer()
    
    # Example test data
    test_samples = [
        {
            "source": "Hello, how are you?",
            "target": "Bonjour, comment allez-vous ?",
            "domain": "greetings"
        },
        {
            "source": "The weather is beautiful today.",
            "target": "Le temps est magnifique aujourd'hui.",
            "domain": "general"
        }
    ]
    
    # Example predictions
    predictions = [
        "Bonjour, comment allez-vous ?",
        "Le temps est beau aujourd'hui."
    ]
    
    # Evaluate
    print("Evaluating translations...")
    avg_metrics = evaluator.evaluate_batch(
        [s["target"] for s in test_samples],
        predictions
    )
    
    print("Average Metrics:")
    for metric, score in avg_metrics.items():
        print(f"  {metric}: {score:.4f}")
    
    # Domain analysis
    domain_results = evaluator.evaluate_by_domain(test_samples, predictions)
    print("\nDomain Analysis:")
    for domain, metrics in domain_results.items():
        print(f"  {domain}: BLEU={metrics['bleu_score']:.4f}")
    
    print("\nEvaluation complete!")
