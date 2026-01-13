"""
Enhanced Visualization Module for Neural Machine Translation

This module provides comprehensive visualization tools for translation analysis,
performance monitoring, and interactive dashboards.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TranslationVisualizer:
    """Enhanced visualization tools for translation analysis."""
    
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
        
        # Color schemes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'light': '#8c564b',
            'dark': '#e377c2'
        }
    
    def plot_metrics_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
        title: str = "Translation Quality Metrics Comparison",
        save_path: Optional[str] = None,
        interactive: bool = True
    ) -> None:
        """
        Plot comparison of metrics across different models/domains.
        
        Args:
            results: Dictionary of results (model/domain -> metrics)
            metrics: List of metrics to plot
            title: Plot title
            save_path: Path to save the plot
            interactive: Whether to create interactive plot
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
        
        if interactive:
            # Interactive plot with Plotly
            fig = px.bar(
                df, 
                x="Model/Domain", 
                y="Score", 
                color="Metric",
                title=title,
                barmode="group",
                color_discrete_sequence=list(self.colors.values())
            )
            fig.update_layout(
                xaxis_title="Model/Domain",
                yaxis_title="Score",
                legend_title="Metrics",
                font=dict(size=12)
            )
            fig.show()
            
            if save_path:
                fig.write_html(self.output_dir / f"{save_path}.html")
        else:
            # Static plot with Matplotlib
            plt.figure(figsize=(12, 8))
            sns.barplot(data=df, x="Model/Domain", y="Score", hue="Metric")
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel("Model/Domain", fontsize=12)
            plt.ylabel("Score", fontsize=12)
            plt.xticks(rotation=45)
            plt.legend(title="Metrics")
            plt.tight_layout()
            
            if save_path:
                plt.savefig(self.output_dir / f"{save_path}.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_domain_analysis(
        self,
        domain_results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
        interactive: bool = True
    ) -> None:
        """
        Plot analysis of translation quality by domain.
        
        Args:
            domain_results: Results grouped by domain
            save_path: Path to save the plot
            interactive: Whether to create interactive plot
        """
        domains = list(domain_results.keys())
        metrics = ["bleu_score", "chrf_score", "rouge_l_score"]
        
        if interactive:
            # Interactive subplot
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=[metric.replace("_", " ").title() for metric in metrics],
                specs=[[{"secondary_y": False}] * 3]
            )
            
            for i, metric in enumerate(metrics, 1):
                values = [domain_results[domain].get(metric, 0) for domain in domains]
                fig.add_trace(
                    go.Bar(
                        x=domains,
                        y=values,
                        name=metric.replace("_", " ").title(),
                        marker_color=self.colors[list(self.colors.keys())[i-1]]
                    ),
                    row=1, col=i
                )
            
            fig.update_layout(
                title="Translation Quality by Domain",
                showlegend=False,
                height=500
            )
            fig.show()
            
            if save_path:
                fig.write_html(self.output_dir / f"{save_path}.html")
        else:
            # Static subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for i, metric in enumerate(metrics):
                values = [domain_results[domain].get(metric, 0) for domain in domains]
                axes[i].bar(domains, values, color=list(self.colors.values())[i])
                axes[i].set_title(f"{metric.replace('_', ' ').title()} by Domain")
                axes[i].set_xlabel("Domain")
                axes[i].set_ylabel("Score")
                axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(self.output_dir / f"{save_path}.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_length_analysis(
        self,
        test_data: List[Dict],
        predictions: List[str],
        evaluator,
        save_path: Optional[str] = None,
        interactive: bool = True
    ) -> None:
        """
        Plot translation quality by text length.
        
        Args:
            test_data: Test dataset
            predictions: Model predictions
            evaluator: Evaluation instance
            save_path: Path to save the plot
            interactive: Whether to create interactive plot
        """
        length_groups = {}
        
        for sample, pred in zip(test_data, predictions):
            source_length = len(sample["source"].split())
            length_category = self._categorize_length(source_length)
            
            if length_category not in length_groups:
                length_groups[length_category] = []
            
            metrics = evaluator.evaluate_single(sample["target"], pred)
            length_groups[length_category].append(metrics.bleu_score)
        
        # Calculate averages
        length_metrics = {}
        for length_cat, scores in length_groups.items():
            length_metrics[length_cat] = np.mean(scores)
        
        if interactive:
            # Interactive plot
            fig = go.Figure(data=[
                go.Bar(
                    x=list(length_metrics.keys()),
                    y=list(length_metrics.values()),
                    marker_color=self.colors['primary']
                )
            ])
            fig.update_layout(
                title="Translation Quality by Text Length",
                xaxis_title="Text Length Category",
                yaxis_title="Average BLEU Score",
                font=dict(size=12)
            )
            fig.show()
            
            if save_path:
                fig.write_html(self.output_dir / f"{save_path}.html")
        else:
            # Static plot
            plt.figure(figsize=(10, 6))
            lengths = list(length_metrics.keys())
            scores = list(length_metrics.values())
            
            plt.bar(lengths, scores, color=self.colors['primary'])
            plt.title("Translation Quality by Text Length", fontsize=16, fontweight='bold')
            plt.xlabel("Text Length Category", fontsize=12)
            plt.ylabel("Average BLEU Score", fontsize=12)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(self.output_dir / f"{save_path}.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_performance_trends(
        self,
        performance_data: List[Dict[str, Any]],
        save_path: Optional[str] = None,
        interactive: bool = True
    ) -> None:
        """
        Plot performance trends over time.
        
        Args:
            performance_data: List of performance records
            save_path: Path to save the plot
            interactive: Whether to create interactive plot
        """
        df = pd.DataFrame(performance_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if interactive:
            # Interactive time series plot
            fig = go.Figure()
            
            metrics = ['bleu_score', 'chrf_score', 'rouge_l_score']
            colors = [self.colors['primary'], self.colors['secondary'], self.colors['success']]
            
            for metric, color in zip(metrics, colors):
                if metric in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df[metric],
                        mode='lines+markers',
                        name=metric.replace('_', ' ').title(),
                        line=dict(color=color)
                    ))
            
            fig.update_layout(
                title="Translation Performance Trends",
                xaxis_title="Time",
                yaxis_title="Score",
                hovermode='x unified'
            )
            fig.show()
            
            if save_path:
                fig.write_html(self.output_dir / f"{save_path}.html")
        else:
            # Static plot
            plt.figure(figsize=(12, 8))
            
            metrics = ['bleu_score', 'chrf_score', 'rouge_l_score']
            colors = [self.colors['primary'], self.colors['secondary'], self.colors['success']]
            
            for metric, color in zip(metrics, colors):
                if metric in df.columns:
                    plt.plot(df['timestamp'], df[metric], 
                            label=metric.replace('_', ' ').title(), 
                            color=color, marker='o')
            
            plt.title("Translation Performance Trends", fontsize=16, fontweight='bold')
            plt.xlabel("Time", fontsize=12)
            plt.ylabel("Score", fontsize=12)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(self.output_dir / f"{save_path}.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_language_pair_usage(
        self,
        usage_data: Dict[str, int],
        save_path: Optional[str] = None,
        interactive: bool = True
    ) -> None:
        """
        Plot language pair usage distribution.
        
        Args:
            usage_data: Dictionary of language pair -> usage count
            save_path: Path to save the plot
            interactive: Whether to create interactive plot
        """
        pairs = list(usage_data.keys())
        counts = list(usage_data.values())
        
        if interactive:
            # Interactive pie chart
            fig = go.Figure(data=[go.Pie(
                labels=pairs,
                values=counts,
                hole=0.3,
                marker_colors=list(self.colors.values())[:len(pairs)]
            )])
            fig.update_layout(
                title="Language Pair Usage Distribution",
                font=dict(size=12)
            )
            fig.show()
            
            if save_path:
                fig.write_html(self.output_dir / f"{save_path}.html")
        else:
            # Static pie chart
            plt.figure(figsize=(10, 8))
            colors = list(self.colors.values())[:len(pairs)]
            
            plt.pie(counts, labels=pairs, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title("Language Pair Usage Distribution", fontsize=16, fontweight='bold')
            plt.axis('equal')
            
            if save_path:
                plt.savefig(self.output_dir / f"{save_path}.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    def create_dashboard(
        self,
        evaluation_results: Dict[str, Any],
        performance_data: List[Dict[str, Any]] = None,
        usage_data: Dict[str, int] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            evaluation_results: Evaluation results
            performance_data: Performance trend data
            usage_data: Language pair usage data
            save_path: Path to save the dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Metrics Comparison",
                "Domain Analysis", 
                "Performance Trends",
                "Language Pair Usage"
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "pie"}]]
        )
        
        # Metrics comparison
        if 'domain_results' in evaluation_results:
            domains = list(evaluation_results['domain_results'].keys())
            bleu_scores = [evaluation_results['domain_results'][d].get('bleu_score', 0) for d in domains]
            
            fig.add_trace(
                go.Bar(x=domains, y=bleu_scores, name="BLEU Score", 
                      marker_color=self.colors['primary']),
                row=1, col=1
            )
        
        # Performance trends
        if performance_data:
            df = pd.DataFrame(performance_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            if 'bleu_score' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=df['bleu_score'], 
                             mode='lines+markers', name="BLEU Trend",
                             line=dict(color=self.colors['secondary'])),
                    row=2, col=1
                )
        
        # Language pair usage
        if usage_data:
            pairs = list(usage_data.keys())
            counts = list(usage_data.values())
            
            fig.add_trace(
                go.Pie(labels=pairs, values=counts, name="Usage"),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Translation System Dashboard",
            height=800,
            showlegend=False
        )
        
        fig.show()
        
        if save_path:
            fig.write_html(self.output_dir / f"{save_path}.html")
    
    def _categorize_length(self, word_count: int) -> str:
        """Categorize text length based on word count."""
        if word_count <= 5:
            return "Short (≤5 words)"
        elif word_count <= 15:
            return "Medium (6-15 words)"
        else:
            return "Long (>15 words)"
    
    def export_visualization_data(
        self,
        data: Dict[str, Any],
        filename: str
    ) -> None:
        """
        Export visualization data to JSON file.
        
        Args:
            data: Data to export
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Recursively convert numpy types
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(item) for item in obj]
            else:
                return convert_numpy(obj)
        
        converted_data = recursive_convert(data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Visualization data exported to {output_path}")


if __name__ == "__main__":
    # Test visualization
    visualizer = TranslationVisualizer()
    
    # Sample data
    sample_results = {
        "Model A": {"bleu_score": 0.75, "chrf_score": 0.80, "rouge_l_score": 0.70},
        "Model B": {"bleu_score": 0.72, "chrf_score": 0.78, "rouge_l_score": 0.68},
        "Model C": {"bleu_score": 0.78, "chrf_score": 0.82, "rouge_l_score": 0.72}
    }
    
    # Test plotting
    visualizer.plot_metrics_comparison(sample_results, interactive=False)
    
    print("Visualization test completed!")
