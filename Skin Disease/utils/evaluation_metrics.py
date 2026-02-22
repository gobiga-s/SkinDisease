"""
Evaluation Metrics and Visualization Utilities for SkinX
Provides comprehensive evaluation metrics and visualization tools for model performance
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization toolkit
    """
    
    def __init__(self, class_names: List[str]):
        """
        Initialize the evaluator
        
        Args:
            class_names (List[str]): List of class names for evaluation
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def compute_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_prob: Optional[np.ndarray] = None) -> Dict:
        """
        Compute basic classification metrics
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_prob (np.ndarray): Prediction probabilities (optional)
            
        Returns:
            Dict: Dictionary containing computed metrics
        """
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision_macro': precision_score(y_true, y_pred, average='macro'),
                'precision_micro': precision_score(y_true, y_pred, average='micro'),
                'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
                'recall_macro': recall_score(y_true, y_pred, average='macro'),
                'recall_micro': recall_score(y_true, y_pred, average='micro'),
                'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
                'f1_macro': f1_score(y_true, y_pred, average='macro'),
                'f1_micro': f1_score(y_true, y_pred, average='micro'),
                'f1_weighted': f1_score(y_true, y_pred, average='weighted')
            }
            
            # Add per-class metrics
            precision_per_class = precision_score(y_true, y_pred, average=None)
            recall_per_class = recall_score(y_true, y_pred, average=None)
            f1_per_class = f1_score(y_true, y_pred, average=None)
            
            for i, class_name in enumerate(self.class_names):
                if i < len(precision_per_class):
                    metrics[f'precision_{class_name}'] = precision_per_class[i]
                    metrics[f'recall_{class_name}'] = recall_per_class[i]
                    metrics[f'f1_{class_name}'] = f1_per_class[i]
            
            # Add AUC if probabilities are provided
            if y_prob is not None:
                if self.num_classes == 2:
                    # Binary classification
                    metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
                    metrics['auc_pr'] = average_precision_score(y_true, y_prob[:, 1])
                else:
                    # Multi-class classification
                    metrics['auc_roc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                    metrics['auc_roc_ovo'] = roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro')
            
            logger.info("Basic metrics computed successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing basic metrics: {str(e)}")
            raise
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Generate detailed classification report
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            
        Returns:
            Dict: Classification report
        """
        try:
            report = classification_report(
                y_true, y_pred, 
                target_names=self.class_names,
                output_dict=True,
                zero_division=0
            )
            
            logger.info("Classification report generated")
            return report
            
        except Exception as e:
            logger.error(f"Error generating classification report: {str(e)}")
            raise
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            normalize: bool = False, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            normalize (bool): Whether to normalize the confusion matrix
            save_path (str): Path to save the plot (optional)
            
        Returns:
            plt.Figure: The confusion matrix plot
        """
        try:
            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                title = 'Normalized Confusion Matrix'
                fmt = '.2f'
            else:
                title = 'Confusion Matrix'
                fmt = 'd'
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot heatmap
            sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                       xticklabels=self.class_names, yticklabels=self.class_names,
                       ax=ax, cbar_kws={'shrink': 0.8})
            
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            
            # Rotate labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            plt.setp(ax.get_yticklabels(), rotation=0)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Confusion matrix saved to: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {str(e)}")
            raise
    
    def plot_roc_curves(self, y_true: np.ndarray, y_prob: np.ndarray, 
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curves for multi-class classification
        
        Args:
            y_true (np.ndarray): True labels
            y_prob (np.ndarray): Prediction probabilities
            save_path (str): Path to save the plot (optional)
            
        Returns:
            plt.Figure: The ROC curves plot
        """
        try:
            if self.num_classes == 2:
                # Binary classification
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                auc_score = roc_auc_score(y_true, y_prob[:, 1])
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.plot(fpr, tpr, color='darkorange', lw=2,
                       label=f'ROC curve (AUC = {auc_score:.2f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                ax.legend(loc="lower right")
            else:
                # Multi-class classification
                from sklearn.preprocessing import label_binarize
                
                # Binarize labels for multi-class ROC
                y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
                
                fig, ax = plt.subplots(figsize=(12, 10))
                
                # Plot ROC curve for each class
                for i, class_name in enumerate(self.class_names):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                    auc_score = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
                    ax.plot(fpr, tpr, lw=2,
                           label=f'{class_name} (AUC = {auc_score:.2f})')
                
                # Plot diagonal line
                ax.plot([0, 1], [0, 1], 'k--', lw=2)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Multi-class ROC Curves')
                ax.legend(loc="lower right", bbox_to_anchor=(1.05, 1))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"ROC curves saved to: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting ROC curves: {str(e)}")
            raise
    
    def plot_precision_recall_curves(self, y_true: np.ndarray, y_prob: np.ndarray,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Precision-Recall curves
        
        Args:
            y_true (np.ndarray): True labels
            y_prob (np.ndarray): Prediction probabilities
            save_path (str): Path to save the plot (optional)
            
        Returns:
            plt.Figure: The PR curves plot
        """
        try:
            if self.num_classes == 2:
                # Binary classification
                precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
                avg_precision = average_precision_score(y_true, y_prob[:, 1])
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.plot(recall, precision, color='blue', lw=2,
                       label=f'PR curve (AP = {avg_precision:.2f})')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title('Precision-Recall Curve')
                ax.legend(loc="lower left")
            else:
                # Multi-class classification
                from sklearn.preprocessing import label_binarize
                
                y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
                
                fig, ax = plt.subplots(figsize=(12, 10))
                
                for i, class_name in enumerate(self.class_names):
                    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                    avg_precision = average_precision_score(y_true_bin[:, i], y_prob[:, i])
                    ax.plot(recall, precision, lw=2,
                           label=f'{class_name} (AP = {avg_precision:.2f})')
                
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title('Multi-class Precision-Recall Curves')
                ax.legend(loc="lower left", bbox_to_anchor=(1.05, 1))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"PR curves saved to: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting PR curves: {str(e)}")
            raise
    
    def plot_class_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot class distribution comparison
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            save_path (str): Path to save the plot (optional)
            
        Returns:
            plt.Figure: The distribution plot
        """
        try:
            # Count occurrences
            true_counts = pd.Series(y_true).value_counts().sort_index()
            pred_counts = pd.Series(y_pred).value_counts().sort_index()
            
            # Create DataFrame for plotting
            df = pd.DataFrame({
                'True': true_counts,
                'Predicted': pred_counts
            })
            df.index = [self.class_names[i] for i in df.index]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x = np.arange(len(df))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, df['True'], width, label='True Distribution', alpha=0.8)
            bars2 = ax.bar(x + width/2, df['Predicted'], width, label='Predicted Distribution', alpha=0.8)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
            
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
            
            ax.set_xlabel('Skin Disease Categories')
            ax.set_ylabel('Count')
            ax.set_title('Class Distribution: True vs Predicted')
            ax.set_xticks(x)
            ax.set_xticklabels(df.index, rotation=45, ha='right')
            ax.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Class distribution plot saved to: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting class distribution: {str(e)}")
            raise
    
    def plot_metrics_comparison(self, metrics_dict: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of different metrics
        
        Args:
            metrics_dict (Dict): Dictionary containing metrics for different models
            save_path (str): Path to save the plot (optional)
            
        Returns:
            plt.Figure: The metrics comparison plot
        """
        try:
            # Extract relevant metrics
            metric_names = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            
            # Create DataFrame
            data = []
            for model_name, metrics in metrics_dict.items():
                for metric in metric_names:
                    if metric in metrics:
                        data.append({
                            'Model': model_name,
                            'Metric': metric.replace('_', ' ').title(),
                            'Value': metrics[metric]
                        })
            
            df = pd.DataFrame(data)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create grouped bar plot
            sns.barplot(data=df, x='Metric', y='Value', hue='Model', ax=ax)
            
            ax.set_title('Model Performance Metrics Comparison', fontsize=16, fontweight='bold')
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Score')
            ax.legend(title='Models', bbox_to_anchor=(1.05, 1))
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Metrics comparison plot saved to: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting metrics comparison: {str(e)}")
            raise
    
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 y_prob: Optional[np.ndarray] = None,
                                 model_name: str = "Model") -> Dict:
        """
        Generate comprehensive evaluation report
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_prob (np.ndarray): Prediction probabilities (optional)
            model_name (str): Name of the model being evaluated
            
        Returns:
            Dict: Comprehensive evaluation report
        """
        try:
            # Compute all metrics
            basic_metrics = self.compute_basic_metrics(y_true, y_pred, y_prob)
            classification_report = self.generate_classification_report(y_true, y_pred)
            
            # Create comprehensive report
            report = {
                'model_name': model_name,
                'basic_metrics': basic_metrics,
                'classification_report': classification_report,
                'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
                'class_names': self.class_names
            }
            
            # Add per-class analysis
            per_class_analysis = {}
            for i, class_name in enumerate(self.class_names):
                if i < len(basic_metrics.get(f'precision_{class_name}', [])):
                    per_class_analysis[class_name] = {
                        'precision': basic_metrics.get(f'precision_{class_name}', 0),
                        'recall': basic_metrics.get(f'recall_{class_name}', 0),
                        'f1_score': basic_metrics.get(f'f1_{class_name}', 0)
                    }
            
            report['per_class_analysis'] = per_class_analysis
            
            logger.info(f"Comprehensive evaluation report generated for {model_name}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating evaluation report: {str(e)}")
            raise
    
    def save_evaluation_report(self, report: Dict, save_path: str):
        """
        Save evaluation report to JSON file
        
        Args:
            report (Dict): Evaluation report
            save_path (str): Path to save the report
        """
        try:
            import json
            
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Evaluation report saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation report: {str(e)}")
            raise


# Utility functions
def create_sample_predictions(num_samples: int = 100, num_classes: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sample predictions for testing evaluation functions
    
    Args:
        num_samples (int): Number of samples
        num_classes (int): Number of classes
        
    Returns:
        tuple: (y_true, y_pred, y_prob)
    """
    np.random.seed(42)
    
    # Generate true labels
    y_true = np.random.randint(0, num_classes, num_samples)
    
    # Generate predicted labels with some noise
    y_pred = y_true.copy()
    # Add some misclassifications
    misclassification_indices = np.random.choice(num_samples, size=int(num_samples * 0.2), replace=False)
    y_pred[misclassification_indices] = np.random.randint(0, num_classes, len(misclassification_indices))
    
    # Generate probabilities
    y_prob = np.random.dirichlet(np.ones(num_classes), size=num_samples)
    
    return y_true, y_pred, y_prob


if __name__ == "__main__":
    # Example usage
    class_names = [
        'Acne', 'Eczema', 'Psoriasis', 'Rosacea', 'Melanoma',
        'Basal Cell Carcinoma', 'Squamous Cell Carcinoma', 'Actinic Keratosis',
        'Dermatitis', 'Viral Infections'
    ]
    
    evaluator = ModelEvaluator(class_names)
    
    # Create sample data
    y_true, y_pred, y_prob = create_sample_predictions()
    
    # Compute metrics
    metrics = evaluator.compute_basic_metrics(y_true, y_pred, y_prob)
    print("Basic Metrics:", metrics)
    
    # Generate classification report
    report = evaluator.generate_classification_report(y_true, y_pred)
    print("Classification Report:", report)
    
    # Plot confusion matrix
    fig_cm = evaluator.plot_confusion_matrix(y_true, y_pred, normalize=True)
    plt.show()
    
    # Plot ROC curves
    fig_roc = evaluator.plot_roc_curves(y_true, y_prob)
    plt.show()
    
    # Generate comprehensive report
    comprehensive_report = evaluator.generate_evaluation_report(
        y_true, y_pred, y_prob, "Sample Model"
    )
    evaluator.save_evaluation_report(comprehensive_report, "evaluation_report.json")
