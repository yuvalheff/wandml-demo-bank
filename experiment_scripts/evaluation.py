import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Tuple
from pathlib import Path
import os

from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, brier_score_loss
)
from sklearn.calibration import calibration_curve

from bank_marketing_campaign_prediction.config import ModelEvalConfig


class ModelEvaluator:
    def __init__(self, config: ModelEvalConfig, output_dir: str):
        self.config: ModelEvalConfig = config
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # App color palette
        self.app_color_palette = [
            'rgba(99, 110, 250, 0.8)',   # Blue
            'rgba(239, 85, 59, 0.8)',    # Red/Orange-Red
            'rgba(0, 204, 150, 0.8)',    # Green
            'rgba(171, 99, 250, 0.8)',   # Purple
            'rgba(255, 161, 90, 0.8)',   # Orange
            'rgba(25, 211, 243, 0.8)',   # Cyan
            'rgba(255, 102, 146, 0.8)',  # Pink
            'rgba(182, 232, 128, 0.8)',  # Light Green
            'rgba(255, 151, 255, 0.8)',  # Magenta
            'rgba(254, 203, 82, 0.8)'    # Yellow
        ]
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                      X_train: pd.DataFrame, y_train: pd.Series,
                      feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive model evaluation with visualizations.
        
        Returns:
        Dict containing all evaluation metrics and plot paths.
        """
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        brier_score = brier_score_loss(y_test, y_pred_proba)
        
        # Generate all plots
        plot_paths = {}
        plot_paths['roc_curve'] = self._plot_roc_curve(y_test, y_pred_proba, roc_auc)
        plot_paths['precision_recall'] = self._plot_precision_recall(y_test, y_pred_proba, avg_precision)
        plot_paths['confusion_matrix'] = self._plot_confusion_matrix(y_test, y_pred)
        plot_paths['calibration'] = self._plot_calibration_curve(y_test, y_pred_proba)
        plot_paths['feature_importance'] = self._plot_feature_importance(model, feature_names)
        plot_paths['prediction_distribution'] = self._plot_prediction_distribution(y_pred_proba, y_test)
        plot_paths['threshold_analysis'] = self._plot_threshold_analysis(y_test, y_pred_proba)
        
        # Compile results
        results = {
            'primary_metric': self.config.primary_metric,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'brier_score': brier_score,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'plot_paths': plot_paths,
            'total_samples': len(y_test),
            'positive_samples': int(sum(y_test == 2)),  # Assuming target is 1/2
            'negative_samples': int(sum(y_test == 1))
        }
        
        return results
    
    def _apply_style(self, fig):
        """Apply consistent styling to plots."""
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8B5CF6', size=12),
            title_font=dict(color='#7C3AED', size=16),
            xaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            yaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)', 
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            legend=dict(font=dict(color='#8B5CF6', size=11))
        )
        return fig
    
    def _plot_roc_curve(self, y_true, y_pred_proba, roc_auc):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba, pos_label=2)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color=self.app_color_palette[0], width=3)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color=self.app_color_palette[1], width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            showlegend=True
        )
        fig = self._apply_style(fig)
        
        filepath = os.path.join(self.plots_dir, "roc_curve.html")
        fig.write_html(filepath, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})
        return filepath
    
    def _plot_precision_recall(self, y_true, y_pred_proba, avg_precision):
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba, pos_label=2)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'PR Curve (AP = {avg_precision:.3f})',
            line=dict(color=self.app_color_palette[0], width=3)
        ))
        
        # Baseline (random classifier)
        baseline = sum(y_true == 2) / len(y_true)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[baseline, baseline],
            mode='lines',
            name=f'Random Classifier (AP = {baseline:.3f})',
            line=dict(color=self.app_color_palette[1], width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            showlegend=True
        )
        fig = self._apply_style(fig)
        
        filepath = os.path.join(self.plots_dir, "precision_recall_curve.html")
        fig.write_html(filepath, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})
        return filepath
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted: No', 'Predicted: Yes'],
            y=['Actual: No', 'Actual: Yes'],
            hoverongaps=False,
            colorscale='Purples',
            showscale=True,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16, "color": "white"}
        ))
        
        fig.update_layout(title='Confusion Matrix')
        fig = self._apply_style(fig)
        
        filepath = os.path.join(self.plots_dir, "confusion_matrix.html")
        fig.write_html(filepath, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})
        return filepath
    
    def _plot_calibration_curve(self, y_true, y_pred_proba):
        """Plot calibration curve."""
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, pos_label=2, n_bins=10
        )
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=mean_predicted_value, y=fraction_of_positives,
            mode='lines+markers',
            name='Calibration Curve',
            line=dict(color=self.app_color_palette[0], width=3),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(color=self.app_color_palette[1], width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Calibration Curve',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            showlegend=True
        )
        fig = self._apply_style(fig)
        
        filepath = os.path.join(self.plots_dir, "calibration_curve.html")
        fig.write_html(filepath, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})
        return filepath
    
    def _plot_feature_importance(self, model, feature_names):
        """Plot feature importance."""
        try:
            importance = model.get_feature_importance()
            if importance is None or len(importance) == 0:
                return None
            
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(len(importance))]
            
            # Create importance dataframe and sort
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(importance)],
                'importance': importance
            }).sort_values('importance', ascending=True)
            
            # Take top 20 features
            importance_df = importance_df.tail(20)
            
            fig = go.Figure(go.Bar(
                x=importance_df['importance'],
                y=importance_df['feature'],
                orientation='h',
                marker_color=self.app_color_palette[0]
            ))
            
            fig.update_layout(
                title='Top 20 Feature Importance',
                xaxis_title='Importance',
                yaxis_title='Features',
                height=600
            )
            fig = self._apply_style(fig)
            
            filepath = os.path.join(self.plots_dir, "feature_importance.html")
            fig.write_html(filepath, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})
            return filepath
        
        except Exception as e:
            print(f"Could not plot feature importance: {e}")
            return None
    
    def _plot_prediction_distribution(self, y_pred_proba, y_true):
        """Plot distribution of prediction probabilities by class."""
        
        fig = go.Figure()
        
        # Positive class predictions
        pos_probs = y_pred_proba[y_true == 2]
        fig.add_trace(go.Histogram(
            x=pos_probs,
            name='Positive Class (Subscribed)',
            opacity=0.7,
            marker_color=self.app_color_palette[0],
            nbinsx=30
        ))
        
        # Negative class predictions
        neg_probs = y_pred_proba[y_true == 1]
        fig.add_trace(go.Histogram(
            x=neg_probs,
            name='Negative Class (Not Subscribed)',
            opacity=0.7,
            marker_color=self.app_color_palette[1],
            nbinsx=30
        ))
        
        fig.update_layout(
            title='Distribution of Predicted Probabilities by True Class',
            xaxis_title='Predicted Probability',
            yaxis_title='Count',
            barmode='overlay',
            showlegend=True
        )
        fig = self._apply_style(fig)
        
        filepath = os.path.join(self.plots_dir, "prediction_distribution.html")
        fig.write_html(filepath, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})
        return filepath
    
    def _plot_threshold_analysis(self, y_true, y_pred_proba):
        """Plot precision, recall, and F1-score vs threshold."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba, pos_label=2)
        
        # Calculate F1 scores
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores)  # Handle division by zero
        
        # Adjust thresholds array to match precision/recall length
        thresholds = np.append(thresholds, 1.0)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=thresholds, y=precision,
            mode='lines',
            name='Precision',
            line=dict(color=self.app_color_palette[0], width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=thresholds, y=recall,
            mode='lines',
            name='Recall',
            line=dict(color=self.app_color_palette[1], width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=thresholds, y=f1_scores,
            mode='lines',
            name='F1-Score',
            line=dict(color=self.app_color_palette[2], width=2)
        ))
        
        fig.update_layout(
            title='Precision, Recall, and F1-Score vs Threshold',
            xaxis_title='Threshold',
            yaxis_title='Score',
            showlegend=True
        )
        fig = self._apply_style(fig)
        
        filepath = os.path.join(self.plots_dir, "threshold_analysis.html")
        fig.write_html(filepath, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})
        return filepath
    
    def find_optimal_threshold(self, y_true, y_pred_proba, metric='f1'):
        """
        Find optimal threshold for classification based on specified metric.
        
        Parameters:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall', 'youden')
        
        Returns:
        Dict containing optimal threshold and corresponding metrics
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba, pos_label=2)
        
        if metric == 'f1':
            # Calculate F1 scores
            f1_scores = 2 * (precision * recall) / (precision + recall)
            f1_scores = np.nan_to_num(f1_scores)
            
            # Find optimal threshold
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[min(optimal_idx, len(thresholds)-1)]
            
            result = {
                'threshold': optimal_threshold,
                'precision': precision[optimal_idx],
                'recall': recall[optimal_idx],
                'f1_score': f1_scores[optimal_idx]
            }
            
        elif metric == 'youden':
            # Youden's J statistic (sensitivity + specificity - 1)
            from sklearn.metrics import roc_curve
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba, pos_label=2)
            youden_scores = tpr - fpr
            optimal_idx = np.argmax(youden_scores)
            optimal_threshold = roc_thresholds[optimal_idx]
            
            result = {
                'threshold': optimal_threshold,
                'tpr': tpr[optimal_idx],
                'fpr': fpr[optimal_idx],
                'youden_j': youden_scores[optimal_idx]
            }
        
        return result
    
    def evaluate_at_thresholds(self, y_true, y_pred_proba, thresholds):
        """
        Evaluate model performance at multiple thresholds.
        
        Parameters:
        y_true: True labels
        y_pred_proba: Predicted probabilities  
        thresholds: List of thresholds to evaluate
        
        Returns:
        Dict containing metrics for each threshold
        """
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        results = {}
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            # Convert predictions to match target labels (1/2 instead of 0/1)
            y_pred_thresh = y_pred_thresh + 1
            
            results[threshold] = {
                'precision': precision_score(y_true, y_pred_thresh, pos_label=2, zero_division=0),
                'recall': recall_score(y_true, y_pred_thresh, pos_label=2, zero_division=0),
                'f1_score': f1_score(y_true, y_pred_thresh, pos_label=2, zero_division=0),
                'accuracy': accuracy_score(y_true, y_pred_thresh)
            }
        
        return results
