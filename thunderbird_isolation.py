#!/usr/bin/env python3
"""
Thunderbird Isolation Forest Anomaly Detection
=============================================

This script implements Isolation Forest anomaly detection for the Thunderbird dataset.
Isolation Forest is an unsupervised anomaly detection algorithm that works by
isolating observations by randomly selecting a feature and then randomly selecting
a split value between the maximum and minimum values of the selected feature.

Author: Your Name
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

import joblib
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThunderbirdIsolationForest:
    def __init__(self, data_path="Thunderbird_features.csv"):
        """Initialize the Thunderbird Isolation Forest detector."""
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        self.model = None
        self.results = {}
        
        # Create results directory
        os.makedirs("thunderbird_isolation_results", exist_ok=True)
        
    def load_data(self):
        """Load and preprocess the Thunderbird feature data."""
        logger.info("Loading Thunderbird feature data...")
        
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.data)} samples with {len(self.data.columns)} features")
            
            # Check if anomaly labels exist
            if 'is_anomaly' in self.data.columns:
                self.y = self.data['is_anomaly']
                logger.info(f"Found anomaly labels: {self.y.sum()} anomalies out of {len(self.y)} samples")
            else:
                self.y = None
                logger.info("No anomaly labels found - using unsupervised evaluation only")
            
            # Prepare features (exclude non-feature columns)
            exclude_cols = ['line_id', 'is_anomaly'] if 'is_anomaly' in self.data.columns else ['line_id']
            feature_cols = [col for col in self.data.columns if col not in exclude_cols]
            self.X = self.data[feature_cols]
            
            # Handle missing values
            self.X = self.X.fillna(0)
            
            # Scale features
            self.X_scaled = self.scaler.fit_transform(self.X)
            
            logger.info(f"Prepared {self.X.shape[1]} features for analysis")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def train_model(self, contamination=0.001, n_estimators=100, max_samples='auto'):  # Very low contamination since no anomalies found
        """Train Isolation Forest model."""
        logger.info("Training Isolation Forest...")
        
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        self.model.fit(self.X_scaled)
        
        # Get predictions and scores
        predictions = self.model.predict(self.X_scaled)
        scores = self.model.score_samples(self.X_scaled)
        
        # Convert to anomaly labels (1 for anomaly, 0 for normal)
        anomaly_predictions = (predictions == -1).astype(int)
        
        self.results = {
            'predictions': anomaly_predictions,
            'scores': scores,
            'contamination': contamination,
            'n_estimators': 100,
            'max_samples': 'auto',
            'anomaly_count': anomaly_predictions.sum(),
            'anomaly_rate': anomaly_predictions.sum() / len(anomaly_predictions)
        }
        
        logger.info(f"Isolation Forest completed. Detected {anomaly_predictions.sum()} anomalies")
        logger.info(f"Anomaly rate: {self.results['anomaly_rate']:.4f}")
    
    def evaluate_model(self):
        """Evaluate the Isolation Forest model."""
        logger.info("Evaluating Isolation Forest model...")
        
        evaluation_results = {
            'predictions': self.results['predictions'],
            'scores': self.results['scores'],
            'anomaly_count': self.results['anomaly_count'],
            'anomaly_rate': self.results['anomaly_rate']
        }
        
        # If we have true labels, calculate supervised metrics
        if self.y is not None:
            from sklearn.metrics import classification_report, confusion_matrix
            
            report = classification_report(self.y, self.results['predictions'], output_dict=True)
            conf_matrix = confusion_matrix(self.y, self.results['predictions'])
            
            # Handle case where no anomalies are detected
            if self.results['anomaly_count'] == 0:
                logger.warning("No anomalies detected - skipping AUC calculation")
                auc = None
            else:
                try:
                    auc = roc_auc_score(self.y, self.results['scores'])
                    logger.info(f"Model AUC: {auc:.4f}")
                except ValueError as e:
                    logger.warning(f"Could not calculate AUC: {e}")
                    auc = None
            
            evaluation_results.update({
                'classification_report': report,
                'confusion_matrix': conf_matrix,
                'auc': auc,
                'true_labels': self.y
            })
        
        self.evaluation_results = evaluation_results
        return evaluation_results
    
    def plot_results(self):
        """Generate visualization plots."""
        logger.info("Generating plots...")
        
        # 1. Score Distribution
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        scores = self.results['scores']
        predictions = self.results['predictions']
        
        plt.hist(scores[predictions == 0], bins=50, alpha=0.7, label='Normal', color='blue', density=True)
        plt.hist(scores[predictions == 1], bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
        plt.title('Isolation Forest Score Distribution')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.legend()
        
        # 2. ROC Curve (if labels available and AUC can be calculated)
        if self.y is not None and self.evaluation_results['auc'] is not None:
            plt.subplot(2, 3, 2)
            fpr, tpr, _ = roc_curve(self.y, self.results['scores'])
            auc = self.evaluation_results['auc']
            
            plt.plot(fpr, tpr, label=f'Isolation Forest (AUC: {auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True)
        elif self.y is not None:
            plt.subplot(2, 3, 2)
            plt.text(0.5, 0.5, 'ROC Curve\nNot Available\n(No anomalies detected)', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('ROC Curve')
            plt.axis('off')
        
        # 3. Confusion Matrix (if labels available)
        if self.y is not None:
            plt.subplot(2, 3, 3)
            conf_matrix = self.evaluation_results['confusion_matrix']
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
        
        # 4. PCA Visualization
        plt.subplot(2, 3, 4)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)
        
        # Plot normal points
        normal_mask = predictions == 0
        plt.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], 
                   c='blue', alpha=0.6, label='Normal', s=20)
        
        # Plot anomaly points
        anomaly_mask = predictions == 1
        plt.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1], 
                   c='red', alpha=0.8, label='Anomaly', s=30)
        
        plt.title('PCA Visualization of Anomalies')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        
        # 5. Feature Importance (based on variance)
        plt.subplot(2, 3, 5)
        feature_variance = np.var(self.X_scaled, axis=0)
        top_features_idx = np.argsort(feature_variance)[-10:]
        feature_names = self.X.columns[top_features_idx]
        
        plt.barh(range(len(top_features_idx)), feature_variance[top_features_idx])
        plt.yticks(range(len(top_features_idx)), feature_names)
        plt.xlabel('Feature Variance')
        plt.title('Top 10 Features by Variance')
        
        # 6. Anomaly Score vs Feature Count
        plt.subplot(2, 3, 6)
        plt.scatter(range(len(scores)), scores, c=predictions, cmap='viridis', alpha=0.6)
        plt.colorbar(label='Anomaly Prediction')
        plt.xlabel('Sample Index')
        plt.ylabel('Anomaly Score')
        plt.title('Anomaly Scores by Sample')
        
        plt.tight_layout()
        plt.savefig('thunderbird_isolation_results/isolation_forest_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Additional detailed plots
        self._plot_detailed_analysis()
    
    def _plot_detailed_analysis(self):
        """Generate additional detailed analysis plots."""
        
        # 1. Score distribution with threshold
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        scores = self.results['scores']
        predictions = self.results['predictions']
        
        # Calculate threshold (score at contamination percentile)
        threshold = np.percentile(scores, (1 - self.results['contamination']) * 100)
        
        plt.hist(scores, bins=50, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(threshold, color='red', linestyle='--', 
                   label=f'Threshold: {threshold:.4f}')
        plt.title('Anomaly Score Distribution with Threshold')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.legend()
        
        # 2. Cumulative distribution
        plt.subplot(2, 2, 2)
        sorted_scores = np.sort(scores)
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        plt.plot(sorted_scores, cumulative, linewidth=2)
        plt.axvline(threshold, color='red', linestyle='--', alpha=0.7)
        plt.title('Cumulative Distribution of Anomaly Scores')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Cumulative Probability')
        plt.grid(True, alpha=0.3)
        
        # 3. Anomaly rate by score percentile
        plt.subplot(2, 2, 3)
        percentiles = np.arange(5, 100, 5)
        anomaly_rates = []
        
        for p in percentiles:
            score_threshold = np.percentile(scores, p)
            anomalies_above = (scores > score_threshold).sum()
            rate = anomalies_above / len(scores)
            anomaly_rates.append(rate)
        
        plt.plot(percentiles, anomaly_rates, marker='o', linewidth=2)
        plt.xlabel('Score Percentile')
        plt.ylabel('Anomaly Rate')
        plt.title('Anomaly Rate vs Score Percentile')
        plt.grid(True, alpha=0.3)
        
        # 4. Model parameters summary
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        summary_text = f"""
        Isolation Forest Results
        
        Contamination: {self.results['contamination']}
        Estimators: {self.results['n_estimators']}
        Max Samples: {self.results['max_samples']}
        
        Anomalies Detected: {self.results['anomaly_count']}
        Anomaly Rate: {self.results['anomaly_rate']:.4f}
        """
        
        if self.y is not None:
            if self.evaluation_results['auc'] is not None:
                summary_text += f"\nAUC Score: {self.evaluation_results['auc']:.4f}"
            else:
                summary_text += "\nAUC Score: Not available (no anomalies detected)"
        
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('thunderbird_isolation_results/detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """Save all results and model."""
        logger.info("Saving results...")
        
        # Save model
        joblib.dump(self.model, 'thunderbird_isolation_results/isolation_forest_model.pkl')
        
        # Save scaler
        joblib.dump(self.scaler, 'thunderbird_isolation_results/scaler.pkl')
        
        # Save results summary
        summary = {
            'model': 'isolation_forest',
            'contamination': self.results['contamination'],
            'n_estimators': self.results['n_estimators'],
            'max_samples': self.results['max_samples'],
            'anomaly_count': self.results['anomaly_count'],
            'anomaly_rate': self.results['anomaly_rate']
        }
        
        if self.y is not None:
            summary['auc'] = self.evaluation_results['auc']
        
        # Save summary to CSV
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv('thunderbird_isolation_results/model_summary.csv', index=False)
        
        # Save detailed results
        with open('thunderbird_isolation_results/detailed_results.txt', 'w') as f:
            f.write("Thunderbird Isolation Forest Anomaly Detection Results\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("MODEL PARAMETERS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Contamination: {self.results['contamination']}\n")
            f.write(f"Number of Estimators: {self.results['n_estimators']}\n")
            f.write(f"Max Samples: {self.results['max_samples']}\n\n")
            
            f.write("DETECTION RESULTS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Anomalies Detected: {self.results['anomaly_count']}\n")
            f.write(f"Anomaly Rate: {self.results['anomaly_rate']:.4f}\n")
            
            if self.y is not None:
                if self.evaluation_results['auc'] is not None:
                    f.write(f"AUC Score: {self.evaluation_results['auc']:.4f}\n\n")
                else:
                    f.write("AUC Score: Not available (no anomalies detected)\n\n")
                f.write("CLASSIFICATION REPORT:\n")
                f.write("-" * 25 + "\n")
                from sklearn.metrics import classification_report
                f.write(classification_report(self.y, self.results['predictions']))
        
        logger.info("Results saved to thunderbird_isolation_results/ directory")
    
    def run_analysis(self, contamination=0.001, n_estimators=100, max_samples='auto'):
        """Run the complete Isolation Forest analysis."""
        logger.info("Starting Thunderbird Isolation Forest Analysis")
        
        # Load data
        self.load_data()
        
        # Train model
        self.train_model(contamination, n_estimators, max_samples)
        
        # Evaluate model
        self.evaluate_model()
        
        # Generate plots
        self.plot_results()
        
        # Save results
        self.save_results()
        
        logger.info("Thunderbird Isolation Forest analysis completed successfully!")
        
        return self.evaluation_results

def main():
    """Main function to run the analysis."""
    detector = ThunderbirdIsolationForest()
    results = detector.run_analysis()
    
    # Print summary
    print("\n" + "="*60)
    print("THUNDERBIRD ISOLATION FOREST RESULTS SUMMARY")
    print("="*60)
    
    print(f"Anomalies Detected: {results['anomaly_count']}")
    print(f"Anomaly Rate: {results['anomaly_rate']:.4f}")
    
    if 'auc' in results and results['auc'] is not None:
        print(f"AUC Score: {results['auc']:.4f}")
    elif 'auc' in results:
        print("AUC Score: Not available (no anomalies detected)")
    
    print(f"\nResults saved in: thunderbird_isolation_results/")

if __name__ == "__main__":
    main() 