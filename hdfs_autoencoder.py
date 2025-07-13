#!/usr/bin/env python3
"""
HDFS Autoencoder Anomaly Detection

Autoencoder for HDFS logs. Finds anomalies by reconstruction error.

Author: Divya
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import joblib
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HDFSAutoencoder:
    def __init__(self, data_path="HDFS_features.csv"):
        """Initialize the HDFS Autoencoder detector."""
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        self.autoencoder_scaler = MinMaxScaler()
        self.model = None
        self.results = {}
        
        # Create results directory
        os.makedirs("hdfs_autoencoder_results", exist_ok=True)
        
    def load_data(self):
        """Load and preprocess the HDFS feature data."""
        logger.info("Loading HDFS feature data...")
        
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
            
            # Scale features for autoencoder
            self.X_scaled = self.scaler.fit_transform(self.X)
            self.X_normalized = self.autoencoder_scaler.fit_transform(self.X_scaled)
            
            logger.info(f"Prepared {self.X.shape[1]} features for analysis")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def build_autoencoder(self, input_dim, encoding_dim=64):
        """Build autoencoder model."""
        model = Sequential([
            # Encoder
            Dense(256, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(encoding_dim, activation='relu', name='encoding'),
            
            # Decoder
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(input_dim, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def train_model(self, encoding_dim=64, epochs=100, batch_size=32):
        """Train Autoencoder model."""
        logger.info("Training Autoencoder...")
        
        # Build model
        self.model = self.build_autoencoder(self.X_normalized.shape[1], encoding_dim)
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
        
        # Train model
        history = self.model.fit(
            self.X_normalized, self.X_normalized,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Get reconstruction error
        reconstructed = self.model.predict(self.X_normalized)
        mse = np.mean(np.power(self.X_normalized - reconstructed, 2), axis=1)
        
        # Determine threshold (95th percentile of reconstruction error)
        threshold = np.percentile(mse, 95)
        anomaly_predictions = (mse > threshold).astype(int)
        
        self.results = {
            'predictions': anomaly_predictions,
            'scores': -mse,  # Negative for consistency (lower = more anomalous)
            'reconstruction_error': mse,
            'threshold': threshold,
            'history': history.history,
            'encoding_dim': encoding_dim,
            'epochs_trained': len(history.history['loss']),
            'anomaly_count': anomaly_predictions.sum(),
            'anomaly_rate': anomaly_predictions.sum() / len(anomaly_predictions)
        }
        
        logger.info(f"Autoencoder completed. Detected {anomaly_predictions.sum()} anomalies")
        logger.info(f"Anomaly rate: {self.results['anomaly_rate']:.4f}")
        logger.info(f"Training completed in {len(history.history['loss'])} epochs")
    
    def evaluate_model(self):
        """Evaluate the Autoencoder model."""
        logger.info("Evaluating Autoencoder model...")
        
        evaluation_results = {
            'predictions': self.results['predictions'],
            'scores': self.results['scores'],
            'reconstruction_error': self.results['reconstruction_error'],
            'threshold': self.results['threshold'],
            'anomaly_count': self.results['anomaly_count'],
            'anomaly_rate': self.results['anomaly_rate']
        }
        
        # If we have true labels, calculate supervised metrics
        if self.y is not None:
            from sklearn.metrics import classification_report, confusion_matrix
            
            report = classification_report(self.y, self.results['predictions'], output_dict=True)
            conf_matrix = confusion_matrix(self.y, self.results['predictions'])
            auc = roc_auc_score(self.y, self.results['scores'])
            
            evaluation_results.update({
                'classification_report': report,
                'confusion_matrix': conf_matrix,
                'auc': auc,
                'true_labels': self.y
            })
            
            logger.info(f"Model AUC: {auc:.4f}")
        
        self.evaluation_results = evaluation_results
        return evaluation_results
    
    def plot_results(self):
        """Generate visualization plots."""
        logger.info("Generating plots...")
        
        # 1. Training History
        plt.figure(figsize=(15, 10))
        
        history = self.results['history']
        
        plt.subplot(2, 3, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Autoencoder Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Reconstruction Error Distribution
        plt.subplot(2, 3, 2)
        reconstruction_error = self.results['reconstruction_error']
        predictions = self.results['predictions']
        
        plt.hist(reconstruction_error[predictions == 0], bins=50, alpha=0.7, 
                label='Normal', color='blue', density=True)
        plt.hist(reconstruction_error[predictions == 1], bins=50, alpha=0.7, 
                label='Anomaly', color='red', density=True)
        plt.axvline(self.results['threshold'], color='red', linestyle='--', 
                   label=f'Threshold: {self.results["threshold"]:.4f}')
        plt.title('Reconstruction Error Distribution')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Density')
        plt.legend()
        
        # 3. ROC Curve (if labels available)
        if self.y is not None:
            plt.subplot(2, 3, 3)
            fpr, tpr, _ = roc_curve(self.y, self.results['scores'])
            auc = self.evaluation_results['auc']
            
            plt.plot(fpr, tpr, label=f'Autoencoder (AUC: {auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True)
        
        # 4. Confusion Matrix (if labels available)
        if self.y is not None:
            plt.subplot(2, 3, 4)
            conf_matrix = self.evaluation_results['confusion_matrix']
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
        
        # 5. PCA Visualization
        plt.subplot(2, 3, 5)
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
        
        # 6. Reconstruction Error vs Sample Index
        plt.subplot(2, 3, 6)
        plt.scatter(range(len(reconstruction_error)), reconstruction_error, 
                   c=predictions, cmap='viridis', alpha=0.6)
        plt.axhline(self.results['threshold'], color='red', linestyle='--', alpha=0.7)
        plt.colorbar(label='Anomaly Prediction')
        plt.xlabel('Sample Index')
        plt.ylabel('Reconstruction Error')
        plt.title('Reconstruction Error by Sample')
        
        plt.tight_layout()
        plt.savefig('hdfs_autoencoder_results/autoencoder_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Additional detailed plots
        self._plot_detailed_analysis()
    
    def _plot_detailed_analysis(self):
        """Generate additional detailed analysis plots."""
        
        # 1. Learning curves with zoom
        plt.figure(figsize=(15, 10))
        
        history = self.results['history']
        
        plt.subplot(2, 3, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Training History (Full)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Zoom in on final epochs
        plt.subplot(2, 3, 2)
        final_epochs = min(50, len(history['loss']))
        plt.plot(history['loss'][-final_epochs:], label='Training Loss')
        plt.plot(history['val_loss'][-final_epochs:], label='Validation Loss')
        plt.title(f'Training History (Last {final_epochs} Epochs)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Reconstruction error analysis
        plt.subplot(2, 3, 3)
        reconstruction_error = self.results['reconstruction_error']
        
        # Box plot
        predictions = self.results['predictions']
        normal_errors = reconstruction_error[predictions == 0]
        anomaly_errors = reconstruction_error[predictions == 1]
        
        plt.boxplot([normal_errors, anomaly_errors], labels=['Normal', 'Anomaly'])
        plt.title('Reconstruction Error Distribution')
        plt.ylabel('Reconstruction Error')
        
        # 3. Cumulative distribution
        plt.subplot(2, 3, 4)
        sorted_errors = np.sort(reconstruction_error)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        plt.plot(sorted_errors, cumulative, linewidth=2)
        plt.axvline(self.results['threshold'], color='red', linestyle='--', alpha=0.7)
        plt.title('Cumulative Distribution of Reconstruction Error')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Cumulative Probability')
        plt.grid(True, alpha=0.3)
        
        # 4. Anomaly rate by error percentile
        plt.subplot(2, 3, 5)
        percentiles = np.arange(5, 100, 5)
        anomaly_rates = []
        
        for p in percentiles:
            error_threshold = np.percentile(reconstruction_error, p)
            anomalies_above = (reconstruction_error > error_threshold).sum()
            rate = anomalies_above / len(reconstruction_error)
            anomaly_rates.append(rate)
        
        plt.plot(percentiles, anomaly_rates, marker='o', linewidth=2)
        plt.xlabel('Error Percentile')
        plt.ylabel('Anomaly Rate')
        plt.title('Anomaly Rate vs Error Percentile')
        plt.grid(True, alpha=0.3)
        
        # 5. Model summary
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        summary_text = f"""
        Autoencoder Results
        
        Encoding Dimension: {self.results['encoding_dim']}
        Epochs Trained: {self.results['epochs_trained']}
        Threshold: {self.results['threshold']:.4f}
        
        Anomalies Detected: {self.results['anomaly_count']}
        Anomaly Rate: {self.results['anomaly_rate']:.4f}
        """
        
        if self.y is not None:
            summary_text += f"\nAUC Score: {self.evaluation_results['auc']:.4f}"
        
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('hdfs_autoencoder_results/detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Feature reconstruction analysis
        self._plot_feature_reconstruction()
    
    def _plot_feature_reconstruction(self):
        """Plot feature-wise reconstruction analysis."""
        plt.figure(figsize=(15, 10))
        
        # Get original and reconstructed data
        reconstructed = self.model.predict(self.X_normalized)
        
        # Convert back to original scale
        reconstructed_scaled = self.autoencoder_scaler.inverse_transform(reconstructed)
        original_scaled = self.autoencoder_scaler.inverse_transform(self.X_normalized)
        
        # Feature-wise reconstruction error
        feature_errors = np.mean(np.power(original_scaled - reconstructed_scaled, 2), axis=0)
        top_error_features_idx = np.argsort(feature_errors)[-10:]
        
        plt.subplot(2, 2, 1)
        feature_names = self.X.columns[top_error_features_idx]
        plt.barh(range(len(top_error_features_idx)), feature_errors[top_error_features_idx])
        plt.yticks(range(len(top_error_features_idx)), feature_names)
        plt.xlabel('Mean Reconstruction Error')
        plt.title('Top 10 Features by Reconstruction Error')
        
        # Sample reconstruction comparison
        plt.subplot(2, 2, 2)
        sample_idx = np.random.choice(len(self.X), min(100, len(self.X)), replace=False)
        original_sample = original_scaled[sample_idx]
        reconstructed_sample = reconstructed_scaled[sample_idx]
        
        plt.scatter(original_sample.flatten(), reconstructed_sample.flatten(), alpha=0.6)
        plt.plot([original_sample.min(), original_sample.max()], 
                [original_sample.min(), original_sample.max()], 'r--', alpha=0.8)
        plt.xlabel('Original Values')
        plt.ylabel('Reconstructed Values')
        plt.title('Original vs Reconstructed Values')
        
        # Reconstruction error by feature count
        plt.subplot(2, 2, 3)
        feature_counts = np.arange(1, len(feature_errors) + 1)
        sorted_errors = np.sort(feature_errors)
        plt.plot(feature_counts, sorted_errors, linewidth=2)
        plt.xlabel('Feature Rank')
        plt.ylabel('Reconstruction Error')
        plt.title('Feature-wise Reconstruction Error Distribution')
        plt.grid(True, alpha=0.3)
        
        # Model architecture visualization
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        arch_text = f"""
        Autoencoder Architecture
        
        Input: {self.X.shape[1]} features
        
        Encoder:
        256 → 128 → {self.results['encoding_dim']}
        
        Decoder:
        {self.results['encoding_dim']} → 128 → 256 → {self.X.shape[1]}
        
        Activation: ReLU (hidden), Sigmoid (output)
        Loss: Mean Squared Error
        """
        
        plt.text(0.1, 0.9, arch_text, transform=plt.gca().transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('hdfs_autoencoder_results/feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """Save all results and model."""
        logger.info("Saving results...")
        
        # Save model
        self.model.save('hdfs_autoencoder_results/autoencoder_model.h5')
        
        # Save scalers
        joblib.dump(self.scaler, 'hdfs_autoencoder_results/scaler.pkl')
        joblib.dump(self.autoencoder_scaler, 'hdfs_autoencoder_results/autoencoder_scaler.pkl')
        
        # Save results summary
        summary = {
            'model': 'autoencoder',
            'encoding_dim': self.results['encoding_dim'],
            'epochs_trained': self.results['epochs_trained'],
            'threshold': self.results['threshold'],
            'anomaly_count': self.results['anomaly_count'],
            'anomaly_rate': self.results['anomaly_rate']
        }
        
        if self.y is not None:
            summary['auc'] = self.evaluation_results['auc']
        
        # Save summary to CSV
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv('hdfs_autoencoder_results/model_summary.csv', index=False)
        
        # Save detailed results
        with open('hdfs_autoencoder_results/detailed_results.txt', 'w') as f:
            f.write("HDFS Autoencoder Anomaly Detection Results\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("MODEL PARAMETERS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Encoding Dimension: {self.results['encoding_dim']}\n")
            f.write(f"Epochs Trained: {self.results['epochs_trained']}\n")
            f.write(f"Threshold: {self.results['threshold']:.4f}\n\n")
            
            f.write("DETECTION RESULTS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Anomalies Detected: {self.results['anomaly_count']}\n")
            f.write(f"Anomaly Rate: {self.results['anomaly_rate']:.4f}\n")
            
            if self.y is not None:
                f.write(f"AUC Score: {self.evaluation_results['auc']:.4f}\n\n")
                
                f.write("CLASSIFICATION REPORT:\n")
                f.write("-" * 25 + "\n")
                from sklearn.metrics import classification_report
                f.write(classification_report(self.y, self.results['predictions']))
        
        logger.info("Results saved to hdfs_autoencoder_results/ directory")
    
    def run_analysis(self, encoding_dim=64, epochs=100, batch_size=32):
        """Run the complete Autoencoder analysis."""
        logger.info("Starting HDFS Autoencoder Analysis")
        
        # Load data
        self.load_data()
        
        # Train model
        self.train_model(encoding_dim, epochs, batch_size)
        
        # Evaluate model
        self.evaluate_model()
        
        # Generate plots
        self.plot_results()
        
        # Save results
        self.save_results()
        
        logger.info("HDFS Autoencoder analysis completed successfully!")
        
        return self.evaluation_results

def main():
    """Main function to run the analysis."""
    detector = HDFSAutoencoder()
    results = detector.run_analysis()
    
    # Print summary
    print("\n" + "="*60)
    print("HDFS AUTOENCODER RESULTS SUMMARY")
    print("="*60)
    
    print(f"Anomalies Detected: {results['anomaly_count']}")
    print(f"Anomaly Rate: {results['anomaly_rate']:.4f}")
    print(f"Threshold: {results['threshold']:.4f}")
    
    if 'auc' in results:
        print(f"AUC Score: {results['auc']:.4f}")
    
    print(f"\nResults saved in: hdfs_autoencoder_results/")

if __name__ == "__main__":
    main() 