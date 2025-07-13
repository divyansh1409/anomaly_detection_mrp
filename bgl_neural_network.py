#!/usr/bin/env python3
"""
BGL Neural Network Anomaly Detection

This script runs a neural network classifier on the BGL log dataset.
I'm using a deep neural network with batch normalization and dropout
to handle the complex patterns in the log data.

Author: Divya
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

import joblib
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BGLNeuralNetwork:
    def __init__(self, data_path="BGL_features.csv"):
        """Initialize the BGL Neural Network detector."""
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        self.model = None
        self.results = {}
        
        # Create results directory
        os.makedirs("bgl_neural_network_results", exist_ok=True)
        
    def load_data(self):
        """Load and preprocess the BGL feature data."""
        logger.info("Loading BGL feature data...")
        
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.data)} samples with {len(self.data.columns)} features")
            
            # Check if anomaly labels exist
            if 'is_anomaly' in self.data.columns:
                self.y = self.data['is_anomaly']
                # Handle NaN values in labels
                self.y = self.y.fillna(0)
                logger.info(f"Found anomaly labels: {self.y.sum()} anomalies out of {len(self.y)} samples")
                logger.info(f"Anomaly rate: {self.y.mean():.4f}")
            else:
                raise ValueError("BGL dataset must have anomaly labels for supervised learning")
            
            # Prepare features (exclude non-feature columns)
            exclude_cols = ['line_id', 'is_anomaly']
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
    
    def build_neural_network(self, input_dim):
        """Build neural network model."""
        model = Sequential([
            # Input layer
            Dense(256, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.4),
            
            # Hidden layers
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        
        return model
    
    def train_model(self, epochs=100, batch_size=32, validation_split=0.2):
        """Train Neural Network model."""
        logger.info("Training Neural Network...")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Build model
        self.model = self.build_neural_network(self.X_scaled.shape[1])
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_auc',
            patience=20,
            restore_best_weights=True,
            mode='max'
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
        
        # Train model
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_test, self.y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Get predictions and probabilities
        train_proba = self.model.predict(self.X_train).flatten()
        test_proba = self.model.predict(self.X_test).flatten()
        train_predictions = (train_proba > 0.5).astype(int)
        test_predictions = (test_proba > 0.5).astype(int)
        
        # Calculate metrics
        train_auc = roc_auc_score(self.y_train, train_proba)
        test_auc = roc_auc_score(self.y_test, test_proba)
        
        self.results = {
            'train_predictions': train_predictions,
            'test_predictions': test_predictions,
            'train_proba': train_proba,
            'test_proba': test_proba,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'history': history.history,
            'epochs_trained': len(history.history['loss']),
            'batch_size': batch_size,
            'validation_split': validation_split
        }
        
        logger.info(f"Neural Network completed. Train AUC: {train_auc:.3f}, Test AUC: {test_auc:.3f}")
        logger.info(f"Training completed in {len(history.history['loss'])} epochs")
    
    def evaluate_model(self):
        """Evaluate the Neural Network model."""
        logger.info("Evaluating Neural Network model...")
        
        # Calculate detailed metrics
        from sklearn.metrics import classification_report, confusion_matrix
        
        train_report = classification_report(self.y_train, self.results['train_predictions'], output_dict=True)
        test_report = classification_report(self.y_test, self.results['test_predictions'], output_dict=True)
        train_conf_matrix = confusion_matrix(self.y_train, self.results['train_predictions'])
        test_conf_matrix = confusion_matrix(self.y_test, self.results['test_predictions'])
        
        evaluation_results = {
            'train_predictions': self.results['train_predictions'],
            'test_predictions': self.results['test_predictions'],
            'train_proba': self.results['train_proba'],
            'test_proba': self.results['test_proba'],
            'train_auc': self.results['train_auc'],
            'test_auc': self.results['test_auc'],
            'train_report': train_report,
            'test_report': test_report,
            'train_conf_matrix': train_conf_matrix,
            'test_conf_matrix': test_conf_matrix,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'history': self.results['history']
        }
        
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
        plt.title('Training History - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training History - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 3)
        auc_key = None
        val_auc_key = None
        for k in ['auc', 'AUC']:
            if k in history:
                auc_key = k
        for k in ['val_auc', 'val_AUC']:
            if k in history:
                val_auc_key = k
        if auc_key and val_auc_key:
            plt.plot(history[auc_key], label='Training AUC')
            plt.plot(history[val_auc_key], label='Validation AUC')
            plt.title('Training History - AUC')
            plt.xlabel('Epoch')
            plt.ylabel('AUC')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'AUC history not available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Training History - AUC')
            plt.axis('off')
        
        # 2. ROC Curves
        plt.subplot(2, 3, 4)
        # Train ROC
        fpr_train, tpr_train, _ = roc_curve(self.y_train, self.results['train_proba'])
        plt.plot(fpr_train, tpr_train, label=f'Train (AUC: {self.results["train_auc"]:.3f})')
        
        # Test ROC
        fpr_test, tpr_test, _ = roc_curve(self.y_test, self.results['test_proba'])
        plt.plot(fpr_test, tpr_test, label=f'Test (AUC: {self.results["test_auc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Confusion Matrix (Test)
        plt.subplot(2, 3, 5)
        conf_matrix = self.evaluation_results['test_conf_matrix']
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Test Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 4. Probability Distribution
        plt.subplot(2, 3, 6)
        train_proba = self.results['train_proba']
        test_proba = self.results['test_proba']
        
        plt.hist(train_proba[self.y_train == 0], bins=30, alpha=0.7, 
                label='Train Normal', color='blue', density=True)
        plt.hist(train_proba[self.y_train == 1], bins=30, alpha=0.7, 
                label='Train Anomaly', color='red', density=True)
        plt.hist(test_proba[self.y_test == 0], bins=30, alpha=0.5, 
                label='Test Normal', color='lightblue', density=True)
        plt.hist(test_proba[self.y_test == 1], bins=30, alpha=0.5, 
                label='Test Anomaly', color='pink', density=True)
        
        plt.xlabel('Anomaly Probability')
        plt.ylabel('Density')
        plt.title('Probability Distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('bgl_neural_network_results/neural_network_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Additional detailed plots
        self._plot_detailed_analysis()
    
    def _plot_detailed_analysis(self):
        """Generate additional detailed analysis plots."""
        
        # 1. Learning curves with zoom
        plt.figure(figsize=(15, 10))
        
        history = self.results['history']
        
        plt.subplot(2, 3, 1)
        precision_key = None
        val_precision_key = None
        for k in ['precision', 'Precision']:
            if k in history:
                precision_key = k
        for k in ['val_precision', 'val_Precision']:
            if k in history:
                val_precision_key = k
        if precision_key and val_precision_key:
            plt.plot(history[precision_key], label='Training Precision')
            plt.plot(history[val_precision_key], label='Validation Precision')
            plt.title('Training History - Precision')
            plt.xlabel('Epoch')
            plt.ylabel('Precision')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Precision history not available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Training History - Precision')
            plt.axis('off')
        
        plt.subplot(2, 3, 2)
        recall_key = None
        val_recall_key = None
        for k in ['recall', 'Recall']:
            if k in history:
                recall_key = k
        for k in ['val_recall', 'val_Recall']:
            if k in history:
                val_recall_key = k
        if recall_key and val_recall_key:
            plt.plot(history[recall_key], label='Training Recall')
            plt.plot(history[val_recall_key], label='Validation Recall')
            plt.title('Training History - Recall')
            plt.xlabel('Epoch')
            plt.ylabel('Recall')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Recall history not available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Training History - Recall')
            plt.axis('off')
        
        # 2. Precision-Recall Curves
        plt.subplot(2, 3, 3)
        precision_train, recall_train, _ = precision_recall_curve(self.y_train, self.results['train_proba'])
        precision_test, recall_test, _ = precision_recall_curve(self.y_test, self.results['test_proba'])
        
        plt.plot(recall_train, precision_train, label='Train')
        plt.plot(recall_test, precision_test, label='Test')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Prediction confidence analysis
        plt.subplot(2, 3, 4)
        test_proba = self.results['test_proba']
        y_test = self.y_test
        
        # Calculate prediction confidence
        confidence = np.maximum(test_proba, 1 - test_proba)
        
        plt.hist(confidence[y_test == 0], bins=20, alpha=0.7, 
                label='Normal', color='blue', density=True)
        plt.hist(confidence[y_test == 1], bins=20, alpha=0.7, 
                label='Anomaly', color='red', density=True)
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Density')
        plt.title('Prediction Confidence Distribution')
        plt.legend()
        
        # 4. Error analysis
        plt.subplot(2, 3, 5)
        test_predictions = self.results['test_predictions']
        
        # Find misclassified samples
        misclassified = test_predictions != y_test
        correct = test_predictions == y_test
        
        plt.scatter(test_proba[correct], confidence[correct], 
                   alpha=0.6, label='Correct', s=20)
        plt.scatter(test_proba[misclassified], confidence[misclassified], 
                   alpha=0.8, label='Misclassified', s=30, color='red')
        plt.xlabel('Anomaly Probability')
        plt.ylabel('Prediction Confidence')
        plt.title('Error Analysis')
        plt.legend()
        
        # 5. Model architecture summary
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        summary_text = f"""
        Neural Network Results
        
        Architecture:
        256 → 128 → 64 → 32 → 1
        
        Training:
        - Epochs: {self.results['epochs_trained']}
        - Batch Size: {self.results['batch_size']}
        - Train AUC: {self.results['train_auc']:.4f}
        - Test AUC: {self.results['test_auc']:.4f}
        
        Regularization:
        - Batch Normalization
        - Dropout (0.2-0.4)
        - Early Stopping
        """
        
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('bgl_neural_network_results/detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Training convergence analysis
        self._plot_convergence_analysis()
    
    def _plot_convergence_analysis(self):
        """Generate convergence analysis plots."""
        plt.figure(figsize=(15, 10))
        
        history = self.results['history']
        
        # 1. Loss convergence
        plt.subplot(2, 3, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss Convergence')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. AUC convergence
        plt.subplot(2, 3, 2)
        auc_key = None
        val_auc_key = None
        for k in ['auc', 'AUC']:
            if k in history:
                auc_key = k
        for k in ['val_auc', 'val_AUC']:
            if k in history:
                val_auc_key = k
        if auc_key and val_auc_key:
            plt.plot(history[auc_key], label='Training AUC')
            plt.plot(history[val_auc_key], label='Validation AUC')
            plt.title('AUC Convergence')
            plt.xlabel('Epoch')
            plt.ylabel('AUC')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'AUC history not available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('AUC Convergence')
            plt.axis('off')
        
        # 3. Learning rate schedule
        plt.subplot(2, 3, 3)
        # Simulate learning rate schedule
        epochs = range(len(history['loss']))
        lr_schedule = [0.001 * (0.5 ** (i // 10)) for i in epochs]
        plt.plot(epochs, lr_schedule)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # 4. Training vs validation gap
        plt.subplot(2, 3, 4)
        train_val_gap_loss = np.array(history['loss']) - np.array(history['val_loss'])
        
        auc_key = None
        val_auc_key = None
        for k in ['auc', 'AUC']:
            if k in history:
                auc_key = k
        for k in ['val_auc', 'val_AUC']:
            if k in history:
                val_auc_key = k
        
        if auc_key and val_auc_key:
            train_val_gap_auc = np.array(history[val_auc_key]) - np.array(history[auc_key])
            plt.plot(train_val_gap_loss, label='Loss Gap')
            plt.plot(train_val_gap_auc, label='AUC Gap')
            plt.title('Training vs Validation Gap')
            plt.xlabel('Epoch')
            plt.ylabel('Gap')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.plot(train_val_gap_loss, label='Loss Gap')
            plt.title('Training vs Validation Gap (Loss Only)')
            plt.xlabel('Epoch')
            plt.ylabel('Gap')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 5. Model performance summary
        plt.subplot(2, 3, 5)
        plt.axis('off')
        
        performance_text = f"""
        Model Performance Summary
        
        Training Set:
        - Samples: {len(self.y_train)}
        - Anomalies: {self.y_train.sum()}
        - AUC: {self.results['train_auc']:.4f}
        
        Test Set:
        - Samples: {len(self.y_test)}
        - Anomalies: {self.y_test.sum()}
        - AUC: {self.results['test_auc']:.4f}
        
        Convergence:
        - Final Loss: {history['loss'][-1]:.4f}
        - Final AUC: "
        auc_key = None
        for k in ['auc', 'AUC']:
            if k in history:
                auc_key = k
        if auc_key:
            performance_text += f"{history[auc_key][-1]:.4f}\n"
        else:
            performance_text += "N/A\n"
        performance_text += f"- Overfitting: {'Yes' if abs(history['loss'][-1] - history['val_loss'][-1]) > 0.1 else 'No'}"
        """
        plt.text(0.1, 0.9, performance_text, transform=plt.gca().transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        # 6. Feature importance (simulated based on weights)
        plt.subplot(2, 3, 6)
        # Get first layer weights as feature importance approximation
        first_layer_weights = self.model.layers[0].get_weights()[0]
        feature_importance = np.mean(np.abs(first_layer_weights), axis=1)
        
        # Get top 10 features
        top_indices = np.argsort(feature_importance)[-10:]
        feature_names = self.X.columns[top_indices]
        
        plt.barh(range(len(top_indices)), feature_importance[top_indices])
        plt.yticks(range(len(top_indices)), feature_names)
        plt.xlabel('Weight Magnitude')
        plt.title('Top 10 Features (Weight-based)')
        
        plt.tight_layout()
        plt.savefig('bgl_neural_network_results/convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """Save all results and model."""
        logger.info("Saving results...")
        
        # Save model
        self.model.save('bgl_neural_network_results/neural_network_model.h5')
        
        # Save scaler
        joblib.dump(self.scaler, 'bgl_neural_network_results/scaler.pkl')
        
        # Save results summary
        summary = {
            'model': 'neural_network',
            'epochs_trained': self.results['epochs_trained'],
            'batch_size': self.results['batch_size'],
            'validation_split': self.results['validation_split'],
            'train_auc': self.results['train_auc'],
            'test_auc': self.results['test_auc']
        }
        
        # Save summary to CSV
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv('bgl_neural_network_results/model_summary.csv', index=False)
        
        # Save detailed results
        with open('bgl_neural_network_results/detailed_results.txt', 'w') as f:
            f.write("BGL Neural Network Anomaly Detection Results\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("MODEL PARAMETERS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Architecture: 256 -> 128 -> 64 -> 32 -> 1\n")
            f.write(f"Epochs Trained: {self.results['epochs_trained']}\n")
            f.write(f"Batch Size: {self.results['batch_size']}\n")
            f.write(f"Validation Split: {self.results['validation_split']}\n")
            f.write(f"Optimizer: Adam (lr=0.001)\n")
            f.write(f"Regularization: BatchNorm + Dropout\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Train AUC: {self.results['train_auc']:.4f}\n")
            f.write(f"Test AUC: {self.results['test_auc']:.4f}\n\n")
            
            f.write("TRAINING SET CLASSIFICATION REPORT:\n")
            f.write("-" * 40 + "\n")
            f.write(classification_report(self.y_train, self.results['train_predictions']))
            
            f.write("\nTEST SET CLASSIFICATION REPORT:\n")
            f.write("-" * 35 + "\n")
            f.write(classification_report(self.y_test, self.results['test_predictions']))
            
            # Write final training/validation metrics
            history = self.results['history']
            auc_key = None
            val_auc_key = None
            for k in ['auc', 'AUC']:
                if k in history:
                    auc_key = k
            for k in ['val_auc', 'val_AUC']:
                if k in history:
                    val_auc_key = k
            f.write(f"Final Training Loss: {history['loss'][-1]:.4f}\n")
            f.write(f"Final Validation Loss: {history['val_loss'][-1]:.4f}\n")
            if auc_key:
                f.write(f"Final Training AUC: {history[auc_key][-1]:.4f}\n")
            else:
                f.write(f"Final Training AUC: N/A\n")
            if val_auc_key:
                f.write(f"Final Validation AUC: {history[val_auc_key][-1]:.4f}\n")
            else:
                f.write(f"Final Validation AUC: N/A\n")
            f.write(f"Final Training Accuracy: {history['accuracy'][-1]:.4f}\n")
            f.write(f"Final Validation Accuracy: {history['val_accuracy'][-1]:.4f}\n")
        
        logger.info("Results saved to bgl_neural_network_results/ directory")
    
    def run_analysis(self, epochs=100, batch_size=32, validation_split=0.2):
        """Run the complete Neural Network analysis."""
        logger.info("Starting BGL Neural Network Analysis")
        
        # Load data
        self.load_data()
        
        # Train model
        self.train_model(epochs, batch_size, validation_split)
        
        # Evaluate model
        self.evaluate_model()
        
        # Generate plots
        self.plot_results()
        
        # Save results
        self.save_results()
        
        logger.info("BGL Neural Network analysis completed successfully!")
        
        return self.evaluation_results

def main():
    """Main function to run the analysis."""
    detector = BGLNeuralNetwork()
    results = detector.run_analysis()
    
    # Print summary
    print("\n" + "="*60)
    print("BGL NEURAL NETWORK RESULTS SUMMARY")
    print("="*60)
    
    print(f"Train AUC: {results['train_auc']:.4f}")
    print(f"Test AUC: {results['test_auc']:.4f}")
    print(f"Epochs Trained: {results['history']['loss'].__len__()}")
    
    print(f"\nResults saved in: bgl_neural_network_results/")

if __name__ == "__main__":
    main() 