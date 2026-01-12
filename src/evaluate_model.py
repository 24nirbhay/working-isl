"""
Model Evaluation and Testing Script
Evaluates trained model on test dataset from fortesting/ directory.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import glob
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Allow optional auto-generation of test_dataset from images
try:
    from .dataset_manager import create_test_dataset_from_fortesting, TEST_DATASET_PATH
except Exception:
    try:
        from dataset_manager import create_test_dataset_from_fortesting, TEST_DATASET_PATH
    except Exception:
        create_test_dataset_from_fortesting = None
        TEST_DATASET_PATH = "data/test_dataset"

try:
    from .preprocess import load_datasets
    from .model_architecture import ReduceSumLayer
except ImportError:
    from preprocess import load_datasets
    from model_architecture import ReduceSumLayer


TEST_DATASET_PATH = "data/test_dataset"
EVAL_RESULTS_PATH = "data/evaluation_results"


def load_test_data(test_path=TEST_DATASET_PATH, maxlen=30):
    """Load test dataset from test_dataset directory.
    If test sequences are missing, optionally generate them from fortesting/ images.
    """
    # If directory missing or empty, attempt auto-conversion from images
    need_convert = (not os.path.exists(test_path)) or not any(os.scandir(test_path))
    if need_convert and create_test_dataset_from_fortesting:
        print(f"Test sequences not found at {test_path}. Auto-generating from fortesting/ images...")
        create_test_dataset_from_fortesting(frames_per_sequence=30, debug=False)

    if not os.path.exists(test_path):
        print(f"Error: Test dataset not found at {test_path}")
        print(f"Run: python app.py convert-test")
        return None, None, None
    
    print(f"\nLoading test data from {test_path}...")
    X_test, y_test = load_datasets([test_path], expected_frame_length=126, maxlen=maxlen)
    
    if len(X_test) == 0:
        print("No test data loaded!")
        return None, None, None
    
    print(f"Loaded {len(X_test)} test samples")
    return X_test, y_test, None


def evaluate_model(model_path, test_X=None, test_y=None):
    """Evaluate trained model on test dataset.
    
    Args:
        model_path: Path to trained model directory
        test_X: Optional pre-loaded test sequences
        test_y: Optional pre-loaded test labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load model
    model_file = os.path.join(model_path, 'model.keras')
    if not os.path.exists(model_file):
        print(f"Error: Model not found at {model_file}")
        return None
    
    print(f"\nLoading model from {model_path}...")
    model = tf.keras.models.load_model(model_file, custom_objects={'ReduceSumLayer': ReduceSumLayer})
    
    # Load label encoder
    label_encoder_file = os.path.join(model_path, 'label_encoder.joblib')
    label_encoder = joblib.load(label_encoder_file)
    
    # Load test data if not provided
    if test_X is None or test_y is None:
        test_X, test_y, _ = load_test_data()
        if test_X is None:
            return None
    
    # Encode labels
    y_test_encoded = label_encoder.transform(test_y)
    
    print(f"Test set size: {len(test_X)}")
    print(f"Classes: {list(label_encoder.classes_)}")
    
    # Make predictions
    print("\nMaking predictions on test set...")
    predictions = model.predict(test_X, verbose=1)
    predicted_indices = np.argmax(predictions, axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_indices)
    
    # Get prediction confidence
    prediction_confidence = np.max(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(test_y, predicted_labels)
    precision = precision_score(test_y, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(test_y, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(test_y, predicted_labels, average='weighted', zero_division=0)
    
    # Per-class metrics
    class_report = classification_report(
        test_y, predicted_labels, 
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(test_y, predicted_labels, labels=label_encoder.classes_)
    
    # Results dictionary
    results = {
        'model_path': model_path,
        'test_samples': len(test_X),
        'num_classes': len(label_encoder.classes_),
        'classes': list(label_encoder.classes_),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': predicted_labels,
        'ground_truth': test_y,
        'confidence': prediction_confidence,
        'predictions_encoded': predicted_indices,
        'ground_truth_encoded': y_test_encoded,
        'class_report': class_report,
        'confusion_matrix': conf_matrix,
        'label_encoder': label_encoder
    }
    
    return results


def print_evaluation_report(results):
    """Print detailed evaluation report."""
    if results is None:
        print("No evaluation results to display")
        return
    
    print(f"\n{'='*80}")
    print(f"MODEL EVALUATION REPORT")
    print(f"{'='*80}\n")
    
    print(f"Model: {results['model_path']}")
    print(f"Test Samples: {results['test_samples']}")
    print(f"Number of Classes: {results['num_classes']}")
    print(f"Classes: {', '.join(results['classes'])}\n")
    
    print(f"{'='*80}")
    print(f"OVERALL METRICS")
    print(f"{'='*80}\n")
    
    print(f"Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1']:.4f}\n")
    
    print(f"{'='*80}")
    print(f"PER-CLASS METRICS")
    print(f"{'='*80}\n")
    
    class_report = results['class_report']
    for gesture in results['classes']:
        if gesture in class_report:
            metrics = class_report[gesture]
            print(f"{gesture:15} - Precision: {metrics['precision']:.4f}, "
                  f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1-score']:.4f}")
    
    print(f"\n{'='*80}")
    print(f"CONFUSION MATRIX")
    print(f"{'='*80}\n")
    
    # Print confusion matrix
    conf_matrix = results['confusion_matrix']
    print("Predicted →")
    print("Ground ↓   ", end="")
    for cls in results['classes']:
        print(f"{cls:8}", end="")
    print()
    
    for i, true_label in enumerate(results['classes']):
        print(f"{true_label:8} ", end="")
        for j in range(len(results['classes'])):
            print(f"{conf_matrix[i,j]:8}", end="")
        print()
    
    print(f"\n{'='*80}\n")


def save_evaluation_report(results, output_dir=EVAL_RESULTS_PATH):
    """Save evaluation results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'metric': ['accuracy', 'precision', 'recall', 'f1'],
        'value': [results['accuracy'], results['precision'], results['recall'], results['f1']]
    })
    metrics_file = os.path.join(output_dir, 'metrics.csv')
    metrics_df.to_csv(metrics_file, index=False)
    print(f"✓ Metrics saved to {metrics_file}")
    
    # Save per-class results
    class_results = []
    for gesture in results['classes']:
        if gesture in results['class_report']:
            metrics = results['class_report'][gesture]
            class_results.append({
                'gesture': gesture,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1-score'],
                'support': int(metrics['support'])
            })
    
    class_df = pd.DataFrame(class_results)
    class_file = os.path.join(output_dir, 'per_class_metrics.csv')
    class_df.to_csv(class_file, index=False)
    print(f"✓ Per-class metrics saved to {class_file}")
    
    # Save predictions
    pred_df = pd.DataFrame({
        'ground_truth': results['ground_truth'],
        'predicted': results['predictions'],
        'confidence': results['confidence'],
        'correct': results['ground_truth'] == results['predictions']
    })
    pred_file = os.path.join(output_dir, 'predictions.csv')
    pred_df.to_csv(pred_file, index=False)
    print(f"✓ Predictions saved to {pred_file}")
    
    # Save confusion matrix
    conf_df = pd.DataFrame(
        results['confusion_matrix'],
        index=results['classes'],
        columns=results['classes']
    )
    conf_file = os.path.join(output_dir, 'confusion_matrix.csv')
    conf_df.to_csv(conf_file)
    print(f"✓ Confusion matrix saved to {conf_file}")
    
    # Generate visualizations
    try:
        # Confusion matrix heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plot_file = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(plot_file, dpi=100)
        plt.close()
        print(f"✓ Confusion matrix plot saved to {plot_file}")
    except Exception as e:
        print(f"⚠ Could not generate confusion matrix plot: {e}")
    
    # Per-class metrics bar chart
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        x_pos = np.arange(len(class_df))
        width = 0.25
        
        ax.bar(x_pos - width, class_df['precision'], width, label='Precision', alpha=0.8)
        ax.bar(x_pos, class_df['recall'], width, label='Recall', alpha=0.8)
        ax.bar(x_pos + width, class_df['f1_score'], width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Gesture')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(class_df['gesture'], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, 'per_class_metrics.png')
        plt.savefig(plot_file, dpi=100)
        plt.close()
        print(f"✓ Per-class metrics plot saved to {plot_file}")
    except Exception as e:
        print(f"⚠ Could not generate per-class metrics plot: {e}")
    
    print(f"\n✓ All evaluation results saved to {output_dir}")


def find_latest_model(models_dir="models"):
    """Find the most recently trained model."""
    model_dirs = glob.glob(os.path.join(models_dir, "model_*"))
    if not model_dirs:
        print(f"No models found in {models_dir}")
        return None
    
    latest_model = max(model_dirs, key=os.path.getmtime)
    return latest_model


if __name__ == "__main__":
    import sys
    
    # Find model
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = find_latest_model()
    
    if model_path is None:
        print("Please specify a model path or ensure models exist")
        print("Usage: python evaluate_model.py [model_path]")
        sys.exit(1)
    
    print(f"Using model: {model_path}")
    
    # Evaluate
    results = evaluate_model(model_path)
    
    if results:
        print_evaluation_report(results)
        save_evaluation_report(results)
        print(f"\nOverall Accuracy: {results['accuracy']*100:.2f}%")
    else:
        print("Evaluation failed")
        sys.exit(1)
