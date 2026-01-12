#!/usr/bin/env python
"""
Diagnostic script to verify model and label encoder consistency.
Run this after training to check if the model output matches the label encoder.
"""

import os
import sys
import joblib
import tensorflow as tf
import numpy as np
from pathlib import Path

def diagnose_model(model_dir):
    """Check if model output classes match label encoder."""
    
    print(f"\n{'='*70}")
    print(f"MODEL DIAGNOSTIC REPORT")
    print(f"Model directory: {model_dir}")
    print(f"{'='*70}\n")
    
    # Check files exist
    model_file = os.path.join(model_dir, 'model.keras')
    encoder_file = os.path.join(model_dir, 'label_encoder.joblib')
    classes_file = os.path.join(model_dir, 'classes.txt')
    
    print("1. File Checks:")
    print(f"   Model file: {model_file}")
    print(f"     {'✓ EXISTS' if os.path.exists(model_file) else '✗ MISSING'}")
    print(f"   Encoder file: {encoder_file}")
    print(f"     {'✓ EXISTS' if os.path.exists(encoder_file) else '✗ MISSING'}")
    print(f"   Classes file: {classes_file}")
    print(f"     {'✓ EXISTS' if os.path.exists(classes_file) else '✗ MISSING'}")
    
    if not all([os.path.exists(f) for f in [model_file, encoder_file, classes_file]]):
        print("\n✗ Missing critical files!")
        return False
    
    # Load label encoder
    print("\n2. Label Encoder:")
    try:
        le = joblib.load(encoder_file)
        num_classes_encoder = len(le.classes_)
        print(f"   Classes loaded: {num_classes_encoder}")
        print(f"   Classes: {list(le.classes_)}")
    except Exception as e:
        print(f"   ✗ Failed to load: {e}")
        return False
    
    # Load classes.txt
    print("\n3. Classes Text File:")
    try:
        with open(classes_file, 'r') as f:
            classes_txt = [line.strip() for line in f.readlines()]
        print(f"   Classes loaded: {len(classes_txt)}")
        print(f"   Classes: {classes_txt}")
        
        # Verify match
        if set(classes_txt) == set(le.classes_):
            print(f"   ✓ Matches label encoder")
        else:
            print(f"   ✗ MISMATCH with label encoder!")
            print(f"   In encoder but not in file: {set(le.classes_) - set(classes_txt)}")
            print(f"   In file but not in encoder: {set(classes_txt) - set(le.classes_)}")
    except Exception as e:
        print(f"   ✗ Failed to load: {e}")
        return False
    
    # Load model and check output shape
    print("\n4. Model Architecture:")
    try:
        from src.model_architecture import ReduceSumLayer
        custom_objects = {'ReduceSumLayer': ReduceSumLayer}
    except:
        custom_objects = {'ReduceSumLayer': None}
    
    try:
        model = tf.keras.models.load_model(model_file, custom_objects=custom_objects)
        print(f"   Model loaded successfully")
        print(f"   Output layer shape: {model.layers[-1].output_shape}")
        
        output_shape = model.layers[-1].output_shape
        if len(output_shape) > 1:
            num_classes_model = output_shape[-1]
        else:
            num_classes_model = 1
            
        print(f"   Number of output classes: {num_classes_model}")
        
        # Check consistency
        if num_classes_model == num_classes_encoder:
            print(f"   ✓ Model output matches label encoder ({num_classes_model} classes)")
        else:
            print(f"   ✗ MISMATCH!")
            print(f"     Model outputs: {num_classes_model} classes")
            print(f"     Label encoder has: {num_classes_encoder} classes")
            return False
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        return False
    
    # Test prediction
    print("\n5. Prediction Test:")
    try:
        # Create dummy input
        dummy_input = np.zeros((1, model.input_shape[1], model.input_shape[2]))
        predictions = model.predict(dummy_input, verbose=0)
        
        print(f"   Dummy prediction output shape: {predictions.shape}")
        print(f"   Predictions: {predictions[0]}")
        
        predicted_idx = np.argmax(predictions[0])
        predicted_class = le.inverse_transform([predicted_idx])[0]
        
        print(f"   Max prediction index: {predicted_idx}")
        print(f"   Mapped class: '{predicted_class}'")
        
        if predicted_class in le.classes_:
            print(f"   ✓ Class exists in encoder")
        else:
            print(f"   ✗ Class NOT in encoder!")
            return False
    except Exception as e:
        print(f"   ✗ Prediction failed: {e}")
        return False
    
    print(f"\n{'='*70}")
    print("✓ ALL CHECKS PASSED - Model and encoder are consistent")
    print(f"{'='*70}\n")
    return True

if __name__ == "__main__":
    # Find latest model
    models_dir = "models"
    if os.path.exists(models_dir):
        model_dirs = sorted([d for d in os.listdir(models_dir) if d.startswith('model_')])
        if model_dirs:
            latest_model = os.path.join(models_dir, model_dirs[-1])
            print(f"Found latest model: {latest_model}")
            success = diagnose_model(latest_model)
            sys.exit(0 if success else 1)
        else:
            print("No models found in models/ directory")
            sys.exit(1)
    else:
        print(f"Models directory not found: {models_dir}")
        sys.exit(1)
