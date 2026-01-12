"""
Debug real-time translator issues.
Shows what the model is actually predicting vs what's expected.
"""

import numpy as np
import cv2
import mediapipe as mp
from src.real_time_translator import GestureTranslator
from src.preprocess import load_datasets, DATA_PATH

print("="*70)
print("REAL-TIME TRANSLATOR DEBUG")
print("="*70)

# 1. Check model and label encoder
print("\n1. Loading model...")
translator = GestureTranslator()

print(f"   Model loaded: {translator.model}")
print(f"   Label encoder classes: {translator.label_encoder.classes_}")
print(f"   Num classes: {len(translator.label_encoder.classes_)}")
print(f"   Max sequence length: {translator.max_sequence_length}")
print(f"   Threshold: {translator.threshold}")

# 2. Load dataset to inspect
print("\n2. Loading training dataset for inspection...")
X, y = load_datasets([DATA_PATH], expected_frame_length=126, maxlen=30)
print(f"   Loaded X shape: {X.shape}")
print(f"   Loaded y shape: {y.shape}")
print(f"   Unique classes in dataset: {np.unique(y)}")

# 3. Test predictions on actual training data
print("\n3. Testing predictions on training data samples...")
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y)

for gesture_class in np.unique(y):
    # Get one sample for this gesture
    mask = y == gesture_class
    sample_idx = np.where(mask)[0][0]
    sample_sequence = X[sample_idx]
    
    # Predict
    input_data = np.expand_dims(sample_sequence, axis=0)
    predictions = translator.model.predict(input_data, verbose=0)
    
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    predicted_gesture = translator.label_encoder.inverse_transform([predicted_idx])[0]
    
    print(f"\n   Gesture: '{gesture_class}'")
    print(f"     Sample sequence shape: {sample_sequence.shape}")
    print(f"     Model prediction: '{predicted_gesture}' (confidence: {confidence:.4f})")
    print(f"     Prediction vector: {predictions[0]}")
    print(f"     Correct: {predicted_gesture == gesture_class}")

# 4. Check if model outputs match encoder
print("\n4. Checking model output shape vs label encoder...")
print(f"   Model output neurons: {predictions[0].shape[0]}")
print(f"   Label encoder classes: {len(translator.label_encoder.classes_)}")
print(f"   Match: {predictions[0].shape[0] == len(translator.label_encoder.classes_)}")

print("\n" + "="*70)
print("Debug complete. Check outputs above for issues.")
print("="*70)
