"""
Test predictions on actual training data to verify model learning.
"""
import tensorflow as tf
from src.model_architecture import ReduceSumLayer
from src.preprocess import load_datasets, DATA_PATH
from joblib import load
import numpy as np

print("="*70)
print("TESTING MODEL PREDICTIONS ON TRAINING DATA")
print("="*70)

# Load model
print("\n1. Loading model...")
le = load('models/model_20260112_130741/label_encoder.joblib')
model = tf.keras.models.load_model(
    'models/model_20260112_130741/model.keras', 
    custom_objects={'ReduceSumLayer': ReduceSumLayer}
)
print(f"   Classes: {le.classes_}")
print(f"   Model output neurons: {model.output_shape[1]}")

# Load training data
print("\n2. Loading training dataset...")
X, y = load_datasets([DATA_PATH], expected_frame_length=126, maxlen=30)
print(f"   Loaded {len(X)} samples")
print(f"   X shape: {X.shape}")

# Test predictions on a few samples from each class
print("\n3. Testing predictions on samples from each class:")
from sklearn.preprocessing import LabelEncoder
le_check = LabelEncoder()
le_check.fit(y)

correct = 0
total = 0

for gesture_class in np.unique(y):
    # Get first 3 samples for this gesture
    mask = y == gesture_class
    sample_indices = np.where(mask)[0][:3]
    
    print(f"\n   Gesture: '{gesture_class}'")
    for sample_idx in sample_indices:
        sample_sequence = X[sample_idx:sample_idx+1]  # Shape: (1, 30, 126)
        
        # Predict
        predictions = model.predict(sample_sequence, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        predicted_gesture = le.inverse_transform([predicted_idx])[0]
        
        is_correct = predicted_gesture == gesture_class
        total += 1
        if is_correct:
            correct += 1
        
        status = "OK" if is_correct else "WRONG"
        print(f"     Sample: {sample_idx:3d} | Pred: '{predicted_gesture:8s}' | Conf: {confidence:.4f} | {status}")

print(f"\n4. Summary:")
print(f"   Accuracy on tested samples: {correct}/{total} = {100*correct/total:.1f}%")

if 100*correct/total < 50:
    print("\nWARNING: Model accuracy is low!")
    print("   This suggests the model may not have learned properly.")
    print("   Possible causes:")
    print("   - Dataset is too small")
    print("   - Training didn't converge")
    print("   - Model architecture not suitable for the task")
else:
    print("\nOK: Model appears to be learning correctly!")

print("\n" + "="*70)
