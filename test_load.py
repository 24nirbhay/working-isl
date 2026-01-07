import os
import sys

# Check what files exist
print("Checking data/dataset:")
for gesture in os.listdir('data/dataset'):
    gesture_path = os.path.join('data/dataset', gesture)
    if os.path.isdir(gesture_path):
        csv_files = [f for f in os.listdir(gesture_path) if f.endswith('.csv')]
        print(f"  {gesture}: {len(csv_files)} CSV files")

print("\nChecking data/sentences:")
if os.path.exists('data/sentences'):
    for label in os.listdir('data/sentences'):
        label_path = os.path.join('data/sentences', label)
        if os.path.isdir(label_path):
            csv_files = [f for f in os.listdir(label_path) if f.endswith('.csv')]
            print(f"  {label}: {len(csv_files)} CSV files")
else:
    print("  Directory not found")

# Now try to load
print("\n" + "="*60)
print("Attempting to load with preprocess module...")
print("="*60)

try:
    from src.preprocess import _load_one_dir, DATA_PATH, SENTENCE_PATH
    import numpy as np
    
    print(f"\nLoading from {DATA_PATH}:")
    X1, y1 = _load_one_dir(DATA_PATH, 126, 30)
    print(f"Loaded: {len(X1)} sequences")
    if len(y1) > 0:
        unique, counts = np.unique(y1, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  {label}: {count} sequences")
    
    print(f"\nLoading from {SENTENCE_PATH}:")
    X2, y2 = _load_one_dir(SENTENCE_PATH, 126, 30)
    print(f"Loaded: {len(X2)} sequences")
    if len(y2) > 0:
        unique, counts = np.unique(y2, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  {label}: {count} sequences")
    
    print(f"\nTotal: {len(X1) + len(X2)} sequences from both directories")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
