import pandas as pd
import numpy as np
import os
from tensorflow.keras.utils import pad_sequences

DATA_PATH = "data/dataset"


def load_dataset(expected_frame_length=126, maxlen=30):
    """Load all CSV sequences from DATA_PATH.

    Each CSV file represents a sequence (N x F) where F should be expected_frame_length.
    If CSVs were saved from single images, they will be 1 x F.
    Returns: X (num_sequences, maxlen, F), y (num_sequences,)
    """
    print(f"Loading dataset from {DATA_PATH}...")
    data, labels = [], []
    skipped_files = 0
    
    for gesture in sorted(os.listdir(DATA_PATH)):
        gesture_path = os.path.join(DATA_PATH, gesture)
        if not os.path.isdir(gesture_path):
            continue
        
        gesture_files = 0
        for file in sorted(os.listdir(gesture_path)):
            file_path = os.path.join(gesture_path, file)
            
            # Try different encodings to handle potential Unicode issues
            arr = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, header=None, encoding=encoding)
                    arr = df.values
                    break
                except Exception as e:
                    continue
            
            if arr is None:
                print(f"Warning: Could not read {file_path}")
                skipped_files += 1
                continue
            
            # Ensure array is 2D and numeric
            arr = np.array(arr, dtype=np.float32, ndmin=2)
            
            # Validate and fix feature dimension
            if arr.shape[1] < expected_frame_length:
                pad_width = expected_frame_length - arr.shape[1]
                arr = np.pad(arr, ((0, 0), (0, pad_width)), mode='constant', constant_values=0.0)
            elif arr.shape[1] > expected_frame_length:
                arr = arr[:, :expected_frame_length]
            
            # Handle very short sequences by repeating frames
            if arr.shape[0] < 5:
                # Repeat frames to reach minimum length
                repeats = (5 // arr.shape[0]) + 1
                arr = np.tile(arr, (repeats, 1))[:5, :]
            
            data.append(arr.astype('float32'))
            labels.append(gesture)
            gesture_files += 1
        
        print(f"  Loaded {gesture_files} sequences for gesture '{gesture}'")
    
    if skipped_files > 0:
        print(f"Skipped {skipped_files} files due to errors or invalid data")

    # Pad sequences (time dimension) to ensure uniform length
    data = pad_sequences(data, padding='post', dtype='float32', maxlen=maxlen)
    
    X = np.array(data)
    y = np.array(labels)
    print(f"Dataset loaded: X shape {X.shape}, y shape {y.shape}")
    print(f"Found labels: {sorted(set(labels))}")
    
    return X, y
