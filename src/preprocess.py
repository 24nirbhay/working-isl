import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

DATA_PATH = "data/dataset"


def load_dataset(expected_frame_length=126, maxlen=30):
    """Load all CSV sequences from DATA_PATH.

    Each CSV file represents a sequence (N x F) where F should be expected_frame_length.
    If CSVs were saved from single images, they will be 1 x F.
    Returns: X (num_sequences, maxlen, F), y (num_sequences,)
    """
    print(f"Loading dataset from {DATA_PATH}...")
    data, labels = [], []
    for gesture in os.listdir(DATA_PATH):
        gesture_path = os.path.join(DATA_PATH, gesture)
        if not os.path.isdir(gesture_path):
            continue
        for file in os.listdir(gesture_path):
            file_path = os.path.join(gesture_path, file)
            # Try different encodings to handle potential Unicode issues
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, header=None, encoding=encoding)
                    arr = df.values
                    break
                except Exception:
                    continue
            else:
                print(f"Warning: Could not read {file_path} with any encoding")
                continue
            
            # Ensure array is 2D and numeric
            arr = np.array(arr, dtype=np.float32, ndmin=2)
            # Normalize shape: if vectors are 3*N (old single-hand 63), expand to expected length by padding zeros
            if arr.shape[1] < expected_frame_length:
                pad_width = expected_frame_length - arr.shape[1]
                arr = np.pad(arr, ((0, 0), (0, pad_width)), mode='constant', constant_values=0.0)
            elif arr.shape[1] > expected_frame_length:
                arr = arr[:, :expected_frame_length]

            data.append(arr.astype('float32'))
            labels.append(gesture)

    # Pad sequences (time dimension) to ensure uniform length
    data = pad_sequences(data, padding='post', dtype='float32', maxlen=maxlen)
    
    X = np.array(data)
    y = np.array(labels)
    print(f"Dataset loaded: X shape {X.shape}, y shape {y.shape}")
    print(f"Found labels: {sorted(set(labels))}")
    
    return X, y
