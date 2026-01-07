import pandas as pd
import numpy as np
import os
from typing import List, Tuple
from tensorflow.keras.utils import pad_sequences

DATA_PATH = "data/dataset"
SENTENCE_PATH = "data/sentences"


def _load_one_dir(root: str, expected_frame_length: int, maxlen: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load all CSV sequences from a single root directory.

    Structure: root/<label>/sequence_*.csv -> sequences (N x F) with F features.
    Returns padded X: (num_sequences, maxlen, F) and y: (num_sequences,).
    """
    if not os.path.exists(root):
        print(f"Warning: dataset root not found: {root}")
        return np.empty((0, maxlen, expected_frame_length), dtype=np.float32), np.empty((0,), dtype=object)

    print(f"Loading dataset from {root}...")
    data, labels = [], []
    skipped_files = 0

    for label in sorted(os.listdir(root)):
        label_path = os.path.join(root, label)
        if not os.path.isdir(label_path):
            continue

        file_count = 0
        for file in sorted(os.listdir(label_path)):
            if not file.endswith('.csv'):
                continue
            file_path = os.path.join(label_path, file)

            # Try different encodings to handle potential Unicode issues
            arr = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, header=None, encoding=encoding)
                    arr = df.values
                    break
                except Exception:
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

            # Ensure minimum sequence length
            if arr.shape[0] < 5:
                repeats = (5 // arr.shape[0]) + 1
                arr = np.tile(arr, (repeats, 1))[:5, :]

            data.append(arr.astype('float32'))
            labels.append(label)
            file_count += 1

        print(f"  Loaded {file_count} sequences for label '{label}'")

    if skipped_files > 0:
        print(f"Skipped {skipped_files} files due to errors or invalid data")

    if not data:
        return np.empty((0, maxlen, expected_frame_length), dtype=np.float32), np.empty((0,), dtype=object)

    data = pad_sequences(data, padding='post', dtype='float32', maxlen=maxlen)
    return np.array(data), np.array(labels)


def load_datasets(roots: List[str], expected_frame_length: int = 126, maxlen: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Load and merge datasets from multiple roots.

    roots: list of directory roots, e.g., [DATA_PATH, SENTENCE_PATH]
    """
    X_list, y_list = [], []
    for root in roots:
        X, y = _load_one_dir(root, expected_frame_length, maxlen)
        if len(X) > 0:
            X_list.append(X)
            y_list.append(y)

    if not X_list:
        return np.empty((0, maxlen, expected_frame_length), dtype=np.float32), np.empty((0,), dtype=object)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    print(f"Merged dataset: X {X.shape}, y {y.shape} from {len(X_list)} root(s)")
    print(f"Found labels: {sorted(set(y.tolist()))}")
    return X, y


def load_dataset(expected_frame_length: int = 126, maxlen: int = 30):
    """Backward-compatible loader from default DATA_PATH.

    Use load_datasets([DATA_PATH, SENTENCE_PATH]) to include sentence streams.
    """
    return _load_one_dir(DATA_PATH, expected_frame_length, maxlen)
