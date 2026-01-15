import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from joblib import dump
import os
import logging
from datetime import datetime
try:
    from .preprocess import load_dataset, load_datasets, DATA_PATH, SENTENCE_PATH
    from .model_architecture import create_sign_language_model
except Exception:
    from preprocess import load_dataset, load_datasets, DATA_PATH, SENTENCE_PATH
    from model_architecture import create_sign_language_model

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to prevent TensorFlow from allocating all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info(f"GPU available: {len(gpus)} GPU(s) detected")
        logging.info(f"GPU devices: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        logging.warning(f"GPU configuration error: {e}")
else:
    logging.warning("No GPU detected. Training will use CPU.")


def train_model(epochs=12, batch_size=10, validation_split=0.2, roots=None):
    logging.info("Starting model training process...")
    
    # Create timestamped model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join("models", f"model_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Load and preprocess dataset
    logging.info("Loading dataset...")
    if roots is None:
        # Default: gestures dataset; include sentences automatically if present
        roots = [DATA_PATH]
        if os.path.isdir(SENTENCE_PATH):
            # Include if it contains at least one label directory with CSVs
            has_data = False
            for label in os.listdir(SENTENCE_PATH):
                label_path = os.path.join(SENTENCE_PATH, label)
                if os.path.isdir(label_path) and any(f.endswith('.csv') for f in os.listdir(label_path)):
                    has_data = True
                    break
            if has_data:
                logging.info("Including sentence streams from data/sentences in training dataset")
                roots.append(SENTENCE_PATH)
            else:
                logging.info("data/sentences present but empty; ignoring")

    if len(roots) == 1:
        X, y = load_datasets(roots, expected_frame_length=126, maxlen=30)
    else:
        X, y = load_datasets(roots, expected_frame_length=126, maxlen=30)
    
    if len(X) == 0 or len(y) == 0:
        raise ValueError("No data loaded. Please check your dataset directory.")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    logging.info(f"Found {len(le.classes_)} unique classes: {list(le.classes_)}")
    
    # Save label encoder immediately (IMPORTANT: before training in case of error)
    label_encoder_path = os.path.join(model_dir, "label_encoder.joblib")
    classes_path = os.path.join(model_dir, "classes.txt")
    try:
        dump(le, label_encoder_path)
        with open(classes_path, "w") as f:
            f.write("\n".join(le.classes_))
        logging.info(f"Label encoder saved. Classes: {list(le.classes_)}")
    except Exception as e:
        logging.error(f"Failed to save label encoder: {e}")
        raise
    
    # Log class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    logging.info(f"Class distribution: {class_distribution}")
    
    # Check for minimum samples per class
    min_samples = min(counts)
    if min_samples < 2:
        logging.error(f"Some classes have only {min_samples} sample(s). Need at least 2 samples per class for training.")
        logging.error(f"Classes with insufficient samples:")
        for label, count in class_distribution.items():
            if count < 2:
                logging.error(f"  - '{label}': {count} sample(s) (need at least 2)")
        raise ValueError(f"Insufficient training data. Collect at least 2 sequences for each gesture.")
    elif min_samples < 5:
        logging.warning(f"Some classes have very few samples (minimum: {min_samples}). Recommend collecting at least 10 samples per gesture for better accuracy.")
  
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=validation_split,
        stratify=y_encoded,
        random_state=42
    )
    
    logging.info(f"Training set shape: {X_train.shape}")
    logging.info(f"Test set shape: {X_test.shape}")
    
    # Create model
    time_steps = X_train.shape[1]
    feature_dim = X_train.shape[2]
    num_classes = len(set(y_encoded))
    
    model = create_sign_language_model(time_steps, feature_dim, num_classes)

    # Setup callbacks
    callbacks = [
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, 'model.keras'),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir, 'logs'),
            histogram_freq=1
        )
    ]
    
    # Train model
    logging.info("Starting training...")
    try:
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
    except Exception as e:
        logging.error(f"Training failed: {e}")
        logging.info(f"Label encoder already saved with all {len(le.classes_)} classes.")
        with open(os.path.join(model_dir, 'training_error.log'), 'w') as f:
            f.write(f"Training error: {str(e)}")
        raise
    
    # Evaluate model
    logging.info("Evaluating model...")
    try:
        eval_results = model.evaluate(X_test, y_test, verbose=0)
        test_loss = eval_results[0]
        test_acc = eval_results[1] if len(eval_results) > 1 else 0.0
        test_top3 = eval_results[2] if len(eval_results) > 2 else 0.0
        logging.info(f"Test loss: {test_loss:.4f}")
        logging.info(f"Test accuracy: {test_acc:.4f}")
        logging.info(f"Top-3 accuracy: {test_top3:.4f}")
    except Exception as e:
        logging.warning(f"Evaluation failed: {e}. Model still trained and saved.")
    
    # Save training history
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(model_dir, 'training_history.csv'), index=False)
    
    # Save model summary to text file
    with open(os.path.join(model_dir, 'model_summary.txt'), 'w', encoding='utf-8') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    logging.info(f"Training complete. Model and artifacts saved to {model_dir}")
    
    return model, history, model_dir


if __name__ == '__main__':
    train_model()
