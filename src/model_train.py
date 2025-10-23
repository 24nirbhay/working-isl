import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from joblib import dump
import os
try:
    # when executed as part of a package
    from .preprocess import load_dataset
except Exception:
    # fallback when running the module as a script: `python src/model_train.py`
    from preprocess import load_dataset


def train_model(epochs=30, batch_size=32):
    # expected_frame_length set to 126 (two hands x 21 landmarks x 3 coords)
    X, y = load_dataset(expected_frame_length=126, maxlen=30)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    os.makedirs("models", exist_ok=True)
    dump(le, "models/tokenizer.pkl")

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded)

    time_steps = X_train.shape[1]
    feature_dim = X_train.shape[2]

    model = tf.keras.Sequential([
        tf.keras.layers.Masking(mask_value=0., input_shape=(time_steps, feature_dim)),
        tf.keras.layers.LSTM(256, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(set(y_encoded)), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        # Keras now expects the native .keras extension for full model saves
        tf.keras.callbacks.ModelCheckpoint('models/isl_seq2seq_model.keras', save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    ]

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=callbacks)

    # final save (ModelCheckpoint already saved best model)
    model.save("models/isl_seq2seq_model.keras")


if __name__ == '__main__':
    train_model()
