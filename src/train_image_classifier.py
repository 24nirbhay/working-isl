import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from joblib import dump

def train_model():
    """
    Trains an image classification model for hand gestures using transfer learning.
    """
    # --- 1. Configuration ---
    # Resolve repository root relative to this file so paths work regardless of CWD
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Use the provided handsigns image dataset
    # The repository contains `data/handsigns/<label>/*.jpg|png`
    DATA_DIR = os.path.join(repo_root, 'data', 'handsigns')

    # Model-specific parameters
    IMG_SIZE = (224, 224) # Input size for MobileNetV2
    BATCH_SIZE = 32       # Number of images to process in a single batch

    # --- 2. Create Datasets (Training and Validation) ---
    print("Creating training dataset...")
    # verify DATA_DIR exists and list label folders + counts
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")

    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    label_dirs = [d for d in sorted(os.listdir(DATA_DIR)) if os.path.isdir(os.path.join(DATA_DIR, d))]
    if not label_dirs:
        raise ValueError(f"No label subdirectories found in {DATA_DIR}. Expected structure: {DATA_DIR}/<label>/*.jpg")

    total_images = 0
    print("Found label directories:")
    for label in label_dirs:
        p = os.path.join(DATA_DIR, label)
        imgs = [f for f in os.listdir(p) if os.path.splitext(f)[1].lower() in image_exts]
        print(f"  {label}: {len(imgs)} files")
        total_images += len(imgs)

    if total_images == 0:
        raise ValueError(f"No image files found under {DATA_DIR}. Check file extensions and that images are inside label folders.")

    # Try the convenient loader; if it fails, fall back to building a dataset from file paths
    try:
        train_dataset = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR,
            validation_split=0.2,  # Use 20% of the data for validation
            subset="training",
            seed=42,               # Seed for reproducibility
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE
        )

        print("Creating validation dataset...")
        validation_dataset = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR,
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE
        )
    except Exception as e:
        print("image_dataset_from_directory failed, falling back to creating dataset from file paths. Error:", e)

        # Build file lists and numeric labels
        filepaths = []
        labels = []
        for label in label_dirs:
            p = os.path.join(DATA_DIR, label)
            for fname in sorted(os.listdir(p)):
                if os.path.splitext(fname)[1].lower() in image_exts:
                    filepaths.append(os.path.join(p, fname))
                    labels.append(label)

        class_names = sorted(list(dict.fromkeys(labels)))
        label_to_idx = {n: i for i, n in enumerate(class_names)}
        numeric_labels = [label_to_idx[l] for l in labels]

        # create tf.data.Dataset
        path_ds = tf.data.Dataset.from_tensor_slices(filepaths)
        label_ds = tf.data.Dataset.from_tensor_slices(numeric_labels)
        ds = tf.data.Dataset.zip((path_ds, label_ds))

        def _load_image(path, label):
            image = tf.io.read_file(path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.image.resize(image, IMG_SIZE)
            return image, label

        ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(1000)
        ds = ds.batch(BATCH_SIZE)
        val_size = int(0.2 * len(filepaths))
        validation_dataset = ds.take(val_size).prefetch(tf.data.AUTOTUNE)
        train_dataset = ds.skip(val_size).prefetch(tf.data.AUTOTUNE)

        print(f"Loaded dataset from filepaths with {len(filepaths)} images and {len(class_names)} classes.")

    # --- 3. Extract and Save Class Names ---
    class_names = train_dataset.class_names
    num_classes = len(class_names)
    print(f"\nFound {num_classes} classes. Saving class names...")
    # Save the class names for use during inference
    models_dir = os.path.join(repo_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    dump(class_names, os.path.join(models_dir, 'image_class_names.joblib'))

    # --- 4. Configure Dataset for Performance ---
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    print("\nData pipeline is ready!")

    # --- 5. Define Data Augmentation and Preprocessing Layers ---
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ], name="data_augmentation")

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    # --- 6. Create the Model using Transfer Learning ---
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    # --- 7. Chain the Layers Together ---
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    # --- 8. Compile the Model ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    model.summary()

    # --- 9. Train the Model ---
    EPOCHS = 10
    print("\nStarting model training...")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS
    )
    print("\nTraining finished!")

    # --- 10. Save the Trained Model ---
    model.save(os.path.join(models_dir, 'image_gesture_recognizer.keras'))
    print(f"\nModel saved to {os.path.join(models_dir, 'image_gesture_recognizer.keras')}")

    # --- 11. Visualize Performance ---
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

if __name__ == '__main__':
    # Ensure models directory exists relative to repo root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    models_dir = os.path.join(repo_root, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    train_model()