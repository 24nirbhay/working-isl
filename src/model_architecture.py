import tensorflow as tf
from tensorflow.keras import layers


class ReduceSumLayer(layers.Layer):
    """Custom layer to sum across time dimension. Replaces Lambda for serialization."""
    
    def __init__(self, axis=1, **kwargs):
        super(ReduceSumLayer, self).__init__(**kwargs)
        self.axis = axis
    
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.axis)
    
    def get_config(self):
        config = super(ReduceSumLayer, self).get_config()
        config.update({'axis': self.axis})
        return config


def create_sign_language_model(time_steps, feature_dim, num_classes):
    """Create a sophisticated model architecture for sign language recognition.
    
    This architecture is designed to learn temporal patterns in hand gestures:
    - Masking layer handles variable-length sequences (padded data)
    - Bidirectional LSTMs learn forward and backward temporal patterns
    - Attention mechanism focuses on important frames
    - Dense layers map learned patterns to gesture classes
    
    Args:
        time_steps: Number of time steps in the sequence (30 frames)
        feature_dim: Number of features per time step (126 for both hands)
        num_classes: Number of output classes (gestures)
        
    Returns:
        Compiled Keras model
    """
    # Input and masking for variable-length sequences
    inputs = layers.Input(shape=(time_steps, feature_dim))
    x = layers.Masking(mask_value=0.)(inputs)
    
    # Batch normalization on input
    x = layers.BatchNormalization()(x)
    
    # First Bidirectional LSTM layer - learns basic temporal patterns
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)
    )(x)
    x = layers.BatchNormalization()(x)
    
    # Second Bidirectional LSTM layer - learns complex gesture patterns  
    lstm_out = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)
    )(x)
    x = layers.BatchNormalization()(lstm_out)
    
    # Attention mechanism - focuses on important frames in the gesture
    attention = layers.Dense(1, activation='tanh')(x)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(128)(attention)  # 64*2 from Bidirectional
    attention = layers.Permute([2, 1])(attention)
    
    # Apply attention weights
    x = layers.Multiply()([lstm_out, attention])
    x = ReduceSumLayer(axis=1)(x)
    
    # Dense layers for classification
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Compile with learning rate schedule
    initial_learning_rate = 0.0005
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=500,
        decay_rate=0.95,
        staircase=True)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_accuracy')]
    )
    
    return model