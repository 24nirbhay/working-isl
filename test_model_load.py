import tensorflow as tf
from src.model_architecture import ReduceSumLayer
from joblib import load

le = load('models/model_20260112_130741/label_encoder.joblib')
print(f'Classes: {le.classes_}')

model = tf.keras.models.load_model(
    'models/model_20260112_130741/model.keras', 
    custom_objects={'ReduceSumLayer': ReduceSumLayer}
)
print(f'Model loaded successfully')
print(f'Model output shape: {model.output_shape}')
print(f'Model input shape: {model.input_shape}')

# Test a prediction
import numpy as np
test_input = np.random.randn(1, 30, 126).astype(np.float32)
pred = model.predict(test_input, verbose=0)
print(f'Prediction shape: {pred.shape}')
print(f'Prediction: {pred[0]}')
