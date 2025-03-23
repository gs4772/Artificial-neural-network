# predict_magnitude.py
import numpy as np
from tensorflow.keras.models import load_model
model = load_model('earthquake_magnitude_model.h5')
X_test = np.load('x_test.npy')
predictions = model.predict(X_test)
print("Sample Predictions:", predictions[:5].flatten())
