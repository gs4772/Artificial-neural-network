import numpy as np
from keras_tuner import RandomSearch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import shutil
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)  # Optional

# Load data
X_train = np.load('x_train.npy')
X_test = np.load('x_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Define model-building function
def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units1', min_value=32, max_value=128, step=32), 
                    input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(units=hp.Int('units2', min_value=16, max_value=64, step=16), 
                    activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=hp.Choice('lr', [1e-2, 1e-3, 1e-4])),
                  loss='mean_squared_error', metrics=['mae'])
    return model

# Clear previous tuning results (optional)
shutil.rmtree('untitled_project', ignore_errors=True)  # Remove old tuner data

# Set up tuner
tuner = RandomSearch(
    build_model,
    objective='val_mae',
    max_trials=5,
    executions_per_trial=3,
    directory='untitled_project',
    project_name='earthquake_tuning'
)

# Run tuning
tuner.search(X_train, y_train, epochs=50, validation_split=0.2, verbose=1)

# Get and evaluate best model
best_model = tuner.get_best_models(num_models=1)[0]
loss, mae = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Best Model Test MAE: {mae:.2f}")

# Save best model
best_model.save('best_earthquake_model.h5')
