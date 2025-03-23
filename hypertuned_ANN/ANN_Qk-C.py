import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
#loading the dataset
df = pd.read_csv('query (4).csv')
df.head()

#Drop the unnecessary columns
columns_to_drop = ['net','id','magType', 'nst', 'gap', 'dmin', 'rms', 'updated', 'place', 'type', 'horizontalError', 'depthError', 'magError', 'magNst', 'status', 'locationSource', 'magSource']
df = df.drop(columns = columns_to_drop)

df = df.dropna() #removing missing data to clean datasets.

#Extracting Date and Time Components from a Timestamp Column in Pandas
df['time'] = pd.to_datetime(df['time'])
df['year'] = df['time'].dt.year.astype('int16')
df['month'] = df['time'].dt.month.astype('int8')
df['Day'] = df['time'].dt.day.astype('int8')
df['hour'] = df['time'].dt.hour.astype('int8')
df = df.drop(columns = ['time'])

#Optimize data
df['latitude'] = df['latitude'].astype('float32')
df['longitude'] = df ['longitude'].astype('float32')
df['depth'] = df['depth'].astype('float32')
df['mag'] = df['mag'].astype('float32')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
print("Full DataFrame shape:", df.shape)
print("Columns in DataFrame:", df.columns.tolist())

#setting features and target
features = df[[	'latitude', 'longitude','depth','mag','year','month','Day','hour']]
target = df['mag']

print("Features shape (raw):", features.shape)
print("Target shape (raw):", target.shape)

#scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

#test_train split
x_train, x_test, y_train, y_test = train_test_split(features_scaled, target, test_size = 0.2, random_state = 42)

print("Features shape:", features_scaled.shape)
print("Target shape:", target.shape)

#Build the ANN model
model = Sequential()
model.add(Dense(64, input_dim = x_train.shape[1], activation = 'relu')) 
model.add(Dense(32, activation = 'relu'))
model.add (Dense(1, activation = 'linear'))
model.compile(loss = 'mean_squared_error', optimizer = 'Adam', metrics = ['mae'])

history = model.fit(x_train, y_train, epochs = 50, batch_size = 128, validation_split = 0.2, verbose = 1)

loss,mae = model.evaluate(x_test, y_test, verbose = 0)
print(f'Test Mean Absolute Error: {mae:.2f}')

from tensorflow.keras.optimizers import Adam
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0001), metrics=['mae'])

#hyper tuning the preprocessed model.
from keras_tuner import RandomSearch
def build_model(hp):
    model = Sequential()
    model.add(Dense(hp.Int('units1', 32, 128, step=32), input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(hp.Int('units2', 16, 64, step=16), activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=Adam(hp.Choice('lr', [1e-2, 1e-3, 1e-4])), metrics=['mae'])
    return model
tuner = RandomSearch(build_model, objective='val_mae', max_trials=5, executions_per_trial=3)
tuner.search(x_train, y_train, epochs=50, validation_split=0.2)
best_model = tuner.get_best_models(num_models=1)[0]

#K-Fold cross validation
from sklearn.model_selection import KFold
kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
mae_scores =[]

for train_idx, test_idx in kf.split(features_scaled):
    x_train, x_test = features_scaled[train_idx], features_scaled[test_idx]
    y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
    model = Sequential()
    model.add(Dense(64, input_dim=x_train.shape[1], activation = 'relu'))
    model.add(Dense(32, activation ='relu'))
    model.add(Dense(1, activation = 'linear'))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train, y_train, epochs = 50, batch_size = 128, verbose = 0)
loss, mae = model.evaluate(x_test, y_test, verbose = 0)
mae_scores.append(mae)
print(f'Mean MAE: {np.mean(mae_scores):.2f}, std:{np.std(mae_scores):.2f}')

#visualization
import matplotlib.pyplot as plt
history = model.fit(x_train, y_train, epochs = 50, batch_size = 128, validation_split = 0.2, verbose = 1)
plt.plot(history.history ['mae'], label = ['Train MAE'])
plt.plot(history.history ['val_mae'], label = ['Val MAE'])
plt.xlabel['epochs']
plt.ylabel['MAE']
plt.legend()
plt.show()

#saving...
model.save('earthquake_magnitude_model.h5')
import joblib
joblib.dump(scaler, 'scaler.pkl')

np.save('features_scaled.npy', features_scaled)
np.save('target.npy', target)
np.save('x_train.npy', x_train)
np.save('x_test.npy', x_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
