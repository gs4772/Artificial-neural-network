## Files

- **`query (4).csv`**:
  - A CSV file containing a processed subset or version of the earthquake dataset used for training and testing the ANN model. It includes features like latitude, longitude, depth, and time (year, month, day, hour), along with the target variable (magnitude).

- **`target.npy`**:
  - A NumPy array file storing the full set of earthquake magnitudes (target variable) before splitting into training and testing sets. Represents the `mag` column from the dataset.

- **`x_train.npy`**:
  - A NumPy array file containing the training features for the ANN model. Includes scaled values of latitude, longitude, depth, year, month, day, and hour for the training subset.

- **`x_test.npy`**:
  - A NumPy array file containing the testing features for the ANN model. Includes scaled values of latitude, longitude, depth, year, month, day, and hour for the testing subset.

- **`y_train.npy`**:
  - A NumPy array file storing the training target values (earthquake magnitudes) corresponding to `x_train.npy`.

- **`y_test.npy`**:
  - A NumPy array file storing the testing target values (earthquake magnitudes) corresponding to `x_test.npy`.
