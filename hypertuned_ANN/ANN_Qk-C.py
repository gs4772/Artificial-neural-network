{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "14512aa9-d677-4f21-aebe-0742a77ff55c",
   "metadata": {},
   "source": [
    "#ANN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c801e801-5bfe-4c82-9987-2e0ccef77262",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense, Dropout\n",
    "from tensorflow.keras.utils import to_categorical"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "d2f2dcc0-9de0-4cfc-98dd-7de6350a402d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>time</th>\n",
       "      <th>latitude</th>\n",
       "      <th>longitude</th>\n",
       "      <th>depth</th>\n",
       "      <th>mag</th>\n",
       "      <th>magType</th>\n",
       "      <th>nst</th>\n",
       "      <th>gap</th>\n",
       "      <th>dmin</th>\n",
       "      <th>rms</th>\n",
       "      <th>...</th>\n",
       "      <th>updated</th>\n",
       "      <th>place</th>\n",
       "      <th>type</th>\n",
       "      <th>horizontalError</th>\n",
       "      <th>depthError</th>\n",
       "      <th>magError</th>\n",
       "      <th>magNst</th>\n",
       "      <th>status</th>\n",
       "      <th>locationSource</th>\n",
       "      <th>magSource</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2025-02-21T10:11:47.799Z</td>\n",
       "      <td>28.5633</td>\n",
       "      <td>87.5409</td>\n",
       "      <td>10.000</td>\n",
       "      <td>4.9</td>\n",
       "      <td>mb</td>\n",
       "      <td>68.0</td>\n",
       "      <td>67.0</td>\n",
       "      <td>2.137</td>\n",
       "      <td>0.97</td>\n",
       "      <td>...</td>\n",
       "      <td>2025-02-27T21:38:11.659Z</td>\n",
       "      <td>98 km NE of Lobuche, Nepal</td>\n",
       "      <td>earthquake</td>\n",
       "      <td>7.89</td>\n",
       "      <td>1.862</td>\n",
       "      <td>0.054</td>\n",
       "      <td>107.0</td>\n",
       "      <td>reviewed</td>\n",
       "      <td>us</td>\n",
       "      <td>us</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2025-02-20T06:02:58.563Z</td>\n",
       "      <td>25.7977</td>\n",
       "      <td>90.6738</td>\n",
       "      <td>10.000</td>\n",
       "      <td>4.3</td>\n",
       "      <td>mb</td>\n",
       "      <td>34.0</td>\n",
       "      <td>154.0</td>\n",
       "      <td>3.911</td>\n",
       "      <td>0.99</td>\n",
       "      <td>...</td>\n",
       "      <td>2025-03-08T12:23:12.040Z</td>\n",
       "      <td>42 km S of Goālpāra, India</td>\n",
       "      <td>earthquake</td>\n",
       "      <td>7.77</td>\n",
       "      <td>1.852</td>\n",
       "      <td>0.106</td>\n",
       "      <td>25.0</td>\n",
       "      <td>reviewed</td>\n",
       "      <td>us</td>\n",
       "      <td>us</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2025-02-17T00:06:52.349Z</td>\n",
       "      <td>28.5887</td>\n",
       "      <td>77.1025</td>\n",
       "      <td>10.000</td>\n",
       "      <td>4.2</td>\n",
       "      <td>mb</td>\n",
       "      <td>32.0</td>\n",
       "      <td>204.0</td>\n",
       "      <td>7.258</td>\n",
       "      <td>0.61</td>\n",
       "      <td>...</td>\n",
       "      <td>2025-03-11T22:13:27.498Z</td>\n",
       "      <td>10 km SSE of Nāngloi Jāt, India</td>\n",
       "      <td>earthquake</td>\n",
       "      <td>7.96</td>\n",
       "      <td>1.946</td>\n",
       "      <td>0.128</td>\n",
       "      <td>17.0</td>\n",
       "      <td>reviewed</td>\n",
       "      <td>us</td>\n",
       "      <td>us</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2025-02-16T03:28:58.344Z</td>\n",
       "      <td>29.0272</td>\n",
       "      <td>87.5405</td>\n",
       "      <td>10.000</td>\n",
       "      <td>4.8</td>\n",
       "      <td>mb</td>\n",
       "      <td>79.0</td>\n",
       "      <td>94.0</td>\n",
       "      <td>2.337</td>\n",
       "      <td>0.75</td>\n",
       "      <td>...</td>\n",
       "      <td>2025-03-07T02:46:18.532Z</td>\n",
       "      <td>132 km W of Rikaze, China</td>\n",
       "      <td>earthquake</td>\n",
       "      <td>8.91</td>\n",
       "      <td>1.865</td>\n",
       "      <td>0.058</td>\n",
       "      <td>94.0</td>\n",
       "      <td>reviewed</td>\n",
       "      <td>us</td>\n",
       "      <td>us</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2025-02-15T17:48:18.144Z</td>\n",
       "      <td>33.4067</td>\n",
       "      <td>73.0563</td>\n",
       "      <td>35.848</td>\n",
       "      <td>4.7</td>\n",
       "      <td>mb</td>\n",
       "      <td>139.0</td>\n",
       "      <td>42.0</td>\n",
       "      <td>3.441</td>\n",
       "      <td>0.64</td>\n",
       "      <td>...</td>\n",
       "      <td>2025-03-05T18:06:52.476Z</td>\n",
       "      <td>21 km S of Rawalpindi, Pakistan</td>\n",
       "      <td>earthquake</td>\n",
       "      <td>8.23</td>\n",
       "      <td>4.417</td>\n",
       "      <td>0.046</td>\n",
       "      <td>149.0</td>\n",
       "      <td>reviewed</td>\n",
       "      <td>us</td>\n",
       "      <td>us</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 22 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                       time  latitude  longitude   depth  mag magType    nst  \\\n",
       "0  2025-02-21T10:11:47.799Z   28.5633    87.5409  10.000  4.9      mb   68.0   \n",
       "1  2025-02-20T06:02:58.563Z   25.7977    90.6738  10.000  4.3      mb   34.0   \n",
       "2  2025-02-17T00:06:52.349Z   28.5887    77.1025  10.000  4.2      mb   32.0   \n",
       "3  2025-02-16T03:28:58.344Z   29.0272    87.5405  10.000  4.8      mb   79.0   \n",
       "4  2025-02-15T17:48:18.144Z   33.4067    73.0563  35.848  4.7      mb  139.0   \n",
       "\n",
       "     gap   dmin   rms  ...                   updated  \\\n",
       "0   67.0  2.137  0.97  ...  2025-02-27T21:38:11.659Z   \n",
       "1  154.0  3.911  0.99  ...  2025-03-08T12:23:12.040Z   \n",
       "2  204.0  7.258  0.61  ...  2025-03-11T22:13:27.498Z   \n",
       "3   94.0  2.337  0.75  ...  2025-03-07T02:46:18.532Z   \n",
       "4   42.0  3.441  0.64  ...  2025-03-05T18:06:52.476Z   \n",
       "\n",
       "                             place        type horizontalError depthError  \\\n",
       "0       98 km NE of Lobuche, Nepal  earthquake            7.89      1.862   \n",
       "1       42 km S of Goālpāra, India  earthquake            7.77      1.852   \n",
       "2  10 km SSE of Nāngloi Jāt, India  earthquake            7.96      1.946   \n",
       "3        132 km W of Rikaze, China  earthquake            8.91      1.865   \n",
       "4  21 km S of Rawalpindi, Pakistan  earthquake            8.23      4.417   \n",
       "\n",
       "   magError  magNst    status  locationSource magSource  \n",
       "0     0.054   107.0  reviewed              us        us  \n",
       "1     0.106    25.0  reviewed              us        us  \n",
       "2     0.128    17.0  reviewed              us        us  \n",
       "3     0.058    94.0  reviewed              us        us  \n",
       "4     0.046   149.0  reviewed              us        us  \n",
       "\n",
       "[5 rows x 22 columns]"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv('query (4).csv')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "b079924a-f241-46fa-bd13-372c6145c4fa",
   "metadata": {},
   "outputs": [],
   "source": [
    "columns_to_drop = ['net','id','magType', 'nst', 'gap', 'dmin', 'rms', 'updated', 'place', 'type', 'horizontalError', 'depthError', 'magError', 'magNst', 'status', 'locationSource', 'magSource']\n",
    "df = df.drop(columns = columns_to_drop)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "3a5d072c-a268-46b5-8477-e87d85cb6dc8",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.dropna()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "0c6ab1ed-d094-4260-948b-da2b1a2845af",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['time'] = pd.to_datetime(df['time'])\n",
    "df['year'] = df['time'].dt.year.astype('int16')\n",
    "df['month'] = df['time'].dt.month.astype('int8')\n",
    "df['Day'] = df['time'].dt.day.astype('int8')\n",
    "df['hour'] = df['time'].dt.hour.astype('int8')\n",
    "df = df.drop(columns = ['time'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "9ac65d72-c224-462b-b2a8-c3fea0a9c245",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>latitude</th>\n",
       "      <th>longitude</th>\n",
       "      <th>depth</th>\n",
       "      <th>mag</th>\n",
       "      <th>year</th>\n",
       "      <th>month</th>\n",
       "      <th>Day</th>\n",
       "      <th>hour</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>28.5633</td>\n",
       "      <td>87.5409</td>\n",
       "      <td>10.000</td>\n",
       "      <td>4.9</td>\n",
       "      <td>2025</td>\n",
       "      <td>2</td>\n",
       "      <td>21</td>\n",
       "      <td>10</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>25.7977</td>\n",
       "      <td>90.6738</td>\n",
       "      <td>10.000</td>\n",
       "      <td>4.3</td>\n",
       "      <td>2025</td>\n",
       "      <td>2</td>\n",
       "      <td>20</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>28.5887</td>\n",
       "      <td>77.1025</td>\n",
       "      <td>10.000</td>\n",
       "      <td>4.2</td>\n",
       "      <td>2025</td>\n",
       "      <td>2</td>\n",
       "      <td>17</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>29.0272</td>\n",
       "      <td>87.5405</td>\n",
       "      <td>10.000</td>\n",
       "      <td>4.8</td>\n",
       "      <td>2025</td>\n",
       "      <td>2</td>\n",
       "      <td>16</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>33.4067</td>\n",
       "      <td>73.0563</td>\n",
       "      <td>35.848</td>\n",
       "      <td>4.7</td>\n",
       "      <td>2025</td>\n",
       "      <td>2</td>\n",
       "      <td>15</td>\n",
       "      <td>17</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   latitude  longitude   depth  mag  year  month  Day  hour\n",
       "0   28.5633    87.5409  10.000  4.9  2025      2   21    10\n",
       "1   25.7977    90.6738  10.000  4.3  2025      2   20     6\n",
       "2   28.5887    77.1025  10.000  4.2  2025      2   17     0\n",
       "3   29.0272    87.5405  10.000  4.8  2025      2   16     3\n",
       "4   33.4067    73.0563  35.848  4.7  2025      2   15    17"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "67884478-503d-491e-9920-039097e4c6b0",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Optimize data\n",
    "df['latitude'] = df['latitude'].astype('float32')\n",
    "df['longitude'] = df ['longitude'].astype('float32')\n",
    "df['depth'] = df['depth'].astype('float32')\n",
    "df['mag'] = df['mag'].astype('float32')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "18cf995c-b420-493b-9f91-517c6f256ded",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "adce58cb-53e8-4d2d-89b5-6406e54f9986",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Full DataFrame shape: (6940, 8)\n",
      "Columns in DataFrame: ['latitude', 'longitude', 'depth', 'mag', 'year', 'month', 'Day', 'hour']\n"
     ]
    }
   ],
   "source": [
    "print(\"Full DataFrame shape:\", df.shape)\n",
    "print(\"Columns in DataFrame:\", df.columns.tolist())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "5478ca3d-37b9-4870-a6e7-ae9362d8c52b",
   "metadata": {},
   "outputs": [],
   "source": [
    "#setting features and target\n",
    "features = df[[\t'latitude', 'longitude','depth','mag','year','month','Day','hour']]\n",
    "target = df['mag']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "3ecefcbe-8457-4bdd-ad15-52e429479b2c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Features shape (raw): (6940, 8)\n",
      "Target shape (raw): (6940,)\n"
     ]
    }
   ],
   "source": [
    "print(\"Features shape (raw):\", features.shape)\n",
    "print(\"Target shape (raw):\", target.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "72fcb479-ed8a-4f4b-ba06-81916bd65701",
   "metadata": {},
   "outputs": [],
   "source": [
    "#scale the features\n",
    "scaler = StandardScaler()\n",
    "features_scaled = scaler.fit_transform(features)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "79d2cff3-faae-4052-8c04-0ed8163ea479",
   "metadata": {},
   "outputs": [],
   "source": [
    "#test_train split\n",
    "x_train, x_test, y_train, y_test = train_test_split(features_scaled, target, test_size = 0.2, random_state = 42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "259bd703-b4b4-4c06-8205-87e8928abc6f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Features shape: (6940, 8)\n",
      "Target shape: (6940,)\n"
     ]
    }
   ],
   "source": [
    "print(\"Features shape:\", features_scaled.shape)\n",
    "print(\"Target shape:\", target.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "691d99e6-ccdf-42d8-8ee2-f523378d2164",
   "metadata": {},
   "source": [
    "Build ANN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "a8870023-4564-4722-be8e-5e805d68258d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Guru G\\AppData\\Roaming\\Python\\Python312\\site-packages\\keras\\src\\layers\\core\\dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
      "  super().__init__(activity_regularizer=activity_regularizer, **kwargs)\n"
     ]
    }
   ],
   "source": [
    "model = Sequential()\n",
    "model.add(Dense(64, input_dim = x_train.shape[1], activation = 'relu')) \n",
    "model.add(Dense(32, activation = 'relu'))\n",
    "model.add (Dense(1, activation = 'linear'))\n",
    "model.compile(loss = 'mean_squared_error', optimizer = 'Adam', metrics = ['mae'])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "429dac42-62eb-4eb6-84d1-4d3c729af9c5",
   "metadata": {},
   "source": [
    "Train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "6e58ac87-4efa-490c-b2ef-f5bab7200bfb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m2s\u001b[0m 12ms/step - loss: 18.8904 - mae: 4.2625 - val_loss: 8.2673 - val_mae: 2.7482\n",
      "Epoch 2/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 5.7638 - mae: 2.1997 - val_loss: 1.4278 - val_mae: 0.9823\n",
      "Epoch 3/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 1.2718 - mae: 0.9129 - val_loss: 0.7626 - val_mae: 0.6899\n",
      "Epoch 4/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.7282 - mae: 0.6745 - val_loss: 0.6472 - val_mae: 0.6352\n",
      "Epoch 5/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.6133 - mae: 0.6171 - val_loss: 0.5858 - val_mae: 0.6015\n",
      "Epoch 6/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.5738 - mae: 0.6038 - val_loss: 0.5302 - val_mae: 0.5727\n",
      "Epoch 7/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.4852 - mae: 0.5572 - val_loss: 0.4868 - val_mae: 0.5492\n",
      "Epoch 8/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.4762 - mae: 0.5457 - val_loss: 0.4511 - val_mae: 0.5289\n",
      "Epoch 9/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.4414 - mae: 0.5288 - val_loss: 0.4170 - val_mae: 0.5080\n",
      "Epoch 10/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - loss: 0.3950 - mae: 0.5004 - val_loss: 0.3880 - val_mae: 0.4905\n",
      "Epoch 11/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.3645 - mae: 0.4809 - val_loss: 0.3588 - val_mae: 0.4719\n",
      "Epoch 12/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.3402 - mae: 0.4594 - val_loss: 0.3283 - val_mae: 0.4514\n",
      "Epoch 13/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.3083 - mae: 0.4401 - val_loss: 0.3026 - val_mae: 0.4333\n",
      "Epoch 14/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - loss: 0.2937 - mae: 0.4315 - val_loss: 0.2755 - val_mae: 0.4145\n",
      "Epoch 15/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.2568 - mae: 0.4051 - val_loss: 0.2518 - val_mae: 0.3959\n",
      "Epoch 16/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.2367 - mae: 0.3884 - val_loss: 0.2275 - val_mae: 0.3772\n",
      "Epoch 17/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - loss: 0.2015 - mae: 0.3547 - val_loss: 0.2016 - val_mae: 0.3557\n",
      "Epoch 18/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.1823 - mae: 0.3416 - val_loss: 0.1769 - val_mae: 0.3334\n",
      "Epoch 19/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.1661 - mae: 0.3227 - val_loss: 0.1550 - val_mae: 0.3127\n",
      "Epoch 20/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.1445 - mae: 0.3045 - val_loss: 0.1327 - val_mae: 0.2898\n",
      "Epoch 21/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.1264 - mae: 0.2835 - val_loss: 0.1123 - val_mae: 0.2673\n",
      "Epoch 22/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.1028 - mae: 0.2571 - val_loss: 0.0957 - val_mae: 0.2453\n",
      "Epoch 23/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - loss: 0.0894 - mae: 0.2393 - val_loss: 0.0780 - val_mae: 0.2223\n",
      "Epoch 24/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - loss: 0.0697 - mae: 0.2104 - val_loss: 0.0623 - val_mae: 0.1986\n",
      "Epoch 25/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0569 - mae: 0.1900 - val_loss: 0.0512 - val_mae: 0.1802\n",
      "Epoch 26/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0467 - mae: 0.1733 - val_loss: 0.0403 - val_mae: 0.1589\n",
      "Epoch 27/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0368 - mae: 0.1515 - val_loss: 0.0324 - val_mae: 0.1417\n",
      "Epoch 28/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0294 - mae: 0.1358 - val_loss: 0.0263 - val_mae: 0.1268\n",
      "Epoch 29/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - loss: 0.0235 - mae: 0.1205 - val_loss: 0.0214 - val_mae: 0.1147\n",
      "Epoch 30/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0192 - mae: 0.1083 - val_loss: 0.0183 - val_mae: 0.1072\n",
      "Epoch 31/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0166 - mae: 0.1001 - val_loss: 0.0152 - val_mae: 0.0956\n",
      "Epoch 32/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0133 - mae: 0.0894 - val_loss: 0.0137 - val_mae: 0.0901\n",
      "Epoch 33/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0119 - mae: 0.0851 - val_loss: 0.0120 - val_mae: 0.0837\n",
      "Epoch 34/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0101 - mae: 0.0778 - val_loss: 0.0105 - val_mae: 0.0799\n",
      "Epoch 35/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0087 - mae: 0.0720 - val_loss: 0.0092 - val_mae: 0.0740\n",
      "Epoch 36/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - loss: 0.0079 - mae: 0.0692 - val_loss: 0.0084 - val_mae: 0.0700\n",
      "Epoch 37/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0072 - mae: 0.0657 - val_loss: 0.0077 - val_mae: 0.0679\n",
      "Epoch 38/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0067 - mae: 0.0630 - val_loss: 0.0071 - val_mae: 0.0655\n",
      "Epoch 39/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0062 - mae: 0.0606 - val_loss: 0.0067 - val_mae: 0.0629\n",
      "Epoch 40/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0056 - mae: 0.0584 - val_loss: 0.0061 - val_mae: 0.0601\n",
      "Epoch 41/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0052 - mae: 0.0550 - val_loss: 0.0057 - val_mae: 0.0581\n",
      "Epoch 42/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - loss: 0.0050 - mae: 0.0547 - val_loss: 0.0053 - val_mae: 0.0566\n",
      "Epoch 43/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - loss: 0.0044 - mae: 0.0525 - val_loss: 0.0050 - val_mae: 0.0550\n",
      "Epoch 44/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0044 - mae: 0.0517 - val_loss: 0.0047 - val_mae: 0.0529\n",
      "Epoch 45/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0041 - mae: 0.0500 - val_loss: 0.0045 - val_mae: 0.0522\n",
      "Epoch 46/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0038 - mae: 0.0483 - val_loss: 0.0042 - val_mae: 0.0509\n",
      "Epoch 47/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0036 - mae: 0.0472 - val_loss: 0.0041 - val_mae: 0.0497\n",
      "Epoch 48/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0034 - mae: 0.0459 - val_loss: 0.0038 - val_mae: 0.0482\n",
      "Epoch 49/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0034 - mae: 0.0454 - val_loss: 0.0037 - val_mae: 0.0468\n",
      "Epoch 50/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0032 - mae: 0.0442 - val_loss: 0.0034 - val_mae: 0.0459\n"
     ]
    }
   ],
   "source": [
    "history = model.fit(x_train, y_train, epochs = 50, batch_size = 128, validation_split = 0.2, verbose = 1)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f4a7fa7a-187f-4dfa-bbf0-0c22ebc05a13",
   "metadata": {},
   "source": [
    "Evaluate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "7823812a-3146-4997-b322-f88512eaf65d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test Mean Absolute Error: 0.05\n"
     ]
    }
   ],
   "source": [
    "loss,mae = model.evaluate(x_test, y_test, verbose = 0)\n",
    "print(f'Test Mean Absolute Error: {mae:.2f}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "53f37848-4a07-4c00-a41e-30bb8bfffe62",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.optimizers import Adam\n",
    "model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0001), metrics=['mae'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "2b0ebaf4-e66a-476b-aec3-8278b50c0bb3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Reloading Tuner from .\\untitled_project\\tuner0.json\n",
      "WARNING:tensorflow:From C:\\Users\\Guru G\\AppData\\Roaming\\Python\\Python312\\site-packages\\keras\\src\\backend\\common\\global_state.py:82: The name tf.reset_default_graph is deprecated. Please use tf.compat.v1.reset_default_graph instead.\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From C:\\Users\\Guru G\\AppData\\Roaming\\Python\\Python312\\site-packages\\keras\\src\\backend\\common\\global_state.py:82: The name tf.reset_default_graph is deprecated. Please use tf.compat.v1.reset_default_graph instead.\n",
      "\n",
      "C:\\Users\\Guru G\\AppData\\Roaming\\Python\\Python312\\site-packages\\keras\\src\\saving\\saving_lib.py:757: UserWarning: Skipping variable loading for optimizer 'adam', because it has 2 variables whereas the saved optimizer has 14 variables. \n",
      "  saveable.load_own_variables(weights_store.get(inner_path))\n"
     ]
    }
   ],
   "source": [
    "from keras_tuner import RandomSearch\n",
    "def build_model(hp):\n",
    "    model = Sequential()\n",
    "    model.add(Dense(hp.Int('units1', 32, 128, step=32), input_dim=x_train.shape[1], activation='relu'))\n",
    "    model.add(Dense(hp.Int('units2', 16, 64, step=16), activation='relu'))\n",
    "    model.add(Dense(1, activation='linear'))\n",
    "    model.compile(loss='mean_squared_error', optimizer=Adam(hp.Choice('lr', [1e-2, 1e-3, 1e-4])), metrics=['mae'])\n",
    "    return model\n",
    "tuner = RandomSearch(build_model, objective='val_mae', max_trials=5, executions_per_trial=3)\n",
    "tuner.search(x_train, y_train, epochs=50, validation_split=0.2)\n",
    "best_model = tuner.get_best_models(num_models=1)[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "3b21f3ce-cbe3-40f1-a59c-b4fc48978cc7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean MAE: 0.03, std:0.00\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import KFold\n",
    "kf = KFold(n_splits = 5, shuffle = True, random_state = 42)\n",
    "mae_scores =[]\n",
    "\n",
    "for train_idx, test_idx in kf.split(features_scaled):\n",
    "    x_train, x_test = features_scaled[train_idx], features_scaled[test_idx]\n",
    "    y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]\n",
    "    model = Sequential()\n",
    "    model.add(Dense(64, input_dim=x_train.shape[1], activation = 'relu'))\n",
    "    model.add(Dense(32, activation ='relu'))\n",
    "    model.add(Dense(1, activation = 'linear'))\n",
    "    model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['mae'])\n",
    "model.fit(x_train, y_train, epochs = 50, batch_size = 128, verbose = 0)\n",
    "loss, mae = model.evaluate(x_test, y_test, verbose = 0)\n",
    "mae_scores.append(mae)\n",
    "print(f'Mean MAE: {np.mean(mae_scores):.2f}, std:{np.std(mae_scores):.2f}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "93339788-2ae4-4115-a53f-795844e42c91",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 8ms/step - loss: 0.0015 - mae: 0.0307 - val_loss: 0.0019 - val_mae: 0.0335\n",
      "Epoch 2/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.0013 - mae: 0.0288 - val_loss: 0.0020 - val_mae: 0.0345\n",
      "Epoch 3/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.0013 - mae: 0.0282 - val_loss: 0.0020 - val_mae: 0.0351\n",
      "Epoch 4/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.0013 - mae: 0.0282 - val_loss: 0.0019 - val_mae: 0.0335\n",
      "Epoch 5/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.0012 - mae: 0.0276 - val_loss: 0.0018 - val_mae: 0.0329\n",
      "Epoch 6/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.0012 - mae: 0.0271 - val_loss: 0.0019 - val_mae: 0.0338\n",
      "Epoch 7/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.0011 - mae: 0.0264 - val_loss: 0.0024 - val_mae: 0.0388\n",
      "Epoch 8/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.0011 - mae: 0.0264 - val_loss: 0.0018 - val_mae: 0.0324\n",
      "Epoch 9/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.0010 - mae: 0.0252 - val_loss: 0.0018 - val_mae: 0.0327\n",
      "Epoch 10/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.0010 - mae: 0.0250 - val_loss: 0.0020 - val_mae: 0.0356\n",
      "Epoch 11/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.0010 - mae: 0.0249 - val_loss: 0.0017 - val_mae: 0.0318\n",
      "Epoch 12/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 9.8085e-04 - mae: 0.0245 - val_loss: 0.0018 - val_mae: 0.0324\n",
      "Epoch 13/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 9.4037e-04 - mae: 0.0238 - val_loss: 0.0018 - val_mae: 0.0328\n",
      "Epoch 14/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 8.7404e-04 - mae: 0.0231 - val_loss: 0.0017 - val_mae: 0.0324\n",
      "Epoch 15/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 8.4478e-04 - mae: 0.0226 - val_loss: 0.0019 - val_mae: 0.0340\n",
      "Epoch 16/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 8.7299e-04 - mae: 0.0230 - val_loss: 0.0020 - val_mae: 0.0358\n",
      "Epoch 17/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 7.8246e-04 - mae: 0.0220 - val_loss: 0.0018 - val_mae: 0.0334\n",
      "Epoch 18/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - loss: 7.6091e-04 - mae: 0.0216 - val_loss: 0.0018 - val_mae: 0.0326\n",
      "Epoch 19/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 7.5065e-04 - mae: 0.0212 - val_loss: 0.0017 - val_mae: 0.0321\n",
      "Epoch 20/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 7.2644e-04 - mae: 0.0210 - val_loss: 0.0017 - val_mae: 0.0319\n",
      "Epoch 21/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 6.9205e-04 - mae: 0.0205 - val_loss: 0.0018 - val_mae: 0.0336\n",
      "Epoch 22/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 6.9050e-04 - mae: 0.0206 - val_loss: 0.0019 - val_mae: 0.0341\n",
      "Epoch 23/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 7.2575e-04 - mae: 0.0211 - val_loss: 0.0015 - val_mae: 0.0306\n",
      "Epoch 24/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 6.5012e-04 - mae: 0.0198 - val_loss: 0.0016 - val_mae: 0.0305\n",
      "Epoch 25/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 6.4149e-04 - mae: 0.0196 - val_loss: 0.0017 - val_mae: 0.0330\n",
      "Epoch 26/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 6.0112e-04 - mae: 0.0190 - val_loss: 0.0015 - val_mae: 0.0303\n",
      "Epoch 27/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 5.7689e-04 - mae: 0.0188 - val_loss: 0.0015 - val_mae: 0.0301\n",
      "Epoch 28/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - loss: 5.8505e-04 - mae: 0.0188 - val_loss: 0.0015 - val_mae: 0.0305\n",
      "Epoch 29/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 5.6455e-04 - mae: 0.0186 - val_loss: 0.0015 - val_mae: 0.0307\n",
      "Epoch 30/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 5.5004e-04 - mae: 0.0183 - val_loss: 0.0015 - val_mae: 0.0307\n",
      "Epoch 31/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - loss: 5.1028e-04 - mae: 0.0176 - val_loss: 0.0015 - val_mae: 0.0299\n",
      "Epoch 32/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - loss: 4.9809e-04 - mae: 0.0174 - val_loss: 0.0015 - val_mae: 0.0302\n",
      "Epoch 33/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - loss: 4.8105e-04 - mae: 0.0171 - val_loss: 0.0015 - val_mae: 0.0298\n",
      "Epoch 34/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 4.6697e-04 - mae: 0.0169 - val_loss: 0.0015 - val_mae: 0.0298\n",
      "Epoch 35/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 4.6577e-04 - mae: 0.0171 - val_loss: 0.0014 - val_mae: 0.0297\n",
      "Epoch 36/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 4.4190e-04 - mae: 0.0163 - val_loss: 0.0015 - val_mae: 0.0305\n",
      "Epoch 37/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 4.5520e-04 - mae: 0.0167 - val_loss: 0.0015 - val_mae: 0.0305\n",
      "Epoch 38/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 4.3707e-04 - mae: 0.0164 - val_loss: 0.0014 - val_mae: 0.0286\n",
      "Epoch 39/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 3.9711e-04 - mae: 0.0156 - val_loss: 0.0015 - val_mae: 0.0292\n",
      "Epoch 40/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 4.0214e-04 - mae: 0.0158 - val_loss: 0.0014 - val_mae: 0.0292\n",
      "Epoch 41/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 4.0736e-04 - mae: 0.0157 - val_loss: 0.0015 - val_mae: 0.0291\n",
      "Epoch 42/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 3.9178e-04 - mae: 0.0153 - val_loss: 0.0014 - val_mae: 0.0284\n",
      "Epoch 43/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - loss: 3.6976e-04 - mae: 0.0150 - val_loss: 0.0014 - val_mae: 0.0286\n",
      "Epoch 44/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 3.9126e-04 - mae: 0.0154 - val_loss: 0.0015 - val_mae: 0.0286\n",
      "Epoch 45/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 3.4010e-04 - mae: 0.0143 - val_loss: 0.0014 - val_mae: 0.0278\n",
      "Epoch 46/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - loss: 3.4283e-04 - mae: 0.0145 - val_loss: 0.0015 - val_mae: 0.0297\n",
      "Epoch 47/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 3.4967e-04 - mae: 0.0147 - val_loss: 0.0015 - val_mae: 0.0295\n",
      "Epoch 48/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 3.4442e-04 - mae: 0.0144 - val_loss: 0.0016 - val_mae: 0.0290\n",
      "Epoch 49/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 3.1042e-04 - mae: 0.0138 - val_loss: 0.0015 - val_mae: 0.0304\n",
      "Epoch 50/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 3.3460e-04 - mae: 0.0144 - val_loss: 0.0014 - val_mae: 0.0277\n"
     ]
    },
    {
     "ename": "TypeError",
     "evalue": "'function' object is not subscriptable",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[24], line 5\u001b[0m\n\u001b[0;32m      3\u001b[0m plt\u001b[38;5;241m.\u001b[39mplot(history\u001b[38;5;241m.\u001b[39mhistory [\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mmae\u001b[39m\u001b[38;5;124m'\u001b[39m], label \u001b[38;5;241m=\u001b[39m [\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mTrain MAE\u001b[39m\u001b[38;5;124m'\u001b[39m])\n\u001b[0;32m      4\u001b[0m plt\u001b[38;5;241m.\u001b[39mplot(history\u001b[38;5;241m.\u001b[39mhistory [\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mval_mae\u001b[39m\u001b[38;5;124m'\u001b[39m], label \u001b[38;5;241m=\u001b[39m [\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mVal MAE\u001b[39m\u001b[38;5;124m'\u001b[39m])\n\u001b[1;32m----> 5\u001b[0m plt\u001b[38;5;241m.\u001b[39mxlabel[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mepochs\u001b[39m\u001b[38;5;124m'\u001b[39m]\n\u001b[0;32m      6\u001b[0m plt\u001b[38;5;241m.\u001b[39mylabel[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mMAE\u001b[39m\u001b[38;5;124m'\u001b[39m]\n\u001b[0;32m      7\u001b[0m plt\u001b[38;5;241m.\u001b[39mlegend()\n",
      "\u001b[1;31mTypeError\u001b[0m: 'function' object is not subscriptable"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjUAAAGhCAYAAACZCkVQAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8hTgPZAAAACXBIWXMAAA9hAAAPYQGoP6dpAABlJElEQVR4nO3dd3hUZcLG4d+kk0oJpEAIoSi9JUqNWDAKroqiImtbO3Zg3bXtfrrqyq7ruq7SVsWCBVgFXVRUUGlCUEpCDUgPJSEESIGQfr4/XpIQM0AmmcmkPPd1zTUnZ86c886hzJO32izLshARERFp4DzcXQARERERZ1CoERERkUZBoUZEREQaBYUaERERaRQUakRERKRRUKgRERGRRkGhRkRERBoFhRoRERFpFBRqREREpFFQqBEREZFGoUahZurUqcTExODn50dsbCzLly8/6/FLly4lNjYWPz8/OnbsyPTp08947OzZs7HZbIwaNarW1xUREZGmw+FQM2fOHMaPH88zzzxDUlIS8fHxjBgxgtTUVLvH7969m5EjRxIfH09SUhJPP/00jz76KHPnzq1y7N69e3n88ceJj4+v9XVFRESkabE5uqDlgAED6N+/P9OmTSvf161bN0aNGsWkSZOqHP/EE08wf/58UlJSyveNGzeO9evXk5iYWL6vpKSEYcOGceedd7J8+XKysrL4/PPPa3xde0pLSzl48CBBQUHYbDZHPraIiIi4iWVZ5ObmEhkZiYfHmetjvBw5aWFhIWvXruXJJ5+stD8hIYGVK1fafU9iYiIJCQmV9l1xxRXMmDGDoqIivL29AXj++edp3bo1d999d5VmpZpcF6CgoICCgoLynw8cOED37t3P/UFFRESk3tm3bx/t2rU74+sOhZrMzExKSkoICwurtD8sLIz09HS770lPT7d7fHFxMZmZmURERLBixQpmzJhBcnKy064LMGnSJP7yl79U2b9v3z6Cg4PP+D4RERGpP3JycoiKiiIoKOisxzkUasr8uunGsqyzNufYO75sf25uLrfeeitvvfUWoaGhTr3uU089xcSJE8t/LrspwcHBCjUiIiINzLm6jjgUakJDQ/H09KxSO5KRkVGlFqVMeHi43eO9vLxo1aoVmzdvZs+ePVx99dXlr5eWlprCeXmxbds2oqKiHL4ugK+vL76+vo58RBEREWmgHBr95OPj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x+rVq6u8pr/fzjVgwABmzpzJeeedx6FDh3jxxRcZPHgwmzdvdvm9VqhxApvNVulny7Kq7BPX0L13vocffpgNGzbw448/VnlN99t5zj//fJKTk8nKymLu3LnccccdLF26tPx13Wvn2bdvH4899hgLFy7Ez8/vjMfpnjvHiBEjyrd79erFoEGD6NSpE++//z4DBw4EXHev1fxUC6GhoXh6elaplcnIyKiSQsW5ynrS69471yOPPML8+fNZvHgx7dq1K9+v++18Pj4+dO7cmbi4OCZNmkSfPn3497//rXvtAmvXriUjI4PY2Fi8vLzw8vJi6dKlvP7663h5eZXfV91z1wgICKBXr15s377d5X+/FWpqwcfHh9jYWBYtWlRp/6JFixg8eLCbStU0xMTEEB4eXuneFxYWsnTpUt37GrAsi4cffph58+bxww8/EBMTU+l13W/XsyyLgoIC3WsXuOyyy9i4cSPJycnlj7i4OG655RaSk5Pp2LGj7rkLFRQUkJKSQkREhOv/fte6q3ETN3v2bMvb29uaMWOGtWXLFmv8+PFWQECAtWfPHncXrcHLzc21kpKSrKSkJAuwXn31VSspKcnau3evZVmW9be//c0KCQmx5s2bZ23cuNEaO3asFRERYeXk5Li55A3PAw88YIWEhFhLliyx0tLSyh95eXnlx+h+O89TTz1lLVu2zNq9e7e1YcMG6+mnn7Y8PDyshQsXWpale10XTh/9ZFm65870+9//3lqyZIm1a9cua9WqVdZvfvMbKygoqPx70ZX3WqHGCaZMmWJFR0dbPj4+Vv/+/cuHwUrtLF682AKqPO644w7LsszQwGeffdYKDw+3fH19rYsuusjauHGjewvdQNm7z4D17rvvlh+j++08d911V/n/Ga1bt7Yuu+yy8kBjWbrXdeHXoUb33HnGjBljRUREWN7e3lZkZKR1/fXXW5s3by5/3ZX32mZZllX7+h4RERER91KfGhEREWkUFGpERESkUVCoERERkUZBoUZEREQaBYUaERERaRQUakRERKRRUKgRERGRRkGhRkRERBoFhRoRERFpFBRqREREpFFQqBEREZFG4f8BUj7+xzrbMzsAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "history = model.fit(x_train, y_train, epochs = 50, batch_size = 128, validation_split = 0.2, verbose = 1)\n",
    "plt.plot(history.history ['mae'], label = ['Train MAE'])\n",
    "plt.plot(history.history ['val_mae'], label = ['Val MAE'])\n",
    "plt.xlabel['epochs']\n",
    "plt.ylabel['MAE']\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "67a8be70-ef24-4bfa-a493-6012b0a844b1",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "['scaler.pkl']"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# In your notebook, after training\n",
    "model.save('earthquake_magnitude_model.h5')\n",
    "import joblib\n",
    "joblib.dump(scaler, 'scaler.pkl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "a99382ed-f157-4140-b152-0c8c2d20d27a",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "np.save('features_scaled.npy', features_scaled)\n",
    "np.save('target.npy', target)\n",
    "np.save('x_train.npy', x_train)\n",
    "np.save('x_test.npy', x_test)\n",
    "np.save('y_train.npy', y_train)\n",
    "np.save('y_test.npy', y_test)\n",
    "\n",
    "# Continue with model training..."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "3f9b637c-a6a7-42a7-8473-33709e953304",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "X_train shape: (5552, 8)\n",
      "X_test shape: (1388, 8)\n",
      "y_train shape: (5552,)\n",
      "y_test shape: (1388,)\n",
      "Epoch 1/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m2s\u001b[0m 12ms/step - loss: 16.0940 - mae: 3.9397 - val_loss: 9.5628 - val_mae: 3.0208\n",
      "Epoch 2/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 3.3526 - mae: 1.6164 - val_loss: 1.9716 - val_mae: 1.2480\n",
      "Epoch 3/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.7755 - mae: 0.7056 - val_loss: 1.0537 - val_mae: 0.8532\n",
      "Epoch 4/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.6106 - mae: 0.6172 - val_loss: 0.8124 - val_mae: 0.7136\n",
      "Epoch 5/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.5233 - mae: 0.5734 - val_loss: 0.8327 - val_mae: 0.7015\n",
      "Epoch 6/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.4718 - mae: 0.5469 - val_loss: 0.9397 - val_mae: 0.7251\n",
      "Epoch 7/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - loss: 0.4302 - mae: 0.5176 - val_loss: 1.0613 - val_mae: 0.7525\n",
      "Epoch 8/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.4104 - mae: 0.5063 - val_loss: 1.1925 - val_mae: 0.7870\n",
      "Epoch 9/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.3708 - mae: 0.4830 - val_loss: 1.2198 - val_mae: 0.7833\n",
      "Epoch 10/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.3512 - mae: 0.4712 - val_loss: 1.3001 - val_mae: 0.8049\n",
      "Epoch 11/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.3224 - mae: 0.4510 - val_loss: 1.2618 - val_mae: 0.7854\n",
      "Epoch 12/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.2918 - mae: 0.4295 - val_loss: 1.3532 - val_mae: 0.8080\n",
      "Epoch 13/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.2684 - mae: 0.4117 - val_loss: 1.3461 - val_mae: 0.8018\n",
      "Epoch 14/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.2417 - mae: 0.3897 - val_loss: 1.3173 - val_mae: 0.7908\n",
      "Epoch 15/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.2241 - mae: 0.3785 - val_loss: 1.3448 - val_mae: 0.7990\n",
      "Epoch 16/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.1982 - mae: 0.3531 - val_loss: 1.2881 - val_mae: 0.7781\n",
      "Epoch 17/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.1773 - mae: 0.3349 - val_loss: 1.3192 - val_mae: 0.7951\n",
      "Epoch 18/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.1558 - mae: 0.3167 - val_loss: 1.3027 - val_mae: 0.7830\n",
      "Epoch 19/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.1342 - mae: 0.2923 - val_loss: 1.2138 - val_mae: 0.7553\n",
      "Epoch 20/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.1145 - mae: 0.2715 - val_loss: 1.2017 - val_mae: 0.7536\n",
      "Epoch 21/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.0990 - mae: 0.2511 - val_loss: 1.1460 - val_mae: 0.7380\n",
      "Epoch 22/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.0836 - mae: 0.2299 - val_loss: 1.1274 - val_mae: 0.7323\n",
      "Epoch 23/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.0743 - mae: 0.2178 - val_loss: 1.0678 - val_mae: 0.7122\n",
      "Epoch 24/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.0614 - mae: 0.1980 - val_loss: 0.9803 - val_mae: 0.6777\n",
      "Epoch 25/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.0495 - mae: 0.1786 - val_loss: 0.8956 - val_mae: 0.6435\n",
      "Epoch 26/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - loss: 0.0405 - mae: 0.1621 - val_loss: 0.8384 - val_mae: 0.6242\n",
      "Epoch 27/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - loss: 0.0332 - mae: 0.1466 - val_loss: 0.7519 - val_mae: 0.5921\n",
      "Epoch 28/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - loss: 0.0273 - mae: 0.1318 - val_loss: 0.6526 - val_mae: 0.5462\n",
      "Epoch 29/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - loss: 0.0223 - mae: 0.1199 - val_loss: 0.6243 - val_mae: 0.5323\n",
      "Epoch 30/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0178 - mae: 0.1069 - val_loss: 0.5533 - val_mae: 0.4974\n",
      "Epoch 31/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0141 - mae: 0.0954 - val_loss: 0.5219 - val_mae: 0.4848\n",
      "Epoch 32/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0113 - mae: 0.0841 - val_loss: 0.4621 - val_mae: 0.4492\n",
      "Epoch 33/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0098 - mae: 0.0787 - val_loss: 0.4257 - val_mae: 0.4287\n",
      "Epoch 34/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0090 - mae: 0.0747 - val_loss: 0.4037 - val_mae: 0.4152\n",
      "Epoch 35/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0076 - mae: 0.0684 - val_loss: 0.3860 - val_mae: 0.4049\n",
      "Epoch 36/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0068 - mae: 0.0645 - val_loss: 0.3684 - val_mae: 0.3936\n",
      "Epoch 37/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0064 - mae: 0.0628 - val_loss: 0.3772 - val_mae: 0.4030\n",
      "Epoch 38/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0059 - mae: 0.0601 - val_loss: 0.3327 - val_mae: 0.3700\n",
      "Epoch 39/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0057 - mae: 0.0589 - val_loss: 0.3181 - val_mae: 0.3579\n",
      "Epoch 40/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0051 - mae: 0.0560 - val_loss: 0.3196 - val_mae: 0.3606\n",
      "Epoch 41/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0050 - mae: 0.0553 - val_loss: 0.2983 - val_mae: 0.3421\n",
      "Epoch 42/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0047 - mae: 0.0538 - val_loss: 0.2958 - val_mae: 0.3455\n",
      "Epoch 43/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0047 - mae: 0.0535 - val_loss: 0.2749 - val_mae: 0.3242\n",
      "Epoch 44/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0043 - mae: 0.0514 - val_loss: 0.2766 - val_mae: 0.3291\n",
      "Epoch 45/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0043 - mae: 0.0511 - val_loss: 0.2773 - val_mae: 0.3283\n",
      "Epoch 46/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0040 - mae: 0.0500 - val_loss: 0.2742 - val_mae: 0.3285\n",
      "Epoch 47/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0039 - mae: 0.0487 - val_loss: 0.2522 - val_mae: 0.3105\n",
      "Epoch 48/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0036 - mae: 0.0468 - val_loss: 0.2502 - val_mae: 0.3100\n",
      "Epoch 49/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 0.0035 - mae: 0.0465 - val_loss: 0.2489 - val_mae: 0.3105\n",
      "Epoch 50/50\n",
      "\u001b[1m35/35\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 4ms/step - loss: 0.0035 - mae: 0.0461 - val_loss: 0.2476 - val_mae: 0.3087\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test Mean Absolute Error: 0.10\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense\n",
    "\n",
    "# Load preprocessed data\n",
    "X_train = np.load('x_train.npy')\n",
    "X_test = np.load('x_test.npy')\n",
    "y_train = np.load('y_train.npy')\n",
    "y_test = np.load('y_test.npy')\n",
    "\n",
    "# Verify shapes\n",
    "print(\"X_train shape:\", X_train.shape)\n",
    "print(\"X_test shape:\", X_test.shape)\n",
    "print(\"y_train shape:\", y_train.shape)\n",
    "print(\"y_test shape:\", y_test.shape)\n",
    "\n",
    "# Define and train hyper-tuned ANN\n",
    "model = Sequential()\n",
    "model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Adjust based on tuning\n",
    "model.add(Dense(32, activation='relu'))\n",
    "model.add(Dense(1, activation='linear'))\n",
    "model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])\n",
    "history = model.fit(X_train, y_train, epochs=50, batch_size=128, validation_split=0.2, verbose=1)\n",
    "\n",
    "# Evaluate\n",
    "loss, mae = model.evaluate(X_test, y_test, verbose=0)\n",
    "print(f\"Test Mean Absolute Error: {mae:.2f}\")\n",
    "\n",
    "# Save model\n",
    "model.save('earthquake_magnitude_model.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "8aba407d-2745-4702-9ea9-c2f7beafbfc0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean MAE: 0.04, Std: 0.01\n"
     ]
    }
   ],
   "source": [
    "#K-Fold Cross-Validation\n",
    "from sklearn.model_selection import KFold\n",
    "X = np.concatenate([np.load('x_train.npy'), np.load('x_test.npy')])\n",
    "y = np.concatenate([np.load('y_train.npy'), np.load('y_test.npy')])\n",
    "kf = KFold(n_splits=5, shuffle=True, random_state=42)\n",
    "mae_scores = []\n",
    "for train_idx, test_idx in kf.split(X):\n",
    "    X_train, X_test = X[train_idx], X[test_idx]\n",
    "    y_train, y_test = y[train_idx], y[test_idx]\n",
    "    model = Sequential([Dense(64, input_dim=X_train.shape[1], activation='relu'),\n",
    "                        Dense(32, activation='relu'), Dense(1, activation='linear')])\n",
    "    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])\n",
    "    model.fit(X_train, y_train, epochs=50, batch_size=128, verbose=0)\n",
    "    loss, mae = model.evaluate(X_test, y_test, verbose=0)\n",
    "    mae_scores.append(mae)\n",
    "print(f\"Mean MAE: {np.mean(mae_scores):.2f}, Std: {np.std(mae_scores):.2f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "9edf28ac-6bb0-4325-8806-037eec5390f7",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "['scaler.pkl']"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.save('earthquake_magnitude_model.h5')\n",
    "import joblib\n",
    "joblib.dump(scaler, 'scaler.pkl')  # If you still have the scaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "f4a51bbf-f78d-407b-86d5-8e2c0a528e2f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m44/44\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step\n"
     ]
    },
    {
     "ename": "ValueError",
     "evalue": "Shape of passed values is (1388, 8), indices imply (1388, 7)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[32], line 6\u001b[0m\n\u001b[0;32m      4\u001b[0m y_test \u001b[38;5;241m=\u001b[39m np\u001b[38;5;241m.\u001b[39mload(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124my_test.npy\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[0;32m      5\u001b[0m predictions \u001b[38;5;241m=\u001b[39m model\u001b[38;5;241m.\u001b[39mpredict(X_test)\u001b[38;5;241m.\u001b[39mflatten()\n\u001b[1;32m----> 6\u001b[0m df_test \u001b[38;5;241m=\u001b[39m pd\u001b[38;5;241m.\u001b[39mDataFrame(X_test, columns\u001b[38;5;241m=\u001b[39m[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mlatitude\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mlongitude\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mdepth\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124myear\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mmonth\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mday\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mhour\u001b[39m\u001b[38;5;124m'\u001b[39m])\n\u001b[0;32m      7\u001b[0m df_test[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mactual_mag\u001b[39m\u001b[38;5;124m'\u001b[39m] \u001b[38;5;241m=\u001b[39m y_test\n\u001b[0;32m      8\u001b[0m df_test[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mpredicted_mag\u001b[39m\u001b[38;5;124m'\u001b[39m] \u001b[38;5;241m=\u001b[39m predictions\n",
      "File \u001b[1;32mD:\\anaconda\\Lib\\site-packages\\pandas\\core\\frame.py:827\u001b[0m, in \u001b[0;36mDataFrame.__init__\u001b[1;34m(self, data, index, columns, dtype, copy)\u001b[0m\n\u001b[0;32m    816\u001b[0m         mgr \u001b[38;5;241m=\u001b[39m dict_to_mgr(\n\u001b[0;32m    817\u001b[0m             \u001b[38;5;66;03m# error: Item \"ndarray\" of \"Union[ndarray, Series, Index]\" has no\u001b[39;00m\n\u001b[0;32m    818\u001b[0m             \u001b[38;5;66;03m# attribute \"name\"\u001b[39;00m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    824\u001b[0m             copy\u001b[38;5;241m=\u001b[39m_copy,\n\u001b[0;32m    825\u001b[0m         )\n\u001b[0;32m    826\u001b[0m     \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m--> 827\u001b[0m         mgr \u001b[38;5;241m=\u001b[39m ndarray_to_mgr(\n\u001b[0;32m    828\u001b[0m             data,\n\u001b[0;32m    829\u001b[0m             index,\n\u001b[0;32m    830\u001b[0m             columns,\n\u001b[0;32m    831\u001b[0m             dtype\u001b[38;5;241m=\u001b[39mdtype,\n\u001b[0;32m    832\u001b[0m             copy\u001b[38;5;241m=\u001b[39mcopy,\n\u001b[0;32m    833\u001b[0m             typ\u001b[38;5;241m=\u001b[39mmanager,\n\u001b[0;32m    834\u001b[0m         )\n\u001b[0;32m    836\u001b[0m \u001b[38;5;66;03m# For data is list-like, or Iterable (will consume into list)\u001b[39;00m\n\u001b[0;32m    837\u001b[0m \u001b[38;5;28;01melif\u001b[39;00m is_list_like(data):\n",
      "File \u001b[1;32mD:\\anaconda\\Lib\\site-packages\\pandas\\core\\internals\\construction.py:336\u001b[0m, in \u001b[0;36mndarray_to_mgr\u001b[1;34m(values, index, columns, dtype, copy, typ)\u001b[0m\n\u001b[0;32m    331\u001b[0m \u001b[38;5;66;03m# _prep_ndarraylike ensures that values.ndim == 2 at this point\u001b[39;00m\n\u001b[0;32m    332\u001b[0m index, columns \u001b[38;5;241m=\u001b[39m _get_axes(\n\u001b[0;32m    333\u001b[0m     values\u001b[38;5;241m.\u001b[39mshape[\u001b[38;5;241m0\u001b[39m], values\u001b[38;5;241m.\u001b[39mshape[\u001b[38;5;241m1\u001b[39m], index\u001b[38;5;241m=\u001b[39mindex, columns\u001b[38;5;241m=\u001b[39mcolumns\n\u001b[0;32m    334\u001b[0m )\n\u001b[1;32m--> 336\u001b[0m _check_values_indices_shape_match(values, index, columns)\n\u001b[0;32m    338\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m typ \u001b[38;5;241m==\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124marray\u001b[39m\u001b[38;5;124m\"\u001b[39m:\n\u001b[0;32m    339\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28missubclass\u001b[39m(values\u001b[38;5;241m.\u001b[39mdtype\u001b[38;5;241m.\u001b[39mtype, \u001b[38;5;28mstr\u001b[39m):\n",
      "File \u001b[1;32mD:\\anaconda\\Lib\\site-packages\\pandas\\core\\internals\\construction.py:420\u001b[0m, in \u001b[0;36m_check_values_indices_shape_match\u001b[1;34m(values, index, columns)\u001b[0m\n\u001b[0;32m    418\u001b[0m passed \u001b[38;5;241m=\u001b[39m values\u001b[38;5;241m.\u001b[39mshape\n\u001b[0;32m    419\u001b[0m implied \u001b[38;5;241m=\u001b[39m (\u001b[38;5;28mlen\u001b[39m(index), \u001b[38;5;28mlen\u001b[39m(columns))\n\u001b[1;32m--> 420\u001b[0m \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mShape of passed values is \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mpassed\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m, indices imply \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mimplied\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m\"\u001b[39m)\n",
      "\u001b[1;31mValueError\u001b[0m: Shape of passed values is (1388, 8), indices imply (1388, 7)"
     ]
    }
   ],
   "source": [
    "import plotly.express as px\n",
    "import pandas as pd\n",
    "X_test = np.load('x_test.npy')\n",
    "y_test = np.load('y_test.npy')\n",
    "predictions = model.predict(X_test).flatten()\n",
    "df_test = pd.DataFrame(X_test, columns=['latitude', 'longitude', 'depth', 'year', 'month', 'day', 'hour'])\n",
    "df_test['actual_mag'] = y_test\n",
    "df_test['predicted_mag'] = predictions\n",
    "df_test = df_test.dropna()\n",
    "df_test['actual_mag'] = df_test['actual_mag'].clip(lower=0)\n",
    "fig = px.scatter_geo(df_test, lat='latitude', lon='longitude', size='actual_mag', \n",
    "                     color='predicted_mag', title='Actual vs Predicted Magnitudes')\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7cb2147d-5937-42ba-9ff4-2f201b6cb1be",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
