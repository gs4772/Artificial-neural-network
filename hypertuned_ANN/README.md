# Earthquake Magnitude Prediction (Hyper-Tuned ANN)

This project predicts earthquake magnitudes using a hyper-tuned Artificial Neural Network (ANN) trained on historical data from 1990 to 2023.

## Overview
The ANN is optimized to predict magnitudes based on features like latitude, longitude, depth, and time (year, month, day, hour). Hyperparameter tuning enhances model performance over a baseline ANN.

## Files
- **`ANN_Qk.ipynb`**: Jupyter Notebook with:
  - Data preprocessing of `Eartquakes-1990-2023.csv`.
  - Hyper-tuned ANN training and evaluation.
  - Visualization of predictions.

## Requirements
pandas
numpy
scikit-learn
tensorflow
keras-tuner
joblib
plotly

Usage
Place Eartquakes-1990-2023.csv in the directory.
Run ANN_Qk.ipynb in Jupyter to train and evaluate the hyper-tuned model.
Model
Tuning: Optimized layers (e.g., 32–128 neurons) and learning rate (e.g., 0.0001–0.01) using keras-tuner.
Loss: Mean Squared Error (MSE).
Optimizer: Adam.
