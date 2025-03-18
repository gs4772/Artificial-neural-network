# Artificial-neural-network
Seismic Hazard Assessment Using ANN

Seismic Hazard Assessment Using Artificial Neural Networks and Geospatial Analysis
 An AI-driven approach to earthquake risk prediction in Northeast India using Artificial Neural Networks (ANN) and geospatial data.

 About This Project
Seismic hazard assessment is crucial for predicting and mitigating earthquake risks, especially in highly seismic regions like Northeast India. Traditional statistical models often fail to capture nonlinear relationships between seismic parameters. This project introduces an Artificial Neural Network (ANN)-based model, integrating historical earthquake data with geospatial features to improve risk prediction accuracy.

🛠 Key Features
 Data Collection & Preprocessing: Integrates USGS earthquake records with geospatial features (fault density, PGA, lithology, etc.).
 Deep Learning Model: Implements a multi-layer ANN to classify seismic risk levels.
 Performance Metrics: Evaluates model accuracy using precision, recall, F1-score, and confusion matrix.
 Optimized for NE India: Focuses on earthquake-prone regions, filling gaps in seismic hazard research.

Dataset & Features
Data Source: USGS Earthquake Catalog, Geological Survey of India (GSI), NASA SRTM DEM.
Key Features:
Magnitude, Depth, PGA, Fault Density, Lithology Factor, Elevation, Slope, Epicenter Distance.
Preprocessing Techniques:
Missing value handling, feature scaling, SMOTE for class balancing.

Model Architecture
 Input Layer: 11 neurons (seismic & geospatial parameters).
 Hidden Layers:

Layer 1: 64 neurons (ReLU activation).
Layer 2: 32 neurons (ReLU activation).
 Output Layer: 25 neurons (Softmax activation).
 Optimizer: Adam | Loss Function: Categorical Cross-Entropy.
📊 Results & Evaluation
Model	Accuracy	Precision	Recall	F1-score
ANN (Proposed Model)	89.7%	0.91	0.88	0.89
Random Forest	83.5%	0.85	0.81	0.83
Logistic Regression	76.4%	0.79	0.75	0.77
📌 ANN achieves the highest accuracy in predicting earthquake risk zones.

License
This project is licensed under the MIT License – see the LICENSE file for details.
