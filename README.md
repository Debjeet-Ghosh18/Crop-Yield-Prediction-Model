🌾 ML-Based Crop Yield Analysis and Prediction Model
📌 Overview

The ML-Based Crop Yield Analysis and Prediction Model is a data-driven system designed to forecast agricultural crop yield using machine learning techniques. By leveraging historical crop data along with environmental indicators such as rainfall, temperature, pesticide usage, crop type, and region, the model predicts yield in hectograms per hectare.
This solution addresses real-world challenges in agriculture—including climate variability, resource optimization, and sustainable planning. The project includes:

A machine learning pipeline for preprocessing, training, and evaluating models

Multiple ML algorithms (Decision Tree, Random Forest, KNN)

A Streamlit dashboard for real-time predictions

Highly modular directory structure for scalability and future enhancements
Project Directory Structure
CROP_YIELD_PREDICTION
├── data
├── model
│   ├── crop_yield_model.pkl
│   ├── model_info.pkl
│   ├── preprocessor.pkl
│   └── yield_df.csv
├── utils
│   └── helpers.py
├── app.py
├── Crop_Yield_Prediction.ipynb
├── dashboard.py
├── README.md
├── requirements.txt
└── setup.py
