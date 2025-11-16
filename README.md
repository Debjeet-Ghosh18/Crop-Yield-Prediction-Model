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
🛠 Technologies Used
Programming & Libraries

Python 3.11

Pandas, NumPy — Data Analysis

Scikit-learn — ML Algorithms (Decision Tree, Random Forest, KNN)

Matplotlib, Seaborn, Plotly — Visualization

Deployment & Tools

Streamlit — Interactive Web App

Pickle — Model Serialization

GitHub — Version Control

VS Code, Jupyter Notebook — Development


🔑 Key Functionalities
✔ 1. Data Preprocessing Pipeline

Automated cleaning, handling of missing values, encoding categorical features, and normalization for improved model accuracy.

✔ 2. Exploratory Data Analysis (EDA)

Includes:

Time-series analysis of yield across years

Country-wise and crop-wise comparison

Heatmaps and correlation analysis

✔ 3. Model Training & Evaluation

Implements and compares:

Decision Tree Regressor

Random Forest Regressor

K-Nearest Neighbors (KNN)

Evaluation Metrics:

R² Score

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

Achieved up to 96.5% accuracy after optimization.

✔ 4. Performance Optimization

Hyperparameter tuning

Cross-validation

Feature engineering

✔ 5. Interactive Prediction Dashboard

Built using Streamlit.
Users input:

Country

Crop Type

Rainfall

Temperature

Pesticide usage



🔮 The system predicts crop yield in real time.

✔ 6. Visual Analytics

Interactive charts generated using Plotly for:

Model performance comparison

Feature importance

Yield trends

✔ 7. Scalable & Extensible Architecture

Easily extendable to include:

Time-series forecasting

Satellite imagery

IoT sensor data

Additional ML or deep learning models


🚀 How to Run the Project
1. Clone the Repository
git clone https://github.com/your-username/CROP_YIELD_PREDICTION.git
cd CROP_YIELD_PREDICTION

2. Install Dependencies
pip install -r requirements.txt

3. Run the Jupyter Notebook (Optional)
jupyter notebook

4. Launch the Streamlit Dashboard
streamlit run dashboard.py




