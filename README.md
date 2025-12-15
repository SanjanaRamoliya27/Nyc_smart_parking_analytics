# 🚗 Smart Parking Analytics System

An end-to-end **Smart Parking Analytics System** that uses machine learning and geospatial analysis to analyze parking meter data, recommend nearby parking spots, and predict parking meter status through an interactive Streamlit dashboard.

---

## 📌 Overview

Urban parking is a major challenge in large cities. This project addresses that challenge by applying machine learning techniques to parking meter data in order to:
- Identify parking zones
- Recommend nearby parking meters
- Predict parking meter status
- Visualize parking insights interactively

The project follows a **complete machine learning workflow**, from data preprocessing to deployment-ready visualization.

---

## 📊 Dataset Source

The dataset used in this project is obtained from **NYC Open Data**, an official open data platform provided by the City of New York.

**Dataset:** Parking Meters Data  
**Source:** https://data.cityofnewyork.us/Transportation/Parking-Meters-Locations-and-Status/693u-uax6/data_preview

The dataset contains detailed information about parking meters across New York City, including:
- Parking meter identifiers
- Geographic coordinates (latitude and longitude)
- Parking rules and enforcement timings
- Maximum parking duration
- Street and borough information

The raw dataset was cleaned, preprocessed, and feature-engineered before being used in the machine learning pipeline.

---

## 🗂️ Project Structure

## Complete Workflow Explanation

The project workflow is divided into **two main phases**:

1. **Machine Learning Pipeline**
2. **Streamlit Application**

---

## Phase 1: Machine Learning Pipeline (`parking_pipeline.py`)

This phase handles **data processing, model training, evaluation, and saving artifacts**.

### Step 1: Data Loading
The pipeline loads the parking dataset containing parking meter details, geographic coordinates, parking rules, and time-based features.

---

### Step 2: Feature Selection
Two feature groups are created:

**Clustering Features**
- Latitude
- Longitude
- Start_Time_Hours
- End_Time_Hours
- Max_Parking_Hours

**Prediction Features**
- Max_Parking_Hours
- Rule_Count
- Enforcement_Duration_Hours
- Start_Time_Hours
- End_Time_Hours
- Latitude
- Longitude
- Cluster labels (DBSCAN & KMeans)

---

### Step 3: DBSCAN Clustering
DBSCAN performs density-based clustering to:
- Identify dense parking regions
- Detect noise and outlier parking meters

Missing values are handled and features are scaled before clustering.

---

### Step 4: KMeans Clustering
KMeans is applied only to non-noise points identified by DBSCAN to divide dense parking regions into structured parking zones.

---

### Step 5: Target Encoding
The target column `Meter_Status` is encoded into numerical values using label encoding.

---

### Step 6: Data Preprocessing
A preprocessing pipeline is created to:
- Handle missing values
- Scale numerical features

This pipeline is reused during inference to maintain consistency.

---

### Step 7: Train-Test Split
The dataset is split into training and testing sets using stratified sampling to preserve class distribution.

---

### Step 8: XGBoost Model Training
An XGBoost classifier is trained to predict parking meter status using time, location, rule-based features, and cluster labels.

---

### Step 9: Model Evaluation
The trained model is evaluated using precision, recall, and F1-score metrics.

---

### Step 10: Saving Artifacts
All trained models and pipelines are saved for deployment:
- DBSCAN pipeline
- KMeans pipeline
- Preprocessing pipeline
- XGBoost model
- Label encoder
- Clean dataset for Streamlit

---

## Phase 2: Streamlit Web Application (`streamlit_app.py`)

The Streamlit application provides an interactive interface for users.

### Step 1: Load Data and Models
The app loads the preprocessed dataset and trained ML models.

---

### Step 2: Parking Meter Selection
Users select a parking meter using its unique identifier from the sidebar.

---

### Step 3: Display Meter Details
All details of the selected parking meter are displayed for transparency.

---

### Step 4: Nearest Parking Recommendation
Nearby parking meters are recommended using the Haversine distance based on geographic proximity.

---

### Step 5: Meter Status Prediction
The selected meter’s features are preprocessed and passed to the XGBoost model to predict its parking meter status.

---

### Step 6: Map Visualization
An interactive map displays:
- The selected parking meter
- Nearby recommended parking meters

---

## ▶️ How to Run the Project

### 1️. Install Dependencies
pip install -r requirements.txt

### 2. Train the Models
python parking_pipeline.py

### 3. Run the Streamlit App
streamlit run streamlit_app.py
