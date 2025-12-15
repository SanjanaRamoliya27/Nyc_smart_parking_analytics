# =====================================================
# SMART PARKING ANALYTICS - FULL ML PIPELINE (FINAL)
# =====================================================

import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN, KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# -----------------------------------------------------
# 1. Load Data
# -----------------------------------------------------
DATA_PATH = "F:/Projects/NYC_TAXI/parking_preprocessed_final.csv"
df = pd.read_csv(DATA_PATH)

print("✅ Data Loaded:", df.shape)

# -----------------------------------------------------
# 2. Feature Selection
# -----------------------------------------------------
cluster_features = [
    "Latitude",
    "Longitude",
    "Start_Time_Hours",
    "End_Time_Hours",
    "Max_Parking_Hours"
]

ml_features = [
    "Max_Parking_Hours",
    "Rule_Count",
    "Enforcement_Duration_Hours",
    "Start_Time_Hours",
    "End_Time_Hours",
    "Latitude",
    "Longitude"
]

# -----------------------------------------------------
# 3. DBSCAN
# -----------------------------------------------------
dbscan_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("dbscan", DBSCAN(eps=0.8, min_samples=5))
])

df["DBSCAN_Cluster"] = dbscan_pipeline.fit_predict(df[cluster_features])
print("✅ DBSCAN clustering completed")

# -----------------------------------------------------
# 4. KMeans (Exclude Noise)
# -----------------------------------------------------
df_kmeans = df[df["DBSCAN_Cluster"] != -1].copy()

kmeans_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=5, random_state=42))
])

df.loc[df_kmeans.index, "KMeans_Cluster"] = (
    kmeans_pipeline.fit_predict(df_kmeans[cluster_features])
)

df["KMeans_Cluster"] = df["KMeans_Cluster"].fillna(-1).astype(int)
print("✅ KMeans clustering completed")

# -----------------------------------------------------
# 5. Encode Target
# -----------------------------------------------------
le = LabelEncoder()
df["Meter_Status_Encoded"] = le.fit_transform(df["Meter_Status"])

# -----------------------------------------------------
# 6. ML Dataset
# -----------------------------------------------------
X = df[ml_features + ["DBSCAN_Cluster", "KMeans_Cluster"]]
y = df["Meter_Status_Encoded"]

ml_preprocess = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

X_processed = ml_preprocess.fit_transform(X)

# -----------------------------------------------------
# 7. Train/Test Split
# -----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------------------------------
# 8. XGBoost
# -----------------------------------------------------
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="mlogloss",
    random_state=42
)

xgb.fit(X_train, y_train)
print("✅ XGBoost training completed")

# -----------------------------------------------------
# 9. Evaluation
# -----------------------------------------------------
y_pred = xgb.predict(X_test)
labels = np.unique(y_test)

print(classification_report(
    y_test,
    y_pred,
    labels=labels,
    target_names=le.inverse_transform(labels)
))

# -----------------------------------------------------
# 10. Save Artifacts
# -----------------------------------------------------
pickle.dump(dbscan_pipeline, open("dbscan_pipeline.pkl", "wb"))
pickle.dump(kmeans_pipeline, open("kmeans_pipeline.pkl", "wb"))
pickle.dump(ml_preprocess, open("ml_preprocess_pipeline.pkl", "wb"))
pickle.dump(xgb, open("xgb_model.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

# Save SAME dataframe for Streamlit
df.to_csv("parking_preprocessed_final_streamlit.csv", index=False)

print("✅ ALL MODELS & STREAMLIT DATA SAVED")
