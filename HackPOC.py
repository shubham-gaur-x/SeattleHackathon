#!/usr/bin/env python
# coding: utf-8

"""
Seattle Emergency Data Analysis and Modeling
This script processes Seattle's emergency data from the city's open dataset to perform data cleaning,
feature engineering, and predictive modeling using Logistic Regression and K-Nearest Neighbors (KNN).
It includes hyperparameter tuning, evaluation metrics, and model saving.
"""

import pandas as pd
import requests
from joblib import load, dump
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------------
# Fetch Data from the Seattle Open Data API
# --------------------------------------------------------------

# URL of the JSON endpoint
url = "https://data.seattle.gov/resource/kzjm-xkqj.json"
response = requests.get(url)
df = pd.read_json(response.text)

# --------------------------------------------------------------
# Preprocess the Dataset
# --------------------------------------------------------------

# Drop irrelevant columns
df = df.drop(df.columns[7:12], axis=1)
# Remove rows with missing values
df = df.dropna()

# Handle outliers for latitude and longitude using IQR
Q1_long, Q3_long = df['longitude'].quantile([0.25, 0.75])
IQR_long = Q3_long - Q1_long
outlier_threshold_long = 1.5 * IQR_long

Q1_lat, Q3_lat = df['latitude'].quantile([0.25, 0.75])
IQR_lat = Q3_lat - Q1_lat
outlier_threshold_lat = 1.5 * IQR_lat

df = df[
    (df['longitude'] >= (Q1_long - outlier_threshold_long)) &
    (df['longitude'] <= (Q3_long + outlier_threshold_long)) &
    (df['latitude'] >= (Q1_lat - outlier_threshold_lat)) &
    (df['latitude'] <= (Q3_lat + outlier_threshold_lat))
]

# Convert and extract date-related features
df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%Y %I:%M:%S %p')
df['Month'] = df['datetime'].dt.month
df['Hour'] = df['datetime'].dt.hour
df['AM_PM'] = df['datetime'].dt.strftime('%p').map({'AM': 1, 'PM': 0})
df = df.drop(['datetime', 'report_location', 'incident_number', 'address'], axis=1)

# Encode the 'type' column
df['Encoded_Type'] = LabelEncoder().fit_transform(df['type'])
df = df.drop(['type'], axis=1)

# Adjust longitude values
def convert_longitude(longitude):
    positive_longitude = 180.0 + longitude
    if positive_longitude > 180.0:
        positive_longitude -= 360.0
    return positive_longitude

df['longitude'] = df['longitude'].apply(convert_longitude)

# --------------------------------------------------------------
# Split Data into Features and Target
# --------------------------------------------------------------

X = df[['latitude', 'longitude', 'Month', 'Hour', 'AM_PM']]
y = df['Encoded_Type']

# --------------------------------------------------------------
# Balance the Dataset
# --------------------------------------------------------------

X_resampled, y_resampled = RandomOverSampler(random_state=42).fit_resample(X, y)

# --------------------------------------------------------------
# Train-Test Split
# --------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Save the test data
csv_path = "/Users/shubhamgaur/Desktop/HackPOCTest.csv"
json_path = "/Users/shubhamgaur/Desktop/HackPOCTest.json"

test_df = pd.DataFrame(X_test, columns=X.columns)
test_df['Encoded_Type'] = y_test
test_df.to_csv(csv_path, index=False)
test_df.to_json(json_path, orient='records')

# --------------------------------------------------------------
# Train and Evaluate Models
# --------------------------------------------------------------

# Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)
log_predictions = logistic_model.predict(X_test)

log_metrics = {
    "Accuracy": accuracy_score(y_test, log_predictions),
    "Precision": precision_score(y_test, log_predictions, average='macro', zero_division=0),
    "Recall": recall_score(y_test, log_predictions, average='macro')
}
print("Logistic Regression Metrics:", log_metrics)

# Save Logistic Regression model
dump(logistic_model, 'logistic_regression_model.joblib')

# K-Nearest Neighbors with Grid Search
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_knn_model = grid_search.best_estimator_
knn_predictions = best_knn_model.predict(X_test)

knn_metrics = {
    "Best Parameters": grid_search.best_params_,
    "Accuracy": accuracy_score(y_test, knn_predictions),
    "Precision": precision_score(y_test, knn_predictions, average='macro', zero_division=0),
    "Recall": recall_score(y_test, knn_predictions, average='macro')
}
print("K-NN Metrics:", knn_metrics)

# Save K-NN model
dump(best_knn_model, 'knn_model.joblib')
