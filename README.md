# Iris Flower Classification Project

## Introduction

This project explores the fascinating world of machine learning through the lens of the Iris flower dataset, one of the most famous datasets used for classification tasks. Our objective is to build a predictive model capable of distinguishing between the three species of Iris flowers — setosa, versicolor, and virginica — based on the physical dimensions of their petals and sepals. By applying machine learning techniques, we aim to uncover the patterns that define the uniqueness of each species.

## Data Loading and Preprocessing

The dataset is loaded from a CSV file named `IRIS.csv`. The preprocessing steps involve separating the features (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) and the target column (`species`). The target column remains in its original categorical form, as the `RandomForestClassifier` can handle categorical labels directly. Additionally, the feature data is standardized using `StandardScaler` to ensure uniform scaling, which improves model performance.

```python
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/IRIS.csv')

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Display dataset information
print("\nDataset Information:")
print(data.info())

# Display basic statistics of the dataset
print("\nDataset Statistics:")
print(data.describe())

# Data preprocessing
X = data.drop('species', axis=1)  # Features
y = data['species']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature importance
feature_importances = model.feature_importances_
features = X.columns
sns.barplot(x=feature_importances, y=features)
plt.title("Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()
