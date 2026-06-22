# AI Project 2
# Data Classification Using Machine Learning

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

print("========== DATA CLASSIFICATION USING AI ==========")

# Load Dataset
iris = load_iris()

X = iris.data
y = iris.target

print("\nDataset Loaded Successfully!")
print("Number of Samples:", len(X))
print("Number of Features:", len(X[0]))

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Size:", len(X_train))
print("Testing Data Size:", len(X_test))

# Create Model
model = DecisionTreeClassifier()

# Train Model
model.fit(X_train, y_train)

print("\nModel Training Completed!")

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# Detailed Report
print("\nClassification Report")
print(classification_report(y_test, predictions))

# Visualization
species = ["Setosa", "Versicolor", "Virginica"]

plt.figure(figsize=(6,4))
plt.bar(species, [50,50,50])
plt.title("Iris Dataset Classes")
plt.xlabel("Flower Species")
plt.ylabel("Count")
plt.show()