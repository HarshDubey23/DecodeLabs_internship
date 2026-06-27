"""
Project 2 - Data Classification Using AI
DecodeLabs AI Internship

Builds a K-Nearest Neighbors classifier on the Iris dataset, following
the Input -> Process -> Output pipeline:
    Input:    Iris dataset + feature scaling
    Process:  Train-test split + KNN algorithm
    Output:   Confusion matrix + F1 score
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score


def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    df.to_csv("dataset.csv", index=False)
    return df, iris.target_names


def main():
    df, target_names = load_data()

    X = df.drop(columns=["target", "species"])
    y = df["target"]

    # Step 1: Train-test split (80/20, shuffled, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 2: Feature scaling (StandardScaler -> mean 0, variance 1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 3: Train the KNN model
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)

    # Step 4: Predict on the test set
    predictions = model.predict(X_test_scaled)

    # Step 5: Evaluate
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("F1 Score (weighted):", f1_score(y_test, predictions, average="weighted"))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=target_names))

    # Step 6: Predict a brand-new sample
    sample = [[5.1, 3.5, 1.4, 0.2]]
    sample_scaled = scaler.transform(sample)
    sample_pred = model.predict(sample_scaled)[0]
    print(f"\nSample prediction for {sample[0]}: {target_names[sample_pred]}")


if __name__ == "__main__":
    main()
