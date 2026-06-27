# Project 2 — Data Classification Using AI

## Overview
A supervised machine learning project that trains a K-Nearest Neighbors
(KNN) classifier on the classic Iris dataset to classify flowers into
Setosa, Versicolor, or Virginica based on their measurements.

## Objective
Build, train, and evaluate a basic classification model end-to-end:
load data → split → scale → train → predict → evaluate.

## Features
- Loads the Iris dataset (150 samples, 4 features, 3 classes) and exports it to `dataset.csv`
- 80/20 stratified train-test split with shuffling
- Feature scaling with `StandardScaler` (mean 0, variance 1)
- K-Nearest Neighbors classifier (k=5)
- Evaluation via Accuracy, F1 Score, Confusion Matrix, and Classification Report
- Predicts the class of a brand-new, unseen sample

## Architecture (IPO Model)

INPUT                    PROCESS                  OUTPUT
Iris dataset      -->    Train/Test Split   -->    Confusion Matrix
                         Feature Scaling          + F1 Score / Accuracy
                         + KNN Algorithm

## Tech Stack
Python, pandas, scikit-learn

## Installation
```bash
pip install -r requirements.txt
```

## Run
```bash
python train_model.py
```
This also (re)generates `dataset.csv` in this folder.

## Screenshots
> Replace this with a real screenshot of your terminal output (accuracy,
> F1 score, confusion matrix).

![Training output](screenshots/output_demo.png)

## Folder Structure

Project-2-Classification/
│── dataset.csv          # Training dataset (auto-generated on run)
│── train_model.py       # Model training & evaluation
│── requirements.txt
│── screenshots/
│── README.md

## Future Improvements
- Try different values of K and plot the "elbow curve"
- Compare KNN against Logistic Regression / Decision Tree / SVM
- Add cross-validation
- Test the model on a completely new, real-world dataset
