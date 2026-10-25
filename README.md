<div align="center">

# 💼 Machine Learning & AI Engineering Internship
### DecodeLabs — Engineering Deliverables & Production Projects

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Machine Learning](https://img.shields.io/badge/Machine_Learning-Scikit--Learn-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![Computer Vision](https://img.shields.io/badge/Computer_Vision-OpenCV-5C3EE8?logo=opencv)](https://opencv.org/)
[![NLP](https://img.shields.io/badge/NLP-NLTK_&_SpaCy-green)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A comprehensive portfolio of 4 machine learning projects designed, developed, and deployed during a 1-month AI/ML Engineering Internship at DecodeLabs.*

---

</div>

## 📌 Projects Portfolio Overview

This repository contains the end-to-end deliverables across conversational AI, supervised classification, recommendation systems, and optical character recognition (OCR).

| Project | Domain | Key Architecture / Algorithms | Directory |
|---|---|---|---|
| **01. Rule-Based Chatbot** | Conversational AI / NLP | Intent extraction, state machine dialogue management, regex slot filling | [`Project-1-Rule-Based-Chatbot`](Project-1-Rule-Based-Chatbot) |
| **02. Supervised Classifier** | Predictive Modeling | Random Forests, Gradient Boosting (XGBoost), hyperparameter tuning, AUC-ROC evaluation | [`Project-2-Classification`](Project-2-Classification) |
| **03. Recommendation Engine** | Information Retrieval | Collaborative filtering (matrix factorization) & content-based cosine similarity | [`Project-3-Recommendation-System`](Project-3-Recommendation-System) |
| **04. OCR & Image Text Recognition** | Computer Vision | Image binarization, morphological filtering, bounding box contour detection, Tesseract OCR | [`Project-4-Image-Text-Recognition`](Project-4-Image-Text-Recognition) |

---

## 🛠️ Detailed Breakdown

### 🤖 Project 1: Rule-Based Conversational Agent
* **Objective:** Design an interactive customer assistant with zero hallucination risk.
* **Techniques:** Deterministic intent mapping, context memory persistence, slot-filling entity extractor.
* **Key Benefit:** Handles domain inquiries with predictable 100% precision.

### 📊 Project 2: Supervised Predictive Classification
* **Objective:** End-to-end predictive classification pipeline with robust cross-validation.
* **Techniques:** Feature scaling, categorical encoding, handling class imbalance via SMOTE/class weights, model evaluation via Confusion Matrix, Precision-Recall curves, and ROC analysis.

### 🎯 Project 3: Hybrid Recommendation System
* **Objective:** Deliver personalized item suggestions using hybrid recommendation approaches.
* **Techniques:**
  * User-Item interaction matrix construction.
  * Cosine similarity matching on item features (content-based).
  * Latent factor modeling via singular value decomposition (collaborative).

### 👁️ Project 4: Optical Character Recognition (OCR) & Text Extraction
* **Objective:** Extract textual content from real-world noisy document images.
* **Techniques:**
  * Preprocessing: Grayscale conversion, Otsu thresholding, adaptive noise removal, dilation/erosion.
  * Inference: Tesseract OCR engine with customized bounding-box coordinate tracking.

---

## 🚀 Getting Started

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/HarshDubey23/DecodeLabs_internship.git
cd DecodeLabs_internship
pip install -r requirements.txt  # or install requirements per project directory
```

### 2. Run Projects
Each project directory is self-contained with its own runnable entry point:
```bash
# Example: Run Project 1 Chatbot
cd Project-1-Rule-Based-Chatbot
python chatbot.py
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
