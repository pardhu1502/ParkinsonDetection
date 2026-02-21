# Parkinson’s Disease Detection using PCA-SVM & SAE-SVM

This project compares two hybrid machine learning models for detecting Parkinson’s Disease using voice measurements:

-PCA + SVM

-Sparse Autoencoder (SAE) + SVM

-Dataset used: UCI Parkinson's Dataset

# 📌 Objective

To evaluate whether deep feature extraction (Sparse Autoencoder) performs better than traditional dimensionality reduction (PCA) for Parkinson’s classification.

## 🛠 Tech Stack

 - Python

 - PyTorch

 - Scikit-learn

 - NumPy

 - Pandas

 - Matplotlib

🏗️ Architecture 

Voice Features
      ↓
Train/Test Split (Stratified)
      ↓
SMOTE (Class Balancing)
      ↓
Standardization
      ↓
Feature Reduction
   ├── PCA
   └── Sparse Autoencoder (PyTorch + L1 Regularization)
      ↓
SVM (RBF Kernel + GridSearchCV)
      ↓
Evaluation (Accuracy, F1, MCC, PR-AUC)

⚙️ Methodology

Load and preprocess dataset

Handle class imbalance using SMOTE

Standardize features

Feature Extraction:

PCA

Sparse Autoencoder (PyTorch)

Classification using SVM

Evaluate using:

Accuracy

F1 Score

MCC

PR AUC

Confusion Matrix

📊 Final Results
🔹 PCA-SVM

Accuracy: 89.74%

F1 Score: 0.9286

MCC: 0.7536

PR AUC: 0.9818

🔹 SAE-SVM

Accuracy: 89.74%

F1 Score: 0.9259

MCC: 0.7847

PR AUC: 0.9854
