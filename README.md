# 🧬 Skin Lesion Classification Pipeline (HAM10000)

This project performs feature-based classification of skin lesions using the HAM10000 dataset. It combines classic computer vision techniques (HOG and LBP), dimensionality reduction (PCA), and traditional machine learning models (SVM, Softmax Regression, OvR) to classify dermatological images.

---

## 🔍 Project Overview

- 📊 **Dataset**: [HAM10000](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)
- 🧠 **Features**: 
  - HOG (Histogram of Oriented Gradients)
  - LBP (Local Binary Patterns)
  - PCA for dimensionality reduction
- 🧪 **Models**:
  - SVM (with and without PCA)
  - Logistic Regression (Softmax)
  - Logistic Regression (OvR)
- 📉 **Evaluation**:
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)

---

## 🧰 Tools & Libraries

- Python
- OpenCV
- scikit-image
- scikit-learn
- TensorFlow
- CuML (GPU-accelerated SVM)
- Matplotlib, Seaborn

---

## ⚙️ Feature Extraction Pipeline

1. Resize images to 64x64
2. Convert to grayscale
3. Extract:
   - HOG features (orientation, blocks, cells)
   - LBP histograms (radius, points, uniform)
4. Combine features
5. Apply PCA to reduce to 50 components

---

## 📈 Classifier Comparison

| Classifier        | Input      | Accuracy | Notes                                  |
|------------------|------------|----------|----------------------------------------|
| SVM              | Raw Images | Low      | High dimensionality, weak performance  |
| SVM              | PCA        | **Best** | High accuracy, robust classification   |
| Logistic (Softmax)| Raw/PCA   | Medium   | Sensitive to feature quality           |
| Logistic (OvR)   | Raw        | Very Low | Weak prediction across all classes     |
| Logistic (OvR)   | PCA        | Medium   | PCA improves OvR significantly         |

---

## 🖼️ Visual Outputs

- Confusion matrices for each model
- HOG image visualizations
- PCA explained variance plot
  
---

## 🧪 Results Summary

- SVM with PCA reached up to **90% accuracy**
- PCA significantly boosts performance across models
- Feature extraction + PCA is crucial for good ML performance on image data

---
