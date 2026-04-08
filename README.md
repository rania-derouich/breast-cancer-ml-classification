# quantum-ai-breast-cancer-prediction

This project implements and compares multiple classical machine learning models
and a neural network for breast cancer diagnosis using the
**Wisconsin Breast Cancer Diagnostic dataset**.

The goal is to benchmark different models, evaluate their predictive performance,
and provide a clean, reproducible ML pipeline suitable for academic and applied use.

---

## 📊 Dataset

- **Source**: Wisconsin Breast Cancer Diagnostic Dataset
- **Task**: Binary classification (Benign vs Malignant)
- **Target variable**: `diagnosis`
  - Malignant → 1
  - Benign → 0

Non-informative identifiers (e.g., `id`) are removed during preprocessing.

---

## 🧠 Models Implemented

- Random Forest
- Gradient Boosting
- Neural Network (Keras / TensorFlow)
- Soft Voting Classifier (ensemble)

---

## ⚙️ Pipeline Overview

1. Data loading and cleaning  
2. Feature scaling using StandardScaler  
3. Train / test split (80/20)  
4. Model training  
5. Performance evaluation  
6. Visualization and model persistence  

---

## 📈 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

---

## 📂 Outputs

- Feature importance plots
- Confusion matrix
- Trained models saved for reuse
- Scaler for reproducibility

All outputs are stored in the `results/` and `models/` directories.

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python main.py
