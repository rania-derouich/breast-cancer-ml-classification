"""
Breast Cancer Classification using Multiple ML Models
Dataset: Wisconsin Breast Cancer Diagnostic
Author: Rania Derouich
Date: 2024-12-31
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.utils import shuffle

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
def load_data(file_path):
    """
    Load and prepare the dataset.

    Args:
        file_path (str): Path to CSV file

    Returns:
        pd.DataFrame: Cleaned dataset
    """
    df = pd.read_csv(file_path)

    # Drop non-informative columns
    columns_to_drop = ['id', 'Unnamed: 32']
    for col in columns_to_drop:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Encode target variable
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    return df


# -------------------------------------------------------------------
# Preprocessing
# -------------------------------------------------------------------
def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Preprocess data and split into train/test sets.

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']

    X, y = shuffle(X, y, random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# -------------------------------------------------------------------
# Neural network model
# -------------------------------------------------------------------
def build_neural_network(input_dim):
    """
    Build and compile a neural network model.
    """
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a model and print metrics.
    """
    y_pred = model.predict(X_test)

    # Handle Keras output
    if y_pred.ndim > 1:
        y_pred = (y_pred > 0.5).astype(int).ravel()

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"{model_name} Evaluation")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'predictions': y_pred
    }


# -------------------------------------------------------------------
# Visualization
# -------------------------------------------------------------------
def plot_feature_importance(model, feature_names, title):
    """
    Plot feature importance for tree-based models.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), feature_names[indices])
    plt.xlabel("Relative Importance")
    plt.title(title)
    plt.tight_layout()


def plot_confusion_matrix(y_true, y_pred, title):
    """
    Plot confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Benign', 'Malignant'],
        yticklabels=['Benign', 'Malignant']
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    print("Breast Cancer Classification Project")
    print("=" * 50)

    import os
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Load data
    df = load_data('data/breast_cancer.csv')
    feature_names = df.drop(columns=['diagnosis']).columns

    # Preprocessing
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Models
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    gb_model.fit(X_train, y_train)

    nn_model = build_neural_network(X_train.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = nn_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    voting_clf = VotingClassifier(
        estimators=[('rf', rf_model), ('gb', gb_model)],
        voting='soft'
    )
    voting_clf.fit(X_train, y_train)

    # Evaluation
    results = {}
    results['Random Forest'] = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    results['Gradient Boosting'] = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")
    results['Neural Network'] = evaluate_model(nn_model, X_test, y_test, "Neural Network")
    results['Voting Classifier'] = evaluate_model(voting_clf, X_test, y_test, "Voting Classifier")

    # Feature importance
    plot_feature_importance(rf_model, feature_names, "Random Forest Feature Importance")
    plt.savefig('results/feature_importance.png', dpi=150)
    plt.close()

    # Best model confusion matrix
    best_model = max(results, key=lambda x: results[x]['accuracy'])
    plot_confusion_matrix(y_test, results[best_model]['predictions'],
                          f"Confusion Matrix - {best_model}")
    plt.savefig('results/confusion_matrix.png', dpi=150)
    plt.close()

    # Save models
    import joblib
    joblib.dump(rf_model, 'models/random_forest.pkl')
    joblib.dump(gb_model, 'models/gradient_boosting.pkl')
    joblib.dump(voting_clf, 'models/voting_classifier.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    nn_model.save('models/neural_network.h5')

    print("\nProject completed successfully.")


if __name__ == "__main__":
    main()
