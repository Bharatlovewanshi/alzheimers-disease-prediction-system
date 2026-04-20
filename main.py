from src.data_loader import load_data
from src.preprocessing import balance_data
from src.train_logistic import train_logistic
from src.train_svm import train_svm
from src.train_dt import train_dt
from src.train_rf import train_rf
from src.train_xgb import train_xgb
from src.evaluate import evaluate

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import joblib
import os

os.makedirs("models", exist_ok=True)

# 1. Load Data
X, y = load_data("data/alzheimers_disease_data.csv")

print("\n📊 TOTAL DATASET SIZE:", len(X))
print("Class Distribution:\n", y.value_counts())

# Save feature columns
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

# 2. Under Sampling
USE_UNDERSAMPLING = True

if USE_UNDERSAMPLING:
    X, y = balance_data(X, y)

print("\n📊 AFTER SAMPLING SIZE:", len(X))
print("Class Distribution:\n", y.value_counts())

# 3.Multiple Test Sizes
test_sizes = [0.2, 0.25, 0.3]

all_results = {}

for test_size in test_sizes:
    print(f"\n\n==============================")
    print(f"TEST SIZE: {test_size}")
    print(f"==============================")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=42
    )

    # Print counts
    print(f"\n📦 Total Samples: {len(X)}")
    print(f"🟢 Train Samples: {len(X_train)}")
    print(f"🔵 Test Samples: {len(X_test)}")

    print("\nTrain Class Distribution:\n", y_train.value_counts())
    print("\nTest Class Distribution:\n", y_test.value_counts())

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save scaler (last one used for UI)
    joblib.dump(scaler, "models/scaler.pkl")

    # Train models
    models = {
        "Logistic": train_logistic(X_train, y_train),
        "SVM": train_svm(X_train, y_train),
        "DecisionTree": train_dt(X_train, y_train),
        "RandomForest": train_rf(X_train, y_train),
        "XGB": train_xgb(X_train, y_train),
    }

    results = {}

    for name, model in models.items():
        print(f"\n===== {name} =====")
        acc, rec = evaluate(model, X_test, y_test)
        results[name] = {"Accuracy": acc, "Recall": rec}

    all_results[test_size] = results

# 4. Final Comparison
print("\n\n===== FINAL COMPARISON =====")

best_overall = None
best_score = 0

for test_size, models in all_results.items():
    print(f"\n--- Test Size: {test_size} ---")
    for name, metrics in models.items():
        print(f"{name} → Acc: {metrics['Accuracy']:.4f}, Rec: {metrics['Recall']:.4f}")

        if metrics["Recall"] > best_score:
            best_score = metrics["Recall"]
            best_overall = (test_size, name)

print(f"\n🔥 BEST CONFIG:")
print(f"Test Size: {best_overall[0]}, Model: {best_overall[1]}")

# 5. Retrain Best Model
best_test_size = best_overall[0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=best_test_size,
    stratify=y,
    random_state=42
)

print("\n📦 FINAL TRAINING")
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")

final_models = {
    "Logistic": train_logistic(X_train, y_train),
    "SVM": train_svm(X_train, y_train),
    "DecisionTree": train_dt(X_train, y_train),
    "RandomForest": train_rf(X_train, y_train),
    "XGB": train_xgb(X_train, y_train),
}

best_model = final_models[best_overall[1]]

joblib.dump(best_model, "models/best_model.pkl")