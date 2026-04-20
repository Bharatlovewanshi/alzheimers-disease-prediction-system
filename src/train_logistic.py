from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_logistic(X_train, y_train):
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Improved Logistic Regression
    model = LogisticRegression(
        solver="liblinear",        # good for small/medium datasets
        class_weight="balanced",   # handle imbalance (important for recall)
        max_iter=1000,             # avoid convergence issues
        random_state=42
    )

    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "models/logistic_model.pkl")

    return model