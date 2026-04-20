from sklearn.tree import DecisionTreeClassifier
import joblib
import os

def train_dt(X_train, y_train):
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Improved Decision Tree
    model = DecisionTreeClassifier(
        max_depth=5,                # control overfitting
        min_samples_split=10,       # avoid very small splits
        min_samples_leaf=5,         # smoother tree
        class_weight="balanced",    # handle imbalance (IMPORTANT)
        random_state=42
    )

    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "models/dt_model.pkl")

    return model