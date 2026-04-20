from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_rf(X_train, y_train):
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Improved Random Forest
    model = RandomForestClassifier(
        n_estimators=200,          # more trees better performance
        max_depth=10,              # control overfitting
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",   # IMPORTANT for recall
        random_state=42,
        n_jobs=-1                  # use all CPU cores
    )

    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "models/rf_model.pkl")

    return model