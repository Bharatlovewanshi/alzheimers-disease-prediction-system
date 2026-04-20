from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os

def train_xgb(X_train, y_train):
    
    os.makedirs("models", exist_ok=True)

    model = GradientBoostingClassifier(
        n_estimators=200,     # more trees
        learning_rate=0.05,   # slower learning = better generalization
        max_depth=3,          # controls overfitting
        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "models/xgb_model.pkl")

    return model