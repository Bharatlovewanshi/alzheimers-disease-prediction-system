from sklearn.svm import SVC
import joblib
import os

def train_svm(X_train, y_train):
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Improved SVM
    model = SVC(
        kernel="rbf",              # non-linear (better for real data)
        C=1.0,                     # regularization
        gamma="scale",             # auto scaling
        probability=True,          # needed for threshold tuning
        class_weight="balanced",   # IMPORTANT for recall
        random_state=42
    )

    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "models/svm_model.pkl")

    return model