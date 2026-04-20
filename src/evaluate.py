from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix

def evaluate(model, X_test, y_test, threshold=0.5):
    
    # Use probability if available (better control)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Print results
    print(f"Accuracy: {acc:.4f}")
    print(f"Recall: {rec:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return acc, rec