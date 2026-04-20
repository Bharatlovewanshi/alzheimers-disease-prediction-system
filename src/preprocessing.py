from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def balance_data(X, y):
    # Combine
    df = pd.concat([X, y], axis=1)

    target_col = y.name  # safer than hardcoding "Diagnosis"

    # Split classes
    majority = df[df[target_col] == 0]
    minority = df[df[target_col] == 1]

    # Check if balancing is needed
    if len(majority) == len(minority):
        print("✅ Dataset already balanced")
        return X, y

    print("⚠️ Applying downsampling...")

    # Downsample majority
    majority_down = resample(
        majority,
        replace=False,
        n_samples=len(minority),
        random_state=42
    )

    # Combine + shuffle
    df_bal = pd.concat([majority_down, minority]).sample(frac=1, random_state=42)

    # Split back
    X_bal = df_bal.drop(target_col, axis=1)
    y_bal = df_bal[target_col]

    return X_bal, y_bal


def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    scaler = StandardScaler()

    # Keep DataFrame structure (IMPORTANT)
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X.columns
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X.columns
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler