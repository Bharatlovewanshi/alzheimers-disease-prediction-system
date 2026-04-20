import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    # Safety check (good practice)
    if "Diagnosis" not in df.columns:
        raise ValueError("Target column 'Diagnosis' not found in dataset")

    # Features and target
    X = df.drop("Diagnosis", axis=1)
    y = df["Diagnosis"]

    return X, y