import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from src.feature_extraction import build_feature_dataframe


def main():
    # Paths
    data_path = os.path.join("data", "urls_dataset.csv")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "url_model.pkl")

    # 1. Load data
    df = pd.read_csv(data_path)

    # Make sure columns exist
    assert 'url' in df.columns and 'label' in df.columns, "CSV must have 'url' and 'label' columns"

    # 2. Extract features
    X = build_feature_dataframe(df)
    y = df['label']

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Train model
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=None,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # 6. Save model
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
