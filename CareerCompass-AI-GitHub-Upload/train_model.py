"""Train CareerCompass AI on the supplied placement dataset."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "placementdata.csv"
MODEL_DIR = ROOT / "model"
NUMERIC_FEATURES = ["CGPA", "Internships", "Projects", "Workshops/Certifications", "AptitudeTestScore", "SoftSkillsRating", "SSC_Marks", "HSC_Marks"]
CATEGORICAL_FEATURES = ["ExtracurricularActivities", "PlacementTraining"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    MODEL_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(DATA_PATH).dropna(subset=FEATURES + ["PlacementStatus"]).copy()
    target = data["PlacementStatus"].map({"Placed": 1, "NotPlaced": 0})
    if target.isna().any():
        raise ValueError("PlacementStatus must contain only 'Placed' or 'NotPlaced'.")
    x_train, x_test, y_train, y_test = train_test_split(data[FEATURES], target, test_size=.20, random_state=42, stratify=target)
    preprocessor = ColumnTransformer([("numeric", StandardScaler(), NUMERIC_FEATURES), ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)])
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", RandomForestClassifier(n_estimators=350, max_depth=10, min_samples_leaf=3, class_weight="balanced", random_state=42))])
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    joblib.dump(pipeline, MODEL_DIR / "placement_model.joblib")
    joblib.dump({"accuracy": accuracy, "features": FEATURES, "report": classification_report(y_test, predictions, output_dict=True)}, MODEL_DIR / "model_metrics.joblib")
    print(f"Trained on {len(data)} dataset rows. Test accuracy: {accuracy:.1%}")


if __name__ == "__main__":
    main()
