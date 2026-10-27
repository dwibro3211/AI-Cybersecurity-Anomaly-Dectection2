import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/real_traffic_sample.csv")
df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()

features = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow IAT Mean",
    "Destination Port"
]

X = df[features].apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
y = (df["Label"] != "BENIGN").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_train_normal = X_train.loc[y_train == 0]

preprocessor = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

X_normal_scaled = preprocessor.fit_transform(X_train_normal)
X_test_scaled = preprocessor.transform(X_test)

model = IsolationForest(
    n_estimators=200,
    contamination=0.03,
    random_state=42,
    n_jobs=-1
)

model.fit(X_normal_scaled)
predictions = (model.predict(X_test_scaled) == -1).astype(int)
scores = model.decision_function(X_test_scaled)

print("\nANOMALY SCORE ANALYSIS")
print("Normal traffic:")
print(pd.Series(scores[y_test.to_numpy() == 0]).describe())

print("\nAttack traffic:")
print(pd.Series(scores[y_test.to_numpy() == 1]).describe())
print("\nREAL NETWORK TRAFFIC RESULTS")
print("Training normal records:", len(X_train_normal))
print("Testing records:", len(X_test))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification report:")
print(classification_report(
    y_test, predictions,
    target_names=["BENIGN", "ATTACK"],
    zero_division=0
))

Path("outputs").mkdir(exist_ok=True)
pd.DataFrame({
    "actual_attack": y_test.to_numpy(),
    "predicted_attack": predictions
}).to_csv("outputs/real_model_predictions.csv", index=False)

print("Saved outputs/real_model_predictions.csv")
