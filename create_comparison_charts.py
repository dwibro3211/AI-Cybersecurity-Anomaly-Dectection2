from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score

out = Path("outputs")
rf = pd.read_csv(out / "random_forest_predictions.csv")
iso = pd.read_csv(out / "real_model_predictions.csv")

results = []
for name, df in [("Isolation Forest V2", iso), ("Random Forest", rf)]:
    y = df["actual_attack"]
    p = df["predicted_attack"]
    results.append({
        "Model": name,
        "Precision": precision_score(y, p, zero_division=0),
        "Recall": recall_score(y, p, zero_division=0),
        "F1": f1_score(y, p, zero_division=0)
    })

metrics = pd.DataFrame(results).set_index("Model")
ax = metrics.plot(kind="bar", figsize=(9, 5), rot=0)
ax.set_title("Attack Detection: Model Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig(out / "model_comparison.png", dpi=180)
plt.close()

cm = confusion_matrix(rf["actual_attack"], rf["predicted_attack"])
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["BENIGN", "ATTACK"]
)
disp.plot(cmap="Blues", values_format="d")
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig(out / "random_forest_confusion_matrix.png", dpi=180)
plt.close()

metrics.to_csv(out / "model_comparison_metrics.csv")
print(metrics.round(4))
print("\nSaved charts and metrics to outputs/")
