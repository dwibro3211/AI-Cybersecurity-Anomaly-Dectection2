# AI-Powered Cybersecurity Anomaly Detection

An educational machine learning portfolio project demonstrating how unsupervised algorithms can identify unusual patterns in **synthetic** network traffic.

## What it does

- Generates 2,500 reproducible simulated traffic records, including 125 deliberately unusual examples.
- Standardizes six numerical network features.
- Fits **Isolation Forest** to flag anomalies, **K-Means** to discover groups, and **DBSCAN** to find density-based noise points.
- Uses **PCA** to visualize high-dimensional data in two dimensions.
- Compares two anomaly-flagging methods against the *synthetic* ground-truth labels with precision, recall and F1.
- Exports a chart, evaluation table and analyzed CSV.

## Run locally

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/run_analysis.py
```

Outputs are written to `outputs/`. Input data is generated in `data/` on each run with a fixed seed.

## Project structure

- `src/run_analysis.py` — data generation, preprocessing, training, evaluation, plotting
- `data/synthetic_network_traffic.csv` — generated synthetic input data
- `outputs/analyzed_traffic.csv` — records with model results
- `outputs/evaluation_metrics.csv` — precision, recall and F1
- `outputs/anomaly_visualization.png` — PCA projection of flagged records

## Model roles

| Technique | Role |
| --- | --- |
| Isolation Forest | Unsupervised anomaly detection |
| K-Means | Clustering into three groups; cluster IDs are not attack labels |
| DBSCAN | Density-based clustering; label -1 is treated as a noise flag |
| PCA | Two-dimensional visualization, not an attack classifier |

## Limitations and responsible interpretation

This is a **demonstration, not a deployed intrusion detection system**. The records are artificially generated, and anomalous does not necessarily mean malicious. The labels were created during synthetic data generation and are used only for demonstration metrics, not to train the unsupervised models. Performance on this dataset does not establish performance on real network traffic. No cloud infrastructure, production deployment, real traffic capture, or multi-GPU training is included.

## Suggested next improvements

1. Compare against an appropriately licensed, publicly available intrusion-detection dataset.
2. Introduce separate training and evaluation datasets and tune parameters without using evaluation labels.
3. Add a reproducible notebook, unit tests, and a simple read-only dashboard.
4. Write up false positives, feature selection, and practical deployment considerations.

## Skills demonstrated

Python, Pandas, NumPy, scikit-learn, feature scaling, unsupervised machine learning, anomaly detection, visualization, evaluation, and technical documentation.

**Portfolio note:** Review and run the code yourself before describing this as a completed personal project in an application.
