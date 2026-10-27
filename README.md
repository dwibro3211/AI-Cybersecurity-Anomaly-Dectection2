# AI-Powered Cybersecurity Anomaly Detection

## Project Overview

A Python-based machine learning portfolio project exploring network intrusion detection using supervised and unsupervised learning techniques.

The project evaluates network traffic patterns from the CIC-IDS2017 dataset, comparing Isolation Forest and Random Forest models for identifying potentially malicious network activity.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Machine Learning
- Network Traffic Analysis
- Data Preprocessing
- Model Evaluation

## Dataset

**Source:** [CIC-IDS2017 — University of New Brunswick](https://www.unb.ca/cic/datasets/ids-2017.html)

The Tuesday traffic dataset contains 445,909 records, including:

- 432,074 benign traffic records
- 7,938 FTP-Patator attack records
- 5,897 SSH-Patator attack records

An exploratory sample of 11,743 records was used for model development and comparison.

## Machine Learning Models

### Isolation Forest

An unsupervised anomaly detection model trained on benign traffic.

Two versions were evaluated using different network feature sets. Neither version successfully identified attack records under the default anomaly threshold in this experiment.

### Random Forest

A supervised classification model trained using labeled benign and attack records.

The model achieved the following preliminary results on the sampled test set:

| Metric | Result |
|---|---|
| Accuracy | 98.67% |
| Attack Precision | 98.34% |
| Attack Recall | 99.36% |
| Attack F1 Score | 98.84% |

**Confusion Matrix Results:**

- True Negatives: 1,466
- False Positives: 34
- False Negatives: 13
- True Positives: 2,010

## Key Findings

Random Forest substantially outperformed Isolation Forest in this experiment. The supervised classifier successfully learned distinctions between the labeled traffic categories, while Isolation Forest struggled to flag attacks as statistical anomalies.

The results demonstrate the importance of selecting appropriate models, evaluating false positives and false negatives, and understanding the limitations of anomaly detection.

## Project Limitations

The dataset was deliberately sampled and split randomly at the record level. Related network flows may appear in both training and testing data, potentially inflating reported performance.

These results are preliminary experimental findings and should not be interpreted as production intrusion-detection performance.

Future improvements include capture-separated evaluation, stronger unsupervised baselines, additional attack categories, and testing on unseen network environments.

## Visualizations

The project includes:

- Model precision, recall, and F1 comparison
- Random Forest confusion matrix
- Synthetic traffic anomaly visualization

## Running the Project

Install the Python dependencies listed in `requirements.txt`, then run:

```bash
python train_real_model.py
python train_real_model_v2.py
python train_random_forest.py
python create_comparison_charts.py
```

The real-data training scripts require `data/real_traffic_sample.csv`, which is derived from the CIC-IDS2017 Tuesday dataset and is not included in this repository.

## Project Purpose

This educational portfolio project demonstrates practical experience in Python programming, machine learning experimentation, network traffic analysis, model evaluation, and technical documentation.
