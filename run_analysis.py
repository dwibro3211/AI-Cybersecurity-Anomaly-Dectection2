"""Reproducible educational network-anomaly detection demo using synthetic data."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
FEATURES = ['duration_ms', 'bytes_sent', 'bytes_received', 'packets', 'failed_logins', 'unique_dest_ports']


def generate_data(seed=42, n=2500):
    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=int)
    labels[rng.choice(n, size=125, replace=False)] = 1
    duration = rng.gamma(2.5, 120, n)
    sent = rng.lognormal(7, 0.65, n)
    received = rng.lognormal(7.3, 0.7, n)
    packets = rng.poisson(35, n).astype(float)
    failed = rng.poisson(0.25, n).astype(float)
    ports = rng.poisson(2, n).astype(float) + 1
    attack = labels == 1
    duration[attack] *= rng.uniform(2, 5, attack.sum())
    sent[attack] *= rng.uniform(3, 9, attack.sum())
    packets[attack] *= rng.uniform(3, 7, attack.sum())
    failed[attack] += rng.integers(4, 15, attack.sum())
    ports[attack] += rng.integers(5, 30, attack.sum())
    frame = pd.DataFrame(dict(zip(FEATURES, [duration, sent, received, packets, failed, ports])))
    frame['simulated_attack_label'] = labels
    frame.to_csv(DATA / 'synthetic_network_traffic.csv', index=False)
    return frame


def main():
    df = generate_data()
    x = StandardScaler().fit_transform(df[FEATURES])
    truth = df['simulated_attack_label'].to_numpy()
    forest = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    forest_flag = (forest.fit_predict(x) == -1).astype(int)
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    km_labels = kmeans.fit_predict(x)
    db_labels = DBSCAN(eps=1.5, min_samples=10).fit_predict(x)
    pca = PCA(n_components=2)
    points = pca.fit_transform(x)
    df['isolation_forest_anomaly'] = forest_flag
    df['kmeans_cluster'] = km_labels
    df['dbscan_noise'] = (db_labels == -1).astype(int)
    df['pca_1'] = points[:, 0]
    df['pca_2'] = points[:, 1]
    df.to_csv(OUT / 'analyzed_traffic.csv', index=False)
    metrics = pd.DataFrame([{'model': 'Isolation Forest', 'precision': precision_score(truth, forest_flag, zero_division=0), 'recall': recall_score(truth, forest_flag, zero_division=0), 'f1': f1_score(truth, forest_flag, zero_division=0), 'flagged': int(forest_flag.sum())}, {'model': 'DBSCAN noise heuristic', 'precision': precision_score(truth, df['dbscan_noise'], zero_division=0), 'recall': recall_score(truth, df['dbscan_noise'], zero_division=0), 'f1': f1_score(truth, df['dbscan_noise'], zero_division=0), 'flagged': int(df['dbscan_noise'].sum())}])
    metrics.to_csv(OUT / 'evaluation_metrics.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(points[forest_flag == 0, 0], points[forest_flag == 0, 1], s=12, alpha=0.5, label='Not flagged')
    ax.scatter(points[forest_flag == 1, 0], points[forest_flag == 1, 1], s=20, alpha=0.85, label='Flagged anomaly')
    ax.set(title='Isolation Forest flags in PCA projection (synthetic traffic)', xlabel='Principal component 1', ylabel='Principal component 2')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'anomaly_visualization.png', dpi=160)
    plt.close(fig)
    print('Saved synthetic dataset, analyzed traffic, evaluation metrics, and chart to data/ and outputs/')
    print(metrics.to_string(index=False))

if __name__ == '__main__':
    main()
