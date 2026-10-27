from pathlib import Path
import pandas as pd

data_dir = Path("data")
files = list(data_dir.glob("Tuesday-WorkingHours*.csv"))

if not files:
    raise FileNotFoundError(
        "Place the CIC-IDS2017 Tuesday CSV in the data folder."
    )

file = files[0]
samples = []

for chunk in pd.read_csv(file, chunksize=50000):
    chunk.columns = chunk.columns.str.strip()
    chunk["Label"] = chunk["Label"].astype(str).str.strip()
    for label, group in chunk.groupby("Label"):
        samples.append(group.sample(
            n=min(len(group), 1000),
            random_state=42
        ))

df = pd.concat(samples, ignore_index=True)
df = df.groupby("Label", group_keys=False).head(5000)
df = df.sample(frac=1, random_state=42)

output = data_dir / "real_traffic_sample.csv"
df.to_csv(output, index=False)

print("Dataset saved:", output)
print(df["Label"].value_counts())
print("Total records:", len(df))
