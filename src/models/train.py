import pandas as pd, numpy as np, mlflow, yaml, json
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve
from xgboost import XGBClassifier
from pathlib import Path

cfg = yaml.safe_load(Path("configs/params.yaml").read_text())
df = pd.read_parquet(cfg["data"]["input_path"])

# Simple label: large billed-to-paid ratio suggests risk (demo only)
y = (df["billed"] > df["paid"] * 1.5).astype(int)
X = df[["billed","paid","units"]].copy()
X["ratio"] = (df["billed"] / (df["paid"] + 1e-6)).clip(0, 20)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=cfg["train"]["test_size"], random_state=cfg["seed"])

mlflow.set_experiment(cfg["experiment"])
Path("out").mkdir(exist_ok=True)
with mlflow.start_run():
    model = XGBClassifier(**cfg["model"]["params"])
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:,1]
    ap = average_precision_score(yte, proba)
    mlflow.log_metric("avg_precision", ap)

    # Choose threshold by PR curve for demo
    p, r, t = precision_recall_curve(yte, proba)
    best_idx = int(np.argmax(0.5 * (p + r)))
    thresh = float(t[max(0, best_idx-1)]) if len(t) else cfg["infer"]["threshold"]

    mlflow.sklearn.log_model(model, "model")
    Path("out/metrics.json").write_text(json.dumps({"avg_precision": ap, "threshold": thresh}, indent=2))
    Path("out/threshold.txt").write_text(str(thresh))
    print({"avg_precision": ap, "threshold": thresh})
