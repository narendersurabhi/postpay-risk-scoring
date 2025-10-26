import pandas as pd
from pathlib import Path

inp = Path("data/synth_claims.parquet")
out = Path("out/preds.parquet")
thr_path = Path("out/threshold.txt")

df = pd.read_parquet(inp)
# Heuristic score for demo (no model load to keep CI fast)
df["risk_score"] = (df["billed"] / (df["paid"] + 1e-6)).clip(0, 20)

thr = 0.7
if thr_path.exists():
    try:
        thr = float(thr_path.read_text().strip())
    except Exception:
        pass

df["triage_flag"] = (df["risk_score"] >= thr).astype(int)
df[["claim_id","risk_score","triage_flag"]].to_parquet(out, index=False)
print(f"Wrote {out}")
