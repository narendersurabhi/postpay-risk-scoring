import numpy as np, pandas as pd, pathlib
from datetime import timedelta

pathlib.Path("data").mkdir(exist_ok=True)
rng = np.random.default_rng(42)

# Providers
prov = pd.DataFrame({
    "provider_id": np.arange(1200),
    "specialty": rng.choice(["IM","PED","CAR","DERM","DEN","OPT","ORTHO"], 1200),
    "zip": rng.integers(10000, 99999, 1200).astype(str)
})

# Members
mem = pd.DataFrame({
    "member_id": np.arange(6000),
    "age": rng.integers(0, 90, 6000),
    "zip": rng.integers(10000, 99999, 6000).astype(str)
})

# Claims
n = 25000
start = pd.Timestamp("2023-01-01")
codes = ["99213","99214","D1110","V2100","93000","72148","J1100","97110","81002"]
claims = pd.DataFrame({
    "claim_id": np.arange(n),
    "provider_id": rng.integers(0, len(prov), n),
    "member_id": rng.integers(0, len(mem), n),
    "proc_code": rng.choice(codes, n, p=np.array([0.2,0.18,0.1,0.06,0.14,0.08,0.1,0.08,0.06])),
    "units": rng.integers(1, 6, n),
    "dos": start + pd.to_timedelta(rng.integers(0, 640, n), unit="D"),
    "billed": np.round(rng.gamma(2.4, 85, n), 2)
})
claims["paid"] = np.round(claims["billed"] * rng.uniform(0.6, 1.05, n), 2)

# Inject anomalies
mask = rng.random(n) < 0.035
claims.loc[mask, "billed"] *= rng.uniform(2.0, 4.0, mask.sum())

# Save
claims.to_parquet("data/synth_claims.parquet", index=False)
prov.to_parquet("data/providers.parquet", index=False)
mem.to_parquet("data/members.parquet", index=False)
claims.sample(10, random_state=0).to_csv("data/sample.csv", index=False)
print("Wrote data/synth_claims.parquet, providers.parquet, members.parquet, and data/sample.csv")
