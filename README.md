# Post-Pay Risk Scoring — Cohorts + Transparent Metrics

Problem
Prioritize paid claims for audit with clear, explainable scores.

Approach
- PCA + KMeans to form peer cohorts per CPT/DRG or similar groupings.
- Risk = α·z-score + β·distance-to-centroid + γ·policy-rule hits.
- Sampling utility to create compact, statistically valid pulls.

Data
Synthetic only. Generator included. No PHI/PII.

Features
- Charge vs peer median, length-of-stay deltas, rarity by specialty.
- Temporal burstiness and reversal rates.
- Geographic outliers and duplicates.

Evaluation
- Precision@K on a labeled synthetic subset.
- Lift curves vs random sampling.
- Estimated auditor-hours saved (simulation).

Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/synthetic_generator.py
````

Train + Score

```bash
python src/models/train.py --config configs/params.yaml
python src/inference/batch.py --input data/synth_paid.parquet --out out/risk_scores.parquet
```

Outputs

* `out/risk_scores.parquet` with claim_id, risk, and component breakdowns.
* `reports/` with lift curves and sampling tables.

Repo layout

```
data/
notebooks/          # 01_eda, 02_pca_kmeans, 03_eval
src/features/
src/models/
src/inference/
configs/
model-card.md
.github/workflows/ci.yml
```

Tech
Python, scikit-learn, Pandas, MLflow

> Disclaimer: Educational code and synthetic data only. No employer code, data, or confidential methods.
