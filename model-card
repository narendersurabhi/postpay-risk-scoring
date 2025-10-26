# Model Card

## Summary
Scope: healthcare claims FWA risk scoring on synthetic data. Non-clinical. Educational only.

## Intended Use
Assist analysts in triage and sampling. Human review required.

## Data
Synthetic distributions mimicking claims, providers, and members. No PHI/PII.

## Performance
Report: precision@K, lift, latency, cost per 1k claims, and calibration error.

## Safety
- Do not deploy on real data without HIPAA controls and legal review.
- Red-team features for proxy bias and leakage.

## Monitoring
- Data drift: PSI, KS tests on key fields.
- Model drift: precision@K delta, AUCPR delta, ECE.
- Bias: group metrics where legally permitted.

## Limitations
- Synthetic results may not transfer to production.
- New codes and small cohorts reduce stability.

## Governance
- Version with MLflow.
- Peer review changes via PR.
