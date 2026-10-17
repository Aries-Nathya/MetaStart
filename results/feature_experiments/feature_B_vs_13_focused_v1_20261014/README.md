# B versus 13-feature sensitivity analysis

This experiment compares the 15-feature B configuration with a 13-feature
variant that omits median absolute numeric skewness and median absolute excess
kurtosis. It is a repeated-seed sensitivity analysis, not an independent
replication on new datasets.

## Protocol

- Ten eligible dataset identities: four local real datasets and six synthetic
  datasets from the same generator family.
- Fifty-one fixed model seeds, with the same leave-one-dataset-identity-out
  folds and baseline draws for both configurations.
- Results are averaged across seeds within each identity before paired
  identity-bootstrap intervals are calculated. The independent unit remains
  the dataset identity (`n = 10`), not the 51 seeds.
- Dataset profiles use the complete dataset. This evaluates dataset-level
  recommendation, not held-out-row prediction.

## Results

| Metric | B: 15 features | 13 features |
| --- | ---: | ---: |
| Mean regret (lower is better) | 0.01583 | 0.01556 |
| Top-1 hit rate | 42.5% | 43.1% |
| Top-3 hit rate | 82.0% | 76.9% |
| Real-subset mean regret (`n = 4`) | 0.01154 | 0.01202 |
| Real-subset top-3 hit rate | 79.9% | 67.2% |

The paired regret difference (13-feature variant minus B) is −0.00028, with a
95% identity-bootstrap interval of [−0.00136, 0.00045]. The results do not
establish that the reduced feature set preserves B's recommendation quality.

## Interpretation and status

The 13-feature variant is a simplification candidate for validation on a more
diverse dataset collection. Six of the ten current identities are synthetic,
so this run provides limited evidence about generalization to independently
sourced real datasets. Keep B as the research reference and A as the production
model; this experiment does not promote or retrain either model.

See `experiment_manifest.json` for the input hashes, feature definitions,
seeds, and protocol. Detailed per-seed, per-identity, paired, and real-subset
outputs are preserved alongside this note.
