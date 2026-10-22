# Verified run-2 analysis: official 0.877

## Evidence boundary

The official score **0.877** was supplied for the completed second run. The execution log
`/Users/bikashkumarsah/Downloads/download (2).txt` completed in 254.2 minutes. The attached
`results.zip` is 952,543,101 bytes with SHA-256
`96ed139413359eda2bc92a9aeb63ec7d4b07bc958a4f8ffc749073df43963fc2`.

The archive passed a full CRC test. All ten checkpoint hashes, the manifest hash, label hash,
and three submission hashes reproduce their recorded receipts. It contains five complete
detail folds, five complete depth folds, 4,407 OOF rows, the corrected label table, and the
selected 70/30 rank-blend submission.

## Aggregate results

| System | All-teacher OOF | Legacy explicit receipt | Gold-58 |
|---|---:|---:|---:|
| Detail, 336 px | 0.83664 | 0.86200 | 0.84145 |
| Depth, 224 px | 0.83007 | — | 0.82132 |
| 70/30 detail/depth | **0.83959** | **0.86479** | 0.84054 |

The legacy explicit receipt used a raw `verdict != "UNK"` mask, which also counted the
twelve blank verdict cells in the appended gold-only coverage row. Run 3 normalizes those
gold-backed blanks to `UNK`; its strictly explicit `YES`/`NO` anchor is therefore **0.86512**.
The legacy 0.86479 value remains stored solely to verify the immutable run-2 lineage.

The 70/30 blend improved all twelve all-teacher target AUCs and macro AUC in every fold. Its
all-teacher gain over detail was +0.00295; the stored 1,000-sample report-group bootstrap
lower 95% bound was +0.00223. Gold changed by only -0.00091. The official improvement from
the earlier 0.835 run was +0.042.

## Target-level diagnosis

| Target | Detail | Depth | 70/30 | Blend gain | Gold 70/30 | UNK rate |
|---|---:|---:|---:|---:|---:|---:|
| ACL | 0.8293 | 0.8140 | 0.8307 | +0.0015 | 0.8701 | 8.1% |
| MCL | 0.7989 | 0.7836 | 0.8003 | +0.0014 | 0.8390 | 9.8% |
| Medial Meniscus | 0.8352 | 0.8196 | 0.8369 | +0.0017 | 0.8005 | 5.7% |
| Lateral Meniscus | 0.7833 | 0.7739 | 0.7863 | +0.0030 | **0.6410** | 10.0% |
| Medial OA | 0.8621 | 0.8562 | 0.8633 | +0.0013 | 0.9225 | 25.5% |
| Lateral OA | 0.8138 | 0.8065 | 0.8156 | +0.0018 | 0.7621 | 33.1% |
| PF OA | 0.7991 | 0.7964 | 0.8012 | +0.0022 | 0.8031 | 18.5% |
| Effusion | 0.8473 | 0.8466 | 0.8497 | +0.0023 | 0.9503 | 9.8% |
| Synovitis | 0.8689 | 0.8645 | 0.8712 | +0.0023 | 0.7587 | **84.2%** |
| Baker's | 0.8880 | 0.8836 | 0.8939 | +0.0059 | 0.9438 | **45.9%** |
| Contusion | 0.8370 | **0.8435** | 0.8449 | **+0.0078** | 0.8718 | 20.8% |
| Fracture | 0.8767 | 0.8725 | 0.8810 | +0.0043 | 0.9236 | **56.4%** |

Detail is the stronger individual model for eleven targets. Depth is nevertheless useful:
contusion is independently stronger, and Baker's/fracture have the lowest detail-depth rank
correlations. This supports keeping run 2 as the anchor and spending run 3 on a learned
through-plane diversity arm rather than retraining two correlated banks.

## Training behavior and run-3 consequences

- Nine of ten folds peaked at epoch 8 or 9; simply shortening training is not justified.
- The depth cache decoded all 20,130 selected slots and all 4,407 studies, so its lower OOF
  score is not a missing-data failure.
- Training used one random window per batch, whereas validation/inference used all windows
  with fixed post-sigmoid pooling. Run 3 replaces this mismatch with differentiable ordered
  window attention and evaluates all ten windows at every OOF checkpoint.
- `UNK` fields were previously soft negative targets. Strict explicit-label re-evaluation
  raises the anchor OOF from 0.83959 to 0.86512, demonstrating that unknown report fields should not
  define the validation target or receive production loss.

The byte-identical run-2 notebook is archived as
`10_experiments/BEST_NOTEBOOK_RSNA_KNEE_0877_2026-08-23.ipynb` with SHA-256
`c2c2a1be7c45790ac56be4408ed7d0705425afa9e09096573f8d4cf08376cc9f`.
