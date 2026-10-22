# Attaching inputs for the super-ensemble

The inputs are declared in `super_ensemble/kernel/{submit,gold58}/kernel-metadata.json` and attached by
`kaggle kernels push`; nothing is attached by hand.

| input | used by | checked in the notebook |
|---|---|---|
| competition `rsna-knee-abnormality-detection` | every stage | `test.csv`, `sample_submission.csv` columns |
| `nartaa/rsna-knee-publication-swa-weights-20261007` | Raptor-384, Raptor-224 | checkpoint sha256 `7e5315da…` / `3394fdd7…` |
| `heliosli/rsna-knee-r6-frozen-fe055406` | R7 | bundle id `8c7b666e27dc0e8f`, every file's sha256 from its manifest |
| `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | R7, S6F | mainline code sha256 pins, `anatomy.pt` sha256 `5904dd16…`, wheels |
| `nartaa/rsna-knee-s6fall-weights` | S6F | model sha256 `0c7297d5…`, run config, DONE marker, payload member set/size/sha |

All five are public and versioned (a non-versioned dataset such as `heliosli/rsna-knee-compact-r6cnxfsss` blocks
submission). Internet off, accelerator T4 x2 (`--accelerator NvidiaTeslaT4`, set by `kaggle_flow.py push`).

## Licences and acknowledgements

- Raptor weights (nartaa): research and competition use; trained with OAI knees — the notebook carries the required
  OAI acknowledgement. Whether OAI is allowed is still unresolved on the forum (topic 749145).
- S6F weights (nartaa): CC BY-NC 4.0, no OAI data.
- R7 and the OrthoFoundation code/anatomy checkpoint (heliosli): public datasets, SKM-TEA anatomy V-Net (Desai et al.).

The previous DINOv2 pipeline's inputs (offline DINOv2-S, label table, `results.zip`, `results_v3.zip`) are no longer
needed; that notebook is archived as `10_experiments/BEST_NOTEBOOK_RSNA_KNEE_0887_2026-08-24.ipynb`.
