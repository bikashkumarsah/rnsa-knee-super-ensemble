# External Asset Licenses

Winner-stage disclosure list (§2.5 / §2.8 of `COMPETITION_RULES.md`). Every external dataset, model, or tool that feeds a submission gets a row here. Assets with licenses **incompatible** with the CC-BY-NC 4.0 winner-license duty are usable under §2.5a's explicit exemption but must be *identified* — they cannot be shipped as part of your licensed code.

Status legend: `OK` = OSI-approved, no commercial-use limits (satisfies §3.6c) · `IDENTIFY` = incompatible/unclear license → exempt-but-disclose under §2.5a · `CHECK` = verify before prize stage.

## Pretrained models

| Asset | License | Status | Used in | Notes |
|---|---|---|---|---|
| DINOv2 ViT-S/B (Meta AI) | Apache-2.0 | OK | `04_dino-single-model/*`, all ensembles | redistributes cleanly as Kaggle dataset |
| DINOv3 ViT-S/16 (Meta AI) | DINOv3 License (gated access) | IDENTIFY | `05_ensembles/dinov3-hybrid/*` | approval-gated weights; identify-only at winner stage |
| RadImageNet ResNet-50 (official **weights** + training code, BMEII-AI/RadImageNet) | **MIT** (verified from repo LICENSE, © 2021 BMEII-AI) | OK | `05_ensembles/read-the-report-lineage` (E10/E13/E11 arms), champions | code+pretrained weights are MIT: commercial use, modification, redistribution all permitted; keep copyright notice. The *database* behind them is request-access via radimagenet.com — but we use only the released checkpoints, not the database, so the DB terms don't attach. Cite Mei et al., Radiol: Artif Intell 2022 (doi:10.1148/ryai.210315). Third-party Kaggle re-uploads of the .h5/.pt files inherit MIT; prefer the official source when packaging winner weights. |
| timm backbones (EffNet-B3 etc.) | Apache-2.0 | OK | baselines, B3 arm | |
| Qwen/Qwen3-14B (report reader, off-notebook) | **Apache-2.0** (verified on HuggingFace model card) | OK | label generation only (never ships in notebook) | output is derived labels, not the model itself |

## Public notebooks & code (all shared on this competition's Kaggle → OSI-deemed per §3.6b)

| Asset | Author | License basis | Used in |
|---|---|---|---|
| rsna-knee-baseline-v1 + rsna-knee-weights (20-member package) | pilkwang | public competition code (§3.6b) | `05_ensembles/pilkwang-anchor-lineage`, champions' parent |
| Bend the Knee to DINOv3 (ensembled) V104 | Mattia Angeli | Apache-2.0 (stated in notebook) | `05_ensembles/champions/head-and-shoulders…` base |
| DINOsaur V3 fold-rank aggregation | romantamrazov | public competition code | adaptive second-pass weighting in champions |
| E10/E11/E13 RadImageNet machinery | cf696666 (fishface), sofiaanjenje, ieshanmeghani | public competition code | RadImageNet arms |
| report-label tables (LLM-read labels v2) | pilkwang, stevenleehans, lixin73 | public datasets | target generation |

## Super-ensemble members (from 2026-10-15; every row feeds `BEST_NOTEBOOK_RSNA_KNEE.ipynb`)

| Asset | Author | License / terms | Status | Notes |
|---|---|---|---|---|
| `nartaa/rsna-knee-publication-swa-weights-20261007` (Raptor CoAtNet 384 + 224 SWA) | nartaa (Danial Zakaria), dreaddevelopment recipe | research/educational use incl. this competition (dataset README); third-party code keeps its licences | IDENTIFY | trained with 2,399 OAI knees: OAI acknowledgement required and carried in the notebook; OAI eligibility under rule 2.6.b still unanswered by the hosts (forum topic 749145) |
| `nartaa/rsna-knee-s6fall-weights` (S6F ConvNeXt-small, all data) | nartaa, heliosli architecture | CC BY-NC 4.0 (weights); Apache-2.0 code extras | IDENTIFY | no OAI data; gold-58 excluded from training |
| `heliosli/rsna-knee-r6-frozen-fe055406` (= R7 bundle `8c7b666e27dc0e8f`) | heliosli | Kaggle dataset licence "other" | CHECK | fastMRI auxiliary training (heliosli's notebook); "No KneeCoT inputs or derivatives; gold never trained" (manifest) |
| `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` (decode code, SKM-TEA anatomy V-Net, wheels) | heliosli; SKM-TEA by Desai et al. | Kaggle dataset licence "other"; vendored MONAI 0.6.0 (Apache-2.0) | CHECK | only the code tree and `anatomy.pt` are used, not the OrthoFoundation weights |
| Public notebook code (Raptor scripts, R7 setup, S6F launcher) | nartaa, heliosli, goodpjw2008, varshithakolupuri | public competition code (§3.6b) | OK | vendored byte-for-byte in `super_ensemble/vendor/`, changes marked `[se]` |

## Data-derived assets

| Asset | License | Status | Notes |
|---|---|---|---|
| Competition DICOMs/reports | RSNA MIRA + competition rules | n/a (competition use) | never redistribute; see `COMPLIANCE_CHECKLIST.md` A |
| Barun's processed RSNA knee volumes (24-slice cache) | CC BY-NC-SA 4.0 | IDENTIFY | used by specialist arm (`head-and-shoulders`); attribution required, no ODC shipping |
| Report-label CSVs derived by me/team | own work | OK | derived from permitted report reading |

## Maintenance rule

Add a row **the moment** a new external asset enters any experiment (one line in `10_experiments/EXPERIMENT_LOG.md` + one row here). At winner stage this file becomes the disclosure appendix — keeping it current as you go is minutes of work; reconstructing it in November is days.
