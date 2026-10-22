# Audit of the 10 public notebooks in `OCT_15_notebook/` (2026-10-15)

All ten are inference-only blends of a handful of public checkpoints; none trains anything. Read with the public
split's noise in mind: nartaa reports an SD of roughly **0.003** on the public split (discussion 749359), so the
0.954 / 0.956 / 0.957 labels below are mostly the same blend family inside noise. Kaggle shows three decimals; heliosli's
"0.9573" is a fourth digit nobody can read off the board.

## The members they share

| member | dataset | what it is | public LB alone |
|---|---|---|---|
| Raptor-384 | `nartaa/rsna-knee-publication-swa-weights-20261007` | CoAtNet-rmlp-2 SWA (epochs 14,11,12), 96-slice 5-slot stack, 94 windows, anatomical L/R mirror TTA; trained with 2,399 OAI knees; gold-58 used for checkpoint selection | 0.949 |
| Raptor-224 | same dataset, second file | same recipe, 0.8 centre crop resized to 224 | 0.945 |
| R7 | `heliosli/rsna-knee-compact-r7cnxfsss` = `heliosli/rsna-knee-r6-frozen-fe055406` (same bundle `8c7b666e27dc0e8f`) | ConvNeXt-small, 4 folds, 384/352 px knee-anchored views (SKM-TEA V-Net), fastMRI auxiliary loss, Effusion fluid stacker + Baker's state stacker, torch.compile, no flip TTA; gold never trained | not published |
| S6 | `heliosli/rsna-knee-compact-v1` (bundle `209a32c8b50a6b1c`) | the earlier ConvNeXt-small bundle, same views, no stackers | not published (0.954 with Raptor-384) |
| S6F | `nartaa/rsna-knee-s6fall-weights` | heliosli's S6 architecture retrained by nartaa on JEV-blend labels, all 4,263 report studies, 320/288 px, no OAI, gold excluded | **0.950** |
| 2.5D reader | `goodpjw2008/rsna-knee-2-5d-convnext-reader` | ConvNeXt-Tiny, 3 folds, 3-slice windows + 2-layer study transformer | 0.929 |
| knee-centred legs | `bliverrigpdoge/rsna-knee-kcd25leg` (**private**), `bliverrigpdoge/rsna-knee-leg-r9all` (public CoAtNet `R9_all`) | knee-centred CoAtNet / ConvNeXt / EffNetV2 crops | not published |

## Notebook by notebook

| file | author | members and weights | claim | verdict |
|---|---|---|---|---|
| `0-9573-latest-public-dataset-score-in-one-click` | heliosli | CoAtNet slot 0.5 = 0.65 R384 + 0.35 R224 (no mirror on 224); R7 0.5 | 0.9573 | **the reference public recipe**; reproduced as the fallback of our blend |
| `rsna-gold-medal` | lavinwins | code identical to the row above | title 0.950 | duplicate of heliosli's recipe |
| `rsna-knee-abnormality-pipeline` | mayankraaz | R384 + R7 50/50, R224 at 35 % of the CoAtNet slot | 0.956+ | same recipe |
| `rsna-knee-abnormality-detection` | degnonguidi | R384 + R7 50/50 | 0.956 | the two-model base |
| `rsna-versia-7` | evgendvorkin | R384 + R7 50/50 | 0.956 | the two-model base |
| `r7-raptor-knee-centred-convnext` | bliverrigpdoge | 0.8 × (R384 + R7) + 0.2 × private knee-centred legs | 0.957 | the +0.001 comes from a **private** dataset; a fork silently falls back to the 0.956 base |
| `rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954` (v20) | goodpjw2008 | CoAtNet slot 0.65 R384 + 0.35 R224 **with mirror**; ConvNeXt slot 0.6 R7 + 0.4 S6; then the 2.5D reader at 1/10 of its per-finding weights | 0.956 ("top 10 % of the 0.956 block") | honest write-up; shows S6 and the reader add little on top of R7 |
| `rsna-knee-research-optimized` | yuvrajlamba01 | R384 + R224 + R7 + r9all with a 156 mm *horizontal-flip* pass, z-logit hybrid, hand-set "co-occurrence smoothing" | ">0.960 target" | ablation table and per-target AUCs are not backed by any run in the notebook; the horizontal flip swaps medial/lateral; not used |
| `rsna-knee-0-956-micro-boost-clinical-tri-reader` | rabari9999 | R384 + R224 + 2.5D reader, probit fusion, temperature 1.02, co-occurrence priors | 0.956+ | the CV/AUC tables are unverifiable and the post-processing cannot change ranks the way described; not used |
| `rsna_knee_s6fall_solo.py` | nartaa | S6F alone (Raptor-384 only for a study S6F cannot read) | 0.950 | strongest single public model; **new member** in our blend |

## What we took

1. heliosli's 0.957 recipe unchanged as the base (and as the exact fallback when a newer member fails).
2. The anatomical mirror on the 224 model (goodpjw2008 / varshithakolupuri "one free flip").
3. nartaa's S6F, run through nartaa's own pinned code tree and launcher, as the second ConvNeXt member: it scores
   0.950 alone and was trained on a different label set than R7, unlike heliosli's S6, which four-model public blends
   showed adds nothing over R7 (rudrpratapsingh: 0.956 with S6 at 50 % of the ConvNeXt slot).

Not taken: the private knee-centred legs (not attachable), the probit/temperature/co-occurrence post-processing (cannot
help a rank metric in the way claimed, never measured), the horizontal flip (breaks the four lateralised findings), and
the 2.5D reader (no measurable gain at any weight on top of R7 per its own author).

## Runtime facts worth keeping

- The ConvNeXt pipelines are **GPU-bound**, not decode-bound. The 25-45 s `decode_s` in 3-study commit logs is each
  spawned loader worker's one-off setup (anatomy V-Net, CUDA init) landing on its first study. Measured by us
  (2026-10-15): heliosli's mainline decode + geometry costs ~8 s/study cold in one process and ~3.25 s/study of
  throughput with 4 processes; on the 58 gold studies R7 needs ~12 s of GPU per study per T4 (+5 min compile), S6F
  ~8 s, Raptor-384 ~1.5 s, Raptor-224 ~0.8 s. Four members take ~4.5 h on the hidden test; a fifth compiled ConvNeXt
  bundle (~2.3 h) still fits.
- The Raptor scripts read with 16-32 I/O threads because Kaggle's input mount is FUSE-latency-bound; they finish the
  hidden test in about 20 minutes (they log every 200 studies, "~3 min at 1x").
- `heliosli/rsna-knee-compact-r6cnxfsss` is a non-versioned dataset and blocks submission; `-compact-v1` and
  `-r6-frozen-fe055406` are versioned copies.
- Hidden test = 1,284 studies (nartaa's fallback caps).
