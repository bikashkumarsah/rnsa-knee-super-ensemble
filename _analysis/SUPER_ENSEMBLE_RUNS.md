# Super-ensemble runs and measurements (2026-10-15)

Kernels (all private, pushed with `super_ensemble/kaggle_flow.py`, T4 x2):

- `bikashsah1230/rsna-knee-super-ensemble` — the submission kernel (`BEST_NOTEBOOK_RSNA_KNEE.ipynb`).
- `bikashsah1230/rsna-knee-super-ensemble-gold58` — the same notebook scoring the 58 labelled training studies.
- `bikashsah1230/rsna-knee-decode-timing` — CPU-only diagnostic of heliosli's mainline decode.

Outputs are fetched into `super_ensemble/runs/<mode>/` (not versioned in git: they are Kaggle outputs).

## Submit kernel v1 (commit run on the 3 visible studies)

All four members ran and covered all three studies; 553 s end to end (Raptor-384 54 s, Raptor-224 26 s, R7 392 s of
which ~5 min is torch.compile, S6F 79 s). Submitted as **57219200** (pending at the time of writing).

## Gold58 kernel v1 (58 labelled studies)

R7 and S6F never trained on these studies (their manifests say so); nartaa selected the Raptor checkpoints on them, so
Raptor's numbers are optimistic. 58 studies, 9-30 positives per finding: standard error ~0.02 on a macro AUC.

| candidate | macro AUC |
|---|---|
| member Raptor-384 | 0.9229 |
| member Raptor-224 | 0.9160 |
| member R7 | 0.9312 |
| member S6F | 0.9314 |
| CoAtNet slot (0.65 / 0.35) | 0.9238 |
| ConvNeXt slot (R7 + S6F) | 0.9377 |
| public 0.957 recipe (CoAtNet slot + R7, 50/50) | 0.9355 |
| **v1** (CoAtNet slot + R7/S6F slot, 50/50) | **0.9370** |
| v1 with slots 0.4 / 0.6 | 0.9391 |
| v1 with slots 0.6 / 0.4 | 0.9351 |

Mean per-finding rank correlation on these studies: Raptor-384 / Raptor-224 0.946; R7 / S6F 0.871; across families
0.856-0.880 — R7 and S6F are as different from each other as from the CoAtNets, which is why S6F is worth a slot.

Paired bootstrap (2,000 resamples of the 58 studies) of the macro-AUC change against v1:

| change | Δ macro | 95 % interval | P(Δ > 0) |
|---|---|---|---|
| CoAtNet slot 0.40 | +0.0022 | [−0.0005, +0.0053] | 0.95 |
| CoAtNet slot 0.30 | +0.0039 | [−0.0001, +0.0084] | 0.97 |
| CoAtNet slot 0.60 | −0.0019 | [−0.0052, +0.0010] | 0.11 |
| S6F share 0.3 / 0.7 of the ConvNeXt slot | −0.0019 / −0.0003 | | 0.08 / 0.43 |
| Raptor-224 share 0.20 / 0.50 of the CoAtNet slot | +0.0009 / −0.0014 | | 0.81 / 0.11 |

Per finding, the CoAtNets are stronger on Fracture (0.958 vs 0.900) and Effusion, the ConvNeXts on Lateral OA (0.884 /
0.851 vs 0.776) and PF OA — the two families are complementary, and the ConvNeXt slot is the stronger one on these 58
studies even though the selection bias favours Raptor.

## Decode diagnostic (CPU kernel)

heliosli's `native.prepare` + geometry bank per study, 214 DICOMs on average:

| condition | prepare | geometry | total |
|---|---|---|---|
| cold, straight from `/kaggle/input` | 7.50 s | 0.72 s | 8.22 s |
| after a 32-thread copy to local disk (copy 0.32 s) | 3.76 s | 0.40 s | 4.16 s |
| warm (second read) | 4.75 s | 0.79 s | 5.53 s |
| 4 cold studies at once in 4 processes | 11.18 s | 0.88 s | 12.06 s each → **3.25 s/study** throughput |

So the 25-45 s `decode_s` in 3-study commit logs is not steady-state decode: each spawned loader worker pays its setup
(anatomy V-Net, CUDA init) on its first study. On the 58 gold studies R7 reported decode 8.6 s, render 2.7 s and GPU
12.1 s per study per worker: R7 is GPU-bound. A DICOM prefetcher was built, tested locally and then dropped — it would
save almost nothing.

Hidden-test estimate for v1 (1,284 studies): ~0.8 h CoAtNets + ~2.3 h R7 + ~1.4 h S6F ≈ 4.5 h of the 9 h.

## Gold58 kernel v2 (optional members: heliosli S6, R7 with flip TTA)

| candidate | macro AUC | Δ vs v1 (bootstrap 95 %) | P(Δ > 0) |
|---|---|---|---|
| member S6 (heliosli, compiled) | 0.9267 | | |
| member R7 with flip TTA | 0.9317 (R7 0.9312; rank corr with R7 0.998) | | |
| v1: CoAtNet 0.5 / (R7 + S6F) 0.5 | 0.9369 | — | — |
| **v2: CoAtNet 0.4 / (R7 + S6F) 0.6** | **0.9390** | +0.0022 [−0.0005, +0.0053] | 0.95 |
| CoAtNet 0.4 / (R7 + S6F + S6) 0.6 | 0.9377 | +0.0009 [−0.0030, +0.0047] | 0.69 |
| CoAtNet 0.4 / (0.4 R7 + 0.4 S6F + 0.2 S6) 0.6 | 0.9386 | +0.0018 [−0.0011, +0.0048] | 0.90 |
| CoAtNet 0.4 / (R7-flip + S6F) 0.6 | 0.9391 | +0.0023 [−0.0005, +0.0055] | 0.95 |

S6 is weaker than R7 and S6F here and correlates 0.93 with R7 (S6F: 0.87), matching the public evidence that S6 adds
nothing on top of R7; flip TTA leaves R7's ranking almost unchanged at twice the GPU time. Both stay off. Decision for
submission v2: the only change from v1 is the slot weight, CoAtNet 0.4 / ConvNeXt 0.6. Timings on the 58 studies: S6
compiled 616 s (GPU 11.9 s/study/T4, like R7), R7-flip 922 s (GPU 22.7 s/study/T4).

## Public leaderboard (2026-10-15)

| submission | blend | public LB |
|---|---|---|
| 55731024 | our DINOv2 run 4 (previous best) | 0.887 |
| 57219200 | v1: CoAtNet 0.5 / ConvNeXt (R7 + S6F) 0.5 | **0.957** (team rank 240 of 6,038) |
| 57221549 | v2: CoAtNet 0.4 / ConvNeXt 0.6 | **0.957** |
| 57221968 | non-OAI hedge: R7 + S6F only | **0.955** |
| 57222270 | probe: CoAtNet 0.3 / ConvNeXt 0.7 | **0.957** (scored 7 h after submission) |

Each hidden-test rerun finished about 3.6-3.7 h after submission. v1 and v2 are indistinguishable at Kaggle's three
decimals, as expected for a 0.002 gold-58 difference against a public-split SD near 0.003. The ConvNeXt pair alone,
with no OAI-trained weights, is only 0.002 behind the full blend, which makes it a credible second final if the hosts
rule against OAI.

The slot-weight curve is flat on the public board (0.5, 0.4 and 0.3 all show 0.957), so v2's 0.4 stays the default: it
sits in the middle of the flat range and is the gold-58 choice.

## Gold-58 v4 diagnostic (2026-10-20, gold58 kernel v3)

The gold58 kernel ran R6 flip TTA, S6F, the student and goodpjw2008's reader on the 58 labelled studies; the CoAtNets'
gold outputs come from the 2026-10-15 run (S6F reproduced it exactly). R6 flip, S6F, R7, the student and the reader
never trained on these studies; Raptor's numbers are optimistic (its epochs were selected on them).

| member | gold-58 macro AUC |
|---|---|
| R6 flip TTA | **0.9366** (best single member) |
| S6F / R7 | 0.9314 / 0.9312 |
| Raptor-384 / Raptor-224 | 0.9228 / 0.9160 |
| student (two seeds) | 0.9168 |
| goodpjw2008's reader (3 folds) | 0.9105 |

| blend (heliosli's per-finding CoAtNet weights) | macro | Δ vs v3 (bootstrap 95 %) | P(Δ > 0) |
|---|---|---|---|
| v3: R6 flip 0.6 + S6F 0.4 | 0.9377 | — | — |
| frontier replica: R6 flip alone | 0.9383 | +0.0006 [−0.0034, +0.0053] | 0.59 |
| v3r: v3 + reader | 0.9378 | +0.0001 [−0.0008, +0.0010] | 0.62 |
| v4: v3 + student 0.15 | 0.9363 | −0.0014 [−0.0036, +0.0005] | 0.08 |
| v4b: v4 + reader | 0.9364 | −0.0013 [−0.0039, +0.0006] | 0.12 |
| hedge v3: R6 flip + S6F, no CoAtNets | 0.9393 | +0.0017 [−0.0070, +0.0102] | 0.66 |
| hedge v4: + student + reader | 0.9389 | +0.0013 [−0.0070, +0.0091] | 0.63 |

Rank correlations: R6 flip / R7 0.979 (near-duplicates: dropping R7 cost nothing), R6 flip / S6F 0.879, student /
reader 0.914. Every difference is inside its interval; the public board (v3 0.958 vs replica 0.957) is the better
judge, so the final picks follow the public scores of v3, v3r, v4b and the hedges.
