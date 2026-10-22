# Path to 0.960: strategy and research review (2026-10-16)

Our best is **0.957** public (super-ensemble v2, submission 57221549; rank 270 of 6,061). The deadline is
2026-10-22 23:59 UTC: 6.5 days, 5 submissions a day, about 48 Kaggle GPU-hours (18.6 h left this week, 30 h from
2026-10-17). This note asks what can realistically lift the score to 0.960 or above, using the leaderboard, the forum,
the public notebooks published since 2026-10-14, recent research and our own gold-58 outputs.

## 1. Verdict

- **Blending public models tops out around 0.958.** The best public notebook, heliosli's "Blend mine in" (2026-10-15),
  reaches 0.958 with R6 + flip TTA and per-finding weights. Every public blend rearranges the same handful of
  checkpoints (Raptor-384, Raptor-224, R6, R7, S6, S6F, a 2.5D reader).
- **The 71 teams at 0.960+ add their own models.** On the forum, every high scorer credits labels: better report labels,
  then image-model pseudo-labels. Single 5-fold models trained this way are reported at 0.953-0.959 (azamat1ch:
  ConvNeXt-T 0.959, EfficientNetV2-S 0.956, CoAtNet 0.952 as single architectures).
- **So 0.960 needs new information, not new weights.** The most promising route is a model of our own trained on
  consensus report labels mixed with soft pseudo-labels from the public ensemble, chosen to be architecturally
  different, then rank-blended with the members. heliosli's fold rule is deterministic (`sha1(uid)[:8] % 4`), so R7 and
  R6 can produce leak-free out-of-fold predictions for every training study: a clean teacher.
- Rough odds by 2026-10-22 (judgement, not measurement): public-only v3 blend reaching 0.958: ~50 %, 0.960: under 10 %;
  v3 plus one good student model reaching 0.960: about 25-35 %. Whatever the public board says, a genuinely different
  strong member is the best bet for the private board, which scores the other ~70 % of the test.

## 2. The leaderboard (2026-10-16 05:21 UTC, 6,061 teams)

| score at least | teams | note |
|---|---|---|
| 0.968 | 1 | CentralyDS |
| 0.965 | 9 | |
| 0.963 | 24 | gold line ≈ rank 22 (top 10 + 0.2 %) |
| 0.960 | 71 | |
| 0.958 | 170 | |
| 0.957 | 573 | 403-team tie block; silver line (rank 303) sits inside it; **we are rank 270** |
| 0.956 | 1,390 | 817-team block of copied public notebooks; bronze line (rank 606) sits inside it |

- Scores are truncated to three decimals; inside a block, earlier submission ranks first. The public split is ~30 % of
  1,284 studies (~390): a single score has an SD near 0.003 (nartaa, varshithakolupuri), so 0.001 steps are mostly
  noise. Rank inside a block follows the unrounded score, but only a submission that beats the team's best moves it.
- Silver is held by 33 ranks of margin and erodes as teams pass 0.957. A real 0.958 would put us around rank 100-170.
- The private board will split today's blocks by their actual private scores. A blend that differs from the copied
  0.956/0.957 recipe (ours includes S6F) moves independently of them, for better or worse.

## 3. What moved scores in this competition (forum and public notebooks)

Read with the noise above in mind; these are participants' reports, not verified runs.

| lever | evidence | source |
|---|---|---|
| LLM report labels | 0.878 vs regex 0.814 macro AUC against gold; 25 % of label cells are "not addressed" and their meaning differs by finding | stevenleehans, topic 733932 |
| Label sets are interchangeable | eight public LLM label sets indistinguishable on gold-58 for 11 of 12 findings; a plain mean of six matches the best | karttikjangid05, 747540 |
| Report labels have a ceiling (~0.89-0.90 vs gold) | gold was read from images: one gold study is Effusion = 1 while its report says "No knee effusion is present"; 38 % of "effusion absent" reports were expert-positive | hengck23 745861, colum2131 745949 |
| **Image-model pseudo-labels** | Effusion on gold: LLM from the report 0.835, LLM with images ~0.92, image model trained on pseudo-labels **0.986**; "with pseudo labels every single model can match our best ensemble CV" | colum2131 745949, Chris Deotte 735304 |
| Soft bootstrapping (labels mixed with predictions) | replacing labels with predictions was worse, a 50/50 mix gave "a clear lift" on gold and LB | Archit Konde, Nicolai Karcher 735304 |
| Agreement filtering | training only on cells where LLM labellers agree strongly (>0.9 or <0.1) removed 25 % of samples and improved CV and LB | Prateek Grover 745861 |
| Teacher-student chains | teachers trained on several weak-label sources, then a student on their pseudo-labels: 0.938 → 0.947 | Raymond Yuen 735304; heliosli's trainer has a `--holdout-fold` for pseudo-label rounds |
| Strong single models | ConvNeXt-T 5-fold 0.959, EffNetV2-S 0.956, CoAtNet 0.952 (azamat1ch); DINOv2-S 224 px one fold 0.952; ConvNeXt-T 2.5D 0.953 / DINOv2-S 0.951 / both 0.955 (colum2131); 0.961 at OOF 0.937 / gold 0.938 (datnt114) | 735304, 748011 |
| Geometry | 140-160 mm field of view beats 100 mm; 224-288 px is enough for ResNet/EffNet; 94 windows vs 24: +0.006 for CoAtNet; anatomical mirror TTA +0.001 | 735304, 746792 |
| External data | 2,399 OAI knees (MOAKS-derived PF OA, Lateral OA, Synovitis) +0.005 for nartaa's CoAtNet; legality still unanswered (hosts on 2026-10-14: registration-only datasets are fine, institutional agreements are not; OAI is geo-blocked for some eligible countries) | 746792, 743416, 749145 |
| Validation | weak-label OOF tracks public LB better (r = 0.87) than gold-58 (r = 0.66) across 15 submissions | colum2131 748011 |
| New public members | heliosli R6 with flip TTA + Raptor pair 0.957 (reported 0.9576); with per-finding Raptor weights (Synovitis 0.8, Lateral OA 0.2, PF OA 0.6, Baker's 0.35, else 0.5) **0.958** | heliosli notebooks, 2026-10-14/15 |

New public assets since our audit of 2026-10-15:

| asset | what it is | status for us |
|---|---|---|
| `heliosli/rsna-knee-r6-frozen-fe055406-w1` | R6 bundle `fe055406fad6a12f`: ConvNeXt-small, 4 hash folds, 16 epochs, crop 352, targets `opusv11b_rr6_plr6` (Opus labels + pseudo-label rounds), `gold_in_training: 0` | **add**; same worker as our R7 stage (heliosli's new stage cell differs from our vendored one only in bundle id, dataset slug and the flip switch) |
| `dkim54/rsna-knee-models-v11*` | ConvNeXt-small 384 px, 5 folds × 2 seeds, attention pooling, own inference code; CC BY-NC 4.0 | no scored public notebook uses it; unknown strength |
| heliosli's OrthoFoundation fold-0 checkpoint (1.2 GB, inside the code dataset we already attach) | heliosli's "mainline" model, weak-label validation AUC 0.932 | never shown on the board; low priority |

## 4. Research relevant to the remaining levers

| work | finding | how it applies here |
|---|---|---|
| [AnyMC3D, CVPR 2026](https://arxiv.org/abs/2512.12887) | 2D foundation models with light per-task adapters beat 3D architectures on a 12-task 3D classification benchmark; multi-view inputs and auxiliary pixel-level supervision help | supports 2.5D windows + per-finding attention (what every strong member does); a student should stay 2.5D |
| [OrthoFoundation (2026)](https://arxiv.org/abs/2601.18250), [OrthoDiffusion (2026)](https://arxiv.org/abs/2602.20752), [KneePreM (2026)](https://www.alphaxiv.org/abs/2609.31461) | knee-specific self-supervised pretraining (DINOv3, diffusion, masked autoencoders) matches supervised baselines with less labelled data | promising backbones, but no confirmed public weights; not actionable in 6 days |
| [CoPAS, Nature Communications 2024](https://smartx.cse.ust.hk/2024/09/08/copas/) | co-plane attention across sequences for 12 knee abnormalities; each abnormality has its optimal plane | per-finding attention over planes/series; already in R6/R7/Raptor heads |
| [LNMBench (2025)](https://arxiv.org/abs/2512.09315) | ten learning-with-noisy-labels methods degrade substantially under real-world medical label noise | prefer better targets (consensus + pseudo-labels) over exotic noise-robust losses |
| [Reed et al., bootstrapping (ICLR-W 2015)](https://arxiv.org/abs/1412.6596); [Noisy Student (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Xie_Self-Training_With_Noisy_Student_Improves_ImageNet_Classification_CVPR_2020_paper); [RoS-KD (2022)](https://arxiv.org/abs/2210.08388) | mixing labels with model predictions, self-training with a noised student, and multi-teacher soft-label distillation all improve learning from noisy labels | the student recipe in section 6 (the forum's 50/50 lift is this) |
| [Domain-specific Noisy Student (2020)](https://arxiv.org/abs/2005.11746) | iterating teacher → student gave modest AUC gains (0.910 → 0.915 → 0.918) | expect one round to matter most; do not plan several |
| [LLM labels for MIMIC-CXR (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12920854) | LLM labels beat the CheXpert labeler; image classifiers were resilient to label noise; the test-set labelling schema changed reported performance | our gold/test labels are image reads, so report-only labels hit a ceiling; image teachers close the gap |
| Label aggregation: [Dependence-aware aggregation of LLM judges (2026)](https://arxiv.org/abs/2601.22336), [CrowdAgent (2025)](https://arxiv.org/abs/2509.14030) | Dawid–Skene/majority vote assume independent labellers, which LLMs sharing data and prompts violate | weight label sets, don't count near-duplicate LLM sets as independent votes |
| [Compositional deep AUC maximisation, ICLR 2022](https://iclr.cc/virtual/2022/poster/6071); [multi-task DAM (2022)](https://arxiv.org/abs/2206.00260); [LibAUC](https://github.com/Optimization-AI/LibAUC) | optimising an AUC surrogate on top of cross-entropy training improves test AUC (CheXpert among the benchmarks) | optional final-epochs loss for a student; small expected gain |
| [Roelofs et al., NeurIPS 2019](https://papers.nips.cc/paper/9117-a-meta-analysis-of-overfitting-in-machine-learning) | little adaptive overfitting across 100+ Kaggle competitions | holds for large test sets; with ~390 public studies, 0.001 steps are still noise |
| Past RSNA winners ([review](https://arxiv.org/abs/2512.12887), [RSNA 2025 results](https://www.rsna.org/news/2025/november/2025-ai-challenge-winners)) | 2023 trauma: 2.5D triplets + GRU + auxiliary segmentation; 2024 spine: localise then classify crops; 2025 aneurysm: segmentation then 3D classifier | the localise-then-classify pattern is what heliosli's SKM-TEA anchors already do |
| [3DReasonKnee (2025)](https://arxiv.org/abs/2510.20967) | 494k MOAKS-based quintuples with 3D boxes over 7,970 OAI knee MRIs (CC BY 4.0) | needs the OAI images, so it inherits the OAI risk; skip |
| [SGMRI-VQA (2026)](https://arxiv.org/abs/2604.15808), Qwen 3.5 2B fine-tune (0.950 on the forum) | VLMs fine-tuned on volumetric MRI | needs large GPUs and days; not feasible here |

## 5. Our own checks on gold-58

`_analysis/gold58_blend_ideas.py` (run on `super_ensemble/runs/gold58`) repeats the notebook's rank arithmetic.
Raptor's numbers are optimistic (nartaa selected its epochs on these studies); 58 studies give an SE near 0.02.

| candidate | macro AUC | Δ vs v2 (95 % bootstrap) | P(Δ > 0) |
|---|---|---|---|
| v2: slots 0.4 / 0.6 (R7 + S6F) | 0.9390 | — | — |
| heliosli's per-finding weights (base 0.5) | 0.9381 | −0.0009 [−0.0050, +0.0032] | 0.34 |
| heliosli's per-finding shifts on our 0.4 base | 0.9394 | +0.0004 [−0.0032, +0.0042] | 0.61 |
| **Lateral OA slot weight 0.2, else 0.4** | **0.9406** | **+0.0016 [+0.0002, +0.0037]** | **0.98** |
| z-scored logit mean instead of rank mean | 0.9404 | +0.0014 [−0.0020, +0.0052] | 0.79 |
| five-member equal rank mean (with S6) | 0.9383 | −0.0007 [−0.0050, +0.0030] | 0.36 |

Per finding, the families are complementary: CoAtNets lead on Fracture (0.962 vs 0.908), Effusion (0.983 vs 0.953)
and Medial Meniscus; ConvNeXts lead on Lateral OA (0.869 vs 0.776), PF OA (0.907 vs 0.861), Lateral Meniscus and ACL.
heliosli's choices agree with gold-58 only on Lateral OA; on PF OA (0.6 Raptor) and Synovitis (0.8 Raptor) they point
the other way. Weights tuned on 29 studies and scored on the other 29 gained +0.0036 on average (P 0.81), so
per-finding weighting carries real signal but cannot be tuned reliably on 58 studies. Use only the change both sources
support: Lateral OA 0.2.

Rank correlations (mean per finding): Raptor-384 / Raptor-224 0.946, R7 / S6 0.931, R7 / S6F 0.871, across families
0.85-0.88. S6F is the diverse ConvNeXt; R6 is R7's predecessor with near-identical training and will likely correlate
like S6 (~0.93).

## 6. Ranked plan for the 6.5 days left

| # | step | expected public gain | GPU cost | risk |
|---|---|---|---|---|
| 1 | **v3 blend:** add R6 to the ConvNeXt slot (R7 0.3, R6 0.3, S6F 0.4) and set the Lateral OA slot weight to 0.2 | +0.000 to +0.001 | ~1.5 h gold-58 (R6 with and without flip), 0.3 h commit | low |
| 2 | **Teacher table on the 4,344 training studies:** R7 out-of-fold (each study through its held-out fold model only) plus Raptor-384 in-sample; R6 out-of-fold and S6F in-sample if quota allows | enables 3 | ~6 h for R7 + Raptor-384 (one fold model per study is a quarter of R7's GPU cost, so decode at ~3.25 s/study sets the pace); R6 or S6F add ~4-5 h each | medium: needs a small patch to heliosli's worker to select the fold |
| 3 | **Student model:** 2.5D, a different backbone (EfficientNetV2-S or DINOv2-S; ConvNeXt-T if time is short), 150 mm field of view, 256-288 px, 24-48 windows over all planes, per-finding attention. Targets: soft mix of the mean public LLM label set and the teacher table (start at 50/50, "not addressed" cells taken from the teacher). 5 folds | +0.001 to +0.004 in the blend if it reaches ≥0.950 alone at rank corr ≤0.88 | 15-25 h (decode the training set once into a JPEG/uint8 cache on CPU kernels, then train) | high: 2-3 days of engineering under a 6-day clock |
| 4 | Finals: the best full blend, plus a non-OAI hedge (ConvNeXt members + student) | protects against an OAI ruling | — | — |

Notes on each step:

1. **v3.** Our compact stage already runs any heliosli bundle (`compact_bundle(tag, bundle_id, slug)`); R6 needs only
   the `-w1` dataset attached. Hidden-test cost: R6 without flip ≈ +2.2 h (≈6 h total, v2 measured 3.6-3.7 h); R6 with
   flip ≈ +4 h, which fits only if R7 is dropped. Choose flip or not on gold-58; our R7 flip test was a wash (rank corr
   0.998). Gate: gold-58 non-inferior to v2, R6/R7 correlation reported. Submit as v3; keep v2 as fallback.
2. **Teacher table.** heliosli's trainer assigns `fold = int(sha1(StudyInstanceUID)[:8], 16) % 4` over the non-gold
   pool, and the R6/R7 manifests say gold was never trained on. So fold k's model never saw the studies of fold k:
   true OOF predictions. Raptor and S6F trained on every study, so their training-set outputs partly echo their own
   training labels; give them less weight.
3. **Student.** Diversity matters more than standalone strength: a near-copy of R7 adds nothing, as S6 showed. The
   forum's working recipe is soft targets (labels and predictions mixed), agreement filtering and per-finding
   attention. Our own 0.887 pipeline was limited by a RAM cache holding only 3 slices per slot at 336 px (see
   `kaggle-run-measurements`), not by DINOv2; a disk-backed cache over the whole stack removes that limit. For the
   non-OAI hedge, train a second variant whose teacher excludes Raptor (OAI-trained), since topic 749145 asks whether
   pseudo-labels from OAI-trained models count as OAI use.
4. **Selection.** Keep the two finals diverse: the best expected blend plus the non-OAI hedge. Do not pick between
   neighbouring weight tweaks on 0.001 public differences.

Not recommended: per-finding probing of the public board (a 0.001 macro step is 0.012 of one finding's AUC, and it
costs a 4-7 h rerun per probe); probit/temperature/"co-occurrence smoothing" post-processing (AUC reads only the order
inside a finding); horizontal flips (they swap medial and lateral); stacking or per-finding weights fitted on gold-58;
downloading OAI now (≈140 GB and unresolved legality); VLM fine-tuning.

## Sources

Forum topics (competition discussion IDs): 733932, 735304, 743416, 745861, 745949, 746792, 747540, 748011, 749145.
Public notebooks: heliosli `blend-mine-in-private-lb-luck-to-all-upvoters` and `this-is-best-public-dataset-new-version-0-9576`,
goodpjw2008 `rsna-knee-coatnet-r6-2-5d-mil-lb-0-957`, hoshinoyumemi `knee-0-939-to-0-956-what-moved-the-score`.
Papers are linked in section 4.

## Update 2026-10-20: what the latest public notebooks changed

- **The public frontier is 0.958** (goodpjw2008 v25/v26 and heliosli's "Blend mine in"), and heliosli's newest notebook
  claims 0.9586 with stronger per-finding weights (CoAtNet share Synovitis 0.85, PF OA 0.65, Baker's 0.30, Lateral OA
  0.15, else 0.5) plus goodpjw2008's 2.5D reader at 0.003-0.035 per finding. All of them use R6 with flip TTA as the
  ConvNeXt member; R7 next to it made no visible difference (goodpjw2008 v26: 0.958 either way), and S6 next to R6 cost
  0.001 (0.957).
- **Gold-58-tuned weights lose on the public board.** goodpjw2008 moved Fracture, Effusion and ACL towards the gold-58
  optimum (0.957) and set every finding to its gold-58 optimum (0.955), against 0.958 for heliosli's weights. So the
  "Lateral OA 0.2 only" change in section 5 is dropped in favour of heliosli's full weights; section 5's own
  split-half check already warned that per-finding weights cannot be tuned on 58 studies.
- **No new public notebooks after 2026-10-16** (the usual final-week quiet), so the remaining gain has to come from
  our own model.

Executed plan (details in `KAGGLE_RUNBOOK.md` §5b and `10_experiments/EXPERIMENT_LOG.md`):

1. v3 = the frontier recipe (CoAtNet pair + R6 flip TTA, heliosli's 0.9586 weights) plus S6F at 0.4 of the ConvNeXt
   slot; a probe kernel runs the frontier recipe without S6F, so the two scores measure S6F's share.
2. Teacher: out-of-fold R7 on all non-gold training studies (two kernels in parallel, folds 0-1 and 2-3).
3. Student: goodpjw2008's public 2.5D ConvNeXt-T MIL reader (its training and inference code is public, Apache 2.0)
   retrained on 0.5 × LLM consensus + 0.5 × R7 OOF, two seeds, never trained on gold-58. Blended at 0.25 of the
   ConvNeXt slot if its gold-58 macro AUC is at least 0.92 (0.15 if at least 0.90), a rule fixed before training.

## Outcome (2026-10-21 04:30 UTC)

| submission | blend | public |
|---|---|---|
| 57435077 | **v3**: CoAtNet pair + R6 flip 0.6 + S6F 0.4, heliosli's per-finding weights | **0.958** |
| 57449314 | v3 + goodpjw2008's reader (heliosli's 0.003-0.035) | 0.958 (not above v3 unrounded: the team's board submission stays v3) |
| 57451999 | v4b: v3 + student 0.15 + reader | 0.957 |
| 57435035 | frontier replica (v3 without S6F) | 0.957 |
| 57443351 | non-OAI hedge: R6 flip + S6F | 0.955 |
| 57457062 | non-OAI hedge v4: + student + reader | 0.955 (tied; the simpler hedge v3 is the pick) |

- **0.960 was not reached.** The plan's main bet, a student distilled from out-of-fold R7 plus LLM consensus labels,
  produced a clean 0.916 gold-58 model (goodpjw2008's original reader: 0.911) that correlates 0.87-0.90 with every
  member; at the predeclared 0.15 weight it cost about 0.001 on the public board, as the gold-58 bootstrap predicted.
  A model that adds to this ensemble must beat ~0.93 on gold-58 with lower correlation, which needs a stronger
  architecture and geometry than the reader's, not just better targets.
- **What did help:** S6F next to R6 (+0.001 visible, 0.957 → 0.958) and heliosli's per-finding weights. goodpjw2008's
  reader added nothing visible.
- **The leaderboard moved under us:** a 1,400+-team block of forked 0.958 notebooks; v3 is at rank 570 of 6,302 (bronze
  line 630, silver 315). On the private board the fork block moves as one score, and v3 (which differs from it by S6F)
  moves on its own.
- **Final picks** (to be marked on the competition's Submissions page; the CLI cannot): v3 (57435077) as the best
  expected blend, plus the non-OAI hedge 57443351 (R6 flip + S6F; hedge v4 tied at 0.955 with the student that cost
  v4b 0.001) in case OAI-trained
  weights are ruled out. Two near-identical OAI blends would add almost no protection.
