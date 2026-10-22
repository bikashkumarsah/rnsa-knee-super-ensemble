# Run 4 analysis and literature rationale

## Verified run-3 evidence

`results_v3.zip` passes CRC and all pinned SHA-256 checks. It contains five complete
hierarchical checkpoints, OOF predictions, manifest, receipt, and submission artifacts.
The completed run took 8,785.49 seconds (146.4 minutes), selected a 30% run-3 vote over the
run-2 anchor, and received the user-confirmed official score 0.883.

The selected OOF metrics are 0.86794 on explicit `YES`/`NO` report observations and 0.85099
on gold-58. The standalone run-3 bank is weaker than the anchor in aggregate (0.85259), but
adds useful diversity. Its largest failure is synovitis: only 694 of 4,407 report labels are
explicit, so the existing zero-loss treatment leaves 84.2% of that target unsupervised.

The log/manifest exposes an implementation defect: a better explicit-label epoch could be
rejected when the tiny per-fold gold subset declined by more than 0.01. This selected epoch
4 instead of the higher-known-AUC epoch 7 in fold 1, and epoch 0 instead of epoch 3 in fold 3.
Run 4 removes this veto. Gold remains recorded and may guard the final full-OOF promotion,
but it cannot select an epoch or checkpoint average.

## Literature synthesis

- Ilse et al.'s [attention-based multiple-instance learning](https://proceedings.mlr.press/v80/ilse18a.html)
  provides the principled basis for learned bag aggregation instead of fixed max/mean rules.
  Run 4 retains target-aware patch, acquisition-slot, and window attention.
- [MRNet](https://journals.plos.org/plosmedicine/article?id=10.1371%2Fjournal.pmed.1002699)
  demonstrates the value of treating knee MRI as a volume rather than disconnected 2D
  images. [SB-SSL](https://arxiv.org/abs/2208.13923) likewise makes slice order/position an
  explicit representation signal. Run 4 keeps normalized through-plane position and exact
  train/inference window positions.
- [CoPAS](https://www.nature.com/articles/s41467-024-51888-4) supports cross-plane spatial
  attention for multi-sequence knee MRI. The current six acquisition slots and masked
  target-specific attention retain that useful inductive bias without importing unavailable
  weights.
- [CheXpert](https://ojs.aaai.org/index.php/AAAI/article/view/3834), Durand et al.'s
  [partial-label ConvNet](https://openaccess.thecvf.com/content_CVPR_2019/html/Durand_Learning_a_Deep_ConvNet_for_Multi-Label_Classification_With_Partial_Labels_CVPR_2019_paper.html),
  and Yu et al.'s [large-scale learning with missing labels](https://proceedings.mlr.press/v32/yu14.html)
  all distinguish unknown/missing labels from observed negatives. Run 4 therefore preserves
  zero primary loss for `UNK`; it never converts `UNK` to `NO`.
- [Noisy Student](https://openaccess.thecvf.com/content_CVPR_2020/html/Xie_Self-Training_With_Noisy_Student_Improves_ImageNet_Classification_CVPR_2020_paper)
  supports teacher-student learning on unlabeled examples. Here the transfer is intentionally
  narrower: only matching-fold out-of-fold run-3 predictions supervise `UNK`, at weight 0.10.
  This prevents in-fold leakage and leaves observed report/gold labels dominant.
- [Model soups](https://proceedings.mlr.press/v162/wortsman22a.html) motivates evaluating a
  uniform late-checkpoint weight average. Run 4 accepts it only when explicit-label OOF
  improves; gold is diagnostic at checkpoint selection and remains a final promotion safety
  check.
- [Deep ensembles](https://papers.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)
  support independently trained predictors as a robust way to reduce model-specific error.
  The extended schedule therefore spends spare GPU time on a second stochastic continuation,
  rather than blindly lengthening one trajectory. Its vote is still conditional on grouped
  OOF evidence.
- [DINOv2](https://arxiv.org/abs/2304.07193) remains the backbone because it is already
  verified in this domain by runs 2 and 3. Changing the backbone and supervision in the same
  one-run experiment would confound attribution.

## Selected experiment

Run 4 is a dual-seed fold-aligned 336 px hierarchical-detail continuation:

- six slots, 130 mm physical crop, six cached slices per slot, four overlapping three-slice
  windows over 20-80% depth, and no horizontal flips;
- the complete matching run-3 model warm-starts each fold;
- two maximally separated windows train each study; all four vote through learned attention
  at OOF/test time;
- ten epochs maximum, head LR 3e-4, backbone LR 2e-6, last six encoder blocks open;
- four-study micro-batches with two-step gradient accumulation preserve effective batch 8;
- seed 9427 completes first; seed 16061 reuses the exact decoded cache and starts only with
  at least 150 training minutes remaining;
- explicit-label OOF alone selects epochs; a late weight average must also improve that metric;
- observed labels/gold use the original weighted BCE; only unsupervised `UNK` cells receive
  0.10-weight BCE to the frozen matching-fold OOF teacher.

The 336 px choice is internal evidence, not a claim borrowed from the papers: run 2's detail
bank beat its depth bank for 11 of 12 targets, while run 3 showed that hierarchical attention
adds diversity. Run 4 combines these verified strengths with one controlled resolution and
supervision change. The two-replica extension uses the measured 2-3 hour runtime slack while
enforcing an 8.5-hour internal ceiling, a 105-minute inference reserve, and a 30-minute margin
below Kaggle's nine-hour limit.

## Promotion and fallback

The target-repaired run-3 anchor is evaluated before training. An equal-rank replica ensemble
must first beat the primary with a non-negative grouped-bootstrap lower bound, at least four
fold wins, and gold safety; otherwise run 4 uses the primary only. That selected run-4
prediction can vote at
15%, 30%, or 45% only if all five folds finish, explicit-label macro AUC improves, the
2,000-sample report-group bootstrap lower 95% bound is non-negative, at least four folds
improve, and gold falls by no more than 0.01. Otherwise the repaired run-3 anchor is used; if
its own gates fail, the exact official uniform-30% 0.883 recipe is used.

No OOF value is presented as a leaderboard result. The sole success criterion is an official
Kaggle score above 0.883.
