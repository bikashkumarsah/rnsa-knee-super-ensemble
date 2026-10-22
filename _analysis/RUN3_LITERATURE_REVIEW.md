# Run-3 literature review and implementation decision

## Directly transferable evidence

| Evidence | Relevant result | Run-3 decision |
|---|---|---|
| [CoPAS, Nature Communications 2024](https://www.nature.com/articles/s41467-024-51888-4) | Multi-plane and cross-sequence attention classified twelve knee abnormalities; its ablations also expose multi-task negative transfer. | Learn target-specific spatial and acquisition-slot attention while retaining one shared DINO encoder. |
| [MPFuseNet](https://arxiv.org/abs/2108.08136) | Multi-plane fusion with spatial attention improved ACL and abnormality detection and provided lesion localization. | Augment CLS/patch-mean/focal context with learned target-aware patch attention. |
| [Attention-based MIL](https://proceedings.mlr.press/v80/ilse18a.html) | A learnable permutation-invariant bag operator identifies influential instances without slice labels. | Treat the windows and acquisition slots as study-level bags, with masks for absent acquisitions. |
| [SB-SSL](https://arxiv.org/abs/2208.13923) | Slice-based self-supervision and transformers exploit through-plane knee MRI context. | Preserve normalized window-centre position and evaluate all ten ordered windows. |
| [MRNet](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699) | Slice-wise features and volume aggregation established a strong weakly supervised knee-MRI baseline. | Retain efficient 2.5D encoding, but learn the aggregation rather than relying on fixed max pooling. |
| [CheXpert](https://ojs.aaai.org/index.php/AAAI/article/download/3834/3712) | Uncertain observations require an explicit policy; masking is a valid alternative to forcing uncertainty to zero or one. | Give `UNK` exactly zero loss and exclude it from epoch-selection AUC. |
| [DINOv2 radiology evaluation](https://arxiv.org/abs/2312.02366) | DINOv2 transfers across X-ray, CT, and MRI under probing and fine-tuning settings. | Retain DINOv2-S and warm-start only the fold-matched, self-trained run-2 backbone. |
| [Model soups](https://proceedings.mlr.press/v162/wortsman22a.html) | Weight averaging can improve fine-tuned-model robustness without inference overhead. | Evaluate a uniform average of the three strongest eligible late checkpoints; keep it only behind OOF and gold safety gates. |

## Methods deliberately deferred

- **Asymmetric loss:** useful for reliable hard multilabel negatives, but this corpus uses
  report-derived soft targets and has extensive missing observations. Changing loss and
  aggregation together would make a failure uninterpretable.
- **Direct deep AUC maximization:** published medical results are promising, but pairwise
  target statistics are unstable with eight studies per batch and rare fracture positives.
- **PCGrad or separate task backbones:** CoPAS reports some negative transfer, but the current
  per-target queries already isolate decision heads. Gradient surgery adds substantial
  implementation and runtime risk without run-2 gradient-conflict evidence.
- **DINOv2-Base, DINOv3, or RadImageNet:** domain-pretraining evidence is relevant, but a
  backbone change would forfeit the verified fold-matched warm start and consume the single
  run's compute/validation budget.
- **Gold-58-directed tuning:** the expert set is essential as a safety diagnostic but too
  small and enriched to select epochs or a target-specific blend.

## Selected design

Run 3 trains one 224 px, twelve-slice bank. Each three-slice window is encoded by DINOv2-S.
Twelve target queries attend over patch tokens, then over the six masked acquisition slots,
then over all ordered windows using normalized depth position. Two widely separated windows
are sampled per study during training; all ten are used for OOF and inference. This makes the
new bank a controlled source of through-plane diversity while the verified 0.877 model
remains an immutable fallback.

Promotion is based on complete five-fold explicit-label OOF, a 2,000-sample report-group
bootstrap, improvement in at least four folds, and a gold-58 non-inferiority veto. Public
leaderboard behavior is not used to tune weights.
