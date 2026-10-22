# Completed 0.835 baseline: execution-log analysis

Source: `knee-best-corpus-v1.log` (218 lines). The notebook completed successfully in
**123.6 minutes** and produced a valid 3-row visible-test `submission.csv` with 13 columns.
The official **0.835** score is the observed Kaggle result supplied for this completed run;
the score itself is not printed inside the execution log.

## What executed correctly

- Both T4 GPUs passed probes; all five folds completed and saved ~86 MB checkpoints.
- Header discovery found 24,371 train and 15 visible-test series.
- Physical laterality was strong: 2,203 train studies used tags, 2,132 used geometry, 72
  remained unresolved, and tag/geometry agreement was 98.5% on 2,135 comparable studies.
- All 20,130 selected train series and 12 selected visible-test series were geometry-ordered;
  none fell back to filename order.
- Slot selection was healthy: mean 4.57 of 6 train slots, zero studies with no selected slot.
- Pixel decoding covered all 4,407 train studies and all three visible-test studies. Train
  decode took 5.5 minutes and filled 20,130 slots.
- The six-slot DINOv2-S model, `cls_mean_focal` pooling, last six open blocks, 336 px input,
  130 mm crop, and no-flip geometry path all ran as intended.

These results justify retaining the header, geometry, laterality, physical crop, six-slot
masking, DINO adaptation, and rank-fusion contracts.

## What limited the score

### 1. Production supervision was lexicon-only

No `report_labels_v2.csv` was mounted. All 4,407 reports were converted by the fallback
lexicon. Its own coverage gauge shows large weak-label blind spots:

| Target | Silent | Indirect | Positive |
|---|---:|---:|---:|
| Medial OA | 47.9% | 0.4% | 39.1% |
| Lateral OA | 52.9% | 0.5% | 33.6% |
| PF OA | 41.5% | 0.5% | 47.7% |
| Synovitis | 0.0% | **87.1%** | 41.8% |
| Baker's | 53.8% | 0.0% | 27.2% |
| Contusion | 62.1% | 0.0% | 25.6% |
| Fracture | **79.4%** | 0.0% | 7.7% |

The successor therefore requires complete public LLM-derived labels and refuses a production
lexicon fallback.

### 2. Memory reduced through-plane coverage to one window

The cache planner saw 29.4 GB available, allocated 12.3 GB, and reduced the requested two
groups to **one group of three slices per slot**. At group size three this yields exactly one
inference window. Overlapping-window TTA and target-specific max/top-two/mean pooling were
therefore present in source but inactive in the completed run.

The successor fixes the two intended operating points explicitly:

- detail: 6 slices at 336 px -> 4 overlapping windows;
- depth: 12 slices at 224 px -> 10 overlapping windows.

It stages one pixel cache at a time and uses disk memmap when RAM is unsafe, rather than
silently reducing slice count.

### 3. Validation was grouped but not multilabel-stratified

The baseline formed 4,265 normalized-report groups, then assigned groups by a simple hash.
Fold sizes ranged from 817 to 924 studies. The successor preserves report-group isolation but
balances study count, positive and negative target mass, and gold-label mass with SHA-256
deterministic assignment.

### 4. Training ran all 10 epochs after the useful plateau

Best teacher-holdout results restored by the five folds were:

| Fold | Studies | Best epoch | Best holdout AUC |
|---:|---:|---:|---:|
| 0 | 881 | 5 | 0.7770 |
| 1 | 908 | 4 | 0.7928 |
| 2 | 817 | 8 | 0.7909 |
| 3 | 924 | 6 | 0.7748 |
| 4 | 877 | 6 | 0.7828 |
| **Mean** | 4,407 | — | **0.7837** |

Loss continued falling after validation plateaued. The successor keeps the 10-epoch ceiling
but stops after three non-improving epochs once at least four epochs have run.

### 5. Gold diagnostics were too sparse for epoch choice

Gold AUC was printed only for folds 1–3: 12–14 gold studies and just 9–11 scorable targets.
Folds 0 and 4 did not meet the reporting threshold. This confirms that gold-58 cannot be a
stable per-fold epoch selector. The successor keeps gold labels separate, lets them override
loss targets, and uses full-OOF gold macro AUC only as a blend safety constraint.

### 6. The run lacked a complete evidence receipt

The run saved folds and a submission, but did not store full OOF predictions, per-target OOF
metrics, label SHA-256, exact bank configuration fingerprints, a bootstrap blend decision,
or submission SHA-256 receipt. The successor makes all of these required artifacts.

## Timing and capacity

| Stage | Observed time |
|---|---:|
| Headers and slot metadata | ~1.9 min |
| Slice geometry ordering | **28.9 min** train + 2 s test |
| Lexicon label extraction | 22.3 s |
| Train pixel decode | 5.5 min |
| Five-fold train/evaluate/infer tail | ~87 min |
| Total | **123.6 min** |

The two-GPU queue trained folds 0/1 and 2/3 concurrently, then left fold 4 as a serial tail.
The completed runtime leaves headroom inside eight hours, but the successor preallocates that
headroom to stronger labels, exact depth coverage, two banks, complete OOF, and final inference
rather than adding unselected epochs.

## Resulting experiment decision

The 0.835 run is a sound geometry/model baseline but not the ceiling of the available
training-first approach. The predeclared successor changes the supervision and depth evidence
while keeping the well-executed physical/model contracts. An official completed score above
0.835 is still required before calling it an improvement.
