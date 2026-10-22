# Compliance Checklist

Working checklist for staying inside `COMPETITION_RULES.md` while forking, experimenting, and submitting. Not legal advice — the Kaggle pages are authoritative.

---

## A. Always-on rules (violate any → disqualification)

### Account & identity
- [ ] **Exactly one Kaggle account** used for everything competition-related (submissions, GPU quota, forum).
- [ ] No old/duplicate account is joined to this competition, even dormant.
- [ ] Profile info is truthful (§3.1d — false info = immediate DQ).

### Team
- [ ] Joined/formally merged into **one team only**, confirmed via Kaggle's team notification.
- [ ] All collaboration happens **inside the official team** — no side channels.
- [ ] Before merging: combined team's submission count ≤ 5/day × days elapsed since start.
- [ ] Merger completed before **Oct 15, 2026**.

### Sharing hygiene
- [ ] Notebooks/weights/label tables/OOF files shared **only** two ways: (a) within the official team, (b) publicly on the competition forum/notebooks.
- [ ] Nothing competition-related sent via DM/Discord/Drive/email to anyone outside the team (§3.5d, §3.6a).
- [ ] Public shares go on **this competition's** Kaggle forum/notebooks (not a personal blog first).

### Data handling
- [ ] Competition DICOMs/reports/derivatives **never re-uploaded** anywhere (no GitHub, HuggingFace, personal datasets) — §2.4b + MIRA license.
- [ ] This repo stays **code-and-docs only** (it currently holds zero data — keep it that way).
- [ ] Anyone I share data with is a registered participant who accepted the rules.
- [ ] No attempt to re-identify patients (MIRA §9); accidental re-identification would be reported to RSNA immediately.
- [ ] No hand-labeling or human prediction of validation/test records (§3.4b). Report-derived labels are fine; manual annotation is not.

### External data & models
- [ ] Every external dataset/model used is **publicly available at no/minimal cost** to all participants (§2.6 Reasonableness Standard).
- [ ] LLM usage stays within reasonableness (small subscription OK; >prize-cost licenses not).
- [ ] Open-source code used is OSI-approved **without commercial-use limits** (§3.6c) — see `LICENSES.md`.
  - [x] Verified: DINOv2, RadImageNet weights+code (MIT), Qwen3-14B (Apache-2.0), timm — all OK.
  - [ ] Only remaining IDENTIFY asset: DINOv3 gated weights + Barun CC BY-NC-SA cache (identify-only at winner stage).
- [ ] Assets with incompatible licenses are tracked in `LICENSES.md` as *identify-only* (§2.5a exemption) — usable, but flagged for winner-stage disclosure.
- [ ] **OAI (super-ensemble):** nartaa's Raptor weights were trained with OAI knees. The hosts said on Sep 21 that OAI is
  fine if generally accessible without institutional sign-off and free; forum topic 749145 (geo restrictions under
  §2.6b) is still unanswered. Re-check before picking finals; the ConvNeXt members (R7, S6F) contain no OAI.

---

## B. Per-submission technical checklist (Submit button won't work otherwise)

- [ ] Submission made through a **committed Notebook** (not CSV upload).
- [ ] Internet **OFF** in the committing session.
- [ ] Runtime ≤ **9 hours** (GPU T4 ×2 recommended; dual-GPU queue already in `BEST_NOTEBOOK_RSNA_KNEE.ipynb`).
- [ ] Output file named exactly **`submission.csv`** at `/kaggle/working/`.
- [ ] Columns exactly: `StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture` — one row per test study, values in [0,1].
- [ ] `python3 super_ensemble/tests/local_smoke.py` passes and `build_notebook.py --check` reports up to date.
- [ ] Commit run (`kaggle_flow.py push submit` → `wait` → `fetch`): `stage_status.json` all `ok`, `blend_receipt.json`
  lists every configured member with coverage 1.0, every sha check in the log passed.
- [ ] Pushed with `--accelerator NvidiaTeslaT4` (the flow does this); submitted with `kaggle_flow.py submit`, never the UI.
- [ ] Daily quota respected: ≤ **5 submissions/day**; each submission adds to the merger-budget ledger if a future merge is possible.

## C. Final-submission selection (before Oct 22)

- [ ] Exactly **2 finals** selected in My Submissions (Kaggle auto-picks 2 latest if you don't).
- [ ] Pick **anchor + hedge**: highest-expected-private-score config + one diversified variant — not two near-identical blend tweaks.
- [ ] Both finals re-verified end-to-end (smoke pass + full commit) on the exact code version submitted.
- [ ] Private LB decides — don't chase ±0.001 public movements (the corpus's own fork-race warning).

## D. If in prize territory (by Nov 5, 2026)

- [ ] Deliver training code + inference code + **model weights as a public Kaggle dataset** linked to the inference code.
- [ ] Environment reproducible: pip `requirements.txt` + corresponding Kaggle image or Dockerfile.
- [ ] Methodology writeup reproduction-grade: architecture, preprocessing, loss, training details, hyperparameters, validation protocol.
- [ ] Winning code licensed **CC-BY-NC 4.0**; incompatible-license assets identified per §2.5a (see `LICENSES.md`) rather than shipped as licensed code.
- [ ] Short **video** presenting the approach; link to open-sourced code+weights posted on the competition forum; final model publicly shareable for open distribution/validation.
- [ ] Tax/eligibility paperwork returned within 2 weeks of notification; respond to winner email within 1 week.
- [ ] Available for recorded winner call/panel with the host.

---

## Quick DQ-trigger reference

| Trigger | Rule | Status |
|---|---|---|
| Second account | §3.5a | permanent ban-level |
| Private sharing outside team | §3.5d, §3.6a | DQ risk |
| Re-uploading competition data | §2.4b + MIRA | DQ risk |
| Hand-labeling test data | §3.4b | DQ risk |
| False identity info | §3.1d | immediate DQ |
| Tampering/scraping the platform | §3.16 | criminal+civil exposure |
| Missing winner deliverables | §2.8 | prize forfeiture (not DQ) |
| Incompatible-license asset unlisted | §2.5 | remediation window or prize loss |

*Forking public notebooks, attaching public weights, using LLMs for label parsing, and running your own experiments are all explicitly permitted — none of these are compliance risks.*
