# Vendored third-party code

These files are other competitors' public Kaggle code, kept byte-for-byte as published and pinned by sha256 in
`super_ensemble/build_notebook.py` (`VENDOR_SHA`). Changes are applied at build time as named patches that must
match exactly once (`PATCHES`), and are marked `[se]` in the generated notebook. Public Kaggle notebooks are released
under the Apache License 2.0; the copyright stays with their authors.

| file | source | author |
|---|---|---|
| `raptor384_run.py` | scored source of [RSNA Knee 0949 Anatomical Mirror](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror) | nartaa (on dreaddevelopment's Raptor recipe) |
| `raptor224m_run.py` | scored source of [RSNA Knee 0945 Efficient 224crop](https://www.kaggle.com/code/nartaa/rsna-knee-0945-efficient-224crop), with the anatomical mirror view | nartaa; mirror variant by goodpjw2008 / varshithakolupuri |
| `r7_setup.py`, `r7_run_reference.py` | setup and run cells of [0.957 latest dataset, score in one click](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click), with heliosli's embedded inference sources | heliosli |
| `nartaa_s6fall_solo.py` | [RSNA Knee S6 all-data solo](https://www.kaggle.com/code/nartaa/rsna-knee-s6fall-solo) (its launcher and pinned code payload) | nartaa (heliosli's S6 architecture) |

The model weights are not here: they are public Kaggle datasets attached at run time (see `ATTACHING_INPUTS.md` and
`LICENSES.md`).
