#!/usr/bin/env python3
"""Build the super-ensemble notebook (BEST_NOTEBOOK_RSNA_KNEE.ipynb) and the two Kaggle kernel folders.

The public components under vendor/ are kept byte-for-byte as published; every change to them is a named patch below
that must match exactly once, so a re-vendored file that moved its anchor fails the build instead of silently skipping.

  python super_ensemble/build_notebook.py           # write the notebook and kernel folders
  python super_ensemble/build_notebook.py --check   # fail if anything on disk is stale
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VENDOR, STAGES = HERE / 'vendor', HERE / 'stages'
BEST = ROOT / 'BEST_NOTEBOOK_RSNA_KNEE.ipynb'
KAGGLE_USER = 'bikashsah1230'
KERNELS = {
    'submit': {'slug': 'rsna-knee-super-ensemble', 'title': 'RSNA Knee Super Ensemble'},
    'gold58': {'slug': 'rsna-knee-super-ensemble-gold58', 'title': 'RSNA Knee Super Ensemble Gold58'},
    'convnext': {'slug': 'rsna-knee-super-ensemble-convnext', 'title': 'RSNA Knee Super Ensemble ConvNeXt'},
    'probe': {'slug': 'rsna-knee-super-ensemble-probe', 'title': 'RSNA Knee Super Ensemble Probe'},
    'probe2': {'slug': 'rsna-knee-super-ensemble-probe2', 'title': 'RSNA Knee Super Ensemble Probe2'},
    'teacher': {'slug': 'rsna-knee-teacher-r7oof', 'title': 'RSNA Knee Teacher R7OOF'},
    'teacherb': {'slug': 'rsna-knee-teacher-r7oof-b', 'title': 'RSNA Knee Teacher R7OOF B'},
    'teachersmoke': {'slug': 'rsna-knee-teacher-r7oof-smoke', 'title': 'RSNA Knee Teacher R7OOF Smoke'},
    'student': {'slug': 'rsna-knee-student', 'title': 'RSNA Knee Student'},
    'studentsmoke': {'slug': 'rsna-knee-student-smoke', 'title': 'RSNA Knee Student Smoke'},
}
DATASETS = [
    'nartaa/rsna-knee-publication-swa-weights-20261007',      # Raptor CoAtNet 384 + 224 (0.949 / 0.945)
    'heliosli/rsna-knee-r6-frozen-fe055406',                  # R7 bundle 8c7b666e27dc0e8f (heliosli's frozen copy)
    'heliosli/rsna-orthofoundation-fold0-epoch7-20261004',    # mainline decode code, SKM-TEA anatomy V-Net, pinned wheels
    'nartaa/rsna-knee-s6fall-weights',                        # S6 ConvNeXt all-data (0.950) + its code payload
    'heliosli/rsna-knee-compact-v1',                          # S6 bundle 209a32c8b50a6b1c (optional member)
    'heliosli/rsna-knee-r6-frozen-fe055406-w1',               # R6 bundle fe055406fad6a12f (heliosli's versioned copy)
]
COMPETITION = 'rsna-knee-abnormality-detection'

# ---- patches: (file, name, old, new) -- old must occur exactly once ----
ROOT_OLD = '    ROOT = find_test_root()\n'
ROOT_NEW = ('    ROOT = os.environ.get("KNEE_DATA_ROOT") or find_test_root()   '
            '# [se] set only in gold58 mode; the hidden-test run is unchanged\n')
RAW224_ANCHOR = ('    print(f"[gpu] done: {done_n[0]}/{N} studies on {len(devs)} device(s) | {time.time()-t0:.0f}s", flush=True)\n')
RAW224_NEW = RAW224_ANCHOR + (
    '    _raw = pd.DataFrame(arm_probs[0].astype(np.float32), columns=LAB)   # [se] raw probabilities; a failed study stays 0.5\n'
    '    _raw.insert(0, "StudyInstanceUID", test_ids)\n'
    '    _raw.to_csv("/kaggle/working/raptor224_raw.csv", index=False)\n')
PIP_OLD = "subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', *wheels], check=True)\n"
PIP_NEW = ("R7_SITE = TMP / 'r7_site'   # [se] heliosli's pinned wheels in a private folder: only this stage's processes see them\n"
           "R7_SITE.mkdir(parents=True, exist_ok=True)\n"
           "subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--target', str(R7_SITE), *wheels], check=True)\n")
PATCHES = {
    'raptor384_run.py': [('data root', ROOT_OLD, ROOT_NEW)],
    'raptor224m_run.py': [('data root', ROOT_OLD, ROOT_NEW), ('raw dump', RAW224_ANCHOR, RAW224_NEW)],
    'r7_setup.py': [('private wheels', PIP_OLD, PIP_NEW)],
}
# sha256 of the vendored files as published: a changed download is refused until it is reviewed
VENDOR_SHA = {
    'raptor384_run.py': '663e33fcf6022c74',
    'raptor224m_run.py': '95de0f171f12381e',
    'r7_setup.py': '5bc33a021a705def',
}


def patched(name):
    text = (VENDOR / name).read_text(encoding='utf-8')
    got = hashlib.sha256(text.encode()).hexdigest()[:16]
    if got != VENDOR_SHA[name]:
        raise SystemExit('%s: vendored file changed (sha256 %s, expected %s)' % (name, got, VENDOR_SHA[name]))
    for label, old, new in PATCHES.get(name, []):
        n = text.count(old)
        if n != 1:
            raise SystemExit('%s: patch %r anchor found %d times' % (name, label, n))
        text = text.replace(old, new)
    return text


def s6f_launcher():
    text = (VENDOR / 'nartaa_s6fall_solo.py').read_text(encoding='utf-8')
    m = re.search(r"^LAUNCHER = (r'''.*?''')\n", text, re.S | re.M)
    if not m:
        raise SystemExit('nartaa launcher not found')
    return m.group(1)


def student_infer_src():
    text = (HERE / 'student' / 'student_infer.py').read_text(encoding='utf-8')
    assert "'''" not in text, 'student_infer.py must not contain triple single quotes'
    return text


def stage(name, **subs):
    text = (STAGES / name).read_text(encoding='utf-8')
    for k, v in subs.items():
        if text.count(k) != 1:
            raise SystemExit('%s: placeholder %s found %d times' % (name, k, text.count(k)))
        text = text.replace(k, v)
    return text


MEMBER_ROWS = {
    'r384': ('CoAtNet', 'Raptor CoAtNet-rmlp-2, 384 px, anatomical mirror TTA, 94 windows', 'nartaa (dreaddevelopment recipe)', '0.949'),
    'r224': ('CoAtNet', 'Raptor CoAtNet, 224 px from a 0.8 centre crop, mirror TTA', 'nartaa', '0.945'),
    'r7': ('ConvNeXt', 'R7 ConvNeXt-small, 4 folds, knee anchors, fastMRI aux, fluid/state stackers', 'heliosli', '(0.957 with the CoAtNet pair)'),
    's6f': ('ConvNeXt', 'S6 ConvNeXt-small trained on all data, JEV-blend labels, no OAI', 'nartaa (heliosli architecture)', '0.950'),
    's6': ('ConvNeXt', 'S6 ConvNeXt-small, 4 folds (bundle 209a32c8b50a6b1c), compiled', 'heliosli', '(0.954 with Raptor-384)'),
    'r7f': ('ConvNeXt', 'R7 with flip TTA (mirrored windows, Medial/Lateral swapped back)', 'heliosli', 'not published'),
    'r6': ('ConvNeXt', 'R6 ConvNeXt-small, 4 folds (bundle fe055406fad6a12f), stackers, compiled', 'heliosli', 'not published'),
    'r6f': ('ConvNeXt', 'R6 with flip TTA, as in heliosli\'s 0.958 / 0.9586 public blends', 'heliosli', '(0.957-0.958 with the CoAtNet pair)'),
    'stu': ('ConvNeXt', 'student: goodpjw2008\'s 2.5D ConvNeXt-T MIL reader retrained on 0.5 LLM consensus + 0.5 out-of-fold R7, '
            '2 seeds', 'ours (goodpjw2008 code, Apache 2.0)', 'not published'),
    'rdr': ('reader', 'goodpjw2008\'s 2.5D ConvNeXt-T MIL reader, his 3 public folds, 384 px', 'goodpjw2008 (Apache 2.0)', '0.929'),
}


def header(cfg):
    wmap = cfg['W_COATNET'] if isinstance(cfg['W_COATNET'], dict) else {'_': cfg['W_COATNET']}
    wc = float(wmap['_'])
    weights = {'r384': wc * 0.65, 'r224': wc * 0.35} if wc > 0 else {}
    tot = sum(cfg['CONVNEXT_MEMBERS'].values())
    weights.update({m: (1 - wc) * w / tot for m, w in cfg['CONVNEXT_MEMBERS'].items()})
    rows = '\n'.join('| %s | %s | %s | %s | %.3f |' % (MEMBER_ROWS[m] + (w,)) for m, w in weights.items())
    names = ' + '.join(m.upper() for m in cfg['CONVNEXT_MEMBERS'])
    if cfg.get('READER_W'):
        names += '; goodpjw2008\'s own reader takes %s of the final rank per finding' % (
            ', '.join('%s %.3f' % (c, w) for c, w in cfg['READER_W'].items()))
    per = ', '.join('%s %.2f' % (c, w) for c, w in wmap.items() if c != '_')
    if per:
        names += ('; the CoAtNet share is set per finding where it differs from %.2f: %s (heliosli\'s weights from his '
                  '0.9586 public blend)' % (wc, per))
    return """# RSNA Knee: super ensemble of public CoAtNet and ConvNeXt readers, one rank blend per finding

Every member is a public, versioned checkpoint run with its authors' own scored inference code, each as an isolated,
time-guarded process; a per-finding percentile-rank blend combines them with weights fixed before any run
(CoAtNet slot %.2f = 0.65 Raptor-384 + 0.35 Raptor-224, ConvNeXt slot %.2f = %s).

| slot | member | author | public LB alone | weight (default finding) |
|---|---|---|---|---|
%s

With only R7 in the ConvNeXt slot and 50/50 slots the blend is exactly the public 0.957 recipe.  S6F joins the ConvNeXt
slot because it is the strongest single public model and was trained on a different label set: on the 58 labelled
studies its rank correlation with R7 (0.87) is no higher than either's with the CoAtNets, and the ConvNeXt pair beats
both CoAtNets there (see `_analysis/SUPER_ENSEMBLE_RUNS.md` in the source repository).

**Time plan** (Kaggle stops at 9 h): measured per study on T4 x2, Raptor-384 ~1.5 s, Raptor-224 ~0.8 s, R6/R7 ~12 s of
GPU per T4 (+5 min compile; ~23 s with flip TTA), S6F ~8 s: about 6.5 h for the v3 members on the 1,284 hidden
studies.  Every stage has its own
deadline; a stage that fails, hangs or runs out of time is dropped and the weights are renormalised;
`blend_receipt.json` records members, coverage and rank correlations.

Set `KNEE_MODE=gold58` (or use the gold58 kernel) to score the 58 radiologist-labelled training studies through the same
code: R7, S6 and S6F never trained on them; nartaa used them to select the Raptor checkpoints, so Raptor's gold AUC is
optimistic.

""" % (wc, 1 - wc, names, rows) + HEADER_CREDITS


HEADER_CREDITS = """## Credits and required acknowledgements

- **nartaa (Danial Zakaria)**: both Raptor CoAtNet checkpoints and their notebooks
  ([0.949 anatomical mirror](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror),
  [0.945 224 crop](https://www.kaggle.com/code/nartaa/rsna-knee-0945-efficient-224crop)), and the
  [S6 all-data weights](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights) with their scored
  [solo notebook](https://www.kaggle.com/code/nartaa/rsna-knee-s6fall-solo) (weights CC BY-NC 4.0).
- **heliosli**: the R7 and S6 compact models, the OrthoFoundation decode pipeline and SKM-TEA anatomy checkpoint, and the
  [0.957 blend](https://www.kaggle.com/code/heliosli/0-9573-latest-public-dataset-score-in-one-click) this extends.
- **dreaddevelopment**: the [Raptor CoAtNet/MIL recipe](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense).
- **varshithakolupuri** and **goodpjw2008**: the 224 checkpoint at 35 % of the CoAtNet slot and its mirror view.
- Weak-label sources: stevenleehans, pilkwang, riadmohamed42 (JEV labels).  SKM-TEA: Desai et al.

*OAI acknowledgement (required for the Raptor weights).* Data and/or research tools used in the preparation of this
manuscript were obtained and analyzed from the controlled access datasets distributed from the Osteoarthritis
Initiative (OAI), a data repository housed within the NIMH Data Archive (NDA). OAI is a collaborative informatics system
created by the National Institute of Mental Health and the National Institute of Arthritis, Musculoskeletal and Skin
Diseases (NIAMS) to provide a worldwide resource to quicken the pace of biomarker identification, scientific
investigation and OA drug development. Dataset identifier(s): 10.15154/0hcg-f676.
"""


def code_cell(src):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': src}


def md_cell(src):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src}


# ---- the predeclared configuration of each kernel (changed only by editing these lines, never from test predictions) ----
READER_DATASET = 'goodpjw2008/rsna-knee-2-5d-convnext-reader'   # goodpjw2008's reader code (Apache 2.0) + DICOM wheels
STUDENT_DATASETS = [READER_DATASET, 'stevenleehans/rsna-knee-llm-report-labels', 'pilkwang/rsna-knee-llm-labels',
                    'lixin73/rsna-knee-llm-report-labels-sol56', 'laymond/rsna-knee-abnormality-qwen3-8b-weak-labels']
TEACHER_DATASETS = ['heliosli/rsna-knee-r6-frozen-fe055406', 'heliosli/rsna-orthofoundation-fold0-epoch7-20261004']
STUDENT_KERNEL = '%s/rsna-knee-student' % KAGGLE_USER
V4_STU_W = 0.15      # predeclared: 0.25 if the student's two-seed gold-58 macro AUC >= 0.92, 0.15 if >= 0.90, else 0;
                     # the student scored 0.9158 (seed 0 0.9155, seed 1 0.9164) -> 0.15
V4_CONVNEXT = {'r6f': round(0.6 * (1 - V4_STU_W), 4), 's6f': round(0.4 * (1 - V4_STU_W), 4), 'stu': V4_STU_W}
READER_W_HELIOSLI = {"Baker's": 0.035, 'Contusion': 0.03, 'Medial OA': 0.025, 'ACL': 0.025, 'Medial Meniscus': 0.02,
                     'MCL': 0.02, 'Lateral Meniscus': 0.008, 'Fracture': 0.008, 'Lateral OA': 0.008, 'PF OA': 0.008,
                     'Effusion': 0.004, 'Synovitis': 0.003}   # goodpjw2008's reader in heliosli's 0.9586 notebook
STAGE_V4 = False     # True rebuilds the 2026-10-20 v4 kernels: submit = v4b (public 0.957), probe = v3r (0.958, tied
                     # with v3), convnext = hedge v4, gold58 = the v4 diagnostic.  False (the default): v3 (0.958), the
                     # best public score, is BEST_NOTEBOOK_RSNA_KNEE.ipynb again
HELIOSLI_W = {'_': 0.5, 'Medial OA': 0.5, 'Lateral OA': 0.15, 'PF OA': 0.65, 'Synovitis': 0.85, "Baker's": 0.30}
def shifted(w, d):
    return {k: round(max(0.0, v - d), 4) for k, v in w.items()}


_OFF = {'R7': '0', 'S6F': '0', 'S6': '0', 'R7F': '0', 'R6': '0', 'R6F': '0', 'STU': '0', 'RDR': '0', 'READER_W': {}}
CONFIG = {
    # v1 (57219200): CoAtNet 0.5 | R7 + S6F.  v2 (57221549): 0.4 -- both 0.957.  v3: the public frontier (heliosli's 0.958 /
    # 0.9586 blends: R6 with flip TTA, his per-finding CoAtNet weights; goodpjw2008 measured that gold-58-tuned weights
    # lose to them on the public board, 0.957 / 0.955 against 0.958) plus S6F, the member our earlier blends added.  R7
    # is off: R6 flip (~4.2 h) + R7 (~2.2 h) + S6F + the CoAtNets would not fit 9 h, and R7 next to R6 flip made no
    # visible difference on the public board (goodpjw2008 v26).
    'submit': dict(_OFF, mode='submit', W_COATNET=HELIOSLI_W, CONVNEXT_MEMBERS={'r6f': 0.6, 's6f': 0.4}, S6F='1', R6F='1',
                   COATNET='1'),
    # gold58 v3: only the new R6 members run; the other members' gold-58 outputs come from the 2026-10-15 run.
    'gold58': dict(_OFF, mode='gold58', W_COATNET=HELIOSLI_W, CONVNEXT_MEMBERS={'r6f': 0.6, 's6f': 0.4}, R6='1', R6F='1',
                   COATNET='0'),
    # the non-OAI hedge: the v3 ConvNeXt slot alone (no Raptor weights attached or run).
    'convnext': dict(_OFF, mode='submit', W_COATNET=0.0, CONVNEXT_MEMBERS={'r6f': 0.6, 's6f': 0.4}, S6F='1', R6F='1',
                     COATNET='0'),
    # v5a / v5b (2026-10-22): v3 with heliosli's per-finding CoAtNet weights lowered uniformly by 0.05 / 0.10 -- our
    # ConvNeXt slot holds S6F as well as R6 flip, so it is stronger than his; one parameter, not per-finding tuning
    # (gold-58: v3 0.9377, -0.05 0.9384, -0.10 0.9397).  The probe kernel's v2 was the frontier replica (0.957).
    'probe': dict(_OFF, mode='submit', W_COATNET=shifted(HELIOSLI_W, 0.05), CONVNEXT_MEMBERS={'r6f': 0.6, 's6f': 0.4},
                  S6F='1', R6F='1', COATNET='1'),
    'probe2': dict(_OFF, mode='submit', W_COATNET=shifted(HELIOSLI_W, 0.10), CONVNEXT_MEMBERS={'r6f': 0.6, 's6f': 0.4},
                   S6F='1', R6F='1', COATNET='1'),
    # teacher table for the student model: out-of-fold R7 on every non-gold training study (stage T), and its smoke run
    'teacher': dict(_OFF, mode='teacher', W_COATNET=0.0, CONVNEXT_MEMBERS={}, COATNET='0', TEACHER_HOURS='11.2',
                    TEACHER_LIMIT='0', TEACHER_FOLDS='0,1', DATASETS=TEACHER_DATASETS),
    'teacherb': dict(_OFF, mode='teacher', W_COATNET=0.0, CONVNEXT_MEMBERS={}, COATNET='0', TEACHER_HOURS='11.2',
                     TEACHER_LIMIT='0', TEACHER_FOLDS='2,3', DATASETS=TEACHER_DATASETS),
    'teachersmoke': dict(_OFF, mode='teacher', W_COATNET=0.0, CONVNEXT_MEMBERS={}, COATNET='0', TEACHER_HOURS='1.5',
                         TEACHER_LIMIT='6', TEACHER_FOLDS='0,1,2,3', DATASETS=TEACHER_DATASETS),
    # the student model's training run (internet on for the timm ImageNet weights; not a submission notebook) and its
    # smoke run (160 studies, one epoch, LLM labels only: it times the steps and checks the real DICOMs)
    'student': dict(_OFF, mode='student', W_COATNET=0.0, CONVNEXT_MEMBERS={}, COATNET='0', INTERNET=True,
                    # smoke run: ~0.9-1.2 s per 2-study step on a T4, cache 0.9 s/study with 4 processes -> 8 epochs
                    # with 6 windows per slot fit ~4.7 h; pushed with --timeout 34200
                    STUDENT_ARGS='--epochs 8 --k 6 --hours 9.0', DATASETS=STUDENT_DATASETS,
                    KERNEL_SOURCES=['%s/rsna-knee-teacher-r7oof' % KAGGLE_USER, '%s/rsna-knee-teacher-r7oof-b' % KAGGLE_USER]),
    'studentsmoke': dict(_OFF, mode='student', W_COATNET=0.0, CONVNEXT_MEMBERS={}, COATNET='0', INTERNET=True,
                         STUDENT_ARGS='--limit 160 --epochs 1 --hours 1.0', DATASETS=STUDENT_DATASETS),
}


if STAGE_V4:
    # v4 = v3 + the student model (stage 4a) at V4_STU_W of the ConvNeXt slot; the hedge keeps the same ConvNeXt slot;
    # gold58 v4 runs R6 flip + S6F + the student on the 58 labelled studies (the CoAtNets' gold outputs are on file)
    # v4b (submit kernel): v3 + the student at V4_STU_W of the ConvNeXt slot + goodpjw2008's reader at heliosli's weights
    CONFIG['submit'] = dict(_OFF, mode='submit', W_COATNET=HELIOSLI_W, CONVNEXT_MEMBERS=V4_CONVNEXT, S6F='1', R6F='1',
                            STU='1', RDR='1', READER_W=READER_W_HELIOSLI, COATNET='1', KERNEL_SOURCES=[STUDENT_KERNEL])
    # the non-OAI hedge v4: the v4 ConvNeXt slot (R6 flip, S6F, student) + the reader, no Raptor
    CONFIG['convnext'] = dict(_OFF, mode='submit', W_COATNET=0.0, CONVNEXT_MEMBERS=V4_CONVNEXT, S6F='1', R6F='1',
                              STU='1', RDR='1', READER_W=READER_W_HELIOSLI, COATNET='0', KERNEL_SOURCES=[STUDENT_KERNEL])
    CONFIG['gold58'] = dict(_OFF, mode='gold58', W_COATNET=HELIOSLI_W, CONVNEXT_MEMBERS=V4_CONVNEXT, S6F='1', R6F='1',
                            STU='1', RDR='1', READER_W=READER_W_HELIOSLI, COATNET='0', KERNEL_SOURCES=[STUDENT_KERNEL])
    # v3r (probe kernel): v3 + the reader only -- with v4b it separates the reader's share from the student's
    CONFIG['probe'] = dict(_OFF, mode='submit', W_COATNET=HELIOSLI_W, CONVNEXT_MEMBERS={'r6f': 0.6, 's6f': 0.4}, S6F='1',
                           R6F='1', RDR='1', READER_W=READER_W_HELIOSLI, COATNET='1')


def build_teacher(cfg):
    return [
        md_cell('# RSNA Knee: teacher table (out-of-fold R7 on the training studies)\n\nheliosli\'s R7 bundle scores every '
                'non-gold training study with only the fold model that never trained on it (his trainer\'s hash folds); '
                'the table is the image teacher of the student model.  Not a submission notebook.'),
        code_cell(stage('s00_setup.py', __KNEE_MODE__=cfg['mode'], __KNEE_S6__='0', __KNEE_R7F__='0', __KNEE_COATNET__='0',
                        __KNEE_R7__='0', __KNEE_S6F__='0', __KNEE_R6__='0', __KNEE_R6F__='0', __KNEE_STU__='0',
                        __KNEE_RDR__='0')),
        code_cell('%%writefile /tmp/r7_setup.py\n' + patched('r7_setup.py')),
        code_cell(stage('s30_compact.py')),
        code_cell(stage('s35_teacher.py', __TEACHER_HOURS__=cfg['TEACHER_HOURS'], __TEACHER_LIMIT__=cfg['TEACHER_LIMIT'],
                        __TEACHER_FOLDS__=cfg['TEACHER_FOLDS'])),
    ]


STUDENT_WHEELS_CELL = '''# goodpjw2008's reader code and its pylibjpeg wheels (compressed DICOMs), as the submission notebook installs them
import os, subprocess, sys
from pathlib import Path
_skip = {'competitions', 'rsna-knee-abnormality-detection', 'train_series', 'test_series'}
_reader = None
for _r, _d, _f in os.walk('/kaggle/input'):
    _d[:] = sorted(x for x in _d if x not in _skip)
    if 'knee.py' in _f:
        _reader = Path(_r)
        break
assert _reader is not None, 'attach goodpjw2008/rsna-knee-2-5d-convnext-reader'
_tag = 'cp%d%d' % sys.version_info[:2]
_wheels = [str(w) for w in sorted(_reader.glob('pylibjpeg*.whl')) if '-py3-none-' in w.name or _tag in w.name]
print('reader', _reader, 'wheels', [Path(w).name for w in _wheels], flush=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--quiet', *_wheels], check=False)
subprocess.run(['nvidia-smi'], check=False)
'''


def build_student(cfg):
    return [
        md_cell('# RSNA Knee: student model training\n\nOur retrain of goodpjw2008\'s public 2.5D ConvNeXt MIL reader '
                '(knee.py / preprocess.py, Apache 2.0, unchanged) on soft targets: the mean of four public LLM label '
                'tables mixed 50/50 with out-of-fold R7 predictions from our teacher kernel.  Never trains on the 58 '
                'gold studies; scores them after every epoch.  Not a submission notebook (internet on for the timm '
                'ImageNet weights).'),
        code_cell(STUDENT_WHEELS_CELL),
        code_cell('%%writefile /kaggle/working/train_student.py\n' + (HERE / 'student' / 'train_student.py').read_text()),
        code_cell("import subprocess, sys\n"
                  "_r = subprocess.run([sys.executable, '-u', '/kaggle/working/train_student.py', '--mode', 'all', *%r.split()])\n"
                  "assert _r.returncode == 0, _r.returncode\n" % cfg['STUDENT_ARGS']),
    ]


def build(kernel):
    cfg = CONFIG[kernel]
    if cfg['mode'] == 'teacher':
        return notebook(build_teacher(cfg))
    if cfg['mode'] == 'student':
        return notebook(build_student(cfg))
    cells = [
        md_cell(header(cfg)),
        code_cell(stage('s00_setup.py', __KNEE_MODE__=cfg['mode'], __KNEE_S6__=cfg['S6'], __KNEE_R7F__=cfg['R7F'],
                        __KNEE_COATNET__=cfg['COATNET'], __KNEE_R7__=cfg['R7'], __KNEE_S6F__=cfg['S6F'],
                        __KNEE_R6__=cfg['R6'], __KNEE_R6F__=cfg['R6F'], __KNEE_STU__=cfg['STU'], __KNEE_RDR__=cfg['RDR'])),
        md_cell('## 1. Raptor CoAtNet, 384 px (nartaa, public 0.949)\nThe published scoring script, unchanged except one '
                '`[se]` line that lets gold58 mode point it at the labelled studies.'),
        code_cell('%%writefile /tmp/raptor_run.py\n' + patched('raptor384_run.py')),
        code_cell(stage('s10_raptor384.py')),
        md_cell('## 2. Raptor CoAtNet, 224 px from a 0.8 centre crop, with the anatomical mirror (nartaa, public 0.945)'),
        code_cell('%%writefile /tmp/raptor224_run.py\n' + patched('raptor224m_run.py')),
        code_cell(stage('s20_raptor224.py')),
        md_cell('## 3. R7 ConvNeXt-small, 4 folds (heliosli)\nheliosli\'s setup cell, written to a file and run with '
                '`exec` so a failure stays inside this stage; its one `[se]` change (private wheel folder) is marked.'),
        code_cell('%%writefile /tmp/r7_setup.py\n' + patched('r7_setup.py')),
        code_cell(stage('s30_compact.py')),
        md_cell('## 4. S6 ConvNeXt, all data (nartaa, public 0.950)'),
        code_cell(stage('s40_s6f.py', __S6F_LAUNCHER_TEXT__=s6f_launcher())),
        md_cell('## 4a. Student model (our retrain of goodpjw2008\'s 2.5D ConvNeXt MIL reader on R7-teacher + LLM targets)'),
        code_cell(stage('s42_student.py', __STUDENT_INFER_SRC__="r'''" + student_infer_src() + "'''")),
        md_cell('## 4b. Optional ConvNeXt members (heliosli S6, R6, flip TTA variants)'),
        code_cell(stage('s45_compact_extra.py')),
        md_cell('## 5. Rank blend'),
        code_cell(stage('s50_blend.py', __W_COATNET__=json.dumps(cfg['W_COATNET']), __READER_W__=json.dumps(cfg['READER_W']),
                        __CONVNEXT_MEMBERS__=json.dumps(cfg['CONVNEXT_MEMBERS']))),
        code_cell(stage('s60_report.py')),
    ]
    return notebook(cells)


def notebook(cells):
    meta = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
        'kaggle': {'accelerator': 'nvidiaTeslaT4', 'isInternetEnabled': False, 'isGpuEnabled': True,
                   'language': 'python', 'sourceType': 'notebook'},
    }
    return {'cells': cells, 'metadata': meta, 'nbformat': 4, 'nbformat_minor': 4}


def kernel_metadata(kernel, code_file):
    k, cfg = KERNELS[kernel], CONFIG[kernel]
    datasets = cfg.get('DATASETS') or [d for d in DATASETS if cfg['COATNET'] == '1' or not d.startswith('nartaa/rsna-knee-publication-swa')]
    if cfg.get('STU') == '1' or cfg.get('RDR') == '1':
        datasets = datasets + [READER_DATASET]
    return {'id': '%s/%s' % (KAGGLE_USER, k['slug']), 'title': k['title'], 'code_file': code_file, 'language': 'python',
            'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
            'enable_internet': bool(cfg.get('INTERNET')), 'machine_shape': 'NvidiaTeslaT4', 'dataset_sources': datasets,
            'competition_sources': [COMPETITION], 'kernel_sources': list(cfg.get('KERNEL_SOURCES', [])),
            'model_sources': []}


def outputs():
    files = {}
    for kernel in KERNELS:
        nb = json.dumps(build(kernel), indent=1, ensure_ascii=False) + '\n'
        name = KERNELS[kernel]['slug'] + '.ipynb'
        files[HERE / 'kernel' / kernel / name] = nb
        files[HERE / 'kernel' / kernel / 'kernel-metadata.json'] = json.dumps(kernel_metadata(kernel, name), indent=2) + '\n'
        if kernel == 'submit':
            files[BEST] = nb
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='fail if a generated file on disk is stale')
    a = ap.parse_args()
    stale = []
    for path, text in outputs().items():
        if a.check:
            if not path.is_file() or path.read_text(encoding='utf-8') != text:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding='utf-8')
            print('wrote', path.relative_to(ROOT), '%d bytes' % len(text.encode()))
    if stale:
        raise SystemExit('stale: ' + ', '.join(stale))
    if a.check:
        print('up to date')


if __name__ == '__main__':
    sys.exit(main())
