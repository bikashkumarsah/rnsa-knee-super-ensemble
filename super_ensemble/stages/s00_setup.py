# Stage 0: clock, mode, paths and helpers shared by every stage.
# Every model stage below runs as its own process group, so a crash, a hang or a timeout in one stage releases its GPU
# memory and can never stop the notebook: the blend uses whatever finished and says so in blend_receipt.json.
import os, sys, time, json, shutil, signal, subprocess, glob, hashlib, math, traceback
from pathlib import Path

NB_T0 = time.time()
KNEE_MODE = os.environ.get('KNEE_MODE', '__KNEE_MODE__')     # 'submit' (hidden test) or 'gold58' (the 58 labelled studies)
assert KNEE_MODE in ('submit', 'gold58', 'teacher'), KNEE_MODE   # teacher: the teacher kernel (stage T only)
INPUT = Path(os.environ.get('KNEE_INPUT_ROOT', '/kaggle/input'))
WORK = Path(os.environ.get('KNEE_WORK_ROOT', '/kaggle/working'))
WORK.mkdir(parents=True, exist_ok=True)
UID = 'StudyInstanceUID'
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
       'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']

# ---- optional members (stage 4b) ----
RUN_COATNET = os.environ.get('KNEE_COATNET', '__KNEE_COATNET__') == '1'   # nartaa's two Raptor CoAtNets (OAI-trained)
RUN_R7 = os.environ.get('KNEE_R7', '__KNEE_R7__') == '1'      # heliosli's R7 bundle
RUN_S6F = os.environ.get('KNEE_S6F', '__KNEE_S6F__') == '1'   # nartaa's S6 all-data model
RUN_S6 = os.environ.get('KNEE_S6', '__KNEE_S6__') == '1'      # heliosli's S6 bundle
RUN_R7F = os.environ.get('KNEE_R7F', '__KNEE_R7F__') == '1'   # R7 with flip TTA
RUN_R6 = os.environ.get('KNEE_R6', '__KNEE_R6__') == '1'      # heliosli's R6 bundle (the 0.958 public blend's model)
RUN_R6F = os.environ.get('KNEE_R6F', '__KNEE_R6F__') == '1'   # R6 with flip TTA
RUN_STU = os.environ.get('KNEE_STU', '__KNEE_STU__') == '1'   # our student model (stage 4a)
RUN_RDR = os.environ.get('KNEE_RDR', '__KNEE_RDR__') == '1'   # goodpjw2008's own reader folds (stage 4a)
_EXTRA = RUN_S6 or RUN_R7F or RUN_R6 or RUN_R6F

# ---- time plan, seconds from the notebook start (Kaggle stops a GPU notebook at 9 h = 32,400 s).  Measured on the 58
# gold studies (gold58 kernel, T4 x2): Raptor-384 ~1.5 s/study, Raptor-224 ~0.8 s, R7 ~12 s of GPU per study per T4
# (GPU-bound, + ~5 min compile), S6F ~8 s per study per T4: on the 1,284 hidden studies about 0.8 h for both CoAtNets,
# 2.3 h for R7, 1.4 h for S6F, so an optional third ConvNeXt bundle (~2.3 h) still fits. ----
H = 3600.0
RAPTOR384_MAX_S = 2.0 * H
RAPTOR224_MAX_S = 1.5 * H
RAPTOR224_START_BY = 3.0 * H   # the 224 model starts only before this
R7_DEADLINE = (4.3 if _EXTRA else 5.6) * H   # compact_infer takes no new study after this; unscored studies keep the others
R7_KILL = R7_DEADLINE + 0.35 * H
S6F_START_BY = (5.8 if _EXTRA else 7.6) * H   # S6F starts only before this
S6F_DEADLINE = (6.2 if _EXTRA else 8.35) * H
S6F_KILL = min(S6F_DEADLINE + 0.25 * H, 8.55 * H)
EXTRA_DEADLINE = 8.35 * H
EXTRA_KILL = 8.55 * H          # the blend needs under a minute; the margin covers the output upload

STATUS = {}


def elapsed():
    return time.time() - NB_T0


def log(*parts):
    print('[%7.1f min]' % (elapsed() / 60.0), *parts, flush=True)


def save_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def stage_begin(name):
    STATUS[name] = {'start_s': round(elapsed(), 1)}
    log('STAGE_BEGIN', name)


def stage_end(name, ok, **info):
    rec = STATUS.setdefault(name, {'start_s': round(elapsed(), 1)})
    rec.update(ok=bool(ok), seconds=round(elapsed() - rec['start_s'], 1), **info)
    log('STAGE_END', name, json.dumps(rec, default=str)[:3000])
    save_json(WORK / 'stage_status.json', STATUS)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def gpu_count():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True,
                             timeout=60).stdout
        return max(1, len([l for l in out.splitlines() if l.strip()]))
    except Exception:
        return 1


def kill_group(proc):
    """SIGKILL the child's whole process group (started with start_new_session, so its pgid is its pid); this also
    reaches DataLoader / Pool workers the child left behind after it exited."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_process(cmd, env=None, kill_at=None, cwd=None, log_path=None, name='child'):
    """Run cmd in its own process group until it exits or the clock reaches kill_at (seconds from NB_T0); on the deadline
    the whole group is killed.  Output goes to the notebook, or to log_path.  Returns (returncode or None when killed,
    seconds)."""
    t = time.time()
    fh = open(log_path, 'w') if log_path else None
    proc = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT if fh else None,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    rc = None
    try:
        while proc.poll() is None:
            if kill_at is not None and elapsed() > kill_at:
                log('KILL', name, 'at its deadline (%.0f s from start)' % kill_at)
                break
            time.sleep(2)
        rc = proc.poll()
        return rc, time.time() - t
    finally:
        kill_group(proc)
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            log('WARNING', name, 'did not exit after SIGKILL')
        if fh:
            fh.close()


def find_competition_root():
    for p in (INPUT / 'competitions' / 'rsna-knee-abnormality-detection', INPUT / 'rsna-knee-abnormality-detection'):
        if (p / 'test.csv').is_file():
            return p
    raise FileNotFoundError('attach the rsna-knee-abnormality-detection competition data')


import numpy as np
import pandas as pd

COMP = find_competition_root()
if KNEE_MODE == 'gold58':
    # Score the 58 radiologist-labelled training studies through the unchanged test-time code of every member: a fake
    # test split whose study folders are symlinks into train_series.  R7, S6 and S6F never trained on these 58 (their
    # manifests say so); nartaa used them to select the Raptor checkpoints, so Raptor's gold AUC is optimistic.
    _train = pd.read_csv(COMP / 'train.csv', dtype={UID: str})
    _gold = _train.dropna(subset=LAB).reset_index(drop=True)
    DATA_ROOT = Path(os.environ.get('KNEE_GOLD_ROOT', '/tmp/gold58_data'))
    if DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)
    (DATA_ROOT / 'test_series').mkdir(parents=True)
    _ids = _gold[UID].tolist()
    _ser = pd.read_csv(COMP / 'train_series.csv', dtype=str)
    _ser[_ser[UID].isin(set(_ids))].to_csv(DATA_ROOT / 'test_series.csv', index=False)
    pd.DataFrame({UID: _ids}).to_csv(DATA_ROOT / 'test.csv', index=False)
    _ss = pd.DataFrame({UID: _ids})
    for _c in LAB:
        _ss[_c] = 0.5
    _ss.to_csv(DATA_ROOT / 'sample_submission.csv', index=False)
    for _u in _ids:
        os.symlink(COMP / 'train_series' / _u, DATA_ROOT / 'test_series' / _u)
    _gold[[UID] + LAB].to_csv(WORK / 'gold58_labels.csv', index=False)
    os.environ['KNEE_DATA_ROOT'] = str(DATA_ROOT)
else:
    DATA_ROOT = COMP
TEST_IDS = pd.read_csv(DATA_ROOT / 'test.csv', dtype={UID: str})[UID].astype(str).tolist()
SUB_COLS = list(pd.read_csv(DATA_ROOT / 'sample_submission.csv', nrows=1).columns)
assert SUB_COLS[0] == UID and sorted(SUB_COLS[1:]) == sorted(LAB), SUB_COLS
N_GPU = gpu_count()
log('mode', KNEE_MODE, '| data', DATA_ROOT, '| studies', len(TEST_IDS), '| gpus', N_GPU, '| python', sys.version.split()[0],
    '| CoAtNets', RUN_COATNET, '| R7', RUN_R7, '| S6F', RUN_S6F, '| optional S6', RUN_S6, '| R7 flip', RUN_R7F, '| R6', RUN_R6, '| R6 flip', RUN_R6F, '| student', RUN_STU, '| reader', RUN_RDR)
for _cmd in (['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used', '--format=csv,noheader'],
             ['df', '-h', str(WORK), '/tmp']):
    try:
        subprocess.run(_cmd, check=False, timeout=60)
    except Exception as _e:
        log('could not run', _cmd[0], repr(_e)[:200])
