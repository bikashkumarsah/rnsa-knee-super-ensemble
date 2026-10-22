"""Student / reader inference (run by the submission notebook's stage 4a as its own process):

  python student_infer.py <data root> <output dir> <deadline, epoch seconds> [student | reader]

student: our checkpoints (student_s<seed>.pt, the student kernel's output); reader: goodpjw2008's own three public folds
(cnxt_v0_fold*.pt in his dataset, scored at his geometry: 384 px x 0.4 mm, <= 64 slices -- the defaults of his
preprocess.py, which his checkpoints do not record).

goodpjw2008's reader code (knee.py / preprocess.py, Apache 2.0) from its public dataset, our student checkpoints from
the student kernel's output, the reader's pylibjpeg wheels (compressed DICOMs) in a private folder.  Each study is
decoded with preprocess.load_series at the checkpoint's geometry; per (plane, fat-sat) slot the series with the most
files (the training cache's choice), 12 evenly spaced 3-slice windows; output = mean of the checkpoints' sigmoids.
A study with no readable series, or one not reached before the deadline, is left out (the blend uses the others).
"""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA, OUT, DEADLINE = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
GROUP = sys.argv[4] if len(sys.argv) > 4 else 'student'
assert GROUP in ('student', 'reader'), GROUP
INPUT = Path(os.environ.get('STU_INPUT', '/kaggle/input'))
SKIP = {'competitions', 'rsna-knee-abnormality-detection', 'train_series', 'test_series'}
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
       'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
UID = 'StudyInstanceUID'
T0 = time.time()
OUT.mkdir(parents=True, exist_ok=True)


def log(*a):
    print('[student %6.1f s]' % (time.time() - T0), *a, flush=True)


def walk(pred):
    hits = []
    for root, dirs, files in os.walk(INPUT, followlinks=True):
        dirs[:] = sorted(d for d in dirs if d not in SKIP)
        hits += [Path(root) / f for f in files if pred(f)]
    return sorted(hits)


knee_py = walk(lambda f: f == 'knee.py')
assert knee_py, 'attach goodpjw2008/rsna-knee-2-5d-convnext-reader'
READER = knee_py[0].parent
if GROUP == 'student':
    CKPTS = walk(lambda f: f.startswith('student_s') and f.endswith('.pt'))
    assert CKPTS, 'attach the student kernel output (student_s<seed>.pt)'
else:
    CKPTS = sorted(READER.glob('cnxt_v0_fold*.pt'))
    assert CKPTS, 'goodpjw2008 reader folds not found next to knee.py'
tag = 'cp%d%d' % sys.version_info[:2]
wheels = [str(w) for w in sorted(READER.glob('pylibjpeg*.whl')) if ('-py3-none-' in w.name or tag in w.name)]
if wheels and os.environ.get('STU_NO_WHEELS') != '1':
    site = Path('/tmp/stu_site')
    if not site.is_dir():
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--quiet', '--target', str(site),
                        *wheels], check=False)
    sys.path.insert(0, str(site))
sys.path.insert(0, str(READER))
import torch  # noqa: E402
import knee  # noqa: E402
import preprocess  # noqa: E402

ck = [torch.load(p, map_location='cpu', weights_only=False) for p in CKPTS]
A = ck[0]['args']
GEOM = {'size': 384, 'mm_per_px': 0.4, 'max_slices': 64}      # goodpjw2008's preprocess.py defaults
GEOM.update({k: A[k] for k in GEOM if k in A})
assert all({k: c['args'].get(k, GEOM[k]) for k in GEOM} == GEOM and c['args']['res'] == A['res'] for c in ck), \
    [c['args'] for c in ck]
preprocess.SIZE, preprocess.MM_PER_PX, preprocess.MAX_SLICES = GEOM['size'], GEOM['mm_per_px'], GEOM['max_slices']
SIZE, K = GEOM['size'], int(A.get('k_eval', 12))
log('group', GROUP, '| reader', READER, '| checkpoints', [p.name for p in CKPTS], '| geometry', GEOM, '| k', K,
    '| gold', [round(c['args'].get('gold_macro', float('nan')), 4) for c in ck], '| wheels', len(wheels))


def fat_flag(v):
    return 1 if str(v).strip().lower() in ('1', '1.0', 'true', 'yes', 'y') else 0


studies = pd.read_csv(DATA / 'test.csv', dtype=str)[UID].tolist()
se = pd.read_csv(DATA / 'test_series.csv', dtype=str)
se = se[se[UID].isin(set(studies))]
root = DATA / 'test_series'
table, plane_of = {}, {}
for st, g in se.groupby(UID):
    best = {}
    for sr, pl, fs in zip(g.SeriesInstanceUID, g.Anatomical_Plane, g.Fat_Suppression):
        if pl not in knee.PLANES:
            continue
        try:
            nfiles = len(os.listdir(root / st / sr))
        except OSError:
            continue
        slot = knee.slot_of(pl, fat_flag(fs))
        if nfiles >= 3 and (slot not in best or nfiles > best[slot][1]):
            best[slot] = (sr, nfiles)
            plane_of[sr] = pl
    if best:
        slots = [[] for _ in range(knee.N_SLOTS)]
        for slot, (sr, _) in best.items():
            slots[slot] = [sr]
        table[st] = slots


class StudyDS(torch.utils.data.Dataset):
    def __len__(self):
        return len(studies)

    def __getitem__(self, i):
        st = studies[i]
        x = np.zeros((knee.N_SLOTS, K, 3, SIZE, SIZE), np.uint8)
        mask, pos = np.zeros(knee.N_SLOTS, bool), np.zeros((knee.N_SLOTS, K), np.float32)
        for s, cands in enumerate(table.get(st, [])):
            if not cands:
                continue
            try:
                vol, _ = preprocess.load_series(str(root / st / cands[0]), plane_of.get(cands[0]))
            except Exception:
                continue
            n = len(vol)
            if n < 3:
                continue
            c = knee.window_centres(n, K, False)
            x[s] = np.stack([vol[np.clip([j - 1, j, j + 1], 0, n - 1)] for j in c])
            mask[s], pos[s] = True, c / max(n - 1, 1)
        return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(pos), i


dev = 'cuda' if torch.cuda.is_available() else 'cpu'
models = []
for c in ck:
    m = knee.KneeNet(c['args']['backbone'].split('.')[0], pretrained=False)
    m.load_state_dict(c['model'])
    models.append(m.to(dev).to(memory_format=torch.channels_last).eval())
dl = torch.utils.data.DataLoader(StudyDS(), batch_size=2, num_workers=int(os.environ.get('STU_WORKERS', '4')),
                                 pin_memory=dev == 'cuda')
rows, n_empty, timed_out = [], 0, False
with torch.no_grad():
    for x, mask, pos, idx in dl:
        if time.time() > DEADLINE:
            timed_out = True
            break
        keep = mask.any(1)
        n_empty += int((~keep).sum())
        if not keep.any():
            continue
        x, mask, pos = x[keep].to(dev, non_blocking=True), mask[keep].to(dev), pos[keep].to(dev)
        with torch.autocast('cuda', dtype=torch.float16, enabled=dev == 'cuda'):
            p = torch.stack([m(x, mask, pos, res=A['res']).float().sigmoid() for m in models]).mean(0).cpu().numpy()
        for j, i in enumerate(idx[keep].tolist()):
            rows.append([studies[i]] + p[j].tolist())
        if len(rows) % 200 < 2:
            log('scored', len(rows), 'of', len(studies))
pred = pd.DataFrame(rows, columns=[UID] + LAB)
pred.to_csv(OUT / 'predictions.csv', index=False, float_format='%.7g')
rec = {'group': GROUP, 'scored': len(pred), 'of': len(studies), 'no_series': n_empty, 'timed_out': timed_out,
       'seconds': round(time.time() - T0, 1), 'checkpoints': [p.name for p in CKPTS], 'size': SIZE, 'res': A['res'], 'k': K}
(OUT / 'receipt.json').write_text(json.dumps(rec, indent=1))
log('done', json.dumps(rec))
