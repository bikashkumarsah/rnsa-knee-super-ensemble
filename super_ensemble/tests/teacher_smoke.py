#!/usr/bin/env python3
"""Local smoke test of the teacher kernel (stage T): fake competition tree, a stub setup cell whose bundle holds four
fold files, and a stub controller that refuses a --models folder with more than one fold model and scores the data
root's test.csv.  Checks the hash folds, the per-fold bundles and the concatenated teacher table.

  python super_ensemble/tests/teacher_smoke.py --scratch <empty dir>
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import textwrap
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from local_smoke import LAB, UID, cells, fake_competition, pick  # noqa: E402

ROOT = HERE.parents[1]
NB = ROOT / 'super_ensemble' / 'kernel' / 'teachersmoke' / 'rsna-knee-teacher-r7oof-smoke.ipynb'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scratch', required=True)
    a = ap.parse_args()
    scratch = Path(a.scratch)
    shutil.rmtree(scratch, ignore_errors=True)
    work, inp, stubs = scratch / 'work', scratch / 'input', scratch / 'stubs'
    for d in (work, stubs):
        d.mkdir(parents=True)
    fake_competition(inp, n_test=3, n_train=120, seed=2)
    os.environ.update(KNEE_MODE='teacher', KNEE_INPUT_ROOT=str(inp), KNEE_WORK_ROOT=str(work))
    src = stubs / 'r7src'
    src.mkdir()
    (src / 'probe_dual.py').write_text(textwrap.dedent("""
        import argparse, glob, json, os, sys
        import numpy as np, pandas as pd
        ap = argparse.ArgumentParser(); ap.add_argument('--output'); ap.add_argument('--bundle'); ap.add_argument('--data')
        ap.add_argument('--worker-args'); a, _ = ap.parse_known_args()
        fold = sorted(glob.glob(os.path.join(a.bundle, 'E_lrprior_s*_f*.pt')))
        assert len(fold) == 1, fold
        assert '--compile' not in json.loads(a.worker_args)
        ids = pd.read_csv(os.path.join(a.data, 'test.csv'), dtype=str).StudyInstanceUID.tolist()
        for u in ids:
            assert os.path.isdir(os.path.join(a.data, 'test_series', u)), u
        k = int(fold[0].rsplit('_f', 1)[1].split('.')[0])
        rng = np.random.default_rng(k)
        d = pd.DataFrame(rng.random((len(ids), 12)), columns=%r); d.insert(0, 'StudyInstanceUID', ids)
        w = os.path.join(a.output, 'worker0'); os.makedirs(w, exist_ok=True)
        d.to_csv(os.path.join(w, 'predictions.csv'), index=False)
        open(os.path.join(w, 'log.txt'), 'w').write('stub fold %%d\\n' %% k)
    """ % (LAB,)))
    (stubs / 'r7_setup.py').write_text(textwrap.dedent("""
        TMP = Path(%r); TMP.mkdir(parents=True, exist_ok=True)
        R7_SITE = TMP / 'r7_site'; CODE = TMP / 'code'; SRC = Path(%r)
        ARM, WORKER = 'E_lrprior', 'compact_infer.py'
        BUNDLE = TMP / 'bundle'
        BUNDLE.mkdir(exist_ok=True)
        for _k in range(4):
            (BUNDLE / ('E_lrprior_s0_f%%d.pt' %% _k)).write_bytes(b'x')
        for _n in ('state_spec.json', 'side_cls.pt', 'fluid_stacker.json', 'state_stacker.json'):
            (BUNDLE / _n).write_text('{}')
        WORKER_ARGS = ['--backbone', 'convnext_small', '--compile']
    """ % (str(stubs / 'r7tmp'), str(src))))
    cs = cells(NB)
    g = {'__name__': '__main__'}
    exec(compile(pick(cs, '# Stage 0:'), 'setup', 'exec'), g)
    exec(compile(pick(cs, '# Stage 3').replace('/tmp/r7_setup.py', str(stubs / 'r7_setup.py')), 'compact', 'exec'), g)
    assert not g['R7_OK']
    exec(compile(pick(cs, '# Stage T'), 'teacher', 'exec'), g)
    tab = pd.read_csv(work / 'teacher_r7oof.csv', dtype={UID: str})
    train = pd.read_csv(inp / 'competitions' / 'rsna-knee-abnormality-detection' / 'train.csv', dtype={UID: str})
    gold = set(train.dropna(subset=LAB)[UID])
    pool = [u for u in train[UID] if u not in gold]
    fold = {u: int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % 4 for u in pool}
    want = [u for k in range(4) for u in [v for v in pool if fold[v] == k][:6]]
    assert sorted(tab[UID]) == sorted(want), (len(tab), len(want))
    assert not set(tab[UID]) & gold
    assert all(fold[u] == f for u, f in zip(tab[UID], tab['fold']))
    assert list(tab.columns) == [UID, 'fold'] + LAB
    print('TEACHER_SMOKE_PASSED', len(tab), 'rows', json.loads((work / 'stage_status.json').read_text())['teacher']['ok'])


if __name__ == '__main__':
    main()
