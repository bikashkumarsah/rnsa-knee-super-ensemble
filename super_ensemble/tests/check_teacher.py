#!/usr/bin/env python3
"""Check a fetched teacher kernel output (runs/<mode>/teacher_r7oof.csv): enough rows, only the kernel's folds, no gold
studies, finite probabilities.  Exit 0 when it is usable for the student.

  python3 super_ensemble/tests/check_teacher.py teacher 0,1
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
       'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
UID = 'StudyInstanceUID'
mode, folds = sys.argv[1], [int(k) for k in sys.argv[2].split(',')]
run = Path(__file__).resolve().parents[1] / 'runs' / mode
t = pd.read_csv(run / 'teacher_r7oof.csv', dtype={UID: str})
gold = set(pd.read_csv(Path(__file__).resolve().parents[1] / 'runs' / 'gold58_1015' / 'gold58_labels.csv', dtype={UID: str})[UID])
v = t[LAB].to_numpy(float)
want_fold = {u: int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % 4 for u in t[UID]}
problems = []
if len(t) < 0.9 * 1087 * len(folds):
    problems.append('only %d rows for folds %s' % (len(t), folds))
if sorted(t['fold'].unique().tolist()) != folds:
    problems.append('folds %s' % sorted(t['fold'].unique().tolist()))
if any(want_fold[u] != f for u, f in zip(t[UID], t['fold'])):
    problems.append('a row in the wrong hash fold')
if set(t[UID]) & gold:
    problems.append('gold studies present')
if not ((v >= 0) & (v <= 1)).all() or pd.isna(v).any():
    problems.append('invalid probabilities')
print('%s: %d rows, folds %s, mean prob %s' % (mode, len(t), sorted(t['fold'].unique().tolist()),
                                               dict(zip(LAB, t[LAB].mean().round(3)))))
if problems:
    print('PROBLEMS', problems)
    sys.exit(1)
print('TEACHER_OK', mode)
