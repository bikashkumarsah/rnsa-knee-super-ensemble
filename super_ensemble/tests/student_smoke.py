#!/usr/bin/env python3
"""Local CPU dry run of the student pipeline: a fake /kaggle/input tree with synthetic DICOM series (pydicom), four
fake LLM label tables, a fake teacher table and goodpjw2008's reader code, then train_student.py --mode all with a
tiny backbone; checks the label mix, the cache, both seeds' checkpoints and the gold-58 summary.  With --infer it also
runs the submission notebook's student stage (s47) against the trained checkpoints.

  <venv>/bin/python super_ensemble/tests/student_smoke.py --scratch <empty dir> --reader <dir with knee.py, preprocess.py>
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
       'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
UID = 'StudyInstanceUID'
PLANES = {'Sagittal': ([0, 1, 0, 0, 0, -1], 0), 'Coronal': ([1, 0, 0, 0, 0, -1], 1), 'Axial': ([1, 0, 0, 0, 1, 0], 2)}


def write_series(d, plane, n, rng, seed):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    iop, axis = PLANES[plane]
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(d / ('%d.dcm' % i)), {}, file_meta=meta, preamble=b'\0' * 128)
        ds.Rows = ds.Columns = 96
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 'MONOCHROME2'
        ds.ImageOrientationPatient = iop
        pos = [0.0, 0.0, 0.0]
        pos[axis] = 3.0 * i
        ds.ImagePositionPatient = pos
        ds.PixelSpacing = [1.6, 1.6]
        ds.InstanceNumber = i + 1
        img = (rng.random((96, 96)) * 400 + 200 * np.sin(np.arange(96) / 9.0 + seed)[None, :]).clip(0, 4000)
        ds.PixelData = img.astype(np.uint16).tobytes()
        ds.save_as(str(d / ('%d.dcm' % i)), enforce_file_format=True)


def fake_input(inp, reader_dir, n_train=72, seed=0):
    rng = np.random.default_rng(seed)
    comp = inp / 'competitions' / 'rsna-knee-abnormality-detection'
    (comp / 'train_series').mkdir(parents=True)
    ids = ['1.2.826.9.%d' % i for i in range(n_train)]
    rows = []
    for k, u in enumerate(ids):
        for j, (plane, fs) in enumerate((('Sagittal', 1), ('Coronal', 1), ('Axial', 0))[: 2 + k % 2]):
            sr = u + '.%d' % j
            rows.append({UID: u, 'SeriesInstanceUID': sr, 'Anatomical_Plane': plane, 'Fat_Suppression': fs})
            write_series(comp / 'train_series' / u / sr, plane, 6 + (k % 4), rng, k)
    pd.DataFrame(rows).to_csv(comp / 'train_series.csv', index=False)
    tr = pd.DataFrame({UID: ids, 'Report': 'x'})
    truth = rng.integers(0, 2, size=(n_train, len(LAB))).astype(float)
    gold = np.full_like(truth, np.nan)
    gold[:20] = truth[:20]
    gold[0], gold[1] = 1, 0
    for j, c in enumerate(LAB):
        tr[c] = gold[:, j]
    tr.to_csv(comp / 'train.csv', index=False)
    def table(noise):
        d = pd.DataFrame((truth * 0.7 + 0.15 + rng.normal(0, noise, truth.shape)).clip(0, 1), columns=LAB)
        d.insert(0, UID, ids)
        return d
    for slug, fname in (('stevenleehans/rsna-knee-llm-report-labels', 'llm_labels_v4_blend.csv'),
                        ('pilkwang/rsna-knee-llm-labels', 'report_labels_v2.csv'),
                        ('lixin73/rsna-knee-llm-report-labels-sol56', 'labels_llm_gpt56sol.csv')):
        (inp / 'datasets' / slug).mkdir(parents=True)
        table(0.1).to_csv(inp / 'datasets' / slug / fname, index=False)
    q = table(0.1)
    q = q.rename(columns={c: c + '__label' for c in LAB})
    (inp / 'datasets' / 'laymond' / 'rsna-knee-abnormality-qwen3-8b-weak-labels').mkdir(parents=True)
    q.to_csv(inp / 'datasets' / 'laymond' / 'rsna-knee-abnormality-qwen3-8b-weak-labels' / 'qwen_knee_weak_labels.csv', index=False)
    t = table(0.05).iloc[20:].copy()
    t.insert(1, 'fold', [k % 4 for k in range(len(t))])
    for name, part in (('rsna-knee-teacher-r7oof', t[t.fold < 2]), ('rsna-knee-teacher-r7oof-b', t[t.fold >= 2])):
        (inp / name).mkdir(parents=True)
        part.to_csv(inp / name / 'teacher_r7oof.csv', index=False)
    rd = inp / 'datasets' / 'goodpjw2008' / 'rsna-knee-2-5d-convnext-reader'
    rd.mkdir(parents=True)
    for f in ('knee.py', 'preprocess.py'):
        shutil.copy(Path(reader_dir) / f, rd / f)
    return comp, ids, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scratch', required=True)
    ap.add_argument('--reader', required=True)
    a = ap.parse_args()
    s = Path(a.scratch)
    shutil.rmtree(s, ignore_errors=True)
    inp, work, cache = s / 'input', s / 'work', s / 'cache'
    work.mkdir(parents=True)
    comp, ids, truth = fake_input(inp, a.reader)
    env = dict(os.environ, STU_INPUT=str(inp), STU_WORK=str(work), STU_CACHE=str(cache), PYTHONDONTWRITEBYTECODE='1')
    cmd = [sys.executable, '-u', str(ROOT / 'super_ensemble' / 'student' / 'train_student.py'), '--mode', 'all',
           '--epochs', '2', '--backbone', 'convnext_atto', '--no-pretrained', '--res', '64', '--k', '2', '--k-eval', '3',
           '--bs', '2', '--accum', '2', '--workers', '0', '--cache-procs', '2', '--hours', '0.5']
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(r.stdout[-4000:])
    if r.returncode:
        print(r.stderr[-6000:])
        raise SystemExit('train_student.py failed')
    lab = pd.read_csv(work / 'student_labels.csv', dtype={UID: str}).set_index(UID)
    rep = json.loads((work / 'student_labels_receipt.json').read_text())
    assert rep['teacher_rows'] == len(ids) - 20 and len(rep['sources']) == 4, rep
    assert (lab.is_gold == 1).sum() == 20 and lab.loc[lab.is_gold == 0, LAB].notna().all().all()
    meta = pd.read_csv(work / 'student_cache_meta.csv', dtype=str)
    assert len(meta) == sum(2 + k % 2 for k in range(len(ids))) and (meta.n.astype(int) >= 3).all(), meta.head()
    v = np.load(cache / (meta.series.iloc[0] + '.npy'))
    assert v.dtype == np.uint8 and v.shape[1:] == (256, 256), v.shape
    import torch
    for seed in (0, 1):
        ck = torch.load(work / ('student_s%d.pt' % seed), map_location='cpu', weights_only=False)
        assert ck['args']['epoch'] == 1 and ck['args']['res'] == 64 and ck['args']['size'] == 256, ck['args']
    summ = json.loads((work / 'student_summary.json').read_text())
    assert set(summ['gold58']) == {'seed0', 'seed1', 'mean'}, summ
    # inference: a fake test split (some training studies) and the checkpoints mounted like a notebook output
    data = s / 'testdata'
    (data / 'test_series').mkdir(parents=True)
    test_ids = ids[30:40]
    se = pd.read_csv(comp / 'train_series.csv', dtype=str)
    se[se[UID].isin(test_ids)].to_csv(data / 'test_series.csv', index=False)
    pd.DataFrame({UID: test_ids}).to_csv(data / 'test.csv', index=False)
    for u in test_ids:
        os.symlink(comp / 'train_series' / u, data / 'test_series' / u)
    (inp / 'rsna-knee-student').mkdir()
    for seed in (0, 1):
        shutil.copy(work / ('student_s%d.pt' % seed), inp / 'rsna-knee-student' / ('student_s%d.pt' % seed))
    out = s / 'stu_pred'
    env2 = dict(env, STU_NO_WHEELS='1', STU_WORKERS='0')
    r = subprocess.run([sys.executable, '-u', str(ROOT / 'super_ensemble' / 'student' / 'student_infer.py'), str(data),
                        str(out), '9999999999'], env=env2, capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.returncode:
        print(r.stderr[-4000:])
        raise SystemExit('student_infer.py failed')
    pr = pd.read_csv(out / 'predictions.csv', dtype={UID: str})
    assert sorted(pr[UID]) == sorted(test_ids) and list(pr.columns) == [UID] + LAB
    v = pr[LAB].to_numpy(float)
    assert np.isfinite(v).all() and (v > 0).all() and (v < 1).all()
    # reader mode: goodpjw2008-style fold files next to knee.py, without geometry keys -> his 384 px x 0.4 mm defaults
    rd = inp / 'datasets' / 'goodpjw2008' / 'rsna-knee-2-5d-convnext-reader'
    ck = torch.load(work / 'student_s0.pt', map_location='cpu', weights_only=False)
    for key in ('size', 'mm_per_px', 'max_slices'):
        ck['args'].pop(key)
    torch.save(ck, rd / 'cnxt_v0_fold0.pt')
    out2 = s / 'rdr_pred'
    r = subprocess.run([sys.executable, '-u', str(ROOT / 'super_ensemble' / 'student' / 'student_infer.py'), str(data),
                        str(out2), '9999999999', 'reader'], env=env2, capture_output=True, text=True)
    print(r.stdout[-800:])
    if r.returncode:
        print(r.stderr[-4000:])
        raise SystemExit('student_infer.py reader mode failed')
    rec = json.loads((out2 / 'receipt.json').read_text())
    assert rec['group'] == 'reader' and rec['size'] == 384 and rec['scored'] == len(test_ids), rec
    print('STUDENT_SMOKE_PASSED', json.dumps(summ['gold58']), '| inference rows', len(pr), '| reader rows', rec['scored'])


if __name__ == '__main__':
    main()
