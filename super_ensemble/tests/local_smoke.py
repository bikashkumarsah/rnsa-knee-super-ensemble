#!/usr/bin/env python3
"""Local smoke test of the generated notebook's glue code (no GPU, no competition data).

Runs the notebook's own cells against a fake /kaggle tree: the setup cell in both modes, every stage runner with stub
model processes, the real S6F input preparation when nartaa's and heliosli's datasets are present under --input, and the
blend/report cells, then checks the blend arithmetic, the fallbacks and the submission contract.

  python super_ensemble/tests/local_smoke.py --scratch <empty dir> [--input <dir with datasets/...>]
"""
import argparse
import json
import os
import shutil
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / 'BEST_NOTEBOOK_RSNA_KNEE.ipynb'
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
       'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
UID = 'StudyInstanceUID'


def cells(nb_path=NB):
    nb = json.loads(Path(nb_path).read_text())
    out = {}
    for c in nb['cells']:
        if c['cell_type'] != 'code':
            continue
        s = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
        first = s.split('\n', 1)[0]
        if first.startswith('%%writefile'):
            out['write:' + first.split()[1]] = s.split('\n', 1)[1]
        else:
            out[first[:60]] = s
    return out


def pick(cs, prefix):
    hits = [v for k, v in cs.items() if k.startswith(prefix)]
    assert len(hits) == 1, (prefix, list(cs))
    return hits[0]


def fake_competition(root, n_test, n_train=80, seed=0):
    rng = np.random.default_rng(seed)
    comp = root / 'competitions' / 'rsna-knee-abnormality-detection'
    for split in ('test', 'train'):
        (comp / (split + '_series')).mkdir(parents=True, exist_ok=True)
    tid = ['1.2.826.0.%d' % i for i in range(n_test)]
    rid = ['1.2.826.9.%d' % i for i in range(n_train)]
    pd.DataFrame({UID: tid}).to_csv(comp / 'test.csv', index=False)
    ss = pd.DataFrame({UID: tid})
    for c in LAB:
        ss[c] = 0.5
    ss.to_csv(comp / 'sample_submission.csv', index=False)
    rows = []
    for split, ids in (('test', tid), ('train', rid)):
        recs = []
        for u in ids:
            for k in range(int(rng.integers(2, 6))):
                recs.append({UID: u, 'SeriesInstanceUID': u + '.%d' % k, 'Anatomical_Plane': ['Sagittal', 'Coronal', 'Axial'][k % 3],
                             'Fat_Suppression': int(k % 2), 'Fluid_Sensitive': int(k % 2)})
            (comp / (split + '_series') / u).mkdir(exist_ok=True)
            for k in range(2):
                sd = comp / (split + '_series') / u / (u + '.%d' % k)
                sd.mkdir(exist_ok=True)
                for f in range(3):
                    (sd / ('%d.dcm' % f)).write_bytes(bytes(rng.integers(0, 255, 64, dtype=np.uint8)))
        pd.DataFrame(recs).to_csv(comp / (split + '_series.csv'), index=False)
    tr = pd.DataFrame({UID: rid, 'Report': 'x'})
    gold = np.zeros((n_train, len(LAB))) * np.nan
    gold[:58] = rng.integers(0, 2, size=(58, len(LAB)))
    gold[:58][0] = 1
    gold[:58][1] = 0
    for j, c in enumerate(LAB):
        tr[c] = gold[:, j]
    tr.to_csv(comp / 'train.csv', index=False)
    return comp


def stub_outputs(ids, seed, fail_rows=(), noise=0.3, truth=None):
    rng = np.random.default_rng(seed)
    base = truth if truth is not None else rng.random((len(ids), len(LAB)))
    p = 1 / (1 + np.exp(-(4 * (base - 0.5) + noise * rng.standard_normal(base.shape))))
    d = pd.DataFrame(p.astype(np.float32), columns=LAB)
    d.insert(0, UID, ids)
    for i in fail_rows:
        d.loc[i, LAB] = 0.5
    return d


def run(mode, scratch, real_input, drop=(), partial=None, extra=False, nb=NB, defaults=False, inject=()):
    """defaults=False runs the classic member set (CoAtNets, R7, S6F; with extra=True also S6, R7F, R6, R6F);
    defaults=True leaves every member flag to the notebook's own configuration."""
    convnext_only = 'convnext' in Path(nb).name
    tag = (mode + '_' + '_'.join(drop) + ('_p%s' % partial if partial else '') + ('_x' if extra else '')
           + ('_cx' if convnext_only else '') + ('_cfg' if defaults else '') + ('_inj' if inject else ''))
    work, inp = scratch / ('work_' + tag), scratch / 'input'
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    fake_competition(inp, n_test=3, seed=1)
    if real_input:
        for p in Path(real_input).glob('datasets/*/*'):
            dst = inp / 'datasets' / p.parent.name / p.name
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(p, dst)
    os.environ.update(KNEE_MODE=mode, KNEE_INPUT_ROOT=str(inp), KNEE_WORK_ROOT=str(work),
                      KNEE_GOLD_ROOT=str(scratch / ('gold58_' + tag)))
    for k in ('KNEE_COATNET', 'KNEE_R7', 'KNEE_S6F', 'KNEE_S6', 'KNEE_R7F', 'KNEE_R6', 'KNEE_R6F'):
        os.environ.pop(k, None)
    if not defaults:
        x = '1' if extra else '0'
        os.environ.update(KNEE_S6=x, KNEE_R7F=x, KNEE_R6=x, KNEE_R6F=x, KNEE_R7='1', KNEE_S6F='1',
                          KNEE_COATNET='0' if convnext_only else '1')
    cs = cells(nb)
    g = {'__name__': '__main__'}
    exec(compile(pick(cs, '# Stage 0:'), 'setup', 'exec'), g)
    if not defaults:
        assert g['RUN_S6'] == extra and g['RUN_R7F'] == extra and g['RUN_R6'] == extra and g['RUN_R6F'] == extra
    convnext_only = not g['RUN_COATNET']
    ids = g['TEST_IDS']
    assert (mode == 'submit' and len(ids) == 3) or (mode == 'gold58' and len(ids) == 58), len(ids)
    truth = None
    if mode == 'gold58':
        truth = pd.read_csv(work / 'gold58_labels.csv', dtype={UID: str}).set_index(UID).loc[ids, LAB].to_numpy(float)
    stubs = scratch / ('stubs_' + tag)
    shutil.rmtree(stubs, ignore_errors=True)
    stubs.mkdir()
    # ---- Raptor stages with stub scripts ----
    for name, out, seed in (('raptor_run.py', 'raptor_raw.csv', 11), ('raptor224_run.py', 'raptor224_raw.csv', 12)):
        df = stub_outputs(ids, seed, fail_rows=[0] if name == 'raptor_run.py' and mode == 'gold58' else (), truth=truth)
        if name == 'raptor224_run.py' and 'r224' in drop:
            script = 'import sys; sys.exit(3)\n'
        else:
            csvp = stubs / out
            df.to_csv(csvp, index=False)
            script = 'import shutil; shutil.copyfile(%r, %r); print("stub", %r)\n' % (str(csvp), str(work / out), name)
        (stubs / name).write_text(script)
    for key in ('# Stage 1', '# Stage 2'):
        exec(compile(pick(cs, key).replace('/tmp/raptor_run.py', str(stubs / 'raptor_run.py'))
                     .replace('/tmp/raptor224_run.py', str(stubs / 'raptor224_run.py')), key, 'exec'), g)
    if convnext_only:
        assert not g['RAPTOR384_OK'] and not g['RAPTOR224_OK'] and not (work / 'raptor_raw.csv').exists()
    else:
        assert g['RAPTOR384_OK'] and (g['RAPTOR224_OK'] == ('r224' not in drop))
    # ---- compact stages: a stub setup cell (with the real bundle anchor) + a stub probe_dual that picks its rows by the
    #      bundle folder and the flip flag, and writes them per worker in its reading order ----
    preds = {'r7': stub_outputs(ids, 13, truth=truth), 's6': stub_outputs(ids, 15, truth=truth),
             'r7f': stub_outputs(ids, 16, truth=truth), 'r6': stub_outputs(ids, 17, truth=truth),
             'r6f': stub_outputs(ids, 18, truth=truth)}
    if partial:
        preds['r7'] = preds['r7'].iloc[:int(len(ids) * partial)]
    for k, d in preds.items():
        d.to_csv(stubs / ('pred_%s.csv' % k), index=False)
    src = stubs / 'r7src'
    src.mkdir()
    (src / 'probe_dual.py').write_text(textwrap.dedent("""
        import argparse, json, os, sys
        import pandas as pd
        ap = argparse.ArgumentParser(); ap.add_argument('--output'); ap.add_argument('--bundle'); ap.add_argument('--worker-args')
        a, _ = ap.parse_known_args()
        args = json.loads(a.worker_args)
        base = os.path.basename(a.bundle).replace('bundle_', '').replace('bundle', 'r7')
        tag = base + 'f' if '--flip-tta' in args else base
        assert tag != 's6' or '--compile' in args, 'S6 must run compiled'
        d = pd.read_csv(os.path.join(%r, 'pred_%%s.csv' %% tag), dtype={'StudyInstanceUID': str})
        ids = d.StudyInstanceUID.tolist()
        for gi, part in enumerate([ids[0::2], ids[1::2]]):
            w = os.path.join(a.output, 'worker%%d' %% gi); os.makedirs(w, exist_ok=True)
            json.dump(part, open(os.path.join(w, 'ids.json'), 'w'))
            open(os.path.join(w, 'log.txt'), 'w').write('stub ' + tag + '\\n')
            d[d.StudyInstanceUID.isin(part)].to_csv(os.path.join(w, 'predictions.csv'), index=False)
        sys.exit(%d if tag == 'r7' else 0)
    """ % (str(stubs), 7 if 'r7' in drop else 0)))
    (stubs / 'r7_setup.py').write_text(textwrap.dedent("""
        TMP = Path(%r); TMP.mkdir(parents=True, exist_ok=True)
        R7_SITE = TMP / 'r7_site'; CODE = TMP / 'code'; SRC = Path(%r)
        ARM, WORKER = 'E_lrprior', 'compact_infer.py'
        BUNDLE = TMP / 'bundle'
        BUNDLE.mkdir(exist_ok=True)
        WORKER_ARGS = ['--bundle-id', BUNDLE_ID if 'BUNDLE_ID' in globals() else 'r7']
    """ % (str(stubs / 'r7tmp'), str(src))))
    if 'r7' in drop:
        preds['r7'].iloc[:0].to_csv(stubs / 'pred_r7.csv', index=False)
    exec(compile(pick(cs, '# Stage 3').replace('/tmp/r7_setup.py', str(stubs / 'r7_setup.py')), 'compact', 'exec'), g)
    assert g['R7_OK'] == (g['RUN_R7'] and 'r7' not in drop), g['STATUS'].get('r7')
    # ---- S6F: real input preparation when the datasets are present, stub model processes ----
    s40 = pick(cs, '# Stage 4:')
    head, tail = s40.split("stage_begin('s6f')\n", 1)
    exec(compile(head, 's6f_defs', 'exec'), g)
    if real_input and (inp / 'datasets' / 'nartaa' / 'rsna-knee-s6fall-weights').exists():
        I = g['s6f_prepare'](scratch / ('s6f_prep_' + tag))
        launcher = Path(I['launcher']).read_text()
        assert "SRC = " + repr(str(I['src'])) in launcher and '__SRC__' not in launcher
        assert (Path(I['src']) / 'compact_infer.py').is_file() and (Path(I['code']) / 'training' / 'native_cache.py').is_file()
        print('S6F_PREPARE_OK', I['src'])
    s6 = stub_outputs(ids, 14, truth=truth)
    (stubs / 's6_pred.csv').write_text(s6.to_csv(index=False))
    fake_prep = {'launcher': stubs / 's6f_stub.py', 'models': stubs, 'anatomy': stubs, 'code': stubs, 'src': stubs}
    (stubs / 's6f_stub.py').write_text(textwrap.dedent("""
        import argparse, json, os, sys
        import pandas as pd
        ap = argparse.ArgumentParser(); ap.add_argument('--output'); ap.add_argument('--assignment'); a, _ = ap.parse_known_args()
        ids = set(json.load(open(a.assignment)))
        d = pd.read_csv(%r, dtype={'StudyInstanceUID': str})
        d[d.StudyInstanceUID.isin(ids)].to_csv(os.path.join(a.output, 'predictions.csv'), index=False)
        json.dump({'status': 'PASS', 'scored': int(d.StudyInstanceUID.isin(ids).sum()), 'timed_out': False},
                  open(os.path.join(a.output, 'receipt.json'), 'w'))
        sys.exit(%d)
    """ % (str(stubs / 's6_pred.csv'), 5 if 's6f' in drop else 0)))
    if 's6f' in drop:
        (stubs / 's6_pred.csv').write_text(UID + ',' + ','.join(LAB) + '\n')
    g['s6f_prepare'] = lambda tmp: fake_prep
    g['N_GPU'] = 2
    exec(compile("stage_begin('s6f')\n" + tail, 's6f_run', 'exec'), g)
    assert g['S6F_OK'] == (g['RUN_S6F'] and 's6f' not in drop), g['STATUS']['s6f']
    # ---- student / reader stages: they need torch + timm, so here they either are disabled or fail cleanly; inject=
    #      writes stand-in rows for them, as if they had run ----
    exec(compile(pick(cs, '# Stage 4a'), 'student', 'exec'), g)
    for k in ('stu', 'rdr'):
        assert g['STATUS'][k]['ok'] is False, g['STATUS'][k]
    for k in inject:
        preds[k] = stub_outputs(ids, {'stu': 19, 'rdr': 20}[k], truth=truth)
        preds[k].to_csv(work / ('%s_scored.csv' % k), index=False)
    # ---- optional compact members ----
    exec(compile(pick(cs, '# Stage 4b'), 'extra', 'exec'), g)
    for k in ('S6', 'R7F', 'R6', 'R6F'):
        assert g[k + '_OK'] == g['RUN_' + k], (k, g['STATUS'].get(k.lower()))
    if g['RUN_S6']:
        assert '--compile' in g['STATUS']['s6']['worker_args']
    for k in ('r7f', 'r6f'):
        if g['RUN_' + k.upper()]:
            assert '--flip-tta' in g['STATUS'][k]['worker_args'], g['STATUS'][k]
    if g['RUN_R6']:
        assert '--flip-tta' not in g['STATUS']['r6']['worker_args']
    # ---- blend + report ----
    exec(compile(pick(cs, '# Stage 5'), 'blend', 'exec'), g)
    exec(compile(pick(cs, '# Stage 6'), 'report', 'exec'), g)
    out = work / ('submission.csv' if mode == 'submit' else 'gold58_blend.csv')
    sub = pd.read_csv(out, dtype={UID: str})
    assert list(sub.columns) == [UID] + LAB and sub[UID].tolist() == ids
    v = sub[LAB].to_numpy(float)
    assert np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all()
    # the arithmetic: recompute the blend from the member files with the notebook's configured weights
    def ranks(df, raptor=False):
        d = df.set_index(UID).reindex(ids)[LAB].astype(float)
        if raptor:
            d[(d - 0.5).abs().lt(1e-12).all(1)] = np.nan
        return d.rank(pct=True).to_numpy()
    mem = {}
    if not convnext_only:
        mem['r384'] = ranks(pd.read_csv(work / 'raptor_raw.csv', dtype={UID: str}), True)
    if 'r224' not in drop and not convnext_only:
        mem['r224'] = ranks(pd.read_csv(work / 'raptor224_raw.csv', dtype={UID: str}), True)
    if g['RUN_R7'] and 'r7' not in drop:
        mem['r7'] = ranks(preds['r7'])
    if g['RUN_S6F'] and 's6f' not in drop:
        mem['s6f'] = ranks(s6)
    for k in ('s6', 'r7f', 'r6', 'r6f'):
        if g['RUN_' + k.upper()]:
            mem[k] = ranks(preds[k])
    for k in inject:
        mem[k] = ranks(preds[k])
    wc, cx = g['W_C'], g['CONVNEXT_MEMBERS']
    assert wc.shape == (len(LAB),)
    if convnext_only:
        assert 'convnext' not in Path(nb).name or (wc == 0.0).all()
    # independent recomputation: per slot the weighted mean of the members that scored the study, then the per-finding
    # slot weights renormalised over the slots that have a value (the notebook's documented rule)
    rw = g['R_W']
    slots = [(wc * (1 - rw), [('r384', 0.65), ('r224', 0.35)]), ((1 - wc) * (1 - rw), list(cx.items()))]
    if g['READER_W']:
        slots.append((rw, [('rdr', 1.0)]))
    num = np.zeros((len(ids), len(LAB)))
    den = np.zeros((len(ids), len(LAB)))
    for w, parts in slots:
        s_num = sum(np.nan_to_num(mem[m]) * mw for m, mw in parts if m in mem and mw > 0) if any(m in mem for m, _ in parts) else 0
        s_den = sum(np.isfinite(mem[m]) * mw for m, mw in parts if m in mem and mw > 0) if any(m in mem for m, _ in parts) else 0
        s_num, s_den = np.asarray(s_num, float) + 0 * num, np.asarray(s_den, float) + 0 * num
        have = s_den > 0
        num += np.where(have, w * s_num / np.where(have, s_den, 1), 0)
        den += np.where(have, w * np.ones_like(num), 0)
    want = np.where(den > 0, num / np.where(den > 0, den, 1), 0.5)
    assert np.allclose(v, want, atol=1e-7), np.abs(v - want).max()
    rec = json.loads((work / 'blend_receipt.json').read_text())
    print('OK', tag, '| members', rec['members_used'], '| coverage', rec['coverage'])
    if mode == 'gold58':
        tb = pd.read_csv(work / 'gold58_auc.csv')
        print(tb[['candidate', 'macro', 'n']].to_string(index=False))
        assert tb['macro'].max() > 0.8, 'stub members built from the labels must score high'
        if extra:
            assert tb['candidate'].str.contains('R7flip').any() and tb['candidate'].str.contains('S6').any()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scratch', required=True)
    ap.add_argument('--input', default=None, help='a folder holding datasets/<owner>/<slug> mounts for the real S6F prep')
    a = ap.parse_args()
    scratch = Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    run('submit', scratch, a.input)
    run('submit', scratch, None, drop=('s6f',))
    run('submit', scratch, None, drop=('r7', 'r224'))
    run('gold58', scratch, a.input)
    run('gold58', scratch, None, partial=0.6)
    run('submit', scratch, None, extra=True)
    run('gold58', scratch, None, extra=True)
    run('submit', scratch, None, nb=ROOT / 'super_ensemble' / 'kernel' / 'convnext' / 'rsna-knee-super-ensemble-convnext.ipynb')
    # every kernel exactly as configured (v3: CoAtNets + R6 flip + S6F with per-finding weights; the hedge; the probe)
    run('submit', scratch, None, defaults=True)
    run('gold58', scratch, None, defaults=True, nb=ROOT / 'super_ensemble' / 'kernel' / 'gold58' / 'rsna-knee-super-ensemble-gold58.ipynb')
    run('submit', scratch, None, defaults=True, nb=ROOT / 'super_ensemble' / 'kernel' / 'convnext' / 'rsna-knee-super-ensemble-convnext.ipynb')
    run('submit', scratch, None, defaults=True, nb=ROOT / 'super_ensemble' / 'kernel' / 'probe' / 'rsna-knee-super-ensemble-probe.ipynb')
    print('ALL_LOCAL_SMOKE_PASSED')


if __name__ == '__main__':
    main()
