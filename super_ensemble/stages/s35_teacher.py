# Stage T (teacher kernel only): out-of-fold R7 predictions for every non-gold training study, the image teacher of the
# student model.  heliosli's trainer splits the non-gold pool into K hash folds, fold = int(sha1(StudyInstanceUID)[:8],
# 16) % K (compact_train.py; R7 has K = 4 fold models), so fold model k never trained on fold k's studies: scoring each
# study with only its held-out fold model is leak-free.  One pass per fold, each with a bundle folder holding that
# fold's model alone (the worker loads every <arm>_s*_f*.pt in --models) and a data root of that fold's studies (symlinks
# into train_series, as in gold58 mode).  Eager: with one fold model per study the loaders, not the GPU, set the pace.
import hashlib as _hl

TEACHER_K = 4
TEACHER_DEADLINE = float('__TEACHER_HOURS__') * H
TEACHER_LIMIT = int('__TEACHER_LIMIT__')          # > 0: a smoke run on that many studies per fold
TEACHER_FOLDS = [int(k) for k in '__TEACHER_FOLDS__'.split(',')]   # the folds this kernel scores (two kernels in parallel)
_train = pd.read_csv(COMP / 'train.csv', dtype={UID: str})
_gold_ids = set(_train.dropna(subset=LAB)[UID])
_pool = [u for u in _train[UID].astype(str) if u not in _gold_ids]
_fold = {u: int(_hl.sha1(u.encode()).hexdigest()[:8], 16) % TEACHER_K for u in _pool}
_ser = pd.read_csv(COMP / 'train_series.csv', dtype=str)
log('teacher pool', len(_pool), 'folds', np.bincount(list(_fold.values()), minlength=TEACHER_K).tolist(),
    'scoring folds', TEACHER_FOLDS, 'limit', TEACHER_LIMIT)


def teacher_root(k, ids):
    root = Path('/tmp/teacher_f%d' % k)
    shutil.rmtree(root, ignore_errors=True)
    (root / 'test_series').mkdir(parents=True)
    _ser[_ser[UID].isin(set(ids))].to_csv(root / 'test_series.csv', index=False)
    pd.DataFrame({UID: ids}).to_csv(root / 'test.csv', index=False)
    ss = pd.DataFrame({UID: ids})
    for c in LAB:
        ss[c] = 0.5
    ss.to_csv(root / 'sample_submission.csv', index=False)
    for u in ids:
        os.symlink(COMP / 'train_series' / u, root / 'test_series' / u)
    return root


def teacher_bundle(k, bundle):
    """A bundle folder with fold k's model only (every other bundle file symlinked unchanged)."""
    fb = TMP / ('bundle_r7_f%d' % k)
    shutil.rmtree(fb, ignore_errors=True)
    fb.mkdir(parents=True)
    models = sorted(Path(bundle).glob(ARM + '_s*_f*.pt'))
    keep = [m for m in models if m.name.endswith('_f%d.pt' % k)]
    assert len(models) == TEACHER_K and len(keep) == 1, [m.name for m in models]
    for f in Path(bundle).iterdir():
        if f in models and f not in keep:
            continue
        os.symlink(f, fb / f.name)
    return fb


stage_begin('teacher')
_t_rows, _t_info = [], {}
try:
    _bundle, _args = compact_setup()
    _args = [a for a in _args if a != '--compile']
    for _k in TEACHER_FOLDS:
        _ids = [u for u in _pool if _fold[u] == _k]
        if TEACHER_LIMIT:
            _ids = _ids[:TEACHER_LIMIT]
        if elapsed() > TEACHER_DEADLINE - 0.25 * H:
            log('teacher fold', _k, 'skipped: out of time')
            continue
        DATA_ROOT, TEST_IDS = teacher_root(_k, _ids), _ids     # run_compact reads both
        _info = run_compact('r7oof_f%d' % _k, teacher_bundle(_k, _bundle), _args, TEACHER_DEADLINE,
                            TEACHER_DEADLINE + 0.3 * H)
        _t_info['fold%d' % _k] = _info
        _sc = pd.read_csv(WORK / ('r7oof_f%d_scored.csv' % _k), dtype={UID: str})
        _sc.insert(1, 'fold', _k)
        _t_rows.append(_sc)
        log('teacher fold', _k, json.dumps({k: _info[k] for k in ('rc', 'scored', 'of', 'process_seconds')}))
    _tab = pd.concat(_t_rows, ignore_index=True) if _t_rows else pd.DataFrame(columns=[UID, 'fold'] + LAB)
    _tab.to_csv(WORK / 'teacher_r7oof.csv', index=False, float_format='%.6g')
    _t_info['rows'] = int(len(_tab))
    _t_info['of'] = sum(1 for u in _pool if _fold[u] in TEACHER_FOLDS) if not TEACHER_LIMIT else len(TEACHER_FOLDS) * TEACHER_LIMIT
except Exception as _e:
    _t_info.update(error=repr(_e)[:500], traceback=traceback.format_exc()[-2500:])
    log('teacher stage failed:', repr(_e)[:500])
stage_end('teacher', bool(_t_rows), **{k: v for k, v in _t_info.items() if k in ('rows', 'of', 'error', 'traceback')})
save_json(WORK / 'teacher_receipt.json', _t_info)
log('RUN_DONE', json.dumps({k: {'ok': v.get('ok'), 'seconds': v.get('seconds')} for k, v in STATUS.items()}))
