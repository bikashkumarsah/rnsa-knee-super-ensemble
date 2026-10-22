# Stage 4a: the student model (super_ensemble/student/ in the source repository): goodpjw2008's public 2.5D ConvNeXt
# MIL reader (Apache 2.0) retrained by us on soft targets that mix four public LLM label tables with out-of-fold R7
# predictions (each training study scored by the heliosli fold model that never trained on it); the 58 gold studies
# were never trained on.  Optionally also goodpjw2008's own three reader folds at his geometry (the reader both public
# 0.958 notebooks blend in at 0.003-0.035 per finding).  Each group is its own process with a deadline; the scored rows
# go to WORK/stu_scored.csv and WORK/rdr_scored.csv.
STUDENT_INFER_SRC = __STUDENT_INFER_SRC__
STUDENT_START_BY = 5.6 * H
STUDENT_DEADLINE = 6.3 * H
STUDENT_KILL = 6.5 * H


def student_group(tag, group, enabled):
    stage_begin(tag)
    ok, info = False, {}
    try:
        if not enabled:
            raise RuntimeError('%s disabled in this kernel' % tag)
        if elapsed() > STUDENT_START_BY:
            raise RuntimeError('%s skipped: %.2f h already used' % (tag, elapsed() / H))
        Path('/tmp/student_infer.py').write_text(STUDENT_INFER_SRC)
        out = Path('/tmp/%s_pred' % tag)
        shutil.rmtree(out, ignore_errors=True)
        rc, sec = run_process([sys.executable, '-u', '/tmp/student_infer.py', str(DATA_ROOT), str(out),
                               str(NB_T0 + STUDENT_DEADLINE), group], env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                              kill_at=STUDENT_KILL, name=tag)
        p = out / 'predictions.csv'
        sc = pd.read_csv(p, dtype={UID: str}) if p.is_file() else pd.DataFrame(columns=[UID] + LAB)
        v = sc[LAB].to_numpy(np.float64) if len(sc) else np.zeros((0, len(LAB)))
        sc = sc[np.isfinite(v).all(1) & (v >= 0).all(1) & (v <= 1).all(1)].drop_duplicates(UID)
        sc = sc[sc[UID].isin(set(TEST_IDS))]
        sc.to_csv(WORK / ('%s_scored.csv' % tag), index=False, float_format='%.9g')
        if (out / 'receipt.json').is_file():
            shutil.copyfile(out / 'receipt.json', WORK / ('%s_receipt.json' % tag))
        info = {'rc': rc, 'process_seconds': round(sec, 1), 'scored': int(len(sc)), 'of': len(TEST_IDS)}
        ok = len(sc) > 0
    except Exception as e:
        info = {'error': repr(e)[:500], 'traceback': traceback.format_exc()[-2500:]}
        log('%s stage failed:' % tag, repr(e)[:500])
    stage_end(tag, ok, **info)
    return ok


STU_OK = student_group('stu', 'student', RUN_STU)
RDR_OK = student_group('rdr', 'reader', RUN_RDR)
