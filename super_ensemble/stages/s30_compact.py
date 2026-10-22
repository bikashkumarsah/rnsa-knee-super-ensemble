# Stage 3: heliosli's compact ConvNeXt-small bundles, set up by heliosli's own cell (written above, run here with exec so
# a failure is contained) and scored by his probe_dual controller on both GPUs.  R7 runs when KNEE_R7 is set; his S6
# and R6 bundles and the flip-TTA variants are optional members (stage flags in stage 0).  Changes, all marked [se]: his pinned wheels go to a
# private --target folder used only by these stages, deadlines follow this notebook's time plan, and the scored rows are
# read from the workers' predictions.csv (a study a worker never scored is left to the other members).
COMPACT = {}


def compact_setup():
    """heliosli's setup cell once: mainline code, pinned wheels, embedded sources and the R7 bundle."""
    if 'r7' not in COMPACT:
        exec(compile(Path('/tmp/r7_setup.py').read_text(), '/tmp/r7_setup.py', 'exec'), globals())
        COMPACT['r7'] = (BUNDLE, list(WORKER_ARGS))
    return COMPACT['r7']


def compact_bundle(tag, bundle_id, c_slug):
    """The bundle half of heliosli's setup cell for another bundle, into its own folder: (bundle folder, worker args)."""
    compact_setup()
    if tag not in COMPACT:
        text = Path('/tmp/r7_setup.py').read_text()
        anchor = "BUNDLE = TMP / 'bundle'\n"
        if text.count(anchor) != 1:
            raise RuntimeError('bundle section of the setup cell not found')
        part = text[text.index(anchor):].replace(anchor, 'BUNDLE = TMP / %r\n' % ('bundle_' + tag), 1)
        globals().update(BUNDLE_ID=bundle_id, C_SLUG=c_slug, FLIP_TTA=False)
        exec(compile(part, '/tmp/r7_setup.py[%s]' % tag, 'exec'), globals())
        COMPACT[tag] = (BUNDLE, list(WORKER_ARGS))
    return COMPACT[tag]


def run_compact(tag, bundle, worker_args, deadline, kill_at):
    """probe_dual over both GPUs with a deadline; the scored rows go to WORK/<tag>_scored.csv."""
    env = dict(os.environ, PYTHONPATH=str(R7_SITE) + os.pathsep + str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', ''),
               PYTHONDONTWRITEBYTECODE='1')
    pred = TMP / ('prediction_' + tag)
    shutil.rmtree(pred, ignore_errors=True)              # [se] never read a previous run's rows
    rc, sec = run_process([sys.executable, '-u', str(SRC / 'probe_dual.py'), '--data', str(DATA_ROOT), '--split', 'test',
                           '--code', str(CODE), '--bundle', str(bundle), '--arm', ARM, '--output', str(pred),
                           '--amp', 'auto', '--loader-workers', '2', '--deadline', str(NB_T0 + deadline),
                           '--worker', WORKER, '--worker-args', json.dumps(worker_args),
                           '--max-fallback-fraction', '1.0'], env=env, kill_at=kill_at, name=tag)
    for lg in sorted(pred.glob('worker*/log.txt')):
        shutil.copyfile(lg, WORK / ('%s_%s_log.txt' % (tag, lg.parent.name)))
    rows = []
    for f in sorted(pred.glob('worker*/predictions.csv')):
        d = pd.read_csv(f, dtype={UID: str})
        if set(LAB) <= set(d.columns):
            rows.append(d[[UID] + LAB])
    sc = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[UID] + LAB)
    v = sc[LAB].to_numpy(np.float64) if len(sc) else np.zeros((0, len(LAB)))
    sc = sc[np.isfinite(v).all(1) & (v >= 0).all(1) & (v <= 1).all(1)].drop_duplicates(UID)
    sc = sc[sc[UID].isin(set(TEST_IDS))]
    sc.to_csv(WORK / ('%s_scored.csv' % tag), index=False, float_format='%.9g')
    rcpt = {}
    if (pred / 'submission_receipt.json').is_file():
        rcpt = json.loads((pred / 'submission_receipt.json').read_text())
        shutil.copyfile(pred / 'submission_receipt.json', WORK / ('%s_receipt.json' % tag))
    return {'rc': rc, 'process_seconds': round(sec, 1), 'scored': int(len(sc)), 'of': len(TEST_IDS),
            'worker_args': worker_args,
            'controller': {k: rcpt.get(k) for k in ('status', 'scored', 'fallback_studies', 'seconds', 'per_study_mean')}}


def compact_stage(tag, deadline, kill_at, prepare):
    stage_begin(tag)
    ok, info = False, {}
    try:
        if elapsed() > deadline - 0.5 * H:
            raise RuntimeError('%s skipped: %.2f h already used' % (tag, elapsed() / H))
        bundle, args = prepare()
        info = run_compact(tag, bundle, args, deadline, kill_at)
        ok = info['scored'] > 0
    except Exception as e:
        info = {'error': repr(e)[:500], 'traceback': traceback.format_exc()[-2500:]}
        log('%s stage failed:' % tag, repr(e)[:500])
    stage_end(tag, ok, **info)
    return ok


R7_OK = compact_stage('r7', R7_DEADLINE, R7_KILL, compact_setup) if RUN_R7 else False
