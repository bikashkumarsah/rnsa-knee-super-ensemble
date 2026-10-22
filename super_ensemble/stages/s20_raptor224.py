# Stage 2: nartaa's second public checkpoint (224 px from a 0.8 centre crop, 0.945 alone) with the same anatomical
# mirror view as the 384 model (goodpjw2008 / varshithakolupuri's "one free flip"), run as its own process.
stage_begin('raptor224')
_rc, _sec = None, 0.0
if not RUN_COATNET:
    log('raptor224 disabled in this kernel')
elif elapsed() > RAPTOR224_START_BY:
    log('raptor224 skipped: %.2f h already used' % (elapsed() / H))
else:
    try:
        _rc, _sec = run_process([sys.executable, '-u', '/tmp/raptor224_run.py'], env=dict(os.environ),
                                kill_at=min(elapsed() + RAPTOR224_MAX_S, S6F_KILL), name='raptor224')
    except Exception as _e:
        log('raptor224 failed to start:', repr(_e))
RAPTOR224_OK = _rc == 0 and (WORK / 'raptor224_raw.csv').is_file()
stage_end('raptor224', RAPTOR224_OK, rc=_rc, process_seconds=round(_sec, 1))
