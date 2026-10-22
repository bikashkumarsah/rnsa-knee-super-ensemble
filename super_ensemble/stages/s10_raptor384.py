# Stage 1: nartaa's public 0.949 CoAtNet (384 px, anatomical mirror TTA, 94 windows), the script written above run
# unchanged as its own process.  It writes raptor_raw.csv (probabilities; a study it failed on stays 0.5 everywhere).
stage_begin('raptor384')
_rc, _sec = None, 0.0
if not RUN_COATNET:
    log('raptor384 disabled in this kernel')
else:
    try:
        _rc, _sec = run_process([sys.executable, '-u', '/tmp/raptor_run.py'], env=dict(os.environ),
                                kill_at=min(RAPTOR384_MAX_S, S6F_KILL), name='raptor384')
    except Exception as _e:
        log('raptor384 failed to start:', repr(_e))
RAPTOR384_OK = _rc == 0 and (WORK / 'raptor_raw.csv').is_file()
stage_end('raptor384', RAPTOR384_OK, rc=_rc, process_seconds=round(_sec, 1))
try:
    subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader'], check=False, timeout=60)
except Exception:
    pass
