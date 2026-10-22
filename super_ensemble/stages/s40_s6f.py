# Stage 4: nartaa's S6 ConvNeXt trained on all data (public 0.950 alone; Heliosli's S6 architecture retrained on JEV-blend
# labels, gold-58 excluded, no OAI), scored exactly as nartaa's rsna-knee-s6fall-solo scores its component B: the same
# pinned code tree (s6_extras.bin payload + Heliosli's files, every file sha-checked), the same launcher and the same
# compact_infer arguments, one process per GPU.  Unlike the solo notebook a deadline is not fatal here: studies this
# stage did not reach keep the other members, and the blend receipt records the coverage.
S6F_SLUG = 'rsna-knee-s6fall-weights'
S6F_HELIOS_SLUG = 'rsna-orthofoundation-fold0-epoch7-20261004'
S6F_MODEL = {'file': 'E_prior_s42_fall.pt', 'sha256': '0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12',
             'run_config': 'E_prior_s42_fall.run_config.json', 'done': 'E_prior_s42_fall.DONE.json', 'arm': 'E_prior',
             'expect_epochs': 16, 'expect_train': 4263}
S6F_ANATOMY_FILE, S6F_ANATOMY_SHA256 = 'anatomy.pt', '5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e'
S6F_EXTRAS_FILE, S6F_PAYLOAD_MAGIC = 's6_extras.bin', b'K1PAYLOAD1\n'
S6F_SOURCE_SHA = {
    'px_cache.py': 'b13e38a7ad3bd496f8027e02209653e224b0926db4f0853eeb0d076a2d6e2632',
    'compact_train.py': '14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80',
    'compact_infer.py': '34f433fe72b732543813165c3397ec80cef01b0d49b25b7b22668714f0b0ef4a',
    'probe_infer.py': 'a6cc31448019c84be5868d0a2ff1d99719b23b0f86110efc03efff3ce6bd6980',
    'knee_anchor_bank_v2.py': 'e564807a129dfcd3c984348917436ee6083066a4d685340587ebd1d3ea14f309',
    'side_common.py': 'bbcc5d027834e6a6fd48c4a8cf228e25661b7522c792587c840fe4d7cd5c0566',
}
S6F_CODE_SHA = {
    'native_cache.py': 'c53ce8985406b3b918d9e821d43518479e89d9f946daebd817f378e3949718b9',
    'native_data.py': 'f59afe205c8e193369fa3ccd22edcce8df3b293205fc48ddb1dd45169ee43b36',
    'input_provenance.py': 'b3a22500c7f976f6ee040239ada5cacecf2326861cb49af1c7d8cfdeeb55fcda',
    'intensity_mapping.py': '19be2d8cbd6f8971c1288d7512a8cca41d143470c2ba6e9b9096fde035fa9d41',
    'coronal_views.py': '841f216d2682ae2c0f8afd82fc902b9fa74924d7145b952069dea8b0bbd03c40',
    'exact_duplicates.py': '51b38db21ce9c5fe2b6278a835983414756b61ac6af81aba380b44b41b075897',
    'dicom_geometry.py': '12aa501f9704aef15b7a3bbd560951acb6d95b89b7de0141d6cf89fc3f051ab1',
    'common.py': '8283fa1f8e7ef8cfacbba52df1715636be3c9ad688e39912814e7d5bcf57105f',
}
S6F_GEOMETRY = {'backbone': 'convnext_small', 'size': 320, 'crop': 288, 'views': 'v4', 'slices': 3, 'phys_mm': 3.5,
                'eval_stride': 1, 'chunk': 192, 'item_timeout': 900}
S6F_AMP, S6F_WORKERS = 'fp16', 2
S6F_LAUNCHER = __S6F_LAUNCHER_TEXT__


def s6f_find_mount(slug, marker):
    cands = [INPUT / slug, INPUT / 'competitions' / slug]
    cands += list(INPUT.glob('datasets/*/' + slug)) + list(INPUT.glob('datasets/*/' + slug + '/*'))
    cands += list(INPUT.glob('*/' + slug)) + list(INPUT.glob(slug + '/*'))
    found = []
    for p in cands:
        if (p / marker).is_file() and p.resolve() not in [f.resolve() for f in found]:
            found.append(p)
    if len(found) != 1:
        raise RuntimeError('cannot resolve %s (marker %s): %s' % (slug, marker, found))
    return found[0]


def s6f_safe_rel(name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise RuntimeError('unsafe payload member %r' % name)
    return p


def s6f_extract_payload(path, target, members):
    """nartaa's payload = magic + an uncompressed tar; exact member set, sizes and sha256 are enforced."""
    import tarfile
    if target.exists():
        shutil.rmtree(target)
    with open(path, 'rb') as f:
        if f.read(len(S6F_PAYLOAD_MAGIC)) != S6F_PAYLOAD_MAGIC:
            raise RuntimeError('payload magic mismatch')
        with tarfile.open(fileobj=f, mode='r:') as tar:
            ms = tar.getmembers()
            if {m.name for m in ms} != set(members) or len(ms) != len(members):
                raise RuntimeError('payload member set differs from the code manifest')
            for m in ms:
                s6f_safe_rel(m.name)
                if not m.isfile() or m.size != members[m.name]['bytes']:
                    raise RuntimeError('payload member %s: not a regular file or size differs' % m.name)
            for m in ms:
                dst = target / s6f_safe_rel(m.name)
                dst.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(m) as src, open(dst, 'xb') as out:
                    shutil.copyfileobj(src, out, 1 << 20)
    for name, rec in members.items():
        if sha256_file(target / name) != rec['sha256']:
            raise RuntimeError('payload member %s sha differs' % name)


def s6f_prepare(tmp):
    """The exact code tree, model folder and launcher of nartaa's scored component B."""
    I = {}
    ds = s6f_find_mount(S6F_SLUG, 'checkpoint-manifest.json')
    helios = s6f_find_mount(S6F_HELIOS_SLUG, S6F_ANATOMY_FILE)
    man = json.loads((ds / 'code_manifest.json').read_text())
    if sha256_file(ds / S6F_MODEL['file']) != S6F_MODEL['sha256']:
        raise RuntimeError('S6F checkpoint sha differs')
    if sha256_file(helios / S6F_ANATOMY_FILE) != S6F_ANATOMY_SHA256:
        raise RuntimeError('anatomy.pt differs from the S6 anchor checkpoint')
    cfg = json.loads((ds / S6F_MODEL['run_config']).read_text())
    g = S6F_GEOMETRY
    want = {'size': g['size'], 'crop': g['crop'], 'views': g['views'], 'slices': g['slices'], 'phys_mm': g['phys_mm'],
            'backbone': g['backbone'], 'spatial_attn': True, 'arms': S6F_MODEL['arm']}
    bad = {k: (cfg.get(k), v) for k, v in want.items() if cfg.get(k) != v}
    if bad:
        raise RuntimeError('S6F run_config differs from the inference arguments: %s' % bad)
    done = json.loads((ds / S6F_MODEL['done']).read_text())
    if done.get('model_sha256') != S6F_MODEL['sha256'] or done.get('epochs') != S6F_MODEL['expect_epochs'] \
            or done.get('train_studies') != S6F_MODEL['expect_train']:
        raise RuntimeError('S6F DONE marker does not record this model')
    code_root = tmp / 'k1src'
    extras = {n: man['members'][n] for n in man['from_this_dataset']}
    s6f_extract_payload(ds / S6F_EXTRAS_FILE, code_root, extras)
    for n, rec in man['members'].items():
        if n in extras:
            continue
        src, dst = helios / s6f_safe_rel(n), code_root / s6f_safe_rel(n)
        if not src.is_file() or src.stat().st_size != rec['bytes']:
            raise RuntimeError('heliosli file %s missing or size differs' % n)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        if sha256_file(dst) != rec['sha256']:
            raise RuntimeError('heliosli file %s sha differs' % n)
    I['src'], I['code'] = code_root / 'heliosli', code_root / 'cloud_code'
    for n, s in S6F_SOURCE_SHA.items():
        if sha256_file(I['src'] / n) != s:
            raise RuntimeError('heliosli/%s differs from the pinned sha' % n)
    for n, s in S6F_CODE_SHA.items():
        if sha256_file(I['code'] / 'training' / n) != s:
            raise RuntimeError('cloud_code/training/%s differs from the pinned sha' % n)
    I['models'] = tmp / 'k1models'
    if I['models'].exists():
        shutil.rmtree(I['models'])
    I['models'].mkdir(parents=True)
    os.symlink(ds / S6F_MODEL['file'], I['models'] / S6F_MODEL['file'])      # the folder holds exactly this one model
    I['anatomy'] = helios / S6F_ANATOMY_FILE
    I['launcher'] = tmp / 'k1launch' / 'k1_b_infer.py'
    I['launcher'].parent.mkdir(parents=True, exist_ok=True)
    I['launcher'].write_text(S6F_LAUNCHER.replace('__SRC__', repr(str(I['src']))).replace('__CODE__', repr(str(I['code']))))
    return I


def s6f_cmd(I, assignment, out_dir, tmp_dir, deadline):
    g = S6F_GEOMETRY
    return [sys.executable, '-u', str(I['launcher']), '--data', str(DATA_ROOT), '--split', 'test', '--code', str(I['code']),
            '--models', str(I['models']), '--arm', S6F_MODEL['arm'], '--backbone', g['backbone'], '--size', str(g['size']),
            '--crop', str(g['crop']), '--views', g['views'], '--slices', str(g['slices']), '--phys-mm', str(g['phys_mm']),
            '--eval-stride', str(g['eval_stride']), '--anchor-weights', str(I['anatomy']), '--amp', S6F_AMP,
            '--chunk', str(g['chunk']), '--workers', str(S6F_WORKERS), '--item-timeout', str(g['item_timeout']),
            '--assignment', str(assignment), '--output', str(out_dir), '--tmp', str(tmp_dir),
            '--dump-logits', str(Path(out_dir) / 'logits.json'), '--deadline', '%.0f' % deadline]


def s6f_env(gi):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gi), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                MALLOC_ARENA_MAX='2', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1')


def s6f_run():
    tmp = Path('/tmp/s6f')
    tmp.mkdir(parents=True, exist_ok=True)
    I = s6f_prepare(tmp)
    log('S6F inputs verified: model', S6F_MODEL['sha256'][:16], '| code', I['src'])
    cost = pd.read_csv(DATA_ROOT / 'test_series.csv', dtype=str, usecols=[UID])[UID].value_counts().to_dict()
    order = sorted(TEST_IDS, key=lambda u: (-cost.get(u, 0), u))          # nartaa's split_parts: big studies first,
    parts = [order[i::N_GPU] for i in range(N_GPU)]                       # spread over the GPUs, then interleaved
    parts = [p[::2] + p[1::2] for p in parts if p]
    procs = []
    for gi, part in enumerate(parts):
        pdir = tmp / ('part%d' % gi)
        if pdir.exists():
            shutil.rmtree(pdir)
        pdir.mkdir(parents=True)
        (pdir / 'ids.json').write_text(json.dumps(part))
        fh = open(pdir / 'log.txt', 'w')
        p = subprocess.Popen(s6f_cmd(I, pdir / 'ids.json', pdir, tmp / ('tmp%d' % gi), NB_T0 + S6F_DEADLINE),
                             env=s6f_env(gi), cwd=str(tmp), stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
        procs.append((gi, p, fh, pdir, part))
        log('S6F part %d: %d studies, pid %d' % (gi, len(part), p.pid))
    last = time.time()
    try:
        while any(p.poll() is None for _, p, _, _, _ in procs):
            if elapsed() > S6F_KILL:
                log('KILL S6F at its hard stop')
                break
            if time.time() - last > 300:
                last = time.time()
                for gi, p, _, pdir, part in procs:
                    tail = [l for l in (pdir / 'log.txt').read_text(errors='replace').splitlines() if l.startswith('{')][-1:]
                    log('S6F part %d of %d: %s' % (gi, len(part), tail[0][:300] if tail else '(starting)'))
            time.sleep(5)
    finally:
        for _, p, fh, _, _ in procs:
            kill_group(p)
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                pass
            fh.close()
    rows, info = [], {}
    for gi, p, _, pdir, part in procs:
        shutil.copyfile(pdir / 'log.txt', WORK / ('s6f_part%d_log.txt' % gi))
        f = pdir / 'predictions.csv'
        if f.is_file():
            d = pd.read_csv(f, dtype={UID: str})
            if set(LAB) <= set(d.columns):
                rows.append(d[[UID] + LAB])
        rc = {}
        if (pdir / 'receipt.json').is_file():
            rc = json.loads((pdir / 'receipt.json').read_text())
        info['part%d' % gi] = {'returncode': p.returncode, 'studies': len(part), 'scored': rc.get('scored'),
                               'timed_out': rc.get('timed_out'), 'seconds': rc.get('seconds'), 'stats': rc.get('stats'),
                               'fallback': dict(list((rc.get('fallback') or {}).items())[:10]),
                               'anchor_methods': rc.get('anchor_methods'), 'amp': rc.get('amp')}
    sc = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[UID] + LAB)
    v = sc[LAB].to_numpy(np.float64) if len(sc) else np.zeros((0, len(LAB)))
    sc = sc[np.isfinite(v).all(1) & (v >= 0).all(1) & (v <= 1).all(1)].drop_duplicates(UID)
    sc = sc[sc[UID].isin(set(TEST_IDS))]
    sc.to_csv(WORK / 's6f_scored.csv', index=False, float_format='%.9g')
    info['scored'] = int(len(sc))
    return info


stage_begin('s6f')
S6F_OK, _s6f_info = False, {}
try:
    if not RUN_S6F:
        raise RuntimeError('S6F disabled in this kernel')
    if elapsed() > S6F_START_BY:
        raise RuntimeError('S6F skipped: %.2f h already used' % (elapsed() / H))
    _s6f_info = s6f_run()
    S6F_OK = _s6f_info.get('scored', 0) > 0
except Exception as _e:
    _s6f_info = {'error': repr(_e)[:500], 'traceback': traceback.format_exc()[-2500:]}
    log('S6F stage failed:', repr(_e)[:500])
stage_end('s6f', S6F_OK, **_s6f_info)
