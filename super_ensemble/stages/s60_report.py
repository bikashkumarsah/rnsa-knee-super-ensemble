# Stage 6: run report.  In gold58 mode: AUC of every member and of fixed alternative blends on the 58 labelled studies
# (58 studies, 9 to 30 positives per finding: a sanity check with a standard error near 0.02, never a tuning target).
def auc(y, s):
    y, s = np.asarray(y, np.float64), np.asarray(s, np.float64)
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float('nan')
    r = pd.Series(s).rank(method='average').to_numpy()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


if KNEE_MODE == 'gold58':
    _gold = pd.read_csv(WORK / 'gold58_labels.csv', dtype={UID: str}).set_index(UID).reindex(TEST_IDS)
    _Y = _gold[LAB].to_numpy(np.float64)
    _cands = {('member ' + m): r for m, r in RANKS.items()}
    _cn = {'w': 0.5, 'members': {'r384': 0.65, 'r224': 0.35}}
    _alts = {'final (this notebook)': BLEND_SLOTS,
             'public 0.957 recipe': {'coatnet': _cn, 'convnext': {'w': 0.5, 'members': {'r7': 1.0}}},
             'coatnet slot only': {'coatnet': _cn}}
    _cx = {'R7+S6F': {'r7': 0.5, 's6f': 0.5}}
    if 's6' in RANKS:
        _cx.update({'R7+S6F+S6': {'r7': 1 / 3, 's6f': 1 / 3, 's6': 1 / 3}, 'R7+S6': {'r7': 0.5, 's6': 0.5},
                    'R7+S6F+0.5 S6': {'r7': 0.4, 's6f': 0.4, 's6': 0.2}})
    if 'r7f' in RANKS:
        _cx.update({'R7flip+S6F': {'r7f': 0.5, 's6f': 0.5}, 'R7flip alone': {'r7f': 1.0}})
        if 's6' in RANKS:
            _cx['R7flip+S6F+S6'] = {'r7f': 1 / 3, 's6f': 1 / 3, 's6': 1 / 3}
    for _a, _b in (('r6', 'R6'), ('r6f', 'R6flip')):
        if _a in RANKS:
            _cx.update({'%s+S6F' % _b: {_a: 0.6, 's6f': 0.4}, '%s alone' % _b: {_a: 1.0},
                        'R7+%s+S6F' % _b: {'r7': 0.3, _a: 0.3, 's6f': 0.4}})
            if 'stu' in RANKS:
                for _sw in (0.15, 0.25, 0.4):
                    _cx['%s+S6F+STU%.2f' % (_b, _sw)] = {_a: 0.6 * (1 - _sw), 's6f': 0.4 * (1 - _sw), 'stu': _sw}
    if 'stu' in RANKS:
        _cx['STU alone'] = {'stu': 1.0}
    # heliosli's per-finding Raptor weights of his 0.9586 public blend (v3), next to flat slot weights
    _hw = np.array([{'Medial OA': 0.5, 'Lateral OA': 0.15, 'PF OA': 0.65, 'Synovitis': 0.85, "Baker's": 0.30}.get(c, 0.5)
                    for c in LAB])
    for _name, _mem in _cx.items():
        for _w in (0.5, 0.4):
            _alts['%s | slots %.1f/%.1f' % (_name, _w, 1 - _w)] = {'coatnet': dict(_cn, w=_w), 'convnext': {'w': 1 - _w, 'members': _mem}}
        _alts['%s | heliosli per-finding' % _name] = {'coatnet': dict(_cn, w=_hw), 'convnext': {'w': 1 - _hw, 'members': _mem}}
        _alts['%s alone' % _name] = {'convnext': {'w': 1.0, 'members': _mem}}
    for _k, _s in _alts.items():
        _cands[_k] = blend(RANKS, _s)[0]
    _rows = []
    for _k, _r in _cands.items():
        _a = [auc(_Y[:, j], _r[:, j]) for j in range(len(LAB))]
        _rows.append(dict({'candidate': _k, 'macro': float(np.nanmean(_a)), 'n': int(np.isfinite(_r[:, 0]).sum())},
                          **{c: round(x, 4) for c, x in zip(LAB, _a)}))
    GOLD_TABLE = pd.DataFrame(_rows).sort_values('macro', ascending=False)
    GOLD_TABLE.to_csv(WORK / 'gold58_auc.csv', index=False)
    with pd.option_context('display.max_columns', 30, 'display.width', 250):
        print(GOLD_TABLE.round(4).to_string(index=False))
log('RUN_DONE', json.dumps({k: {'ok': v.get('ok'), 'seconds': v.get('seconds')} for k, v in STATUS.items()}))
