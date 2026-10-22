# Stage 5: per-finding percentile-rank blend.  AUC reads only the order inside each finding, and the members are calibrated
# very differently, so every member becomes ranks over the studies it actually scored, and ranks are averaged in two slots:
#   CoAtNet slot  = 0.65 rank(Raptor-384) + 0.35 rank(Raptor-224)        (heliosli / varshithakolupuri, public 0.957)
#   ConvNeXt slot = CONVNEXT_MEMBERS (v1: 0.5 rank(R7) + 0.5 rank(S6F all-data); v3: 0.6 R6 flip TTA + 0.4 S6F)
#   final         = w_c CoAtNet slot + (1 - w_c) ConvNeXt slot, one w_c per finding (W_COATNET: a number, or a map with
#                   '_' for every finding it does not name; v3 uses heliosli's per-finding Raptor weights of his 0.9586
#                   public blend)
# Weights are fixed before the run and never fitted on test predictions.  A member that failed (or a study it did not
# reach) drops out and the weights are renormalised inside its slot, then across slots: with only R7 left in the ConvNeXt
# slot and W_COATNET = 0.5 this is exactly the public 0.957 recipe.
W_COATNET = json.loads("""__W_COATNET__""")
CONVNEXT_MEMBERS = json.loads('__CONVNEXT_MEMBERS__')
if isinstance(W_COATNET, dict):
    assert '_' in W_COATNET and set(W_COATNET) - {'_'} <= set(LAB), W_COATNET
    W_C = np.array([float(W_COATNET.get(c, W_COATNET['_'])) for c in LAB])
else:
    W_C = np.full(len(LAB), float(W_COATNET))
assert ((W_C >= 0) & (W_C <= 1)).all(), W_C
# optional third slot: goodpjw2008's own reader at a small weight per finding (as in both public 0.958 notebooks); the
# CoAtNet / ConvNeXt slots share the rest.  If the reader did not run, its slot drops out and this is the two-slot blend.
READER_W = json.loads("""__READER_W__""")
R_W = np.array([float(READER_W.get(c, READER_W.get('_', 0.0))) for c in LAB]) if READER_W else np.zeros(len(LAB))
assert ((R_W >= 0) & (R_W < 0.5)).all(), R_W
BLEND_SLOTS = {
    'coatnet': {'w': W_C * (1.0 - R_W), 'members': {'r384': 0.65, 'r224': 0.35}},
    'convnext': {'w': (1.0 - W_C) * (1.0 - R_W), 'members': CONVNEXT_MEMBERS},
}
if READER_W:
    BLEND_SLOTS['reader'] = {'w': R_W, 'members': {'rdr': 1.0}}
MEMBER_FILES = {'r384': WORK / 'raptor_raw.csv', 'r224': WORK / 'raptor224_raw.csv', 'r7': WORK / 'r7_scored.csv',
                's6f': WORK / 's6f_scored.csv', 's6': WORK / 's6_scored.csv', 'r7f': WORK / 'r7f_scored.csv',
                'r6': WORK / 'r6_scored.csv', 'r6f': WORK / 'r6f_scored.csv', 'stu': WORK / 'stu_scored.csv',
                'rdr': WORK / 'rdr_scored.csv'}
RAPTOR_KIND = {'r384', 'r224'}           # these files hold every study; a failed study is left at 0.5 in all 12 columns
MIN_COVERAGE = 0.25                       # a member scoring fewer studies than this is dropped
SANITY_MIN_N, SANITY_MIN_RHO = 100, 0.45  # hidden test: a member whose mean rank corr with the others is below this is broken


def pct_rank(v):
    """Average-tie percentile ranks of the finite entries of v (NaN elsewhere)."""
    v = np.asarray(v, np.float64)
    out = np.full(len(v), np.nan)
    ok = np.isfinite(v)
    if ok.sum() == 1:
        out[ok] = 0.5
    elif ok.any():
        out[ok] = pd.Series(v[ok]).rank(method='average', pct=True).to_numpy()
    return out


def load_member(name):
    f = MEMBER_FILES[name]
    if not f.is_file():
        return None, 'no output file'
    d = pd.read_csv(f, dtype={UID: str})
    if not set(LAB) <= set(d.columns):
        return None, 'columns differ'
    d = d.drop_duplicates(UID).set_index(UID).reindex(TEST_IDS)[LAB]
    v = d.to_numpy(np.float64)
    bad = ~(np.isfinite(v) & (v >= 0) & (v <= 1))
    v[bad] = np.nan
    if name in RAPTOR_KIND:
        v[(np.abs(v - 0.5) < 1e-12).all(1)] = np.nan             # Raptor's own fallback row
    v[np.isnan(v).any(1)] = np.nan                                # a study is scored in all 12 findings or not at all
    cov = float(np.isfinite(v[:, 0]).mean()) if len(v) else 0.0
    if cov < MIN_COVERAGE:
        return None, 'coverage %.3f below %.2f' % (cov, MIN_COVERAGE)
    ok = np.isfinite(v[:, 0])
    if ok.sum() >= SANITY_MIN_N and any(np.nanstd(v[ok, j]) == 0 for j in range(len(LAB))):
        return None, 'a constant finding'
    return v, 'ok coverage %.4f' % cov


stage_begin('blend')
MEMBERS, member_notes = {}, {}
for _m in MEMBER_FILES:
    _v, _note = load_member(_m)
    member_notes[_m] = _note
    if _v is not None:
        MEMBERS[_m] = _v
RANKS = {m: np.stack([pct_rank(v[:, j]) for j in range(len(LAB))], 1) for m, v in MEMBERS.items()}

# sanity floor on the hidden test: drop a member that disagrees with every other member (a broken pipeline, not a weak
# model: the weakest public member still has mean rank corr ~0.8 with the others)
rho = {}
for a in RANKS:
    for b in RANKS:
        if a < b:
            ok = np.isfinite(RANKS[a][:, 0]) & np.isfinite(RANKS[b][:, 0])
            if ok.sum() >= 3:
                rho[a + '|' + b] = [float(pd.Series(RANKS[a][ok, j]).corr(pd.Series(RANKS[b][ok, j]), method='spearman'))
                                    for j in range(len(LAB))]
if len(TEST_IDS) >= SANITY_MIN_N and len(RANKS) >= 3:
    for m in list(RANKS):
        pairs = [np.nanmean(r) for k, r in rho.items() if m in k.split('|')]
        if pairs and max(pairs) < SANITY_MIN_RHO:
            member_notes[m] = 'dropped: max mean rank corr with another member %.3f < %.2f' % (max(pairs), SANITY_MIN_RHO)
            log('BLEND_DROP', m, member_notes[m])
            del RANKS[m]


def blend(ranks, slots=BLEND_SLOTS):
    n = len(TEST_IDS)
    num, den = np.zeros((n, len(LAB))), np.zeros((n, len(LAB)))
    for s in slots.values():
        snum, sden = np.zeros((n, len(LAB))), np.zeros((n, len(LAB)))
        for m, w in s['members'].items():
            if m in ranks and w > 0:
                r = ranks[m]
                ok = np.isfinite(r)
                snum += np.where(ok, r * w, 0.0)
                sden += np.where(ok, w, 0.0)
        have = sden > 0
        num += np.where(have, s['w'] * snum / np.where(have, sden, 1.0), 0.0)
        den += np.where(have, s['w'], 0.0)
    out = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.5)
    return out, den


FINAL, _den = blend(RANKS)
FINAL = np.clip(FINAL, 0.0, 1.0)   # per-finding weights can leave a rank at 1 + 1e-16 after the renormalisation
assert FINAL.shape == (len(TEST_IDS), len(LAB)) and np.isfinite(FINAL).all()
assert ((FINAL >= 0) & (FINAL <= 1)).all()
sub = pd.DataFrame(FINAL, columns=LAB)
sub.insert(0, UID, TEST_IDS)
sub = sub[SUB_COLS]
_out = WORK / ('submission.csv' if KNEE_MODE == 'submit' else 'gold58_blend.csv')
sub.to_csv(_out.with_suffix('.tmp'), index=False, float_format='%.9g')
os.replace(_out.with_suffix('.tmp'), _out)
coverage = {m: float(np.isfinite(r[:, 0]).mean()) for m, r in RANKS.items()}
no_member = int((_den[:, 0] == 0).sum())
_slots_json = {k: {'w': dict(zip(LAB, np.broadcast_to(np.asarray(v['w'], float), (len(LAB),)).round(4).tolist())),
                   'members': v['members']} for k, v in BLEND_SLOTS.items()}
BLEND_RECEIPT = {'mode': KNEE_MODE, 'rows': len(TEST_IDS), 'slots': _slots_json, 'members_used': sorted(RANKS),
                 'member_notes': member_notes, 'coverage': coverage, 'studies_without_any_member': no_member,
                 'rank_corr_mean': {k: round(float(np.nanmean(v)), 4) for k, v in rho.items()},
                 'stage_status': STATUS, 'total_seconds': round(elapsed(), 1), 'output': str(_out),
                 'output_sha256': sha256_file(_out)}
save_json(WORK / 'blend_receipt.json', BLEND_RECEIPT)
log('BLEND', json.dumps({k: BLEND_RECEIPT[k] for k in ('members_used', 'coverage', 'studies_without_any_member',
                                                         'rank_corr_mean', 'output')}, default=str))
_wanted = {m for s in BLEND_SLOTS.values() if np.any(np.asarray(s['w']) > 0) for m, w in s['members'].items() if w > 0}
if not _wanted <= set(RANKS):
    log('BLEND_FALLBACK members used: %s, missing %s (notes %s)' % (sorted(RANKS), sorted(_wanted - set(RANKS)), member_notes))
stage_end('blend', True, members=sorted(RANKS))
print(sub.head())
