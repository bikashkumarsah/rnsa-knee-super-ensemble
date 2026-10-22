"""Cheap blend ideas evaluated on the 58 labelled studies (our gold58 kernel outputs).

  python3 _analysis/gold58_blend_ideas.py super_ensemble/runs/gold58

Mirrors super_ensemble/stages/s50_blend.py: per-member percentile ranks, CoAtNet slot = 0.65 r384 + 0.35 r224,
ConvNeXt slot = members, final = w_c(finding) * CoAtNet + (1 - w_c) * ConvNeXt.  Reports macro AUC and a paired
bootstrap of the change against v2 (w_c = 0.4 everywhere, ConvNeXt = 0.5 R7 + 0.5 S6F).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RUN = Path(sys.argv[1])
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
       'Synovitis', "Baker's", 'Contusion', 'Fracture']
UID = 'StudyInstanceUID'
y = pd.read_csv(RUN / 'gold58_labels.csv', dtype={UID: str}).set_index(UID)[LAB]
ids = y.index
Y = y.to_numpy()
files = {'r384': 'raptor_raw.csv', 'r224': 'raptor224_raw.csv', 'r7': 'r7_scored.csv', 's6f': 's6f_scored.csv',
         's6': 's6_scored.csv', 'r7f': 'r7f_scored.csv'}
P = {m: pd.read_csv(RUN / f, dtype={UID: str}).set_index(UID).reindex(ids)[LAB].to_numpy(np.float64)
     for m, f in files.items()}
for m, v in P.items():
    assert np.isfinite(v).all(), m


def ranks(v):
    return np.stack([pd.Series(v[:, j]).rank(pct=True).to_numpy() for j in range(v.shape[1])], 1)


R = {m: ranks(v) for m, v in P.items()}
logit = {m: np.log(np.clip(v, 1e-7, 1 - 1e-7) / (1 - np.clip(v, 1e-7, 1 - 1e-7))) for m, v in P.items()}
Z = {m: (l - l.mean(0)) / (l.std(0) + 1e-9) for m, l in logit.items()}


def macro(pred, idx=None):
    idx = np.arange(len(Y)) if idx is None else idx
    out = []
    for j in range(len(LAB)):
        yy = Y[idx, j]
        if 0 < yy.sum() < len(yy):
            out.append(roc_auc_score(yy, pred[idx, j]))
    return float(np.mean(out))


def per_finding(pred):
    return [roc_auc_score(Y[:, j], pred[:, j]) for j in range(len(LAB))]


def blend(wc, conv=None, src=R):
    conv = conv or {'r7': 0.5, 's6f': 0.5}
    cs = 0.65 * src['r384'] + 0.35 * src['r224']
    tot = sum(conv.values())
    vs = sum(w * src[m] for m, w in conv.items()) / tot
    wc = np.array([wc.get(c, wc.get('_', 0.4)) if isinstance(wc, dict) else wc for c in LAB])
    return wc * cs + (1 - wc) * vs


rng = np.random.default_rng(0)
BOOT = [rng.integers(0, len(Y), len(Y)) for _ in range(2000)]


def compare(a, b):
    d = []
    for idx in BOOT:
        try:
            d.append(macro(a, idx) - macro(b, idx))
        except ValueError:
            pass
    d = np.array(d)
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()


base = blend(0.4)
H = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, "Baker's": 0.35, '_': 0.5}
H40 = {'Medial OA': 0.4, 'Lateral OA': 0.1, 'PF OA': 0.5, 'Synovitis': 0.7, "Baker's": 0.25, '_': 0.4}
cands = {
    'v2: slots 0.4/0.6 (R7+S6F)': base,
    'v1: slots 0.5/0.5': blend(0.5),
    'heliosli per-finding weights (base 0.5)': blend(H),
    'heliosli deltas on base 0.4': blend(H40),
    'Lateral OA 0.2 only, else 0.4': blend({'Lateral OA': 0.2, '_': 0.4}),
    'agree-only: LatOA 0.2, LatMen 0.3, Effusion/Fracture/MedMen 0.5, else 0.4':
        blend({'Lateral OA': 0.2, 'Lateral Meniscus': 0.3, 'Effusion': 0.5, 'Fracture': 0.5, 'Medial Meniscus': 0.5, '_': 0.4}),
    'z-logit mean instead of rank mean, 0.4/0.6': blend(0.4, src=Z),
    'v2 + S6 at 0.2 of the ConvNeXt slot': blend(0.4, {'r7': 0.4, 's6f': 0.4, 's6': 0.2}),
    'equal 5-member rank mean (r384,r224,r7,s6f,s6)': (R['r384'] + R['r224'] + R['r7'] + R['s6f'] + R['s6']) / 5,
}
rows = []
for k, v in cands.items():
    m, lo, hi, p = compare(v, base) if v is not base else (0, 0, 0, np.nan)
    rows.append((k, macro(v), m, lo, hi, p))
print('%-78s %7s %8s %18s %6s' % ('candidate', 'macro', 'delta', '95% interval', 'P>0'))
for k, mac, m, lo, hi, p in rows:
    print('%-78s %.4f %+.4f [%+.4f, %+.4f] %5.2f' % (k, mac, m, lo, hi, p))

# per-finding AUC of the two slots and the members
cs = 0.65 * R['r384'] + 0.35 * R['r224']
vs = 0.5 * R['r7'] + 0.5 * R['s6f']
tab = pd.DataFrame({'pos': Y.sum(0).astype(int), 'coatnet slot': per_finding(cs), 'convnext slot': per_finding(vs),
                    'r384': per_finding(P['r384']), 'r7': per_finding(P['r7']), 's6f': per_finding(P['s6f']),
                    'v2': per_finding(base)}, index=LAB)
# per-finding best slot weight on gold-58 (grid) -- optimistic by construction
grid = np.linspace(0, 1, 21)
best = []
for j, c in enumerate(LAB):
    a = [roc_auc_score(Y[:, j], w * cs[:, j] + (1 - w) * vs[:, j]) for w in grid]
    best.append(grid[int(np.argmax(a))])
tab['gold-best w_coatnet'] = best
tab['heliosli w_raptor'] = [H.get(c, 0.5) for c in LAB]
print()
print(tab.round(3).to_string())

# nested check: does choosing per-finding weights on half the studies help the other half?
gains = []
for rep in range(300):
    perm = rng.permutation(len(Y))
    tr, te = perm[:29], perm[29:]
    w = []
    for j in range(len(LAB)):
        if not (0 < Y[tr, j].sum() < len(tr)):
            w.append(0.4); continue
        a = [roc_auc_score(Y[tr, j], g * cs[tr, j] + (1 - g) * vs[tr, j]) for g in grid]
        w.append(grid[int(np.argmax(a))])
    w = np.array(w)
    tuned = w * cs + (1 - w) * vs
    try:
        gains.append(macro(tuned, te) - macro(base, te))
    except ValueError:
        pass
gains = np.array(gains)
print('\nper-finding weights tuned on 29 studies, scored on the other 29 (300 splits): mean %+.4f, P(gain>0) %.2f'
      % (gains.mean(), (gains > 0).mean()))

# rank correlations (mean over findings) between members
names = ['r384', 'r224', 'r7', 's6f', 's6', 'r7f']
C = pd.DataFrame(index=names, columns=names, dtype=float)
for a in names:
    for b in names:
        C.loc[a, b] = np.mean([pd.Series(R[a][:, j]).corr(pd.Series(R[b][:, j]), method='spearman') for j in range(len(LAB))])
print('\nmean per-finding Spearman correlation on gold-58')
print(C.round(3).to_string())
