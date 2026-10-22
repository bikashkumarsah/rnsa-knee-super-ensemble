#!/usr/bin/env python3
"""Student model for the super-ensemble: goodpjw2008's public 2.5D ConvNeXt MIL reader (knee.py / preprocess.py from
goodpjw2008/rsna-knee-2-5d-convnext-reader, Apache 2.0, imported unchanged) retrained on soft targets that mix the mean
of four public LLM label tables with out-of-fold R7 predictions (our teacher kernel: each training study scored by the
heliosli fold model that never trained on it).

Runs in a Kaggle GPU session (T4 x2; internet on for the timm ImageNet weights -- this is not a submission notebook):

  python train_student.py --mode all [--limit N --epochs E ...]

  labels  -> WORK/student_labels.csv (soft targets, gold columns kept apart; never trained on the 58 gold studies)
  cache   -> /tmp/stu_cache/<series>.npy, the reader's canonical uint8 volumes at 256 px x 0.6 mm (its 153.6 mm field
             of view), at most 32 slices, the series with the most slices per (plane, fat-sat) slot
  train   -> one all-data model per GPU (seeds 0 and 1): fp16 autocast + GradScaler (bf16 is slow / wrong on T4),
             batch 2 studies x 4 accumulation steps, EMA weights, gold-58 macro AUC after every epoch;
             WORK/student_s<seed>.pt in the reader's infer.load_models format, rewritten after every epoch
"""
import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
       'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
UID = 'StudyInstanceUID'
INPUT = Path(os.environ.get('STU_INPUT', '/kaggle/input'))
WORK = Path(os.environ.get('STU_WORK', '/kaggle/working'))
CACHE = Path(os.environ.get('STU_CACHE', '/tmp/stu_cache'))
SIZE, MM_PER_PX, MAX_SLICES = 256, 0.6, 32
T_START = time.time()
SKIP_DIRS = {'competitions', 'rsna-knee-abnormality-detection', 'train_series', 'test_series'}


def log(*a):
    print('[%6.1f min]' % ((time.time() - T_START) / 60), *a, flush=True)


def find(name, required=True):
    """First file called `name` under the attached datasets / notebook outputs (the competition tree is skipped)."""
    for root, dirs, files in os.walk(INPUT):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        if name in files:
            return Path(root) / name
    if required:
        raise FileNotFoundError(name)
    return None


def find_all(name):
    hits = []
    for root, dirs, files in os.walk(INPUT):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        if name in files:
            hits.append(Path(root) / name)
    return hits


def comp_root():
    for p in (INPUT / 'competitions' / 'rsna-knee-abnormality-detection', INPUT / 'rsna-knee-abnormality-detection'):
        if (p / 'train.csv').is_file():
            return p
    raise FileNotFoundError('attach the competition data')


def reader():
    d = find('knee.py').parent
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    import knee
    import preprocess
    preprocess.SIZE, preprocess.MM_PER_PX, preprocess.MAX_SLICES = SIZE, MM_PER_PX, MAX_SLICES
    return preprocess, knee


# ---------------------------------------------------------------- labels
def build_labels(teacher_w):
    comp = comp_root()
    tr = pd.read_csv(comp / 'train.csv', dtype={UID: str})
    ids = tr[UID].tolist()
    gold = tr.set_index(UID)[LAB]
    is_gold = gold.notna().all(1)
    srcs = {}

    def add(name, fname, fix=None):
        f = find(fname, required=False)
        if f is None:
            log('label source missing:', fname)
            return
        d = pd.read_csv(f, dtype={UID: str})
        if fix:
            d = fix(d)
        if not set(LAB) <= set(d.columns):
            log('label source', fname, 'lacks', sorted(set(LAB) - set(d.columns)))
            return
        v = d.drop_duplicates(UID).set_index(UID)[LAB].apply(pd.to_numeric, errors='coerce').reindex(ids)
        srcs[name] = v.where((v >= 0) & (v <= 1))
    add('steven_v4', 'llm_labels_v4_blend.csv')
    add('pilkwang_v2', 'report_labels_v2.csv')
    add('lixin_gpt56', 'labels_llm_gpt56sol.csv')
    add('qwen3_8b', 'qwen_knee_weak_labels.csv',
        lambda q: q[[UID] + [c + '__label' for c in LAB]].set_axis([UID] + LAB, axis=1))
    assert srcs, 'no label source attached'
    num = sum(v.fillna(0) for v in srcs.values())
    den = sum(v.notna().astype(float) for v in srcs.values())
    cons = (num / den.where(den > 0)).clip(0.01, 0.99)
    tfiles = find_all('teacher_r7oof.csv')                  # one table per teacher kernel (folds 0-1 and 2-3)
    teach = None
    if tfiles:
        t = pd.concat([pd.read_csv(f, dtype={UID: str}) for f in tfiles], ignore_index=True).drop_duplicates(UID)
        log('teacher tables', [str(f) for f in tfiles], 'rows', len(t), 'folds', sorted(t['fold'].unique().tolist()))
        teach = t.set_index(UID)[LAB].reindex(ids).clip(0.001, 0.999)
        teach[is_gold.values] = np.nan                       # the teacher never scores gold; keep it that way
    y = cons.copy()
    if teach is not None:
        y = y.mask(cons.notna() & teach.notna(), (1 - teacher_w) * cons + teacher_w * teach)
        y = y.mask(cons.isna() & teach.notna(), teach)
    out = y.copy()
    out.insert(0, 'is_gold', is_gold.astype(int).values)
    for c in LAB:
        out['gold_' + c] = gold[c]
    out.to_csv(WORK / 'student_labels.csv')
    rep = {'sources': sorted(srcs), 'teacher_rows': int(teach.notna().all(1).sum()) if teach is not None else 0,
           'teacher_w': teacher_w if teach is not None else 0.0, 'train_rows': int((~is_gold).sum()),
           'cells_without_target': int(y[~is_gold.values].isna().sum().sum())}
    # diagnostic only: how each source ranks the 58 gold studies (the teacher never sees them, so it has no gold column)
    from sklearn.metrics import roc_auc_score
    g = is_gold.values
    for name, v in dict(srcs, consensus=cons).items():
        a = [roc_auc_score(gold[c][g], v[c][g].fillna(0.5)) for c in LAB if 0 < gold[c][g].sum() < g.sum()]
        rep['gold_auc_' + name] = round(float(np.mean(a)), 4)
    log('LABELS', json.dumps(rep))
    (WORK / 'student_labels_receipt.json').write_text(json.dumps(rep, indent=1))
    return out


# ---------------------------------------------------------------- cache
def _cache_job(job):
    st, sr, plane, sdir = job
    out = CACHE / (sr + '.npy')
    if out.is_file():
        return sr, int(np.load(out, mmap_mode='r').shape[0]), None
    try:
        preprocess, _ = reader()
        vol, _info = preprocess.load_series(sdir, plane)
        np.save(str(out) + '.tmp.npy', vol)
        os.replace(str(out) + '.tmp.npy', out)
        return sr, int(vol.shape[0]), None
    except Exception as e:  # an unreadable series is left out; its slot is then empty for that study
        return sr, 0, '%s: %s' % (type(e).__name__, str(e)[:160])


def fat_flag(v):
    return 1 if str(v).strip().lower() in ('1', '1.0', 'true', 'yes', 'y') else 0


def build_cache(studies, procs):
    _, knee = reader()
    comp = comp_root()
    CACHE.mkdir(parents=True, exist_ok=True)
    se = pd.read_csv(comp / 'train_series.csv', dtype=str)
    se = se[se[UID].isin(set(studies))]
    rows, jobs = [], []
    for st, g in se.groupby(UID):
        best = {}
        for sr, pl, fs in zip(g.SeriesInstanceUID, g.Anatomical_Plane, g.Fat_Suppression):
            if pl not in knee.PLANES:
                continue
            d = comp / 'train_series' / st / sr
            try:
                nfiles = len(os.listdir(d))
            except OSError:
                continue
            slot = knee.slot_of(pl, fat_flag(fs))
            if nfiles >= 3 and (slot not in best or nfiles > best[slot][1]):
                best[slot] = (sr, nfiles, pl, fat_flag(fs))
        for slot, (sr, nfiles, pl, fs) in best.items():
            rows.append({'study': st, 'series': sr, 'plane': pl, 'fs': fs, 'slot': slot})
            jobs.append((st, sr, pl, str(d.parent / sr)))
    log('cache: %d series of %d studies' % (len(jobs), len(set(r['study'] for r in rows))))
    n, errors = {}, {}
    with Pool(procs) as pool:
        for i, (sr, k, err) in enumerate(pool.imap_unordered(_cache_job, jobs, chunksize=8)):
            n[sr] = k
            if err:
                errors[sr] = err
            if i % 2000 == 0:
                log('cache %d/%d' % (i, len(jobs)))
    meta = pd.DataFrame(rows)
    meta['n'] = meta.series.map(n).fillna(0).astype(int)
    meta.to_csv(WORK / 'student_cache_meta.csv', index=False)
    log('cache done: %d series, %d errors%s' % (len(meta), len(errors), (' e.g. ' + next(iter(errors.values()))) if errors else ''))
    return meta


# ---------------------------------------------------------------- training (one process per GPU)
def train(a):
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    _, knee = reader()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    lab = pd.read_csv(WORK / 'student_labels.csv', dtype={UID: str}).set_index(UID)
    meta = pd.read_csv(WORK / 'student_cache_meta.csv', dtype={'study': str, 'series': str})
    meta = meta[meta.n >= 3]
    se = pd.DataFrame({UID: meta.study, 'SeriesInstanceUID': meta.series, 'Fat_Suppression': meta.fs,
                       'Anatomical_Plane': meta.plane})
    table = knee.build_study_table(se, dict(zip(meta.series, meta.n)))
    tr_ids = [u for u in lab.index[lab.is_gold == 0] if u in table and lab.loc[u, LAB].notna().any()]
    gold_ids = [u for u in lab.index[lab.is_gold == 1] if u in table]
    if a.limit:
        tr_ids = tr_ids[:a.limit]
    y_tr = lab.loc[tr_ids, LAB].to_numpy(np.float32)
    y_gold = lab.loc[gold_ids, ['gold_' + c for c in LAB]].to_numpy(np.float64)
    say = open(WORK / ('student_s%d_log.txt' % a.seed), 'a')

    def note(msg):
        log(msg)
        say.write(msg + '\n')
        say.flush()

    note('seed %d: train %d, gold %d, device %s, args %s' % (a.seed, len(tr_ids), len(gold_ids), dev, json.dumps(vars(a))))
    loader = lambda s: np.load(CACHE / (s + '.npy'), mmap_mode='r')   # noqa: E731  (fork workers: no pickling)
    dl_kw = dict(num_workers=a.workers, pin_memory=dev == 'cuda', persistent_workers=a.workers > 0)
    train_dl = torch.utils.data.DataLoader(knee.StudyDataset(tr_ids, table, str(CACHE), y_tr, a.k, train=True, loader=loader),
                                           batch_size=a.bs, shuffle=True, drop_last=True, **dl_kw)
    gold_dl = torch.utils.data.DataLoader(knee.StudyDataset(gold_ids, table, str(CACHE), None, a.k_eval, loader=loader),
                                          batch_size=a.bs, shuffle=False, **dl_kw)
    model = knee.KneeNet(a.backbone, pretrained=not a.no_pretrained).to(dev).to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    enc = list(model.enc.parameters())
    head = [p for n, p in model.named_parameters() if not n.startswith('enc.')]
    opt = torch.optim.AdamW([{'params': enc, 'lr': a.lr}, {'params': head, 'lr': a.lr_head}], weight_decay=a.wd)
    per_epoch = max(1, len(train_dl) // a.accum)
    steps, warm = a.epochs * per_epoch, per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
    use_amp = dev == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    @torch.no_grad()
    def predict(m):
        m.eval()
        out = []
        for x, mask, pos, _ in gold_dl:
            x, mask, pos = x.to(dev, non_blocking=True), mask.to(dev), pos.to(dev)
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                out.append(m(x, mask, pos, res=a.res).float().sigmoid().cpu())
        return torch.cat(out).numpy()

    def gold_auc(p):
        aucs = [roc_auc_score(y_gold[:, j], p[:, j]) for j in range(len(LAB)) if 0 < y_gold[:, j].sum() < len(y_gold)]
        return float(np.mean(aucs)), aucs

    step, ep_times = 0, []
    for ep in range(a.epochs):
        if ep_times and time.time() + 1.1 * max(ep_times) > a.deadline:
            note('stop before epoch %d: not enough time left' % ep)
            break
        t0, tot, n = time.time(), 0.0, 0
        model.train()
        opt.zero_grad(set_to_none=True)
        for i, (x, mask, pos, y) in enumerate(train_dl):
            x, mask, pos, y = x.to(dev, non_blocking=True), mask.to(dev), pos.to(dev), y.to(dev)
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                logit = model(x, mask, pos, res=a.res, train=True)
            w = torch.isfinite(y).float()
            loss_el = F.binary_cross_entropy_with_logits(logit.float(), torch.nan_to_num(y, nan=0.0), reduction='none')
            loss = (loss_el * w).sum() / w.sum().clamp_min(1.0)
            scaler.scale(loss / a.accum).backward()
            if (i + 1) % a.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                d = min(a.ema, (1 + step) / (10 + step))
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.lerp_(pm, 1 - d)
                    for be, bm in zip(ema.buffers(), model.buffers()):
                        be.copy_(bm)
            tot += float(loss)
            n += 1
            if n % 100 == 0 or (ep == 0 and n in (5, 10, 20, 50)):
                mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev == 'cuda' else 0.0
                note('  ep %d step %d/%d loss %.4f %.2f s/it peak %.1f GB' % (ep, n, len(train_dl), tot / n,
                                                                         (time.time() - t0) / n, mem))
            if time.time() > a.deadline:
                note('deadline inside epoch %d' % ep)
                break
        p = predict(ema)
        g, each = gold_auc(p)
        ep_times.append(time.time() - t0)
        note('ep %2d loss %.4f time %.0fs | gold %.4f [%s]' % (ep, tot / max(n, 1), ep_times[-1], g,
                                                              ' '.join('%.3f' % x for x in each)))
        ck = {'model': ema.state_dict(), 'args': {'backbone': a.backbone, 'res': a.res, 'size': SIZE, 'mm_per_px': MM_PER_PX,
                                                 'max_slices': MAX_SLICES, 'k_eval': a.k_eval, 'seed': a.seed, 'epoch': ep,
                                                 'gold_macro': g, 'train_studies': len(tr_ids)}}
        torch.save(ck, str(WORK / ('student_s%d.pt' % a.seed)) + '.tmp')
        os.replace(str(WORK / ('student_s%d.pt' % a.seed)) + '.tmp', WORK / ('student_s%d.pt' % a.seed))
        gp = pd.DataFrame(p, columns=LAB)
        gp.insert(0, UID, gold_ids)
        gp.to_csv(WORK / ('student_s%d_gold.csv' % a.seed), index=False)
        if time.time() > a.deadline:
            break
    note('seed %d done after %d optimizer steps' % (a.seed, step))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='all', choices=['all', 'labels', 'cache', 'train'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--seeds', default='0,1', help="mode all: one training process per seed, on GPU i for the i-th seed")
    ap.add_argument('--limit', type=int, default=0, help='smoke: use only this many training studies')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--hours', type=float, default=11.0, help='mode all: the whole run must end within this')
    ap.add_argument('--deadline', type=float, default=0.0, help='mode train: absolute end time (epoch seconds)')
    ap.add_argument('--teacher-w', type=float, default=0.5)
    ap.add_argument('--backbone', default='convnext_tiny')
    ap.add_argument('--no-pretrained', action='store_true')
    ap.add_argument('--res', type=int, default=256)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--k-eval', type=int, default=12)
    ap.add_argument('--bs', type=int, default=2)
    ap.add_argument('--accum', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lr-head', type=float, default=5e-4)
    ap.add_argument('--wd', type=float, default=0.02)
    ap.add_argument('--ema', type=float, default=0.998)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--cache-procs', type=int, default=4)
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if a.mode == 'train':
        return train(a)
    lab = build_labels(a.teacher_w)
    if a.mode == 'labels':
        return
    studies = list(lab.index[lab.is_gold == 1])
    pool = [u for u in lab.index[lab.is_gold == 0] if lab.loc[u, LAB].notna().any()]
    studies += pool[:a.limit] if a.limit else pool
    build_cache(studies, a.cache_procs)
    if a.mode == 'cache':
        return
    deadline = T_START + a.hours * 3600
    procs = []
    for i, seed in enumerate(int(s) for s in a.seeds.split(',')):
        cmd = [sys.executable, '-u', __file__, '--mode', 'train', '--seed', str(seed), '--deadline', str(deadline)]
        for k in ('limit', 'epochs', 'teacher_w', 'backbone', 'res', 'k', 'k_eval', 'bs', 'accum', 'lr', 'lr_head', 'wd',
                  'ema', 'workers'):
            cmd += ['--' + k.replace('_', '-'), str(getattr(a, k))]
        if a.no_pretrained:
            cmd.append('--no-pretrained')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        procs.append(subprocess.Popen(cmd, env=env))
    rcs = [p.wait() for p in procs]
    log('training processes exited', rcs)
    # gold-58 macro AUC of each seed and of their mean (the student's clean estimate: neither labels nor teacher saw gold)
    from sklearn.metrics import roc_auc_score
    y = lab[lab.is_gold == 1][['gold_' + c for c in LAB]]
    got = {}
    preds = []
    for seed in a.seeds.split(','):
        f = WORK / ('student_s%s_gold.csv' % seed)
        if f.is_file():
            p = pd.read_csv(f, dtype={UID: str}).set_index(UID).reindex(y.index)[LAB]
            preds.append(p.rank(pct=True))
            got['seed' + seed] = float(np.mean([roc_auc_score(y['gold_' + c], p[c]) for c in LAB]))
    if len(preds) > 1:
        m = sum(preds) / len(preds)
        got['mean'] = float(np.mean([roc_auc_score(y['gold_' + c], m[c]) for c in LAB]))
    log('STUDENT_GOLD58', json.dumps(got))
    (WORK / 'student_summary.json').write_text(json.dumps({'rcs': rcs, 'gold58': got, 'args': vars(a)}, indent=1))
    if any(rcs):
        raise SystemExit('a training process failed: %s' % rcs)


if __name__ == '__main__':
    main()
