#!/usr/bin/env python3
"""Kaggle CLI workflow for the super-ensemble kernels: push, wait, fetch logs/outputs, submit, list scores.

Uses the Kaggle CLI 2.x in .venv-kaggle (it can pin the T4 x2 accelerator; the base install 1.7 cannot).  Credentials
come from ~/.kaggle/kaggle.json as usual.

  python super_ensemble/kaggle_flow.py push submit          # new kernel version, run on T4 x2
  python super_ensemble/kaggle_flow.py wait submit          # poll until the version finishes
  python super_ensemble/kaggle_flow.py fetch submit         # logs + output files into runs/<mode>/
  python super_ensemble/kaggle_flow.py submit --version 3 -m "super ensemble v1"
  python super_ensemble/kaggle_flow.py scores               # this competition's submissions and scores
  python super_ensemble/kaggle_flow.py wait-score 57219200  # poll until that submission is scored
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COMPETITION = 'rsna-knee-abnormality-detection'
CLI = os.environ.get('KAGGLE_CLI') or str(ROOT / '.venv-kaggle' / 'bin' / 'kaggle')
ACCELERATOR = 'NvidiaTeslaT4'
KERNEL_CHOICES = ['submit', 'gold58', 'convnext', 'probe', 'probe2', 'teacher', 'teacherb', 'teachersmoke', 'student',
                  'studentsmoke']


def kernel_id(mode):
    return json.loads((HERE / 'kernel' / mode / 'kernel-metadata.json').read_text())['id']


def cli(*args, capture=True, check=True):
    cmd = [CLI, *args]
    r = subprocess.run(cmd, capture_output=capture, text=True)
    if check and r.returncode != 0:
        sys.stderr.write((r.stdout or '') + (r.stderr or ''))
        raise SystemExit('kaggle %s failed (exit %d)' % (' '.join(args[:2]), r.returncode))
    return r.stdout if capture else ''


SUBMISSION_KERNELS = {'submit', 'convnext', 'probe', 'probe2'}


def cmd_push(a):
    """Without --timeout Kaggle reserves a full 12 h session of GPU quota; the training kernels are pushed with a cap.
    Submission kernels are never capped: a capped version could carry the cap into its hidden-test rerun."""
    folder = HERE / 'kernel' / a.mode
    args = ['kernels', 'push', '-p', str(folder), '--accelerator', ACCELERATOR]
    if a.timeout:
        if a.mode in SUBMISSION_KERNELS:
            raise SystemExit('no --timeout for submission kernels (%s)' % a.mode)
        args += ['--timeout', str(int(a.timeout))]
    out = cli(*args)
    print(out.strip())


def status(mode):
    out = cli('kernels', 'status', kernel_id(mode), check=False)
    return out.strip()


def cmd_status(a):
    print(status(a.mode))


def cmd_wait(a):
    t0 = time.time()
    while True:
        s = status(a.mode)
        low = s.lower()
        print('[%5.1f min] %s' % ((time.time() - t0) / 60, s), flush=True)
        if any(k in low for k in ('complete', 'error', 'cancel')):
            return
        if time.time() - t0 > a.timeout:
            raise SystemExit('timed out waiting')
        time.sleep(a.every)


def cmd_fetch(a):
    dest = HERE / 'runs' / a.mode
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    log = cli('kernels', 'logs', kernel_id(a.mode), check=False)
    (dest / 'kernel_log.txt').write_text(log)
    print(cli('kernels', 'output', kernel_id(a.mode), '-p', str(dest), check=False).strip()[-2000:])
    print('saved to', dest.relative_to(ROOT))


def cmd_submit(a):
    out = cli('competitions', 'submit', COMPETITION, '-k', kernel_id(a.kernel), '-v', str(a.version),
              '-f', 'submission.csv', '-m', a.message)
    print(out.strip())


def cmd_scores(a):
    print(cli('competitions', 'submissions', COMPETITION).strip())


def submission_row(ref):
    out = cli('competitions', 'submissions', COMPETITION, '--csv', check=False)
    for line in out.splitlines():
        if line.startswith(str(ref) + ','):
            return line
    return None


def cmd_wait_score(a):
    """Poll until submission <ref> leaves PENDING; prints its row (status, public score)."""
    t0 = time.time()
    while True:
        row = submission_row(a.ref) or '(not listed yet)'
        if 'PENDING' not in row and row != '(not listed yet)':
            print('[%5.1f min] %s' % ((time.time() - t0) / 60, row), flush=True)
            return
        if time.time() - t0 > a.timeout:
            raise SystemExit('timed out waiting for submission %s' % a.ref)
        time.sleep(a.every)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name, fn in (('push', cmd_push), ('status', cmd_status), ('fetch', cmd_fetch)):
        p = sub.add_parser(name)
        p.add_argument('mode', choices=KERNEL_CHOICES)
        if name == 'push':
            p.add_argument('--timeout', type=float, default=0, help='seconds; caps the run and its quota reservation')
        p.set_defaults(fn=fn)
    p = sub.add_parser('wait')
    p.add_argument('mode', choices=KERNEL_CHOICES)
    p.add_argument('--every', type=int, default=60)
    p.add_argument('--timeout', type=int, default=4 * 3600)
    p.set_defaults(fn=cmd_wait)
    p = sub.add_parser('submit')
    p.add_argument('--version', required=True, help='the kernel version whose submission.csv is submitted')
    p.add_argument('--kernel', default='submit', choices=sorted(SUBMISSION_KERNELS), help='which submission kernel')
    p.add_argument('-m', '--message', required=True)
    p.set_defaults(fn=cmd_submit)
    p = sub.add_parser('scores')
    p.set_defaults(fn=cmd_scores)
    p = sub.add_parser('wait-score')
    p.add_argument('ref', help='submission ref from `scores`')
    p.add_argument('--every', type=int, default=300)
    p.add_argument('--timeout', type=int, default=10 * 3600)
    p.set_defaults(fn=cmd_wait_score)
    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
