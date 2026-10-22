# Kaggle submission guide — super-ensemble, CLI only

Submissions go through the Kaggle CLI 2.x in `.venv-kaggle`, wrapped by `super_ensemble/kaggle_flow.py`; no web-UI
clicks. The kernel is private: `bikashsah1230/rsna-knee-super-ensemble`.

1. Rebuild after any change and run the local smoke test:

   ```bash
   python3 super_ensemble/build_notebook.py
   ```

   ```bash
   python3 super_ensemble/tests/local_smoke.py --scratch /tmp/knee_smoke
   ```

2. Push a new version (always T4 x2) and wait for its commit run:

   ```bash
   python3 super_ensemble/kaggle_flow.py push submit
   ```

   ```bash
   python3 super_ensemble/kaggle_flow.py wait submit
   ```

3. Fetch the log and outputs; confirm `blend_receipt.json` lists all four members with coverage 1.0:

   ```bash
   python3 super_ensemble/kaggle_flow.py fetch submit
   ```

4. Submit that version's `submission.csv` (one of the 5 daily submissions):

   ```bash
   python3 super_ensemble/kaggle_flow.py submit --version <N> -m "super-ensemble vN"
   ```

5. Read the score once the hidden-test rerun finishes (about 4.5 h), then log it in
   `10_experiments/EXPERIMENT_LOG.md`:

   ```bash
   python3 super_ensemble/kaggle_flow.py wait-score <submission ref>
   ```

   ```bash
   python3 super_ensemble/kaggle_flow.py scores
   ```

The non-OAI hedge kernel is pushed and submitted the same way with `push convnext` and
`submit --kernel convnext --version <N> -m "..."`.

Do not submit a version whose commit run shows a stage that is not `ok`, fewer than four members, or a constant
column. The public split's SD is about 0.003: treat 0.001-0.002 differences as noise and keep the two final picks
diverse (best expected blend + a hedge), not two neighbouring weight tweaks.

## Final selection (before 2026-10-22 23:59 UTC)

Kaggle scores both final submissions on the private split and keeps the better one; if none are marked it falls back
to the best public scores, which would drop the non-OAI hedge. Mark them on the competition's **Submissions** page (the
CLI has no command for it):

1. **57435077** — super-ensemble v3, public 0.958 (the team's public-board submission).
2. **57443351** — the non-OAI hedge (R6 flip TTA + S6F, no OAI-trained weights), public 0.955. Hedge v4 (57457062,
   with the student and the reader) also shows 0.955; the simpler one is preferred because the student cost the full
   blend about 0.001 (v4b 0.957 vs v3 + reader 0.958).

`kaggle competitions team-submissions <team id>` shows which submission Kaggle currently uses on the public board (the
one with the highest unrounded score): useful when two submissions display the same three decimals.
