# Stage 4b (optional members, stage flags in stage 0): heliosli's S6 bundle 209a32c8b50a6b1c with the same embedded
# code as R7 (his own S6 notebook runs it with these sources), compiled like R7 (the worker checks compiled against eager
# and falls back to eager), and R7 with heliosli's flip TTA (--flip-tta: mirrored windows, Medial/Lateral outputs
# swapped back, MCL from the plain pass).  Each scores into its own <tag>_scored.csv; the blend uses whichever is listed.
# R6 is heliosli's bundle fe055406fad6a12f from his versioned copy rsna-knee-r6-frozen-fe055406-w1 (4 hash folds, gold
# never trained; the model of his 0.958 public blend, which scores it with flip TTA); his newest notebook runs it through
# this same setup cell with only the bundle id, the dataset slug and the flip switch changed.


def _s6_prepare():
    bundle, args = compact_bundle('s6', '209a32c8b50a6b1c', 'rsna-knee-compact-v1')
    return bundle, args + ([] if '--compile' in args else ['--compile'])


def _r7f_prepare():
    bundle, args = compact_setup()
    return bundle, args + ([] if '--flip-tta' in args else ['--flip-tta'])


def _r6_prepare():
    return compact_bundle('r6', 'fe055406fad6a12f', 'rsna-knee-r6-frozen-fe055406-w1')


def _r6f_prepare():
    bundle, args = _r6_prepare()
    return bundle, args + ([] if '--flip-tta' in args else ['--flip-tta'])


S6_OK = compact_stage('s6', EXTRA_DEADLINE, EXTRA_KILL, _s6_prepare) if RUN_S6 else False
R7F_OK = compact_stage('r7f', EXTRA_DEADLINE, EXTRA_KILL, _r7f_prepare) if RUN_R7F else False
R6_OK = compact_stage('r6', EXTRA_DEADLINE, EXTRA_KILL, _r6_prepare) if RUN_R6 else False
R6F_OK = compact_stage('r6f', EXTRA_DEADLINE, EXTRA_KILL, _r6f_prepare) if RUN_R6F else False
