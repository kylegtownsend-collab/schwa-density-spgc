"""Priority 4: bucket-exclusion sensitivity analysis.

Reviewer concern: Brown's 40% register exclusion (N<30) may be driving T1.
Check: rerun T1 η² at N≥20, N≥30, N≥50 thresholds across all 4 corpora.
If η² and pass/fail are stable across thresholds, T1 is robust to the
bucket rule rather than an artifact of it.
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path('/home/kyle/schwa_spgc')
CORPORA = ['nltk_multi', 'spgc', 'brown', 'oanc']
THRESHOLDS = [20, 30, 50]

def eta_squared(values, groups):
    values = np.asarray(values); groups = np.asarray(groups)
    gm = values.mean()
    sst = np.sum((values - gm) ** 2)
    if sst == 0: return 0.0
    ssb = sum(len(values[groups == g]) * (values[groups == g].mean() - gm) ** 2
              for g in np.unique(groups))
    return ssb / sst

def bootstrap_eta2(values, groups, n_boot=1000, seed=42, ci=0.95):
    rng = np.random.default_rng(seed)
    values = np.asarray(values); groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        sv, sg = [], []
        for g in unique_groups:
            idx = np.where(groups == g)[0]
            samp = rng.choice(idx, size=len(idx), replace=True)
            sv.append(values[samp]); sg.append(np.full(len(samp), g))
        boot[b] = eta_squared(np.concatenate(sv), np.concatenate(sg))
    alpha = (1 - ci) / 2
    return float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))

rows = []
for corpus in CORPORA:
    fp = ROOT / f'{corpus}_features.csv'
    if not fp.exists():
        print(f"skip {corpus}")
        continue
    df = pd.read_csv(fp)
    if 'register' not in df.columns or 'schwa_v1_AH0' not in df.columns:
        continue
    counts = df['register'].value_counts()
    total_buckets = len(counts)

    for thr in THRESHOLDS:
        qual = counts[counts >= thr].index.tolist()
        dq = df[df['register'].isin(qual)]
        if len(dq) < thr or len(qual) < 2:
            rows.append({'corpus': corpus, 'N_min': thr,
                         'n_buckets': len(qual), 'n_excluded': total_buckets - len(qual),
                         'n_texts': len(dq), 'eta2': None, 'ci_lo': None, 'ci_hi': None,
                         'pass_t1': None})
            continue
        e = eta_squared(dq['schwa_v1_AH0'].values, dq['register'].values)
        lo, hi = bootstrap_eta2(dq['schwa_v1_AH0'].values, dq['register'].values)
        rows.append({'corpus': corpus, 'N_min': thr,
                     'n_buckets': len(qual), 'n_excluded': total_buckets - len(qual),
                     'n_texts': len(dq),
                     'eta2': round(e, 4),
                     'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4),
                     'pass_t1': lo > 0.04})

out = pd.DataFrame(rows)
out.to_csv(ROOT / 'bucket_sensitivity.csv', index=False)
print(out.to_string(index=False))
print()
print("T1 pass/fail consistency across thresholds:")
for c in CORPORA:
    sub = out[out['corpus'] == c]
    passes = sub['pass_t1'].tolist()
    print(f"  {c}: {passes}")
