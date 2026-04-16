"""Pre-registered confirmatory tests T1, T2, T3 per prereg §4.

Reusable across Brown, NLTK multi-source, SPGC, OANC.

Usage:
    python confirmatory_tests.py FEATURES.csv CORPUS_NAME [--out summary.csv]

Tests (locked, do not modify):
    T1 — Minimum effect: η²(schwa) > 0.04 with 95% bootstrap CI lower bound > 0.04
    T2 — Non-inferiority vs FK: (η²_schwa − η²_FK) > -0.05 with 90% bootstrap CI lower > -0.05
    T3 — Correlation replication: |r(schwa, cond_H)| > 0.36 with one-sided p < 0.05

Bucket rule: registers with N < 30 excluded from primary tests (per prereg §2).
Random seed 42, 1000 bootstrap resamples (per prereg §9).
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SEED = 42
N_BOOT = 1000
MIN_BUCKET_N = 30
T1_THRESHOLD = 0.04
T2_MARGIN = -0.05
T3_THRESHOLD = 0.36


def eta_squared(values, groups):
    """One-way ANOVA η²."""
    values = np.asarray(values)
    groups = np.asarray(groups)
    grand_mean = values.mean()
    ss_total = np.sum((values - grand_mean) ** 2)
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    for g in np.unique(groups):
        gv = values[groups == g]
        ss_between += len(gv) * (gv.mean() - grand_mean) ** 2
    return ss_between / ss_total


def bootstrap_eta2(values, groups, n_boot=N_BOOT, seed=SEED, ci=0.95):
    """Within-group resampling bootstrap for η²."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        sampled_v, sampled_g = [], []
        for g in unique_groups:
            idx = np.where(groups == g)[0]
            samp = rng.choice(idx, size=len(idx), replace=True)
            sampled_v.append(values[samp])
            sampled_g.append(np.full(len(samp), g))
        v = np.concatenate(sampled_v)
        gg = np.concatenate(sampled_g)
        boot[b] = eta_squared(v, gg)
    alpha = (1 - ci) / 2
    return float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha)), boot


def bootstrap_eta2_diff(v_schwa, v_fk, groups, n_boot=N_BOOT, seed=SEED, ci=0.90):
    """Bootstrap CI for (η²_schwa − η²_fk). Same resamples for both to preserve pairing."""
    rng = np.random.default_rng(seed)
    v_schwa, v_fk, groups = map(np.asarray, (v_schwa, v_fk, groups))
    unique_groups = np.unique(groups)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        s_v, f_v, s_g = [], [], []
        for g in unique_groups:
            idx = np.where(groups == g)[0]
            samp = rng.choice(idx, size=len(idx), replace=True)
            s_v.append(v_schwa[samp])
            f_v.append(v_fk[samp])
            s_g.append(np.full(len(samp), g))
        sv, fv, gg = np.concatenate(s_v), np.concatenate(f_v), np.concatenate(s_g)
        diffs[b] = eta_squared(sv, gg) - eta_squared(fv, gg)
    alpha = (1 - ci) / 2
    return float(np.quantile(diffs, alpha)), float(np.quantile(diffs, 1 - alpha)), diffs


def run_tests(df, corpus_name, schwa_col='schwa_v1_AH0', fk_col='fk_grade',
              cond_col='cond_entropy', register_col='register'):
    print(f"\n{'='*70}\nCorpus: {corpus_name}   N total: {len(df)}\n{'='*70}")

    counts = df[register_col].value_counts()
    qualifying = counts[counts >= MIN_BUCKET_N].index.tolist()
    excluded = counts[counts < MIN_BUCKET_N].to_dict()
    df_q = df[df[register_col].isin(qualifying)].copy()
    print(f"Qualifying buckets (N>={MIN_BUCKET_N}): {sorted(qualifying)}")
    if excluded:
        print(f"Excluded buckets (N<{MIN_BUCKET_N}): {excluded}")
    print(f"N qualifying: {len(df_q)}")
    print()

    # ---- T1 ----
    e_schwa = eta_squared(df_q[schwa_col], df_q[register_col])
    lo, hi, _ = bootstrap_eta2(df_q[schwa_col].values, df_q[register_col].values, ci=0.95)
    t1_pass = lo > T1_THRESHOLD
    print(f"T1  η²(schwa) = {e_schwa:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
          f"threshold > {T1_THRESHOLD}  {'PASS' if t1_pass else 'FAIL'}")

    # ---- T2 ----
    e_fk = eta_squared(df_q[fk_col], df_q[register_col])
    diff = e_schwa - e_fk
    d_lo, d_hi, _ = bootstrap_eta2_diff(df_q[schwa_col].values, df_q[fk_col].values,
                                         df_q[register_col].values, ci=0.90)
    t2_pass = d_lo > T2_MARGIN
    print(f"T2  η²(schwa)−η²(FK) = {diff:+.4f}  90% CI [{d_lo:+.4f}, {d_hi:+.4f}]  "
          f"margin > {T2_MARGIN}  {'PASS' if t2_pass else 'FAIL'}  "
          f"(η²_FK={e_fk:.4f})")

    # ---- T3 (uses full df, not just qualifying — T3 doesn't need register) ----
    df_t3 = df.dropna(subset=[schwa_col, cond_col])
    r, p_two = stats.pearsonr(df_t3[schwa_col], df_t3[cond_col])
    p_one = p_two / 2 if r < 0 else 1 - p_two / 2  # one-sided
    t3_pass = abs(r) > T3_THRESHOLD and p_one < 0.05
    print(f"T3  |r(schwa, cond_H)| = {abs(r):.4f}  (r={r:+.4f})  "
          f"p_one_sided = {p_one:.2e}  threshold > {T3_THRESHOLD}  "
          f"{'PASS' if t3_pass else 'FAIL'}")

    # ---- Marg-vs-cond divergence (sensitivity, not pre-registered) ----
    if 'marg_entropy' in df.columns:
        r_marg, _ = stats.pearsonr(df_t3[schwa_col], df_t3['marg_entropy'])
        div = abs(r - r_marg)
        print(f"     [sensitivity] r(schwa, marg_H) = {r_marg:+.4f}  "
              f"divergence from cond = {div:.4f}")

    return [
        {'corpus': corpus_name, 'test': 'T1', 'observed': round(e_schwa, 4),
         'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4),
         'threshold': T1_THRESHOLD, 'pass': t1_pass, 'n': len(df_q)},
        {'corpus': corpus_name, 'test': 'T2', 'observed': round(diff, 4),
         'ci_lo': round(d_lo, 4), 'ci_hi': round(d_hi, 4),
         'threshold': T2_MARGIN, 'pass': t2_pass, 'n': len(df_q)},
        {'corpus': corpus_name, 'test': 'T3', 'observed': round(abs(r), 4),
         'ci_lo': None, 'ci_hi': None,
         'threshold': T3_THRESHOLD, 'pass': t3_pass, 'n': len(df_t3),
         'p_one_sided': float(p_one)},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('features_csv')
    ap.add_argument('corpus_name')
    ap.add_argument('--out', help='Append results to this summary CSV')
    args = ap.parse_args()

    df = pd.read_csv(args.features_csv)
    results = run_tests(df, args.corpus_name)

    if args.out:
        out_path = Path(args.out)
        new_df = pd.DataFrame(results)
        if out_path.exists():
            existing = pd.read_csv(out_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(out_path, index=False)
        print(f"\nWrote summary to {out_path}")


if __name__ == '__main__':
    main()
