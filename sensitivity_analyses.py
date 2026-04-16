"""Pre-registered sensitivity analyses (§§2, 3, 5 of prereg) plus the
schwa-syllables independence check from second-instance critique.

For each corpus:
  A. r(schwa, mean_syllables) and r(schwa, mean_word_length) — proxy diagnosis
  B. Partial r(schwa, cond_entropy) controlling for ttr+mwl+msl+latinate+syll  (prereg §5)
  C. Partial η²(schwa, register | mean_syllables)  — does schwa carry signal beyond syllables?
  D. 5-fold CV logistic regression accuracy with each single predictor (prereg §3 sens)
  E. r(schwa, marg_H) vs r(schwa, cond_H) divergence
"""
import sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

SEED = 42
MIN_BUCKET_N = 30


def eta2(values, groups):
    grand = np.mean(values)
    ss_total = np.sum((values - grand) ** 2)
    if ss_total == 0:
        return 0.0
    ss_between = sum(len(values[groups == g]) * (values[groups == g].mean() - grand) ** 2
                     for g in np.unique(groups))
    return ss_between / ss_total


def partial_eta2(target, group_col, control_cols, df):
    """η² of target on group, after regressing target on control_cols."""
    X = sm.add_constant(df[control_cols].values)
    res = sm.OLS(df[target].values, X).fit()
    resid = df[target].values - res.predict(X)
    return eta2(resid, df[group_col].values)


def partial_r(x, y, controls, df):
    """Partial Pearson r(x, y | controls)."""
    X = sm.add_constant(df[controls].values)
    rx = df[x].values - sm.OLS(df[x].values, X).fit().predict(X)
    ry = df[y].values - sm.OLS(df[y].values, X).fit().predict(X)
    return stats.pearsonr(rx, ry)[0]


def cv_accuracy(predictor, target, df, n_splits=5):
    """5-fold CV logistic regression accuracy with single standardized predictor."""
    X = StandardScaler().fit_transform(df[[predictor]].values)
    y = df[target].astype(str).values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = []
    for tr, te in skf.split(X, y):
        # Skip if any class missing in train fold
        if len(set(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[tr], y[tr])
        scores.append(clf.score(X[te], y[te]))
    return np.mean(scores) if scores else float('nan')


def majority_baseline(df, target='register'):
    return df[target].value_counts(normalize=True).max()


def analyze(name, df):
    print(f"\n{'='*72}\n{name}   N total: {len(df)}\n{'='*72}")
    counts = df['register'].value_counts()
    qual = counts[counts >= MIN_BUCKET_N].index.tolist()
    df_q = df[df['register'].isin(qual)].copy().reset_index(drop=True)
    excluded = counts[counts < MIN_BUCKET_N].to_dict()
    print(f"Qualifying registers (N>={MIN_BUCKET_N}): {sorted(qual)}")
    if excluded:
        print(f"Excluded (N<{MIN_BUCKET_N}): {excluded}")
    print(f"N qualifying: {len(df_q)}")
    print()

    # A. Schwa as syllables proxy
    r_syll, _ = stats.pearsonr(df_q['schwa_v1_AH0'], df_q['mean_syllables'])
    r_mwl, _ = stats.pearsonr(df_q['schwa_v1_AH0'], df_q['mean_word_length'])
    r_lat, _ = stats.pearsonr(df_q['schwa_v1_AH0'], df_q['latinate_ratio'])
    print(f"A.  r(schwa, syllables)  = {r_syll:+.4f}     r(schwa, mwl) = {r_mwl:+.4f}     r(schwa, latinate) = {r_lat:+.4f}")

    # B. Partial r(schwa, cond_H) controlling for surface features (prereg §5)
    pr = partial_r('schwa_v1_AH0', 'cond_entropy',
                   ['ttr', 'mean_word_length', 'mean_sentence_length', 'latinate_ratio', 'mean_syllables'],
                   df_q)
    print(f"B.  partial r(schwa, cond_H | ttr+mwl+msl+lat+syll) = {pr:+.4f}")

    # C. Partial η²(schwa, register | mean_syllables) — the load-bearing test
    raw_e = eta2(df_q['schwa_v1_AH0'].values, df_q['register'].values)
    pe_syll = partial_eta2('schwa_v1_AH0', 'register', ['mean_syllables'], df_q)
    pe_full = partial_eta2('schwa_v1_AH0', 'register',
                            ['mean_syllables', 'mean_word_length', 'latinate_ratio', 'ttr'], df_q)
    print(f"C.  η²(schwa, register)                          = {raw_e:.4f}  (raw)")
    print(f"    partial η²(schwa, register | syllables)      = {pe_syll:.4f}  ← load-bearing")
    print(f"    partial η²(schwa, register | syll+mwl+lat+ttr) = {pe_full:.4f}")

    # D. 5-fold CV head-to-head
    print(f"D.  5-fold CV logistic regression (single predictor → register):")
    base = majority_baseline(df_q)
    print(f"    majority-class baseline:    {base:.4f}")
    for pred in ['schwa_v1_AH0', 'fk_grade', 'mean_syllables', 'mean_word_length', 'latinate_ratio']:
        acc = cv_accuracy(pred, 'register', df_q)
        print(f"    {pred:20s}        {acc:.4f}")

    # E. Marg vs cond divergence
    rc, _ = stats.pearsonr(df_q['schwa_v1_AH0'], df_q['cond_entropy'])
    rm, _ = stats.pearsonr(df_q['schwa_v1_AH0'], df_q['marg_entropy'])
    print(f"E.  r(schwa, cond_H) = {rc:+.4f}   r(schwa, marg_H) = {rm:+.4f}   divergence = {abs(rc-rm):.4f}")


def main():
    corpora = [('Brown', 'brown_features.csv'),
               ('NLTK_multi', 'nltk_multi_features.csv'),
               ('SPGC', 'spgc_features.csv'),
               ('OANC', 'oanc_features.csv')]
    for name, path in corpora:
        try:
            df = pd.read_csv(path)
            analyze(name, df)
        except FileNotFoundError:
            print(f"Skipping {name}: {path} not found")


if __name__ == '__main__':
    main()
