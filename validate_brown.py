"""Validate analyzer against known Brown numbers. Hard gate: r(schwa, cond_H) ≈ −0.92."""
import sys
import nltk
import pandas as pd
import numpy as np
from scipy import stats
from nltk.corpus import cmudict

print("=== Environment ===")
print(f"Python: {sys.version.split()[0]}")
print(f"NLTK: {nltk.__version__}")
print(f"CMUdict size: {len(cmudict.dict())} entries")
print()

df = pd.read_csv('brown_features.csv')
print(f"=== Brown features: N={len(df)} ===")
print(f"Mean OOV: {df['oov_rate'].mean():.2%}  (max {df['oov_rate'].max():.2%})")
print(f"Mean schwa_v1 (AH0): {df['schwa_v1_AH0'].mean():.4f}  sd {df['schwa_v1_AH0'].std():.4f}")
print(f"Mean cond_entropy: {df['cond_entropy'].mean():.4f}")
print(f"Mean marg_entropy: {df['marg_entropy'].mean():.4f}")
print(f"r(marg, cond): {df[['marg_entropy','cond_entropy']].corr().iloc[0,1]:.4f}  (handoff says ~0.99)")
print()

print("=== HARD GATE: r(schwa_v1, cond_entropy) ===")
r, p = stats.pearsonr(df['schwa_v1_AH0'], df['cond_entropy'])
print(f"Observed r = {r:+.4f}  (Brown reference: −0.92)")
print(f"p = {p:.2e}")
gate_pass = abs(r - (-0.92)) <= 0.05
print(f"GATE: |{r:+.4f} − (−0.92)| = {abs(r-(-0.92)):.4f}  {'PASS' if gate_pass else 'FAIL'} (tol 0.05)")
print()

print("=== η²(schwa_v1, register) — informational ===")
def eta2(values, groups):
    grand = np.mean(values)
    ss_total = np.sum((values - grand)**2)
    ss_between = 0.0
    for g in set(groups):
        gv = values[groups == g]
        ss_between += len(gv) * (gv.mean() - grand)**2
    return ss_between / ss_total

# 15-way (full Brown categories)
vals = df['schwa_v1_AH0'].values
grps = df['register'].values
e15 = eta2(vals, grps)
print(f"15-category η² = {e15:.4f}")

# 5-way standard Francis-Kučera grouping
GROUP5 = {
    'news':'press', 'editorial':'press', 'reviews':'press',
    'religion':'general', 'hobbies':'general', 'lore':'general', 'belles_lettres':'general', 'government':'general',
    'learned':'learned',
    'fiction':'fiction', 'mystery':'fiction', 'science_fiction':'fiction',
    'adventure':'fiction', 'romance':'fiction', 'humor':'fiction',
}
df['register5'] = df['register'].map(GROUP5)
e5 = eta2(df['schwa_v1_AH0'].values, df['register5'].values)
print(f"5-bucket (press/general/learned/fiction) η² = {e5:.4f}")
print(f"  (Brown reference: 0.558 with unknown 5-way grouping)")
print()

# Same for FK
e15_fk = eta2(df['fk_grade'].values, df['register'].values)
e5_fk = eta2(df['fk_grade'].values, df['register5'].values)
print(f"FK η² 15-cat: {e15_fk:.4f}   5-bucket: {e5_fk:.4f}")
print(f"  (Brown reference: 0.552)")

sys.exit(0 if gate_pass else 1)
