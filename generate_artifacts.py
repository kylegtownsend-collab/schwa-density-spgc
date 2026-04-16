"""Generate final artifacts: results table, figures, summary memo.

Figure priorities (per framing revision):
  Fig 1 HEADLINE  — η²(schwa) vs η²(FK) across corpora
  Fig 2           — Schwa boxplots by register on SPGC (most heterogeneous
                    pre-registered corpus)
  Fig S1 SUPP     — |r(schwa, cond_H)| bars with 0.36 line (pipeline check)
  Fig S2 SUPP     — schwa-vs-cond_H scatter with structural-confound caption
"""
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

OUT_DIR = Path('/home/kyle/schwa_spgc')
plt.rcParams.update({'savefig.dpi': 300, 'figure.dpi': 100,
                     'font.size': 10, 'axes.titlesize': 11})


def eta2(v, g):
    v = np.asarray(v); g = np.asarray(g)
    grand = v.mean()
    ss_t = ((v - grand) ** 2).sum()
    if ss_t == 0: return 0.0
    ss_b = sum(len(v[g == k]) * (v[g == k].mean() - grand) ** 2 for k in np.unique(g))
    return ss_b / ss_t


def partial_eta2(target, group_col, controls, df):
    X = sm.add_constant(df[controls].values)
    res = sm.OLS(df[target].values, X).fit()
    resid = df[target].values - res.predict(X)
    return eta2(resid, df[group_col].values)


def load(path):
    return pd.read_csv(path)


def qualifying(df, min_n=30):
    counts = df['register'].value_counts()
    qual = counts[counts >= min_n].index.tolist()
    return df[df['register'].isin(qual)].reset_index(drop=True)


def brown_5bucket(df):
    g5 = {
        'news':'press','editorial':'press','reviews':'press',
        'religion':'general','hobbies':'general','lore':'general',
        'belles_lettres':'general','government':'general',
        'learned':'learned',
        'fiction':'fiction','mystery':'fiction','science_fiction':'fiction',
        'adventure':'fiction','romance':'fiction','humor':'fiction',
    }
    out = df.copy()
    out['register'] = out['register'].map(g5)
    return out


def main():
    corpora = {
        'Brown': load('brown_features.csv'),
        'NLTK_multi': load('nltk_multi_features.csv'),
        'SPGC': load('spgc_features.csv'),
        'OANC': load('oanc_features.csv'),
    }

    # Brown gets both 6-bucket (locked rule) and 5-bucket (for reference)
    brown_5 = qualifying(brown_5bucket(corpora['Brown']))
    qualified = {name: qualifying(df) for name, df in corpora.items()}

    # ---------- Results table ----------
    rows = []
    for name, dfq in qualified.items():
        r_cond, _ = stats.pearsonr(dfq['schwa_v1_AH0'], dfq['cond_entropy'])
        r_marg, _ = stats.pearsonr(dfq['schwa_v1_AH0'], dfq['marg_entropy'])
        r_syll, _ = stats.pearsonr(dfq['schwa_v1_AH0'], dfq['mean_syllables'])
        e_s = eta2(dfq['schwa_v1_AH0'], dfq['register'])
        e_f = eta2(dfq['fk_grade'], dfq['register'])
        pe_j3 = partial_eta2('schwa_v1_AH0', 'register',
                              ['mean_syllables','mean_word_length','latinate_ratio'], dfq)
        rows.append({
            'corpus': name, 'n_qualifying': len(dfq),
            'n_buckets': dfq['register'].nunique(),
            'eta2_schwa': round(e_s, 4),
            'eta2_fk': round(e_f, 4),
            'eta2_gap': round(e_s - e_f, 4),
            'partial_eta2_schwa_given_joint3': round(pe_j3, 4),
            'retained_pct': f"{pe_j3/e_s*100:.0f}%" if e_s else "—",
            'r_schwa_condH': round(r_cond, 4),
            'r_schwa_margH': round(r_marg, 4),
            'margcond_divergence': round(abs(r_cond - r_marg), 4),
            'r_schwa_syllables': round(r_syll, 4),
        })
    # Add Brown 5-bucket
    r_cond, _ = stats.pearsonr(brown_5['schwa_v1_AH0'], brown_5['cond_entropy'])
    rows.append({
        'corpus': 'Brown_5bucket', 'n_qualifying': len(brown_5),
        'n_buckets': brown_5['register'].nunique(),
        'eta2_schwa': round(eta2(brown_5['schwa_v1_AH0'], brown_5['register']), 4),
        'eta2_fk': round(eta2(brown_5['fk_grade'], brown_5['register']), 4),
        'eta2_gap': None,
        'partial_eta2_schwa_given_joint3': round(partial_eta2(
            'schwa_v1_AH0','register',
            ['mean_syllables','mean_word_length','latinate_ratio'], brown_5), 4),
        'retained_pct': None,
        'r_schwa_condH': round(r_cond, 4),
        'r_schwa_margH': None,
        'margcond_divergence': None,
        'r_schwa_syllables': None,
    })
    # Add handoff-reference GITenberg row (not re-verified)
    rows.append({
        'corpus': 'GITenberg(handoff_ref)', 'n_qualifying': 142,
        'n_buckets': None,
        'eta2_schwa': 0.318, 'eta2_fk': 0.117, 'eta2_gap': 0.201,
        'partial_eta2_schwa_given_joint3': None, 'retained_pct': None,
        'r_schwa_condH': -0.94, 'r_schwa_margH': None,
        'margcond_divergence': None, 'r_schwa_syllables': None,
    })
    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUT_DIR / 'results_table.csv', index=False)
    print("Wrote results_table.csv")

    # ---------- Per-register summary (SPGC) for fig2 and supplementary table ----------
    spgc_q = qualified['SPGC']
    register_summary = spgc_q.groupby('register').agg(
        n=('text_id', 'count'),
        schwa_mean=('schwa_v1_AH0', 'mean'),
        schwa_sd=('schwa_v1_AH0', 'std'),
        fk_mean=('fk_grade', 'mean'),
        cond_H_mean=('cond_entropy', 'mean'),
        syll_mean=('mean_syllables', 'mean'),
    ).round(4).sort_values('schwa_mean')
    register_summary.to_csv(OUT_DIR / 'spgc_results_table.csv')
    print("Wrote spgc_results_table.csv")

    # ---------- Pre-registered tests summary (already exists) ----------
    # tests_summary.csv was accumulated by confirmatory_tests.py runs; fine as-is

    # ---------- Fig 1: η² comparison (HEADLINE) ----------
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ordered = ['Brown', 'NLTK_multi', 'SPGC', 'OANC']
    schwa_etas = [results_df[results_df['corpus']==c]['eta2_schwa'].values[0] for c in ordered]
    fk_etas = [results_df[results_df['corpus']==c]['eta2_fk'].values[0] for c in ordered]
    x = np.arange(len(ordered))
    w = 0.38
    ax.bar(x - w/2, schwa_etas, w, label='η²(schwa)', color='#2b5f8a')
    ax.bar(x + w/2, fk_etas, w, label='η²(FK)', color='#d47642')
    ax.axhline(0.04, ls=':', color='gray', lw=0.8, label='Crud floor (T1 threshold)')
    ax.set_xticks(x); ax.set_xticklabels(ordered)
    ax.set_ylabel('η² (register discrimination)')
    ax.set_title('Register discrimination by single feature across four corpora')
    ax.legend(loc='best')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for i, (s, f) in enumerate(zip(schwa_etas, fk_etas)):
        ax.text(i - w/2, s + 0.01, f'{s:.2f}', ha='center', fontsize=8)
        ax.text(i + w/2, f + 0.01, f'{f:.2f}', ha='center', fontsize=8)
    ax.set_ylim(0, max(max(schwa_etas), max(fk_etas)) * 1.15)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig1_eta_comparison.png')
    plt.close(fig)
    print("Wrote fig1_eta_comparison.png")

    # ---------- Fig 2: SPGC schwa boxplots by register ----------
    fig, ax = plt.subplots(figsize=(9, 5))
    register_order = spgc_q.groupby('register')['schwa_v1_AH0'].median().sort_values().index.tolist()
    data = [spgc_q[spgc_q['register'] == r]['schwa_v1_AH0'].values for r in register_order]
    bp = ax.boxplot(data, labels=register_order, patch_artist=True, showmeans=True,
                    meanprops={'marker':'D','markerfacecolor':'white','markeredgecolor':'black','markersize':4})
    for patch in bp['boxes']:
        patch.set_facecolor('#9bc5e8')
        patch.set_edgecolor('#2b5f8a')
    ax.set_ylabel('Schwa density (proportion of vowel phones that are AH0)')
    ax.set_title(f'Schwa density by register on SPGC (N={len(spgc_q)}, 12 LCSH buckets)')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig2_schwa_by_register_spgc.png')
    plt.close(fig)
    print("Wrote fig2_schwa_by_register_spgc.png")

    # ---------- Fig S1: correlation bars (supplementary) ----------
    fig, ax = plt.subplots(figsize=(6.5, 4))
    cond_rs = [abs(results_df[results_df['corpus']==c]['r_schwa_condH'].values[0]) for c in ordered]
    ax.bar(ordered, cond_rs, color='#6a9c78')
    ax.axhline(0.36, ls=':', color='red', lw=1.2,
               label='Small-telescopes threshold (|r|=0.36)')
    for i, r in enumerate(cond_rs):
        ax.text(i, r + 0.02, f'{r:.2f}', ha='center', fontsize=9)
    ax.set_ylabel('|r(schwa density, conditional vowel entropy)|')
    ax.set_title('Pipeline-consistency check: schwa–entropy correlation across corpora')
    ax.set_ylim(0, 1.05)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(loc='lower left')
    fig.text(0.5, -0.02, 'NB: high correlations are partly structural (Shannon entropy mechanically\ndrops as one vowel category dominates). See Discussion §3.',
             ha='center', fontsize=8, style='italic')
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'figS1_correlation_check.png', bbox_inches='tight')
    plt.close(fig)
    print("Wrote figS1_correlation_check.png")

    # ---------- Fig S2: schwa vs cond_H scatter (SPGC) ----------
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for reg in sorted(spgc_q['register'].unique()):
        sub = spgc_q[spgc_q['register'] == reg]
        ax.scatter(sub['schwa_v1_AH0'], sub['cond_entropy'], s=8, alpha=0.5, label=reg)
    ax.set_xlabel('Schwa density (AH0 / total vowels)')
    ax.set_ylabel('Conditional vowel entropy H(V_n | V_{n-1})')
    ax.set_title('Schwa density vs conditional vowel entropy, SPGC by register\n'
                 '(r = −0.94; marg_H divergence = 0.01 — partly structural)')
    ax.legend(loc='best', fontsize=7, ncol=2, markerscale=2, framealpha=0.9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'figS2_schwa_vs_entropy_spgc.png')
    plt.close(fig)
    print("Wrote figS2_schwa_vs_entropy_spgc.png")

    print("\nAll artifacts written to", OUT_DIR)


if __name__ == '__main__':
    main()
