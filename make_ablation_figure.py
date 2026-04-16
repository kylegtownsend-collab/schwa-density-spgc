"""Generate fig3_ablation_unmasking.png — shows per-register schwa density
distribution unmasked vs masked, side by side, for all four corpora.

The visual point: after masking NLTK stopwords, register means spread apart
(noise floor removed), which is the geometric explanation for the η² gain.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path('/home/kyle/schwa_spgc')

CORPORA = [
    ('NLTK multi-source', 'nltk_multi'),
    ('SPGC',              'spgc'),
    ('Brown',             'brown'),
    ('OANC',              'oanc'),
]

fig, axes = plt.subplots(4, 2, figsize=(11, 13), sharex=False)
fig.suptitle('Schwa density by register, before and after function-word masking\n'
             '(NLTK stopwords, 198 words removed)',
             fontsize=13, y=0.995)

for row, (label, stem) in enumerate(CORPORA):
    orig = pd.read_csv(ROOT / f'{stem}_features.csv')
    masked = pd.read_csv(ROOT / f'{stem}_features_masked.csv')

    if 'register' not in orig.columns or 'register' not in masked.columns:
        continue

    counts_o = orig['register'].value_counts()
    qual = counts_o[counts_o >= 30].index.tolist()
    orig_q = orig[orig['register'].isin(qual)].copy()
    masked_q = masked[masked['register'].isin(qual)].copy()

    # Order registers by unmasked mean, ascending
    order = orig_q.groupby('register')['schwa_v1_AH0'].mean().sort_values().index.tolist()

    for col, (df, subtitle) in enumerate([(orig_q, 'Unmasked'), (masked_q, 'Masked')]):
        ax = axes[row, col]
        data = [df[df['register'] == r]['schwa_v1_AH0'].values for r in order]
        positions = np.arange(len(order))
        bp = ax.boxplot(data, positions=positions, widths=0.6, vert=True,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='black', linewidth=1.2),
                        boxprops=dict(facecolor='#b0c4de', edgecolor='#334'),
                        whiskerprops=dict(color='#334'),
                        capprops=dict(color='#334'))
        means = [d.mean() for d in data]
        ax.plot(positions, means, 'o', color='#c0392b', markersize=5, zorder=5,
                label='mean')
        # Range markers for spread
        mean_range = max(means) - min(means)
        ax.set_xticks(positions)
        ax.set_xticklabels(order, rotation=35, ha='right', fontsize=8)
        ax.set_ylabel('schwa v1 (AH0 / vowels)' if col == 0 else '')
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        title = f'{label} — {subtitle}'
        if col == 1:
            title += f'  (mean spread: {mean_range:.3f})'
        else:
            title += f'  (mean spread: {mean_range:.3f})'
        ax.set_title(title, fontsize=10)

    # Sync y-axis across the two panels in this row
    ymin = min(axes[row, 0].get_ylim()[0], axes[row, 1].get_ylim()[0])
    ymax = max(axes[row, 0].get_ylim()[1], axes[row, 1].get_ylim()[1])
    axes[row, 0].set_ylim(ymin, ymax)
    axes[row, 1].set_ylim(ymin, ymax)

plt.tight_layout(rect=[0, 0, 1, 0.985])
out = ROOT / 'fig3_ablation_unmasking.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'wrote {out}')
