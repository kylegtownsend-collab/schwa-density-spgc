"""Priority 2: empirical validation of the SPGC ℓ̄=20 heuristic.

Uses NLTK's bundled Project Gutenberg sample (18 books with full punctuation)
as a stand-in for SPGC's source material. Reports actual mean-sentence-length
distribution across those books and whether ℓ̄=20 falls near the central
tendency.

Output: sentence_length_validation.csv
"""
import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import numpy as np

try:
    gutenberg.fileids()
except LookupError:
    nltk.download('gutenberg', quiet=True)

rows = []
for fid in gutenberg.fileids():
    text = gutenberg.raw(fid)
    sents = sent_tokenize(text)
    words_total = 0
    for s in sents:
        words_total += len([w for w in word_tokenize(s) if w.isalpha()])
    if not sents: continue
    msl = words_total / len(sents)
    rows.append({'book': fid, 'n_sents': len(sents),
                 'n_alpha_words': words_total,
                 'mean_sentence_length': round(msl, 2)})

df = pd.DataFrame(rows).sort_values('mean_sentence_length').reset_index(drop=True)
df.to_csv('/home/kyle/schwa_spgc/sentence_length_validation.csv', index=False)

print(f"NLTK Gutenberg sample (N={len(df)} books)")
print(df.to_string(index=False))
print()
msls = df['mean_sentence_length'].values
print(f"Mean of book-level MSL: {msls.mean():.2f}")
print(f"Median: {np.median(msls):.2f}")
print(f"SD: {msls.std():.2f}")
print(f"Range: [{msls.min():.1f}, {msls.max():.1f}]")
print(f"Fraction in [15, 25]: {((msls >= 15) & (msls <= 25)).mean():.1%}")
print(f"Heuristic ℓ̄=20 sits at percentile: {(msls <= 20).mean()*100:.0f}")
