"""Build 2K stratified subsample of SPGC English books using LCSH-derived
register buckets.

Multi-match tiebreak: priority order below, with 'fiction' last so e.g.
'historical fiction' → history, 'Christian poetry' → poetry, 'biographical
drama' → drama. Documented in deviations.md.
"""
import ast
import re
import csv
from pathlib import Path
from collections import Counter
import pandas as pd

SEED = 42
CAP_PER_BUCKET = 250
META_PATH = Path('/home/kyle/schwa_spgc/SPGC-metadata-2018-07-18.csv')
OUT_PATH = Path('/home/kyle/schwa_spgc/spgc_sample_metadata.csv')

# Priority order: more specific first, fiction LAST as tiebreak
BUCKET_PRIORITY = [
    'children',    # juvenile literature gets its own bucket regardless of genre
    'drama',
    'poetry',
    'religion',
    'philosophy',
    'science',
    'history',
    'biography',
    'travel',
    'essays',
    'letters',
    'fiction',     # catch-all literary
]

PATTERNS = {
    'fiction':    r'\bFiction\b|\bnovels?\b|\bShort stories\b|\bRomance\b|\bSatire\b',
    'poetry':     r'\bPoetry\b|\bPoems\b|\bSonnets\b',
    'drama':      r'\bDrama\b|\bplays?\b|\bTheater\b|\bTragedies\b|\bComedies\b',
    'biography':  r'\bBiograph|\bAutobiography\b|\bMemoirs\b',
    'history':    r'\bHistor',
    'science':    r'\bScience\b|\bScientific\b|\bMathematics\b|\bPhysics\b|\bChemistry\b|\bBiology\b|\bAstronomy\b|\bGeology\b',
    'religion':   r'\bReligion\b|\bChristian\b|\bBible\b|\bSermons\b|\bTheology\b|\bChurch\b',
    'philosophy': r'\bPhilosoph|\bEthics\b|\bMetaphysics\b|\bLogic\b',
    'children':   r'\bjuvenile\b|\bChildren|\bFairy tales\b|\bNursery\b',
    'travel':     r'\bTravel\b|\bDescription and travel\b|\bVoyages\b',
    'essays':     r'\bEssays\b',
    'letters':    r'\bCorrespondence\b|\bLetters\b',
}


def assign_bucket(subjects_str):
    """Return single bucket name or None."""
    if pd.isna(subjects_str) or subjects_str == 'set()':
        return None
    matches = []
    for cat, pat in PATTERNS.items():
        if re.search(pat, subjects_str, re.I):
            matches.append(cat)
    if not matches:
        return None
    # Pick first match in priority order
    for b in BUCKET_PRIORITY:
        if b in matches:
            return b
    return None


def main():
    df = pd.read_csv(META_PATH, low_memory=False)
    print(f"Total rows: {len(df)}")

    # Filter English-only — actual format is "['en']" not handoff's "{'en'}"
    en = df[df['language'] == "['en']"].copy()
    print(f"English-only: {len(en)}")

    en['bucket'] = en['subjects'].apply(assign_bucket)
    bucketed = en.dropna(subset=['bucket'])
    print(f"Books with assigned bucket: {len(bucketed)}")
    print()

    print("Available per bucket:")
    for b, n in bucketed['bucket'].value_counts().items():
        print(f"  {b:12s} {n:6d}")
    print()

    # Stratified sample, cap CAP_PER_BUCKET per bucket
    parts = []
    for b, grp in bucketed.groupby('bucket'):
        parts.append(grp.sample(min(len(grp), CAP_PER_BUCKET), random_state=SEED))
    sample = pd.concat(parts, ignore_index=True)
    print(f"Subsample size: {len(sample)}")
    print()
    print("Sampled per bucket:")
    for b, n in sample['bucket'].value_counts().items():
        marker = '' if n >= 30 else '  (BELOW N=30 — will be excluded from primary tests)'
        print(f"  {b:12s} {n:5d}{marker}")
    print()

    out = sample[['id', 'title', 'author', 'bucket', 'subjects']].copy()
    out = out.rename(columns={'bucket': 'register'})
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} rows to {OUT_PATH}")


if __name__ == '__main__':
    main()
