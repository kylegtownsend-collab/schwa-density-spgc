"""Priority 3: OANC rerun with espeak-ng G2P fallback for OOV words.

Workflow:
1. Scan oanc_texts, collect unique alphabetic tokens not in CMUdict
2. Batch-phonemize via espeak-ng (phonemizer library)
3. Parse IPA output: count schwa phones (ə, ɚ) and total vowels
4. Rerun OANC feature extraction with the lookup table filling OOV
5. Compare η²(schwa, register) with vs without G2P fallback

Output:
    oanc_oov_phones.json        — OOV → (n_schwa, n_vowels)
    oanc_features_g2p.csv       — feature table with G2P-filled OOV
    g2p_comparison.csv          — η² with vs without
"""
import json
import sys
import re
from pathlib import Path
from collections import Counter

sys.path.insert(0, '/home/kyle/schwa_spgc')

import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import cmudict
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend

CMU = cmudict.dict()
ROOT = Path('/home/kyle/schwa_spgc')

# IPA vowel set (roughly — covers espeak-ng en-us output)
IPA_VOWELS = set('aɑɐæəɛɜɚɪiɔoʊuʌɒɤɯ')
IPA_SCHWA = {'ə', 'ɚ'}  # schwa and r-colored schwa

def ipa_vowels_and_schwa(ipa_string):
    """Return (n_schwa, n_vowels) for an IPA string."""
    n_schwa = sum(1 for c in ipa_string if c in IPA_SCHWA)
    n_vow = sum(1 for c in ipa_string if c in IPA_VOWELS)
    return n_schwa, n_vow

def collect_oov_words(texts_dir, max_words_per_text=50000):
    """Walk every .txt under texts_dir and return Counter of OOV alpha words."""
    oov = Counter()
    files = list(Path(texts_dir).rglob('*.txt'))
    for i, fp in enumerate(files):
        if i % 500 == 0 and i > 0:
            print(f"  scanned {i}/{len(files)}  unique oov so far={len(oov)}")
        try:
            text = fp.read_text(encoding='utf-8', errors='replace')[:200000]
        except Exception:
            continue
        try:
            words = word_tokenize(text)
        except Exception:
            continue
        for w in words:
            if not w.isalpha(): continue
            wl = w.lower()
            if wl not in CMU:
                oov[wl] += 1
    return oov

def phonemize_batch(words, batch_size=2000):
    """Phonemize words in batches to avoid memory issues."""
    out = {}
    backend = EspeakBackend('en-us', preserve_punctuation=False, with_stress=False)
    for i in range(0, len(words), batch_size):
        batch = words[i:i+batch_size]
        print(f"  phonemizing batch {i}-{i+len(batch)} of {len(words)}")
        try:
            ipas = backend.phonemize(batch)
        except Exception as e:
            print(f"  batch error: {e}; skipping")
            continue
        for w, ipa in zip(batch, ipas):
            n_s, n_v = ipa_vowels_and_schwa(ipa)
            out[w] = (n_s, n_v, ipa)
    return out

def build_oov_lookup(oov_counter, top_n=20000):
    most = oov_counter.most_common(top_n)
    words = [w for w, _ in most]
    print(f"phonemizing {len(words)} OOV words (of {len(oov_counter)} total unique)")
    phones = phonemize_batch(words)
    return phones

def vowels_with_stress_from_cmu(phones):
    out = []
    vowels = {'AA','AE','AH','AO','AW','AY','EH','EY','IH','IY','OW','OY','UH','UW'}
    for p in phones:
        m = re.match(r'^([A-Z]+)([012])?$', p)
        if m and m.group(1) in vowels:
            out.append((m.group(1), m.group(2)))
    return out

def process_text_g2p(text_id, text, oov_lookup, min_words=1000, max_oov_handled=0.20):
    """Same as main analyzer but uses OOV lookup for missing CMU entries."""
    try:
        words = word_tokenize(text[:200000])
    except Exception:
        return {'text_id': text_id, '_error': 'tokenize'}
    alpha = [w for w in words if w.isalpha()]
    if len(alpha) < min_words:
        return {'text_id': text_id, '_error': 'too_short'}

    n_AH0 = 0
    total_vowels = 0
    n_oov_covered = 0
    n_oov_miss = 0

    for w in alpha:
        wl = w.lower()
        if wl in CMU:
            phs = CMU[wl][0]
            vseq = vowels_with_stress_from_cmu(phs)
            total_vowels += len(vseq)
            n_AH0 += sum(1 for b, s in vseq if b == 'AH' and s == '0')
        elif wl in oov_lookup:
            n_s, n_v, _ = oov_lookup[wl]
            total_vowels += n_v
            n_AH0 += n_s
            n_oov_covered += 1
        else:
            n_oov_miss += 1

    if total_vowels < 500:
        return {'text_id': text_id, '_error': 'too_few_vowels'}

    return {
        'text_id': text_id,
        'n_words': len(alpha),
        'n_vowels': total_vowels,
        'schwa_v1_AH0': n_AH0 / total_vowels,
        'oov_g2p_filled': n_oov_covered,
        'oov_still_missing': n_oov_miss,
        'oov_fill_rate': n_oov_covered / max(1, n_oov_covered + n_oov_miss),
    }

def eta_squared(values, groups):
    values = np.asarray(values); groups = np.asarray(groups)
    gm = values.mean()
    sst = np.sum((values - gm) ** 2)
    if sst == 0: return 0.0
    ssb = sum(len(values[groups == g]) * (values[groups == g].mean() - gm) ** 2
              for g in np.unique(groups))
    return ssb / sst

def main():
    print("=== Priority 3: OANC with G2P fallback ===\n")

    oov_json = ROOT / 'oanc_oov_phones.json'
    oov_lookup = {}
    if oov_json.exists():
        print(f"Loading cached OOV lookup from {oov_json}")
        with open(oov_json) as f:
            raw = json.load(f)
            oov_lookup = {k: (v[0], v[1], v[2]) for k, v in raw.items()}
        print(f"  {len(oov_lookup)} words cached")
    else:
        print("Collecting OOV words from oanc_texts...")
        oov_counter = collect_oov_words(ROOT / 'oanc_texts')
        print(f"Found {len(oov_counter)} unique OOV words, {sum(oov_counter.values())} total occurrences")
        print(f"Top 10: {oov_counter.most_common(10)}")
        oov_lookup = build_oov_lookup(oov_counter)
        with open(oov_json, 'w') as f:
            json.dump({k: list(v) for k, v in oov_lookup.items()}, f)
        print(f"  wrote {oov_json}")

    # Load OANC metadata
    oanc_meta = pd.read_csv(ROOT / 'oanc_metadata.csv', dtype=str, low_memory=False)
    meta = {}
    for _, row in oanc_meta.iterrows():
        meta[str(row['id']).strip()] = str(row.get('register', 'unknown'))

    print("\nProcessing OANC texts with G2P fallback...")
    files = sorted((ROOT / 'oanc_texts').rglob('*.txt'))
    rows = []
    for i, fp in enumerate(files):
        if i % 1000 == 0 and i > 0:
            print(f"  {i}/{len(files)}  kept={len(rows)}")
        try:
            text = fp.read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue
        r = process_text_g2p(fp.stem, text, oov_lookup)
        if '_error' in r:
            continue
        r['register'] = meta.get(fp.stem, meta.get(fp.name, 'unknown'))
        rows.append(r)

    df = pd.DataFrame(rows)
    out_csv = ROOT / 'oanc_features_g2p.csv'
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {len(df)} rows to {out_csv}")
    print(f"Mean fill rate: {df['oov_fill_rate'].mean():.2%}")
    print(f"Avg OOV filled per text: {df['oov_g2p_filled'].mean():.1f}")

    # Compare η² vs original
    orig = pd.read_csv(ROOT / 'oanc_features.csv')
    for d, label in [(orig, 'unmasked (orig, no G2P)'), (df, 'G2P fallback')]:
        counts = d['register'].value_counts()
        qual = counts[counts >= 30].index.tolist()
        dq = d[d['register'].isin(qual)]
        e = eta_squared(dq['schwa_v1_AH0'].values, dq['register'].values)
        print(f"  {label}: N_qual={len(dq)}, buckets={len(qual)}, η² = {e:.4f}")

    # Save comparison
    comp = pd.DataFrame([
        {'corpus': 'OANC', 'variant': 'original (OOV excluded)',
         'n_qual': sum(orig['register'].isin(orig['register'].value_counts()[lambda x: x >= 30].index)),
         'eta2': round(eta_squared(orig[orig['register'].isin(orig['register'].value_counts()[lambda x: x >= 30].index)]['schwa_v1_AH0'].values,
                                   orig[orig['register'].isin(orig['register'].value_counts()[lambda x: x >= 30].index)]['register'].values), 4)},
        {'corpus': 'OANC', 'variant': 'G2P fallback (espeak-ng)',
         'n_qual': sum(df['register'].isin(df['register'].value_counts()[lambda x: x >= 30].index)),
         'eta2': round(eta_squared(df[df['register'].isin(df['register'].value_counts()[lambda x: x >= 30].index)]['schwa_v1_AH0'].values,
                                   df[df['register'].isin(df['register'].value_counts()[lambda x: x >= 30].index)]['register'].values), 4)}
    ])
    comp.to_csv(ROOT / 'g2p_comparison.csv', index=False)
    print(f"\nWrote g2p_comparison.csv")
    print(comp.to_string(index=False))

if __name__ == '__main__':
    main()
