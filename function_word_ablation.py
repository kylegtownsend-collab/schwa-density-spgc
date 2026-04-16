"""Function-word ablation — response to peer review critique.

Reruns feature extraction on all four corpora with function words (NLTK
stopwords, 179 English words) masked out BEFORE schwa density and FK are
computed. Then runs T1 (η² register discrimination) masked vs unmasked.

If T1 η² persists after masking, the phonological grounding claim is
strengthened. If it collapses, schwa density is a function-word-frequency
proxy and the paper needs rescoping.

Usage:
    python function_word_ablation.py

Outputs:
    {corpus}_features_masked.csv  — per corpus
    ablation_comparison.csv        — side-by-side T1 table
"""
import sys
import os
import math
import re
from pathlib import Path
from collections import Counter

sys.path.insert(0, '/home/kyle/schwa_spgc')

import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import cmudict, stopwords

try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    STOPWORDS = set(stopwords.words('english'))

CMU = cmudict.dict()
CMU_VOWELS = {'AA','AE','AH','AO','AW','AY','EH','EY','IH','IY','OW','OY','UH','UW'}
VIDX = {v: i for i, v in enumerate(sorted(CMU_VOWELS))}
LATINATE = ('tion','ity','ance','ence','ous','ment','ive','al','ary','ory','ism','ist')

def get_phones(word):
    p = CMU.get(word.lower())
    return p[0] if p else None

def vowels_with_stress(phones):
    out = []
    for p in phones:
        m = re.match(r'^([A-Z]+)([012])?$', p)
        if m and m.group(1) in CMU_VOWELS:
            out.append((m.group(1), m.group(2)))
    return out

def syll_count(word):
    p = get_phones(word)
    if p is None:
        w = word.lower(); v = "aeiouy"; c = 0; pv = False
        for ch in w:
            iv = ch in v
            if iv and not pv: c += 1
            pv = iv
        if w.endswith('e') and c > 1: c -= 1
        return max(1, c)
    return sum(1 for ph in p if re.match(r'^[A-Z]+[012]$', ph))

def strip_gutenberg_boilerplate(text):
    start_pat = re.compile(r'\*\*\*\s*START OF.*?\*\*\*', re.I | re.S)
    end_pat = re.compile(r'\*\*\*\s*END OF.*?\*\*\*', re.I | re.S)
    m = start_pat.search(text)
    if m: text = text[m.end():]
    m = end_pat.search(text)
    if m: text = text[:m.start()]
    return text.strip()

def process_text_masked(text_id, text, mask_words, tokenized_input=False,
                        min_words=500, max_oov=0.15, max_chars=200000,
                        strip_pg=True):
    if strip_pg:
        text = strip_gutenberg_boilerplate(text)
    if max_chars:
        text = text[:max_chars]

    if tokenized_input:
        words = [ln.strip() for ln in text.split('\n') if ln.strip()]
        sents = None
    else:
        try:
            words = word_tokenize(text)
            sents = sent_tokenize(text)
        except Exception as e:
            return {'text_id': text_id, '_error': f'tokenize: {e}'}

    alpha_words_all = [w for w in words if w.isalpha()]
    # Mask: drop function words
    alpha_words = [w for w in alpha_words_all if w.lower() not in mask_words]
    n_words = len(alpha_words)
    n_words_orig = len(alpha_words_all)
    mask_ratio = 1.0 - (n_words / max(1, n_words_orig))

    # After masking, floor for viability is lower (half of original min_words)
    if n_words < min_words:
        return {'text_id': text_id, '_error': f'too_short_after_mask ({n_words}<{min_words})'}

    vseq = []
    n_oov = 0
    for w in alpha_words:
        ph = get_phones(w)
        if ph is None:
            n_oov += 1
            continue
        vseq.extend(vowels_with_stress(ph))

    oov_rate = n_oov / n_words if n_words else 0
    if oov_rate > max_oov:
        return {'text_id': text_id, '_error': f'oov_too_high ({oov_rate:.2%})'}
    if len(vseq) < 250:
        return {'text_id': text_id, '_error': 'too_few_vowels_after_mask'}

    total = len(vseq)
    n_AH0 = sum(1 for b, s in vseq if b == 'AH' and s == '0')

    if sents is None:
        msl = 20.0
    else:
        msl = n_words_orig / max(1, len(sents))

    syll = sum(syll_count(w) for w in alpha_words) / n_words
    n_lat = sum(1 for w in alpha_words if w.lower().endswith(LATINATE))
    lat = n_lat / n_words
    mwl = sum(len(w) for w in alpha_words) / n_words
    fk = 0.39 * msl + 11.8 * syll - 15.59

    return {
        'text_id': text_id,
        'n_words_masked': n_words,
        'n_words_orig': n_words_orig,
        'mask_ratio': mask_ratio,
        'n_vowels': total,
        'oov_rate': oov_rate,
        'schwa_v1_AH0': n_AH0 / total,
        'mean_word_length': mwl,
        'mean_sentence_length': msl,
        'latinate_ratio': lat,
        'mean_syllables': syll,
        'fk_grade': fk,
    }

def eta_squared(values, groups):
    values = np.asarray(values)
    groups = np.asarray(groups)
    grand_mean = values.mean()
    ss_total = np.sum((values - grand_mean) ** 2)
    if ss_total == 0: return 0.0
    ss_between = 0.0
    for g in np.unique(groups):
        gv = values[groups == g]
        ss_between += len(gv) * (gv.mean() - grand_mean) ** 2
    return ss_between / ss_total

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

# ---- Corpus processing ----

CORPORA = [
    # (name, texts_dir, metadata_csv, text_col, register_col, tokenized)
    ('nltk_multi', 'nltk_multi_texts', 'nltk_multi_metadata.csv', 'id', 'register', False),
    ('spgc',       'spgc_texts',       'spgc_sample_metadata.csv', 'id', 'register', True),
    ('brown',      'brown_texts',      'brown_metadata.csv',       'id', 'register', False),
    ('oanc',       'oanc_texts',       'oanc_metadata.csv',        'id', 'register', False),
]

def process_corpus(name, texts_dir, metadata_csv, text_col, register_col, tokenized):
    print(f"\n=== {name} ===")
    root = Path('/home/kyle/schwa_spgc')
    tdir = root / texts_dir
    if not tdir.exists():
        print(f"  skip: {tdir} not found")
        return None

    meta = {}
    if (root / metadata_csv).exists():
        mdf = pd.read_csv(root / metadata_csv, dtype=str, low_memory=False)
        if text_col in mdf.columns and register_col in mdf.columns:
            for _, row in mdf.iterrows():
                meta[str(row[text_col]).strip()] = str(row[register_col]) if pd.notna(row[register_col]) else 'unknown'

    files = sorted(tdir.rglob('*.txt'))
    print(f"  {len(files)} files, {len(meta)} metadata rows")

    rows = []
    for i, fp in enumerate(files):
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{len(files)}  kept={len(rows)}")
        try:
            text = fp.read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue
        tid = fp.stem
        r = process_text_masked(tid, text, STOPWORDS, tokenized_input=tokenized)
        if r is None or '_error' in r:
            continue
        r['register'] = meta.get(tid, meta.get(fp.name, 'unknown'))
        rows.append(r)

    if not rows:
        print("  no rows")
        return None

    df = pd.DataFrame(rows)
    out_csv = root / f'{name}_features_masked.csv'
    df.to_csv(out_csv, index=False)
    print(f"  wrote {len(df)} rows → {out_csv}")
    return df

def run_t1(df, name, schwa_col='schwa_v1_AH0'):
    counts = df['register'].value_counts()
    qual = counts[counts >= 30].index.tolist()
    dfq = df[df['register'].isin(qual)].copy()
    if len(dfq) < 30 or len(qual) < 2:
        return {'corpus': name, 'n_qual': len(dfq), 'n_buckets': len(qual),
                'eta2_masked': None, 'ci_lo': None, 'ci_hi': None}
    e = eta_squared(dfq[schwa_col], dfq['register'])
    lo, hi = bootstrap_eta2(dfq[schwa_col].values, dfq['register'].values)
    return {'corpus': name, 'n_qual': len(dfq), 'n_buckets': len(qual),
            'eta2_masked': round(e, 4), 'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4),
            'pass_t1': lo > 0.04}

def main():
    print(f"Function-word mask: {len(STOPWORDS)} NLTK English stopwords")
    print(f"Sample: {sorted(list(STOPWORDS))[:20]}")

    results = []
    for name, *rest in CORPORA:
        df = process_corpus(name, *rest)
        if df is not None:
            res = run_t1(df, name)
            # Attach unmasked T1 from existing features CSV
            orig_csv = Path('/home/kyle/schwa_spgc') / f'{name}_features.csv'
            if orig_csv.exists():
                od = pd.read_csv(orig_csv)
                if 'register' in od.columns:
                    counts = od['register'].value_counts()
                    qual = counts[counts >= 30].index.tolist()
                    odq = od[od['register'].isin(qual)]
                    if len(odq) and len(qual) >= 2:
                        e_un = eta_squared(odq['schwa_v1_AH0'], odq['register'])
                        lo_un, hi_un = bootstrap_eta2(odq['schwa_v1_AH0'].values, odq['register'].values)
                        res['eta2_unmasked'] = round(e_un, 4)
                        res['ci_lo_unmasked'] = round(lo_un, 4)
                        res['retention'] = round(res['eta2_masked'] / e_un, 3) if e_un > 0 else None
            results.append(res)

    out = pd.DataFrame(results)
    out.to_csv('/home/kyle/schwa_spgc/ablation_comparison.csv', index=False)
    print("\n" + "=" * 70)
    print("FUNCTION-WORD ABLATION — T1 η² COMPARISON")
    print("=" * 70)
    print(out.to_string(index=False))
    print("\nRetention > 0.70  → phonological claim holds; schwa is not a function-word proxy")
    print("Retention < 0.30  → schwa density largely reduces to function-word frequency")

if __name__ == '__main__':
    main()
