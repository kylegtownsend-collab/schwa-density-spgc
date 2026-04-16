#!/usr/bin/env python3
"""
Schwa Density Analyzer — standalone script.

Takes a directory of text files (plus optional metadata CSV) and produces
a features CSV ready for statistical analysis.

Usage:
    python schwa_analyzer.py --input <text_dir> --output features.csv \
                              [--metadata metadata.csv] \
                              [--text-column file_name] \
                              [--register-column lcc] \
                              [--max-files 1000] \
                              [--encoding utf-8]

Examples:
    # OANC structure (recursive search for .txt under data/)
    python schwa_analyzer.py --input ~/OANC/data --output oanc_features.csv \
        --metadata ~/OANC/index.csv --text-column path --register-column genre

    # SPGC (Standardized Project Gutenberg Corpus from Zenodo)
    python schwa_analyzer.py --input ~/SPGC/data --output spgc_features.csv \
        --metadata ~/SPGC/SPGC-metadata-2018-07-18.csv \
        --text-column id --register-column subjects --recursive

    # Plain directory, no metadata (just text files)
    python schwa_analyzer.py --input ./my_texts --output features.csv

Output columns:
    text_id, n_words, n_vowels, oov_rate, schwa_v1_AH0, schwa_v2_AH0_IH0,
    schwa_v3_all_unstressed, schwa_v4_AH_any, cond_entropy, marg_entropy,
    ttr, mean_word_length, mean_sentence_length, latinate_ratio,
    mean_syllables, fk_grade, [register if metadata provided]

Dependencies (auto-installed if missing):
    nltk, numpy, pandas

Tested on Python 3.8+. Should work in Google Colab without modification.
"""
import argparse
import csv
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

# ----- Auto-install dependencies (helps in Colab and clean envs) ---------
def _ensure_deps():
    needed = []
    try: import nltk
    except ImportError: needed.append('nltk')
    try: import numpy
    except ImportError: needed.append('numpy')
    try: import pandas
    except ImportError: needed.append('pandas')
    if needed:
        print(f"Installing missing deps: {needed}", file=sys.stderr)
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + needed)

_ensure_deps()

import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import cmudict

def _ensure_nltk_data():
    """Download NLTK data if missing. Tries default download then GitHub fallback."""
    needed = [('corpora/cmudict', 'cmudict'),
              ('tokenizers/punkt_tab', 'punkt_tab'),
              ('tokenizers/punkt', 'punkt')]
    for path, pkg in needed:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception as e:
                print(f"Standard NLTK download failed for {pkg}: {e}", file=sys.stderr)
                print(f"Try manual install: python -m nltk.downloader {pkg}", file=sys.stderr)
                sys.exit(1)

_ensure_nltk_data()

CMU = cmudict.dict()
CMU_VOWELS = {'AA','AE','AH','AO','AW','AY','EH','EY','IH','IY','OW','OY','UH','UW'}
VIDX = {v: i for i, v in enumerate(sorted(CMU_VOWELS))}
LATINATE = ('tion','ity','ance','ence','ous','ment','ive','al','ary','ory','ism','ist')

# ----- Phonemic analysis -------------------------------------------------
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
        # Heuristic fallback
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
    if m:
        text = text[m.end():]
    m = end_pat.search(text)
    if m:
        text = text[:m.start()]
    return text.strip()

def process_text(text_id, text, min_words=1000, max_oov=0.15,
                 strip_pg=True, max_chars=None, skip_chars=0,
                 tokenized_input=False):
    """Process a single text. Returns dict of features or None if rejected.

    tokenized_input=True: text is one token per line (e.g. SPGC). Bypasses
    word_tokenize/sent_tokenize. Sentence boundaries are unrecoverable, so
    mean_sentence_length is set to a constant 20 (publication-typical) and
    fk_grade becomes effectively a syllables-per-word measure. Reported as
    fk_grade for table consistency; mean_syllables is the unconfounded
    comparator (per SPGC handoff §"Things that will trip you up").
    """
    if tokenized_input:
        if strip_pg:
            text = strip_gutenberg_boilerplate(text)
        if skip_chars:
            text = text[skip_chars:]
        if max_chars:
            text = text[:max_chars]
        words = [ln.strip() for ln in text.split('\n') if ln.strip()]
        sents = None
    else:
        if strip_pg:
            text = strip_gutenberg_boilerplate(text)
        if skip_chars:
            text = text[skip_chars:]
        if max_chars:
            text = text[:max_chars]
        try:
            words = word_tokenize(text)
            sents = sent_tokenize(text)
        except Exception as e:
            return {'text_id': text_id, '_error': f'tokenize: {e}'}

    alpha_words = [w for w in words if w.isalpha()]
    n_words = len(alpha_words)
    if n_words < min_words:
        return {'text_id': text_id, '_error': f'too_short ({n_words}<{min_words})'}

    vseq = []
    n_oov = 0
    for w in alpha_words:
        ph = get_phones(w)
        if ph is None:
            n_oov += 1
            continue
        vseq.extend(vowels_with_stress(ph))

    oov_rate = n_oov / n_words
    if oov_rate > max_oov:
        return {'text_id': text_id, '_error': f'oov_too_high ({oov_rate:.2%}>{max_oov:.0%}; likely non-English)'}
    if len(vseq) < 500:
        return {'text_id': text_id, '_error': 'too_few_vowels'}

    total = len(vseq)
    bases = [b for b, s in vseq]

    # Schwa variants
    n_AH0 = sum(1 for b, s in vseq if b == 'AH' and s == '0')
    n_IH0 = sum(1 for b, s in vseq if b == 'IH' and s == '0')
    n_unstr = sum(1 for b, s in vseq if s == '0')
    n_AH = sum(1 for b, s in vseq if b == 'AH')

    counts = Counter(bases)
    N = sum(counts.values())
    H_marg = -sum((c/N) * math.log2(c/N) for c in counts.values() if c > 0)

    M = np.zeros((14, 14))
    for a, b in zip(bases[:-1], bases[1:]):
        M[VIDX[a], VIDX[b]] += 1
    total_bg = M.sum()
    H_cond = 0.0
    if total_bg > 0:
        for i in range(14):
            rs = M[i].sum()
            if rs == 0: continue
            p_i = rs / total_bg
            rH = sum(-(M[i,j]/rs) * math.log2(M[i,j]/rs) for j in range(14) if M[i,j] > 0)
            H_cond += p_i * rH

    types = set(w.lower() for w in alpha_words)
    ttr = len(types) / n_words
    mwl = sum(len(w) for w in alpha_words) / n_words
    if sents is None:
        msl = 20.0  # heuristic; sentence boundaries unrecoverable from token stream
        msl_estimated = True
    else:
        msl = n_words / max(1, len(sents))
        msl_estimated = False
    n_lat = sum(1 for w in alpha_words if w.lower().endswith(LATINATE))
    lat = n_lat / n_words
    syll = sum(syll_count(w) for w in alpha_words) / n_words
    fk = 0.39 * msl + 11.8 * syll - 15.59

    return {
        'text_id': text_id,
        'n_words': n_words, 'n_vowels': total, 'oov_rate': oov_rate,
        'schwa_v1_AH0': n_AH0/total,
        'schwa_v2_AH0_IH0': (n_AH0+n_IH0)/total,
        'schwa_v3_all_unstressed': n_unstr/total,
        'schwa_v4_AH_any': n_AH/total,
        'cond_entropy': H_cond, 'marg_entropy': H_marg,
        'ttr': ttr, 'mean_word_length': mwl, 'mean_sentence_length': msl,
        'msl_estimated': msl_estimated,
        'latinate_ratio': lat, 'mean_syllables': syll, 'fk_grade': fk,
    }

# ----- Corpus iteration --------------------------------------------------
def find_text_files(input_dir, recursive=True, extensions=('.txt',)):
    """Yield text file paths under input_dir."""
    p = Path(input_dir)
    if not p.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    pattern = '**/*' if recursive else '*'
    for f in p.glob(pattern):
        if f.is_file() and f.suffix.lower() in extensions:
            yield f

def load_metadata(metadata_path, text_column, register_column=None):
    """Load metadata CSV. Returns dict mapping text_id_value -> {register: ...}."""
    if not metadata_path:
        return {}
    df = pd.read_csv(metadata_path, encoding='utf-8', encoding_errors='replace',
                     low_memory=False, dtype=str)
    if text_column not in df.columns:
        print(f"WARN: --text-column '{text_column}' not found in metadata. "
              f"Available: {list(df.columns)[:10]}", file=sys.stderr)
        return {}
    out = {}
    for _, row in df.iterrows():
        key = str(row[text_column]).strip()
        rec = {'meta_id': key}
        if register_column and register_column in df.columns:
            val = row.get(register_column)
            rec['register'] = str(val) if pd.notna(val) else 'unknown'
        for c in df.columns:
            if c not in (text_column, register_column):
                rec[f'meta_{c}'] = row[c] if pd.notna(row[c]) else ''
        out[key] = rec
    return out

def match_metadata(file_path, metadata_dict):
    """Try to find a metadata row for this file. Tries: filename stem, full name, parent dir."""
    if not metadata_dict:
        return {}
    p = Path(file_path)
    candidates = [p.stem, p.name, p.parent.name, str(p)]
    for c in candidates:
        if c in metadata_dict:
            return metadata_dict[c]
    return {}

# ----- Main --------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Compute schwa-density features for a text corpus.")
    ap.add_argument('--input', required=True, help='Directory containing text files')
    ap.add_argument('--output', required=True, help='Output CSV path')
    ap.add_argument('--metadata', help='Optional metadata CSV file')
    ap.add_argument('--text-column', default='id',
                    help='Column in metadata that matches text filenames (stem). Default: id')
    ap.add_argument('--register-column', help='Column in metadata to use as register label')
    ap.add_argument('--max-files', type=int, default=0, help='Limit number of files (0=all)')
    ap.add_argument('--max-chars', type=int, default=200000,
                    help='Truncate each text to N chars (default 200000 ~30K words)')
    ap.add_argument('--skip-chars', type=int, default=0,
                    help='Skip N leading chars per file (default 0)')
    ap.add_argument('--min-words', type=int, default=1000,
                    help='Minimum word count to include text (default 1000)')
    ap.add_argument('--max-oov', type=float, default=0.15,
                    help='Maximum CMUdict OOV rate (default 0.15)')
    ap.add_argument('--no-recursive', action='store_true',
                    help='Do not recurse into subdirectories')
    ap.add_argument('--no-strip-pg', action='store_true',
                    help='Do not strip Project Gutenberg boilerplate')
    ap.add_argument('--encoding', default='utf-8', help='Text file encoding (default utf-8)')
    ap.add_argument('--extensions', default='.txt',
                    help='Comma-separated extensions to process (default .txt)')
    ap.add_argument('--errors-out', help='Write rejected files + reasons to this CSV')
    ap.add_argument('--tokenized-input', action='store_true',
                    help='Input files are one-token-per-line (e.g. SPGC). '
                         'Bypasses NLTK tokenizers; mean_sentence_length set to '
                         'constant 20 (sentence boundaries unrecoverable).')
    args = ap.parse_args()

    extensions = tuple(e.strip() if e.startswith('.') else f'.{e.strip()}'
                       for e in args.extensions.split(','))

    print(f"Loading metadata from {args.metadata}..." if args.metadata else "No metadata.")
    metadata = load_metadata(args.metadata, args.text_column, args.register_column)
    print(f"  Loaded {len(metadata)} metadata rows.")

    print(f"Scanning {args.input} for {extensions} files (recursive={not args.no_recursive})...")
    files = list(find_text_files(args.input, not args.no_recursive, extensions))
    print(f"  Found {len(files)} files.")
    if args.max_files:
        files = files[:args.max_files]
        print(f"  Limiting to first {args.max_files}.")

    rows = []
    errors = []
    matched_metadata = 0
    for i, fp in enumerate(files):
        if i % 50 == 0 and i > 0:
            print(f"  {i}/{len(files)} processed; {len(rows)} kept, {len(errors)} rejected")
        try:
            with open(fp, 'r', encoding=args.encoding, errors='replace') as f:
                text = f.read()
        except Exception as e:
            errors.append({'text_id': str(fp), '_error': f'read: {e}'})
            continue

        text_id = fp.stem
        meta = match_metadata(fp, metadata)
        if meta:
            matched_metadata += 1

        result = process_text(text_id, text,
                              min_words=args.min_words,
                              max_oov=args.max_oov,
                              strip_pg=not args.no_strip_pg,
                              max_chars=args.max_chars,
                              skip_chars=args.skip_chars,
                              tokenized_input=args.tokenized_input)

        if result is None or '_error' in result:
            errors.append(result if result else {'text_id': text_id, '_error': 'unknown'})
            continue

        # Attach metadata fields
        for k, v in meta.items():
            result[k] = v
        result['source_path'] = str(fp)
        rows.append(result)

    print(f"\nDone. {len(rows)} kept, {len(errors)} rejected.")
    if metadata:
        print(f"Metadata matched: {matched_metadata}/{len(files)} files.")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(args.output, index=False)
        print(f"Wrote {len(df)} rows × {len(df.columns)} cols to {args.output}")
        print(f"  Columns: {', '.join(df.columns[:12])}...")
        # Quick descriptive stats
        print(f"\nQuick check:")
        print(f"  Mean OOV rate: {df['oov_rate'].mean():.2%}")
        print(f"  Mean schwa_v1: {df['schwa_v1_AH0'].mean():.3f} (sd {df['schwa_v1_AH0'].std():.3f})")
        if 'register' in df.columns:
            print(f"  Unique registers: {df['register'].nunique()}")
            print(f"  Top registers: {df['register'].value_counts().head(8).to_dict()}")
    else:
        print("No texts processed successfully. Check inputs and error log.")

    if args.errors_out and errors:
        pd.DataFrame(errors).to_csv(args.errors_out, index=False)
        print(f"Wrote {len(errors)} rejection records to {args.errors_out}")

if __name__ == '__main__':
    main()
