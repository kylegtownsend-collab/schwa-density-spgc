"""Export NLTK Brown corpus to one .txt per file with category as register label."""
import os
from pathlib import Path
import csv
import nltk
from nltk.corpus import brown

OUT_DIR = Path('/home/kyle/schwa_spgc/brown_texts')
OUT_DIR.mkdir(exist_ok=True)
META_PATH = Path('/home/kyle/schwa_spgc/brown_metadata.csv')

rows = []
for fid in brown.fileids():
    cat = brown.categories(fid)[0]
    words = brown.words(fid)
    text = ' '.join(words)
    stem = fid.replace('/', '_').replace('.', '_')
    (OUT_DIR / f'{stem}.txt').write_text(text, encoding='utf-8')
    rows.append({'id': stem, 'register': cat, 'fileid': fid})

with META_PATH.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['id', 'register', 'fileid'])
    w.writeheader()
    w.writerows(rows)

print(f"Exported {len(rows)} Brown files. Categories: {sorted(set(r['register'] for r in rows))}")
