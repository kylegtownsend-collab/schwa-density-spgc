"""Extract OANC .txt files and build register metadata."""
import zipfile
from pathlib import Path
import csv

ZIP_PATH = Path('/home/kyle/OANC_GrAF.zip')
OUT_DIR = Path('/home/kyle/schwa_spgc/oanc_texts')
META_PATH = Path('/home/kyle/schwa_spgc/oanc_metadata.csv')
OUT_DIR.mkdir(exist_ok=True)

# Path level-2 (under data/) collapses to register bucket.
# spoken/face-to-face → spoken_conv
# spoken/telephone → spoken_phone
# written_1/fiction → fiction
# written_1/journal → journal
# written_1/letters → letters
# written_2/non-fiction → nonfiction
# written_2/technical → technical
# written_2/travel_guides → travel
REGISTER_MAP = {
    ('spoken', 'face-to-face'): 'spoken_conv',
    ('spoken', 'telephone'):   'spoken_phone',
    ('written_1', 'fiction'):  'fiction',
    ('written_1', 'journal'):  'journal',
    ('written_1', 'letters'):  'letters',
    ('written_2', 'non-fiction'): 'nonfiction',
    ('written_2', 'technical'):  'technical',
    ('written_2', 'travel_guides'): 'travel',
}

records = []
with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    for name in z.namelist():
        if not name.endswith('.txt'):
            continue
        # Path: OANC-GrAF/data/{level1}/{level2}/.../basename.txt
        parts = Path(name).parts
        try:
            data_idx = parts.index('data')
        except ValueError:
            continue
        if len(parts) < data_idx + 3:
            continue
        level1 = parts[data_idx + 1]
        level2 = parts[data_idx + 2]
        register = REGISTER_MAP.get((level1, level2))
        if register is None:
            continue
        # Make a unique id from path (replace / with _)
        rel = '/'.join(parts[data_idx + 1:])
        text_id = rel[:-4].replace('/', '_')  # drop .txt and flatten
        target = OUT_DIR / f'{text_id}.txt'
        with z.open(name) as src:
            content = src.read()
        target.write_bytes(content)
        records.append({'id': text_id, 'register': register, 'source_path': rel})

# Write metadata
with META_PATH.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['id', 'register', 'source_path'])
    w.writeheader()
    w.writerows(records)

from collections import Counter
counts = Counter(r['register'] for r in records)
print(f"Extracted {len(records)} .txt files")
print("Per-register counts:")
for reg, n in counts.most_common():
    marker = '' if n >= 30 else '  (BELOW N=30 — will be excluded from primary tests)'
    print(f"  {reg:14s} {n:5d}{marker}")
