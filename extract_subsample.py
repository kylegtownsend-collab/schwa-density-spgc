"""Extract just the subsample's token files from SPGC zip."""
import zipfile
from pathlib import Path
import pandas as pd

ZIP_PATH = Path('/home/kyle/schwa_spgc/SPGC-tokens-2018-07-18.zip')
OUT_DIR = Path('/home/kyle/schwa_spgc/spgc_texts')
META_PATH = Path('/home/kyle/schwa_spgc/spgc_sample_metadata.csv')

OUT_DIR.mkdir(exist_ok=True)
sample = pd.read_csv(META_PATH)
ids = sample['id'].tolist()
print(f"Extracting {len(ids)} files...")

extracted = 0
missing = []
with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    names_set = set(z.namelist())
    for pg_id in ids:
        member = f"SPGC-tokens-2018-07-18/{pg_id}_tokens.txt"
        if member not in names_set:
            missing.append(pg_id)
            continue
        target = OUT_DIR / f"{pg_id}.txt"
        with z.open(member) as src, open(target, 'wb') as dst:
            dst.write(src.read())
        extracted += 1
        if extracted % 500 == 0:
            print(f"  {extracted}/{len(ids)}")

print(f"Done. Extracted {extracted}, missing {len(missing)}")
if missing:
    print(f"First missing: {missing[:5]}")
