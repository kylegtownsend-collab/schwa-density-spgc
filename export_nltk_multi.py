"""Export NLTK multi-source corpus per prereg §2 stratification.
Sources: gutenberg, shakespeare, inaugural+state_union, reuters+abc,
movie_reviews, webtext+nps_chat. Random seed 42.
"""
import csv
import random
from pathlib import Path
import nltk

for pkg in ['gutenberg', 'shakespeare', 'inaugural', 'state_union',
            'reuters', 'abc', 'movie_reviews', 'webtext', 'nps_chat']:
    nltk.download(pkg, quiet=True)

from nltk.corpus import (gutenberg, shakespeare, inaugural, state_union,
                         reuters, abc, movie_reviews, webtext, nps_chat)

random.seed(42)
OUT = Path('/home/kyle/schwa_spgc/nltk_multi_texts')
OUT.mkdir(exist_ok=True)
META = Path('/home/kyle/schwa_spgc/nltk_multi_metadata.csv')

records = []

def add(corpus_obj, fid, register, source):
    if hasattr(corpus_obj, 'raw'):
        try:
            text = corpus_obj.raw(fid)
        except Exception:
            text = ' '.join(corpus_obj.words(fid))
    else:
        text = ' '.join(corpus_obj.words(fid))
    stem = f"{source}_{fid}".replace('/', '_').replace('.', '_').replace(':', '_').replace(' ', '_')
    (OUT / f'{stem}.txt').write_text(text, encoding='utf-8')
    records.append({'id': stem, 'register': register, 'source': source, 'fileid': fid})

# literary_fiction (gutenberg, all 18)
for fid in gutenberg.fileids():
    add(gutenberg, fid, 'literary_fiction', 'gutenberg')

# drama (shakespeare, all)
for fid in shakespeare.fileids():
    add(shakespeare, fid, 'drama', 'shakespeare')

# oratorical: inaugural (~58) + state_union (~64), sample 60 from union
oratorical = []
for fid in inaugural.fileids():
    oratorical.append(('inaugural', fid))
union_files = list(state_union.fileids())
random.shuffle(union_files)
for fid in union_files[:60]:
    oratorical.append(('state_union', fid))
random.shuffle(oratorical)
for src, fid in oratorical[:60]:
    cobj = inaugural if src == 'inaugural' else state_union
    add(cobj, fid, 'oratorical', src)

# news: reuters has 10K+ tiny docs — group into batches of 50 to clear 1K-word floor
# Use first 50 reuters categories as 'documents' (each is multi-doc concat)
# OR just take 50 longer single docs from reuters + abc
reuters_files = list(reuters.fileids())
random.shuffle(reuters_files)
# Concat reuters docs in groups of 30 to ensure >1000 words each
group_size = 30
news_count = 0
for i in range(0, min(len(reuters_files), 50 * group_size), group_size):
    batch = reuters_files[i:i+group_size]
    if len(batch) < group_size:
        break
    text = '\n'.join(reuters.raw(f) for f in batch)
    stem = f"reuters_batch_{i//group_size:04d}"
    (OUT / f'{stem}.txt').write_text(text, encoding='utf-8')
    records.append({'id': stem, 'register': 'news', 'source': 'reuters', 'fileid': f'batch_{i}'})
    news_count += 1
    if news_count >= 40:
        break
# Add abc (a few files)
for fid in abc.fileids():
    add(abc, fid, 'news', 'abc')

# reviews (movie_reviews, sample 50)
mr_files = list(movie_reviews.fileids())
random.shuffle(mr_files)
# Concat in groups to clear 1000-word floor (each review is ~700 words)
for i in range(0, 50 * 2, 2):
    batch = mr_files[i:i+2]
    if len(batch) < 2:
        break
    text = '\n'.join(movie_reviews.raw(f) for f in batch)
    stem = f"movie_reviews_batch_{i//2:04d}"
    (OUT / f'{stem}.txt').write_text(text, encoding='utf-8')
    records.append({'id': stem, 'register': 'reviews', 'source': 'movie_reviews', 'fileid': f'batch_{i}'})

# web_informal: webtext + nps_chat
for fid in webtext.fileids():
    add(webtext, fid, 'web_informal', 'webtext')
for fid in nps_chat.fileids():
    add(nps_chat, fid, 'web_informal', 'nps_chat')

with META.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['id', 'register', 'source', 'fileid'])
    w.writeheader()
    w.writerows(records)

from collections import Counter
counts = Counter(r['register'] for r in records)
print(f"Exported {len(records)} files")
print("Per-register counts:", dict(counts))
