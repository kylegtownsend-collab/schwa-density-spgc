# Schwa Density Replication Study — Claude Code Handoff Document

## Project context

This document hands off a replication study of a finding about **schwa density as a single-feature register classifier in English text**. An exploratory study (Brown corpus, N=500) and two confirmatory replications (NLTK multi-source N=273; GITenberg N=241) all support the finding. This handoff covers a fourth replication on a much larger and properly catalogued corpus: the Standardized Project Gutenberg Corpus (SPGC, ~55K books).

The deliverables are:
1. A reproducible analysis pipeline that runs from raw SPGC files to publication-ready figures and statistics
2. A pre-registered confirmatory test report
3. Optionally, a workshop-paper draft

This is **not** an open-ended research project. The hypotheses are pre-specified (see §5), the thresholds are locked (see §6), and the goal is to determine whether the finding survives at the largest scale tested so far.

## Background — what's been established already

A pilot study found that schwa density (proportion of CMUdict AH0 phones in a text's vowel stream) correlates very strongly with conditional vowel transition entropy, and discriminates English-language registers about as well as Flesch-Kincaid grade level. The exploratory phase (Brown N=500) found:

- Pearson r(schwa, conditional vowel entropy) = −0.92
- η²(schwa, register) = 0.558 across 5 Brown registers
- η²(FK, register) = 0.552 — schwa essentially tied with FK on Brown

Two confirmatory replications followed:

| Corpus | N | r(schwa, cond_H) | η²(schwa) | η²(FK) | schwa−FK |
|---|---|---|---|---|---|
| NLTK multi-source | 223 | −0.91 | 0.529 | 0.191 | +0.338 |
| GITenberg sample | 142 | −0.94 | 0.318 | 0.117 | +0.201 |

All three pre-registered tests (T1 minimum effect, T2 non-inferiority vs FK, T3 small-telescopes correlation replication) passed on both confirmatory corpora. The defensible publishable claim is:

> Schwa density is a phonologically motivated single-feature register classifier that matches Flesch-Kincaid grade level on educational/journalistic prose and outperforms it on broader register collections.

The SPGC replication is the strongest available test of this claim because (a) N is much larger, (b) SPGC ships with proper Library of Congress Classification (LCC) subject codes for ~85% of books, (c) it's the standard cleaned corpus used by computational humanities researchers, and (d) it's been used in published work, so referees recognize it.

**One critical caveat from prior work:** marginal vowel entropy correlates with conditional vowel entropy at r ≈ 0.99 in all corpora examined. The "vowel transition structure" framing of the original work was misleading — the schwa-entropy effect runs primarily through unigram frequency, not transition predictability. The defensible claim is about register classification, not about vowel transitions specifically. Do not let this slip back into the writeup.

## Inputs you should have on the VPS

```
SPGC-tokens-2018-07-18.zip       # 6.4 GB — book contents
SPGC-metadata-2018-07-18.csv     # 10 MB — catalog
SPGC-counts-2018-07-18.zip       # 1.5 GB — IGNORE, not needed
schwa_analyzer.py                # the standalone pipeline (already written)
preregistration.md               # locked analysis plan from prior phase
brown_schwa_results.csv          # exploratory results for comparison
confirmatory_results.csv         # NLTK confirmatory results
gitenberg_with_registers.csv     # GITenberg confirmatory results
```

If any of those are missing, ask before proceeding. The four CSVs are needed to produce the cross-corpus comparison figure.

## Critical SPGC format check (do this FIRST)

The SPGC tokens are pre-tokenized — typically one token per line, lowercased. The existing `schwa_analyzer.py` expects running prose and uses `nltk.word_tokenize`. Before running the full pipeline, verify the format and patch the analyzer if needed:

```bash
unzip -p SPGC-tokens-2018-07-18.zip "SPGC-tokens/PG2701_tokens.txt" | head -30
```

Three possible formats and what to do for each:

1. **One token per line, no punctuation, all lowercased.** This is the most likely format. The analyzer needs a small patch: add a `--tokenized-input` flag that bypasses `word_tokenize` and reads one word per line, and reconstructs sentences using a heuristic (every N=20 words, or split on `.` if punctuation tokens are kept). Mean sentence length will be unreliable in this case — flag it as a known limitation rather than dropping the metric, since the other features don't depend on it.

2. **Running prose with paragraph breaks.** No patch needed — run analyzer as-is.

3. **JSON or other structured format.** Patch the analyzer to parse it.

Whatever you find, document it in the deviation log (§9 below). Do NOT silently tweak the analyzer without recording the change.

## Pipeline to run

### Step 1 — Setup (on the VPS)

```bash
mkdir -p ~/schwa_spgc && cd ~/schwa_spgc
# Place SPGC files and analyzer script here
python3 -m venv .venv && source .venv/bin/activate
pip install nltk numpy pandas scipy statsmodels scikit-learn matplotlib seaborn
python -c "import nltk; [nltk.download(p, quiet=True) for p in ['cmudict','punkt','punkt_tab']]"

unzip -q SPGC-tokens-2018-07-18.zip
ls SPGC-tokens/ | wc -l   # should be ~55K files
```

### Step 2 — Stratified subsampling

Don't run on all 55K books — overkill, and 6.4 GB is a lot of disk churn. Pull a stratified sample of 2,000 books across LCC top-level classes, balanced as much as the catalog allows.

The LCC code is in the metadata `subjects` column, formatted as `set('PR', 'PS', 'PZ', ...)` per book. Top-level letters are: A (general), B (philosophy/religion), C (history aux), D (world history), E/F (US/Americas history), G (geography/anthropology), H (social sci), J (political sci), K (law), L (education), M (music), N (fine arts), P (language/lit), Q (science), R (medicine), S (agriculture), T (technology), U (military), V (naval), Z (bibliography).

For a register study, group these into broader buckets:

```python
LCC_TO_REGISTER = {
    'P': 'literary',      # language and literature (P, PA, PB, PC, PD, PE, PF, PG, PH, PJ, PK, PL, PM, PN, PQ, PR, PS, PT, PZ)
    'B': 'philosophy_religion',
    'C': 'history',  'D': 'history',  'E': 'history',  'F': 'history',
    'G': 'geography_travel',
    'H': 'social_science',  'J': 'social_science',
    'K': 'law',
    'L': 'education',
    'M': 'arts',  'N': 'arts',
    'Q': 'science',  'R': 'medicine',  'S': 'agriculture',  'T': 'technology',
}
```

The "literary" bucket subdivides further if you want — PR is British literature, PS American, PZ juvenile fiction, PN drama/criticism. Whether to keep that subdivision is a judgment call. **First pass: use the coarse buckets above.** Second pass: split P into PR, PS, PZ, PN as separate registers if N permits.

Subsample target: 2,000 books with balanced N per register bucket where possible (cap at 250 per bucket so philosophy or technology don't get overwhelmed by literature). Use `random_state=42` everywhere.

```python
import pandas as pd, random
random.seed(42)
meta = pd.read_csv('SPGC-metadata-2018-07-18.csv')
meta = meta[meta['language'] == "{'en'}"]  # English only
# parse subjects column into top LCC letter
def lcc_top(s):
    if pd.isna(s): return None
    # subjects column format varies — inspect first
    ...
meta['register'] = meta['subjects'].apply(lambda s: LCC_TO_REGISTER.get(lcc_top(s)))
sample = meta.dropna(subset=['register']).groupby('register', group_keys=False).apply(
    lambda g: g.sample(min(len(g), 250), random_state=42))
sample.to_csv('sample_metadata.csv', index=False)
```

You'll need to inspect the `subjects` column format first — it may be a stringified Python set, a JSON list, or pipe-separated strings. Adjust the parser accordingly. Document the parsing rule in the deviation log.

### Step 3 — Run the analyzer

```bash
python schwa_analyzer.py \
    --input SPGC-tokens/ \
    --output spgc_features.csv \
    --metadata sample_metadata.csv \
    --text-column id \
    --register-column register \
    --max-files 2000 \
    --errors-out spgc_errors.csv
```

Expected wall time: ~30-60 minutes for 2K books on a modest VPS. Monitor output for OOV rate — if mean OOV is above 5% something is wrong (likely tokenization mismatch or non-English texts that slipped through).

### Step 4 — Run the pre-registered tests

Use the same test code as in the prior confirmatory phase. Pseudocode:

```python
import pandas as pd, numpy as np
from scipy import stats
import statsmodels.api as sm

df = pd.read_csv('spgc_features.csv')
np.random.seed(42)

# Apply the locked N >= 30 bucket rule
counts = df['register'].value_counts()
qualifying = counts[counts >= 30].index.tolist()
df_q = df[df['register'].isin(qualifying)].copy()

# T1: minimum effect, schwa η² > 0.04 with 95% bootstrap CI lower bound > 0.04
# T2: non-inferiority, (schwa η² - FK η²) lower 90% CI > -0.05
# T3: |Pearson r(schwa, cond_entropy)| > 0.36 with one-sided p < 0.05

# All three tests pass means the SPGC result reinforces the publishable claim.
# Any one failing means investigate why before reporting.
```

The full test code is in `/mnt/user-data/outputs/confirmatory_test.py` from the prior phase — copy and adapt it.

### Step 5 — Generate the comparison artifacts

You should produce four artifacts at the end:

1. **`spgc_results_table.csv`** — one row per LCC bucket with N, mean schwa, mean FK, mean cond_entropy, schwa SD.

2. **`pre_registered_tests_summary.csv`** — one row per test (T1, T2, T3) with: corpus, N, observed value, threshold, CI bounds, pass/fail. Rows for Brown, NLTK, GITenberg, SPGC so the four-corpus replication record is in one table.

3. **`fig1_correlation_replication.png`** — bar chart of |Pearson r(schwa, cond_H)| across the four corpora with the 0.36 small-telescopes threshold marked.

4. **`fig2_eta_comparison.png`** — grouped bar chart of η²(schwa) vs η²(FK) across the four corpora.

These are the figures that go in the paper. Use matplotlib defaults — don't get fancy. Save at 300 DPI.

### Step 6 — Write the deviation log

Document every place where you departed from this handoff, the prior pre-registration, or the original analyzer. For example:

- "SPGC tokens turned out to be one-word-per-line; analyzer was patched as described in §3 case 1."
- "subjects column was stringified Python set; parser used `ast.literal_eval` then took first element."
- "Two LCC buckets (K=law, U=military) had N<30 after subsampling and were excluded from primary tests per pre-reg §3."

Save as `deviations.md`. This is the file that lets a referee evaluate whether your departures from the locked plan were principled.

## Pre-registered tests (LOCKED — do not modify)

These tests were locked before SPGC data was available. You may not change thresholds. If a test fails, report the failure honestly and discuss it in the deviation log; do not adjust criteria post hoc.

### T1 — Minimum effect against the textual crud floor

- H0: η²(schwa, register) ≤ 0.04
- H1: η²(schwa, register) > 0.04
- Decision rule: reject H0 if the lower bound of the 95% bootstrap CI (1000 resamples, within-group resampling) for η² exceeds 0.04
- Brown ref: 0.558. NLTK: 0.529. GITenberg: 0.318. All passed.

### T2 — Non-inferiority vs Flesch-Kincaid

- H0: η²(schwa) is more than 0.05 (absolute) lower than η²(FK)
- H1: η²(schwa) is within 0.05 of FK or higher
- Decision rule: reject H0 if (η²_schwa − η²_FK) > −0.05 with 90% bootstrap CI lower bound > −0.05
- Brown: +0.006. NLTK: +0.338. GITenberg: +0.201. All passed.

### T3 — Replication of schwa-entropy correlation

- H0: |Pearson r(schwa, conditional vowel entropy)| ≤ 0.36
- H1: |r| > 0.36
- Decision rule: reject H0 if observed |r| > 0.36 and one-sided p < 0.05
- Threshold derivation: smallest r that could have been significant in the original N=30 study (Lakens 2022 §9.12 / Simonsohn 2015 small telescopes)
- Brown: −0.92. NLTK: −0.91. GITenberg: −0.94. All passed.

## Sensitivity analyses to include

These are not pre-registered but should be reported alongside the locked tests:

1. **All four schwa definitions** (v1=AH0, v2=AH0+IH0, v3=all unstressed, v4=all AH). v1 is primary; the others are reported for transparency, never selected post hoc.

2. **Partial correlation** between schwa and cond_entropy controlling for TTR, mean word length, mean sentence length, latinate ratio, mean syllables. Brown gave −0.69; NLTK gave −0.57. Descriptive only.

3. **Cross-validated 5-fold logistic regression accuracy** with each of {schwa, FK, mean word length, latinate ratio} as a single predictor for register. Compare to majority-class baseline.

4. **LCC subdivision sensitivity:** if the primary analysis used coarse buckets, also report results with P split into PR/PS/PZ/PN. If conclusions are robust to this choice, say so. If not, that's important.

## Things that will trip you up

**Memory.** 55K books unzipped is large. Do not load all of them at once. Stream through the analyzer one file at a time (which the existing analyzer already does) and discard text after feature extraction.

**Encoding.** SPGC files should be UTF-8 but some Gutenberg source texts have curly quotes, em dashes, and Latin-1 holdovers. The analyzer uses `errors='replace'` which is correct — don't change it to `errors='strict'`.

**The `language` column in metadata.** It's a stringified set: `"{'en'}"`. Filter with `meta['language'] == "{'en'}"` (string match), not `'en' in meta['language']` which would also match `"{'en', 'fr'}"` multilingual texts that we want to exclude.

**The `subjects` column.** Same format issue. Inspect first, then write a parser. Books missing this field should be excluded from register-discrimination tests but included in the correlation test (T3 doesn't need register labels).

**Books that are translations.** SPGC includes English translations of non-English originals. The schwa story is about English phonology, so translations should arguably stay (the translator's English usage is what we're measuring). But document the decision either way. The metadata column `authoryearofbirth` won't help; check if there's a `translator` column.

**Tokenization affecting downstream features.** If SPGC strips punctuation, then `mean_sentence_length` is unrecoverable and Flesch-Kincaid grade is partially broken (FK uses sentences/word ratio). Two options: (a) report FK with the caveat that sentence length is unreliable, or (b) recompute FK using a constant sentence length of 20 (publication-typical), making it equivalent to a syllables-per-word measure. Prefer (a) and explain. If FK becomes uninterpretable, the headline T2 comparison may need to switch from FK to mean-syllables-per-word as the comparator. Note this would be a meaningful deviation requiring discussion.

**Don't treat T2 as an "obvious win" if FK is broken by tokenization.** The whole point of T2 is a fair head-to-head comparison. If SPGC's preprocessing handicaps FK, the comparison isn't fair. Report it both ways: with FK as-is, and with the closest equivalent that doesn't depend on sentence boundaries.

## What to do if a test fails

The previous three corpora all passed all three tests. If SPGC fails any test, that's interesting and important — do not try to make it pass.

- **T3 failure** (correlation < 0.36): something is fundamentally different about SPGC's text. Likely culprits: tokenization removed something the CMUdict pipeline depends on, or the language filter let non-English in. Investigate before reporting.

- **T2 failure** (schwa worse than FK): possibly real, possibly a tokenization artifact (see above). Report both with-FK and with-syllables comparisons.

- **T1 failure** (schwa η² ≤ 0.04): would be surprising given prior corpora. Check that register labels are being applied correctly and that you haven't accidentally given every book the same register.

In all failure cases, write a clear "what we learned" section. A failed replication on the largest corpus is publishable as a corrective if the failure modes are characterized properly.

## Optional: workshop paper draft

If you complete the analysis successfully, draft a 4-6 page paper targeting one of:

- *Corpora* journal (best fit, methods-friendly)
- LREC short paper or workshop
- *Digital Scholarship in the Humanities* journal

Suggested structure:

1. **Introduction** (~0.5 pp): the problem of register classification, the existing toolkit (Biber MDA, FK, SMOG), and the gap (no phoneme-level single-feature measure).
2. **Method** (~1 pp): schwa density definition, computation pipeline, four corpora (Brown, NLTK multi-source, GITenberg, SPGC).
3. **Pre-registration** (~0.5 pp): note that hypotheses and thresholds were locked before confirmatory data collection. Cite Lakens, Scheel & Isager (2018) for the SESOI framework.
4. **Results** (~2 pp): the four-corpus replication table, the two figures, sensitivity analyses, partial correlations. Honest discussion of the marginal-vs-conditional entropy issue (§Background caveat).
5. **Discussion** (~1 pp): when does FK underperform schwa (broad register spread), when do they tie (educational/journalistic), what schwa is actually measuring (vowel reduction → register), limitations (English only, written text, CMUdict OOV bias toward proper nouns).
6. **Limitations explicit:** corpus availability, English-only, CMUdict gaps, the marginal/conditional entropy redundancy that means the "transition structure" framing of the original work was incorrect.

References to include: Biber (1988) MDA, Lakens et al (2018) equivalence testing, Simonsohn (2015) small telescopes, Plecháč (2021) versification stylometry, Grieve, Clarke & Chiang (2023) stylometric register, McLaughlin (1969) SMOG, Treiman et al (2019) entropy of vowels.

Do NOT draft a paper if any of the locked tests fail — first investigate and write up the failure.

## Deliverable summary

Send back to the user (kyle):
1. `spgc_features.csv` — the raw features
2. `spgc_results_table.csv` — register summaries
3. `pre_registered_tests_summary.csv` — four-corpus test record
4. `fig1_correlation_replication.png`
5. `fig2_eta_comparison.png`
6. `deviations.md` — deviation log
7. (Optional) `paper_draft.md` — workshop draft if all tests pass
8. The patched `schwa_analyzer.py` if you patched it for SPGC tokenization

Plus a one-page summary memo: did all three tests pass on SPGC? what were the values? what deviated? what's the recommended next step?

## Things NOT to do

- Don't change the pre-registered thresholds.
- Don't drop the GITenberg or NLTK results from the comparison table because they're "smaller corpora" — they're independent corpora with their own pre-registered tests, and they belong in the record.
- Don't reframe the finding as being about vowel transition structure. It isn't, and the prior phase established this clearly. The finding is about register classification.
- Don't expand to non-English corpora. The CMUdict pipeline is English-specific and the result wouldn't generalize cleanly.
- Don't run the analyzer on more than 5K books without checking with kyle first. Wall time and storage should be planned, not improvised.
- Don't promote one of the schwa variants (v2, v3, v4) to primary if it happens to perform better on SPGC. v1 is primary; the others are sensitivity reporting.
- Don't write a paper draft if T3 fails. A failed correlation replication means the basic finding is in question, not that a paper is needed.

## Contact and clarification

If anything in this handoff is ambiguous, ask kyle directly rather than guessing. The pre-registration (§5, §6) and the deviation logging (§9, end of §3) are the two places where guessing wrong has the highest cost.
