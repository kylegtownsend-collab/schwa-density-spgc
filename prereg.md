# Pre-registration: Schwa Density as a Single-Feature Register Classifier

**Authors:** Kyle [+ Claude as analysis collaborator]
**Date locked:** 2026-04-15
**Status:** Pre-data-look, confirmatory analysis plan

---

## 1. Background and prior work

A pilot study (N=30 canonical Anglophone literary texts) found Pearson r = −0.89 between schwa density (proportion of vowel phones that are unstressed schwa, CMUdict AH0) and conditional vowel transition entropy H(V_n | V_{n-1}). An exploratory replication on the Brown corpus (N=500) found r = −0.92, with schwa density discriminating Brown's five coarse registers at η² = 0.558 — competitive with Flesch-Kincaid grade level (η² = 0.552) and mean word length (η² = 0.514).

The exploratory phase also surfaced a critical interpretive constraint: marginal vowel entropy correlates with conditional vowel entropy at r = 0.987, indicating the "vowel transition structure" framing of the original work is misleading. The schwa effect is essentially mediated through unigram vowel-frequency distribution, not transition predictability. The defensible claim the confirmatory phase tests is therefore narrower:

> **Schwa density is a phonologically motivated single-feature register classifier that performs at least as well as Flesch-Kincaid grade level on independent corpora.**

This pre-registration locks the analysis plan **before** the confirmatory corpus is touched.

---

## 2. Confirmatory corpus

**Corpus:** A multi-source register-stratified collection assembled from NLTK's bundled corpora, completely independent of the Brown corpus used in the exploratory phase.

**Inclusion criteria (locked before any data look):**
- English language, original (not translation).
- Document length ≥ 1,000 alphabetic word tokens after tokenization. (Lower than Brown's effective floor because we want broader register coverage; schwa% is more robust than entropy at small N.)
- CMU dictionary OOV rate ≤ 15% (the threshold from the original handoff document).

**Sources and target stratification (N target ≥ 200):**
| Register bucket | NLTK source | Target N |
|---|---|---|
| literary_fiction | gutenberg | up to 18 (all available) |
| drama | shakespeare | up to 7 (all available) |
| oratorical | inaugural + state_union | up to 60 (sampled) |
| news | reuters + abc | 50 (random sample) |
| reviews | movie_reviews | 50 (random sample) |
| web_informal | webtext + nps_chat | up to 10 |

If stratification yields < 30 documents in any bucket after applying inclusion criteria, that bucket is reported but excluded from the register-discrimination test (with the deviation noted explicitly).

**Random seed for sampling:** 42 (locked here).

---

## 3. Primary measures (locked operationalization)

**Schwa density (primary):** count of CMUdict phones equal to `AH0` divided by count of all vowel phones (the 14 ARPAbet vowels with stress digit stripped). First-pronunciation rule for words with multiple CMUdict entries. OOV words excluded from the count rather than back-filled with espeak (espeak unavailable in this environment).

**Schwa density (sensitivity):** three additional definitions reported alongside primary, but **not** selectable as primary post hoc:
- v2: (AH0 + IH0) / total vowels
- v3: all unstressed vowels (any vowel with stress digit 0) / total
- v4: all AH (any stress) / total

**Comparison measures:**
- Flesch-Kincaid grade level (computed via CMUdict syllable count)
- Mean word length (characters per alphabetic token)
- Type-token ratio
- Mean sentence length (alphabetic tokens per sentence)
- Latinate ratio (suffix match: tion, ity, ance, ence, ous, ment, ive, al, ary, ory, ism, ist)
- Mean syllables per word

**Outcome variables for the three locked tests:**
- η² for register discrimination across stratification buckets
- Pearson r between schwa density and conditional vowel entropy
- Partial r between schwa density and conditional vowel entropy controlling for the five comparison measures above (excluding FK to avoid colinearity since it's derived from MWL and syllables)

---

## 4. Locked hypothesis tests

Three tests, each with explicit pass/fail criteria. The Lakens et al. (2018) tutorial framework is followed: each test is either a minimum-effect test (rejecting effects that are too small to matter) or a non-inferiority test (rejecting effects worse than an established comparator by a meaningful margin). Justifications follow each test.

### Test 1 — Minimum effect test against the "crud" floor (primary scientific claim)

**H0:** η² for schwa density across register buckets ≤ 0.04 (the crud floor; corresponds roughly to r = 0.20).
**H1:** η² > 0.04.
**Decision rule:** Reject H0 if the lower bound of the 95% bootstrap CI for η² (1,000 resamples) exceeds 0.04.
**Justification of threshold:** Lakens (2022) cites Ferguson & Heene (2021) suggesting r ≥ 0.10 as a minimum effect for psychology correlations to exceed crud. r = 0.20 squared = η² = 0.04 is a stricter version, justified because text-derived measures correlate with each other at non-trivial levels almost automatically (everything derives from the same word distribution), so the crud floor for textual measures should be set higher than for psychological correlations.
**Brown reference:** η² = 0.558. A drop to < 0.04 would indicate the Brown result was a corpus-specific artifact.

### Test 2 — Non-inferiority test against Flesch-Kincaid (publishability claim)

**H0:** η² for schwa density is more than 0.05 (absolute) lower than η² for Flesch-Kincaid grade on the same corpus and stratification.
**H1:** η² for schwa is within 0.05 of FK or higher.
**Decision rule:** Reject H0 if (η²_schwa − η²_FK) > −0.05 with 90% bootstrap CI excluding −0.05 on the lower side.
**Justification of margin:** A 0.05 absolute difference in variance-explained corresponds to roughly a 9% relative difference at the η² ≈ 0.55 range. Below this threshold, the difference is unlikely to be practically meaningful for choosing between measures in applied stylistic work. This is a cost-benefit-style margin (Lakens 2022, §9.10) — it's the smallest gap I'd consider a reason to prefer FK over schwa given that schwa is conceptually simpler (no syllable counting heuristics for OOV words, no sentence boundary requirement).
**Brown reference:** schwa η² − FK η² = +0.006 in Brown (essentially tied).

### Test 3 — Replication of the schwa-entropy correlation (small-telescopes test)

**H0:** |Pearson r between schwa density and conditional vowel entropy| ≤ 0.36.
**H1:** |r| > 0.36.
**Decision rule:** Reject H0 if observed |r| > 0.36 and one-sided p < 0.05.
**Justification of threshold:** Original study had N=30. Critical r for two-tailed p<0.05 with N=30 is r=±0.361 (also ≈ the small-telescopes threshold based on 33% power per Simonsohn 2015). This is the smallest effect that could have been statistically significant in the original study, which Lakens (2022, §9.12) recommends as a defensible SESOI for replications when no theoretically motivated threshold is available.
**Brown reference:** r = −0.92 (well past the threshold).

---

## 5. Confound model

OLS regression locked specification:

```
conditional_entropy ~ ttr + mean_word_length + mean_sentence_length 
                    + latinate_ratio + mean_syllables
```

Schwa density is then regressed on the same predictors. Residuals from both regressions are correlated to obtain the partial r between schwa density and conditional entropy.

**Reported as supplementary, not as a pass/fail test:** the partial r is informative about whether schwa contributes unique variance beyond surface lexical features. A partial |r| ≥ 0.40 is the descriptive benchmark (Brown gave 0.69), but no decision hinges on it.

---

## 6. Sensitivity reporting (no selection)

All four schwa definitions (v1–v4) are computed and reported in a single table. The primary test uses v1 (AH0). Variants are reported for transparency only — none can be promoted to primary based on results.

---

## 7. Deviations and unforeseen issues

Any deviation from this pre-registration will be:
1. Listed explicitly in the results section.
2. Tagged as either (a) inevitable given the data (e.g., a register bucket fell below the N=30 floor) or (b) a researcher decision (e.g., excluding a specific text after seeing it).
3. Reported with both the pre-registered analysis and the deviated analysis where feasible.

---

## 8. What would constitute a publishable finding

If Test 1 rejects (schwa exceeds crud) AND Test 2 rejects (schwa non-inferior to FK), the headline claim "schwa density is a phonologically motivated single-feature register classifier competitive with Flesch-Kincaid" is supported. This is the publishable result.

If Test 1 rejects but Test 2 fails, the finding is "schwa density discriminates register but is meaningfully worse than FK" — a partial result, probably not publishable as a methodological contribution.

If Test 1 fails, the Brown result was a corpus-specific artifact and the project should be abandoned or radically rethought.

Test 3 is a courtesy replication of the original N=30 finding for completeness; it's not load-bearing for the publishable claim.

---

## 9. Pre-commitments

- Random seed: 42
- Number of bootstrap resamples: 1,000
- α level: 0.05
- All schwa variants reported regardless of which performs best
- No exclusion of texts after seeing results except via the locked inclusion criteria
- Code preserved as-is; any modifications post-data-look are deviations
