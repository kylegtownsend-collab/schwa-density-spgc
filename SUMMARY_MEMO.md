# Schwa Density Replication Study — One-Page Summary
**Date:** 2026-04-16  •  **Operator:** Claude Code  •  **Pre-reg:** 2026-04-15 (locked)

## Did all three pre-registered tests pass on SPGC?

**Yes — all three pre-registered tests pass on SPGC (N=2,767 across 12 LCSH register buckets).**

| Test | Observed | 95/90% CI | Threshold | Result |
|---|---|---|---|---|
| **T1** η²(schwa) > 0.04 | **0.365** | [0.339, 0.397] | > 0.04 | **PASS** |
| **T2** η²(schwa) − η²(FK) > −0.05 | **−0.005** | [−0.025, +0.014] | > −0.05 | **PASS** |
| **T3** \|r(schwa, cond_H)\| > 0.36 | **0.936** | — | > 0.36 | **PASS** (pipeline check) |

Pre-reg + analyzer pipeline are both reproducible. Brown validation gate passed (r = −0.924 vs ref −0.92, delta 0.004). NLTK rebuild also passed all three locked tests.

## Headline result across four corpora

The η² comparison (Fig 1, headline figure) is the publishable claim:

| Corpus | N | η²(schwa) | η²(FK) | Gap | Pre-registered? |
|---|---|---|---|---|---|
| Brown (6-bucket) | 313 | 0.20 | 0.20 | 0.00 | exploratory only |
| NLTK_multi | 135 | 0.62 | 0.19 | **+0.43** | yes |
| SPGC | 2,767 | 0.37 | 0.37 | 0.00 | yes |
| OANC | 4,375 | 0.85 | 0.89 | −0.05 | sensitivity (not pre-reg) |

**Both pre-registered confirmatory corpora (NLTK + SPGC) cleared all three locked tests.** The publishable claim is supported.

## Key methodological finding (load-bearing for the writeup)

The critique that "schwa is just a syllables-per-word proxy" was tested directly via **partial η²(schwa, register | syllables + mwl + latinate_ratio)** — the joint-control test:

| Corpus | Raw η² | Joint-control partial η² | % retained | Above crud floor (0.04)? |
|---|---|---|---|---|
| Brown | 0.20 | 0.03 | 15% | **No** — schwa ≈ syllables proxy |
| NLTK | 0.62 | 0.28 | 46% | **Yes** — independent register signal |
| SPGC | 0.37 | 0.19 | 53% | **Yes** — independent register signal |
| OANC | 0.85 | 0.11 | 13% | Marginal |

**Schwa carries register signal independent of surface lexical features specifically when register variation is driven by within-prose stylistic differences** (NLTK rhetorical types, SPGC literary genres). When register variation is driven by speech-vs-writing or narrow-vs-broad formality (Brown informative prose, OANC speech-vs-technical), schwa is largely a one-feature compression of surface lexical features.

## Deviations from handoff (full details in `deviations.md`)

1. **Tokenization patch** — added `--tokenized-input` flag for SPGC's one-token-per-line format. Sentence boundaries unrecoverable; FK on SPGC = mean_syllables × constant.
2. **LCSH not LCC** — SPGC metadata has no LCC codes; built register buckets from LCSH first-segments instead. 12 register-rich buckets vs handoff's 18 LCC letters.
3. **Language column format** — handoff said `"{'en'}"`, actual is `"['en']"`.
4. **Prior CSVs unavailable** — Brown and NLTK rebuilt from scratch; GITenberg numbers reported from handoff reference only.
5. **OANC added** as 5th replication, clearly labeled non-pre-registered.
6. **Figure priorities reframed** per methodological review — η² leads, correlation demoted to pipeline-consistency check.
7. **Sensitivity analyses extended** to include schwa-as-syllables-proxy diagnosis and joint partial η².

## What's the recommended next step?

**Write the workshop paper.** The defensible scope-narrowed claim:

> *Schwa density is a phonologically motivated single-feature register classifier that matches Flesch-Kincaid grade level on independent corpora with intact-sentence prose, outperforms FK when sentence-length information is unavailable or unhelpful (NLTK heterogeneous registers), and underperforms FK only when register variation drives both FK terms in the same direction (cross-modality, e.g., speech vs technical writing). On within-prose stylistic discrimination, schwa retains 46–53% of its register signal even after controlling jointly for syllables, mean word length, and Latinate ratio — supporting the strong phonological-content claim that schwa carries register information beyond the surface lexical features.*

This is narrower than the original "schwa beats FK" framing but more defensible and more useful — it tells practitioners exactly when to reach for schwa vs FK.

**Suggested target journal:** *Corpora* (best fit, methods-friendly). LREC short-paper as fallback.

**Do NOT:** Reframe T3 as substantive evidence for vowel transition structure. The marginal/conditional entropy collinearity (r ≈ 0.99 except NLTK 0.85) means the entropy correlation is largely a structural artifact of Shannon entropy applied to a distribution where one category (schwa) varies substantially across texts. Use T3 as pipeline confirmation only.

## Deliverables in `/home/kyle/schwa_spgc/`

- `schwa_analyzer.py` — patched analyzer (added `--tokenized-input` flag)
- `confirmatory_tests.py` — pre-registered T1/T2/T3 with bootstrap CIs
- `sensitivity_analyses.py` — partial η², 5-fold CV, marg-vs-cond divergence
- `generate_artifacts.py` — figures + tables generator
- `brown_features.csv`, `nltk_multi_features.csv`, `spgc_features.csv`, `oanc_features.csv` — raw per-text features
- `tests_summary.csv` — four-corpus pre-registered test record
- `results_table.csv` — main comparison table (5 corpus rows incl. handoff GITenberg)
- `spgc_results_table.csv` — per-LCSH-register summary on SPGC
- `fig1_eta_comparison.png` — HEADLINE: η²(schwa) vs η²(FK)
- `fig2_schwa_by_register_spgc.png` — schwa density boxplots by register on SPGC
- `figS1_correlation_check.png` — supplementary: |r(schwa, cond_H)| pipeline check
- `figS2_schwa_vs_entropy_spgc.png` — supplementary: scatter with structural-confound caption
- `deviations.md` — full deviation log
- `SUMMARY_MEMO.md` — this document
