# Schwa Density Paper — Revision Plan

Peer review received 2026-04-16. Paused for usage budget; resume at noon.

Current deployed version: https://papers.letsharkness.com/schwa-density/ (HTML + PDF + DOCX)
Source of truth: `/home/kyle/schwa_spgc/paper_draft.tex`
Inlined citation version (what pandoc rendered from): `/home/kyle/schwa_spgc/paper_inlined.tex`

---

## Priority 1 — Function-word ablation — DONE 2026-04-16

Script: `function_word_ablation.py`. Result: `ablation_comparison.csv`.

Used all 198 NLTK English stopwords (more aggressive than the reviewer's "top 50"). T1 η² on register with function words masked vs unmasked:

| Corpus | η² masked | η² unmasked | retention |
|---|---|---|---|
| nltk_multi | 0.781 | 0.616 | **1.27** |
| spgc | 0.434 | 0.365 | **1.19** |
| brown | 0.250 | 0.202 | **1.24** |
| oanc | 0.786 | 0.847 | 0.93 |

**Outcome: rebuttal is decisive.** On three of four corpora, masking function words *increases* register discrimination. OANC dips slightly (retention 0.93) — consistent with existing paper text that OANC registers partly differ on function-word density itself (speech vs writing).

Interpretation for the response-to-reviewer: function words carry near-constant heavy schwa across registers; they're a noise floor, not the signal. Content-word schwa variation (likely Latinate-suffix-driven) is where the register information lives. Masking removes the drag-to-mean effect and amplifies between-register variance.

Writeup addition: new supplementary section "Function-word ablation" with the table above; one-sentence addition to abstract: "Masking NLTK's 198 English stopwords preserves or amplifies register discrimination on all four corpora (retention 0.93–1.27), ruling out function-word frequency as a confound."

---

## Priority 1b — Elevate ablation analysis — DONE 2026-04-16

All four reviewer upgrades shipped in `paper_draft.tex` and redeployed to https://papers.letsharkness.com/schwa-density/ :

- **Title:** "Schwa Density as a Phonological Stylistic Classifier — Primary Stylistic, Secondary Modality"
- **Abstract:** rewritten to include the ablation finding + Primary/Secondary regime framing
- **New Results §:** "Function-word sensitivity: is schwa density a stopword-frequency proxy?" (`\label{sec:ablation}`) — includes full ablation table and `fig3_ablation_unmasking.png`
- **Discussion §4.1** renamed "Two regimes: Primary Stylistic and Secondary Modality" (`\label{sec:regimes}`) — explicitly names the two regimes with three empirical signatures each
- **Limitations §"Sentence boundary fragility on SPGC":** honest admission that SPGC T2 is schwa-vs-syllables; phonological claim no longer leans on it, so we can scope it as exploratory and future work
- **Conclusion:** rewritten around the two-regime framing + ablation as confound rebuttal
- Figure `fig3_ablation_unmasking.png` in web dir; referenced and captioned in tex

Side effects noted: the deployed HTML title/nav bar must be re-applied after every pandoc rebuild (pandoc regenerates the header). Edit pattern documented in project_papers.md. No compute needed for 1b beyond the ablation figure (fast matplotlib, no re-tokenisation).

---

## Priority 1b-orig (for reference — reviewer's original language)

Reviewer saw the ablation result and upgraded guidance. Act on all four:

**1b.1 — Promote ablation from reviewer-response to Results section.** Add a new subsection "Function Word Sensitivity" in the Results (not Discussion, not supplementary). It's empirical proof that schwa density isn't a high-frequency-lexical-filler proxy — belongs with T1/T2/T3, not after them.

**1b.2 — Formalize the two-regime theory using the OANC divergence.** The paper already gestures at this in §4.1; make it explicit. Proposed framing:

> Schwa density operates in two regimes. As a *Primary Stylistic Feature* (NLTK, SPGC, Brown), it captures content-word phonological variation and is unmasked — actually amplified — by stopword removal (retention 1.19–1.27). As a *Secondary Modality Feature* (OANC speech-vs-writing), it partially depends on function-word frequency and retention drops to 0.93. This is not a weakness; it is a theoretical boundary the measure discovers.

Title/abstract rewrite should reflect the two-regime framing, not just "stylistic classifier."

**1b.3 — Ablation visualization (new figure, call it `fig3_ablation_unmasking.png`).**
- Side-by-side panels: schwa density distribution per register, unmasked (left) vs masked (right)
- Violin or ridgeline plots, one per corpus (4 rows × 2 cols grid) — or just the two pre-reg corpora if grid is too busy
- Show visually that register means spread apart after masking (supports the "noise floor" interpretation)
- Use existing masked CSVs; no new compute

**1b.4 — SPGC FK constant (Priority 2 below) can now be honestly conceded.** With the phonological signal proven robust in content words, admitting the FK-on-SPGC limitation doesn't sink the paper. Rewrite that section as "limitation" rather than hiding it.

Reviewer's words: *"Now that you've proven the phonological signal is robust in content words, you can more confidently admit the FK constant is a limitation without it sinking the paper's value."*

---

## Priority 2 — SPGC FK baseline rectification — DONE 2026-04-16

SPGC tokens ship with ALL punctuation stripped (verified: 0 punctuation-only tokens in sampled files). Punkt recovery on the token stream is impossible by construction.

Empirical validation instead: ran punkt on NLTK's bundled Gutenberg sample (18 books, full punctuation). Actual mean sentence length distribution: mean 20.23, median 18.33, SD 7.45, range [11.9, 43.2], 67% in [15,25]. Our ℓ̄=20 constant sits at the 61st percentile — empirically defensible as a central-tendency choice.

This does not fix the FK degeneracy (per-text constant still collapses FK to linear-in-syll/word) but bounds the bias. Limitations section rewritten accordingly. SPGC T2 is now scoped as exploratory; the phonological-grounding claim rests on T1 (unaffected), the function-word ablation, and NLTK/Brown joint partial-η².

Script: `validate_sentence_heuristic.py`. Output: `sentence_length_validation.csv`.

---

## Priority 3 — G2P fallback for OOV — DONE 2026-04-16

espeak-ng (via `phonemizer` 3.3.0) as G2P fallback for OANC OOV. Workflow: scan OANC → collect 20k+ unique OOV tokens → batch phonemize → count `ə`/`ɚ` as schwa and IPA vowels as total vowels → rerun feature extraction with lookup table filling OOV slots.

**Result:**
| Variant | N qualifying | η² |
|---|---|---|
| Original (OOV excluded) | 4375 | 0.847 |
| G2P fallback (espeak-ng) | 4406 | 0.810 |

Retention 0.96. Fill rate 68% (remaining OOV mostly proper nouns, non-English, tokenization artifacts). Per-register ordering preserved, all registers remain above T1 crud floor. Reviewer's OOV-bias concern is empirically addressed: the paper's OANC finding is robust to OOV handling.

Limitations section rewritten to cite empirical retention numbers and describe the G2P protocol.

Scripts: `g2p_oanc_rerun.py`. Outputs: `oanc_oov_phones.json` (1.1MB), `oanc_features_g2p.csv`, `g2p_comparison.csv`.

---

## Priority 2-orig — SPGC FK baseline rectification (original text)

**The critique:** Using ℓ̄=20 constant sentence length makes FK = `11.8 × syll/word − 7.79`, a pure linear rescaling. The T2 non-inferiority test on SPGC is mathematically trivial, not empirical.

**Action:**
1. Run `nltk.tokenize.punkt` on the original SPGC texts in `spgc_texts/` to recover real sentence boundaries
2. Recompute FK with actual sentences
3. Rerun T2 on SPGC with the corrected FK
4. If punkt fails on pre-modern orthography (possible for 19thC Gutenberg), **drop T2 from SPGC entirely** and re-scope the paper as "confirmatory on NLTK multi-source; SPGC FK comparison is exploratory because sentence boundaries were not reliably recoverable"

**T1 (effect-size floor) on SPGC is unaffected — stays as-is.**

Also update abstract + discussion to flag the SPGC sentence-boundary limitation explicitly, whichever path we take.

---

## Priority 3 — G2P fallback for OOV (existential for OANC)

**The critique:** Excluding OOV words rather than using a G2P model creates "literary bias." Technical/jargon-heavy registers (OANC 911 transcripts especially) are unfairly penalized. Evidence: `oanc_errors.csv` is 290KB vs `spgc_errors.csv` 2KB.

**Action:**
- Use `espeak-ng` (already installed on VPS from Joy project) via the `phonemizer` Python package as a G2P fallback
- Pipeline: try CMUdict first; on miss, run espeak-ng; map espeak IPA back to CMU phone set (or just detect schwa-like vowels: ə, ɐ, unstressed reductions)
- Rerun OANC especially; spot-check SPGC and Brown OOV rates for sanity

**Files touched:** `schwa_analyzer.py` — add G2P fallback path; regenerate OANC features.

---

## Priority 4 — Bucket-threshold sensitivity — DONE 2026-04-16

Added new Results subsection `\label{sec:bucket-sens}`: "Bucket-threshold sensitivity." Full table comparing T1 η² at N≥20, N≥30 (locked), N≥50 on all 4 corpora. Script: `bucket_sensitivity.py`. Output: `bucket_sensitivity.csv`.

**Key result:** T1 passes at every feasible threshold on every corpus. SPGC and OANC are essentially insensitive (η² varies by <0.01). Brown is the interesting case — including smaller registers (N≥20) RAISES η² from 0.202 to 0.594. The locked threshold is conservative for Brown, not permissive. Reverses the reviewer's "40% exclusion is driving the result" concern.

Title + scope refinement is already shipped in Priority 1b (title now "Schwa Density as a Phonological Stylistic Classifier — Primary Stylistic, Secondary Modality").

---

## Priority 4-orig — Polish (not existential, kept for reference)

### Bucket-exclusion sensitivity
Rerun T1 with N≥20 and N≥50 bucket thresholds (currently N≥30). Report as supplementary table. Addresses reviewer's concern that Brown's 40% exclusion rate might be driving the effect.

### Title + scope refinement
Current: *Schwa Density as a Single-Feature Register Classifier*
Proposed: *Schwa Density as a Phonological Stylistic Classifier: A Four-Corpus Pre-Registered Replication*

Rationale: OANC result genuinely shows modality collapse. "Stylistic" is the honest scope — schwa works within-prose, not across speech/writing modalities.

Abstract: add one sentence explicitly flagging that modality-driven register variation degrades schwa to a syllables-per-word proxy. (Discussion §4.1 already says this; surface it upstream.)

---

## What we're pushing back on (not acting)

### CMUdict first-entry bias
For a text-based classifier, measuring lexical-potential schwa IS the correct target. No performance data exists for written text because writing isn't speech. Note as a scope limitation in Discussion; do not redesign. Acknowledge more seriously for OANC speech transcripts specifically.

### "Schwa = syll/word + Latinate suffixes"
Joint-η² already statistically controls for both simultaneously. If schwa were additively decomposable, retention would be ≈0. Observed 46–53% is the direct rebuttal. No heavier morphological parser needed.

---

## Deployment after revision

1. Regenerate `paper_inlined.tex` from updated `paper_draft.tex` via `/tmp/inline_cites.py`
2. Pandoc → `paper.docx` and `paper.html`
3. Playwright → `paper.pdf` via `/tmp/html2pdf.js`
4. Copy all three + updated figures to `/var/www/papers/schwa-density/`
5. Re-apply header nav (← papers | DOCX | Download PDF) to `index.html` since pandoc overwrites it — Edit pattern is documented in project_papers.md memory

---

## Reviewer summary (condensed)

| # | Critique | Severity | Our response |
|---|----------|----------|--------------|
| 1 | SPGC FK collapses to syll/word with constant ℓ | Fatal | Fix via punkt or drop T2 |
| 2 | Function-word confound | Fatal | Ablation required |
| 3 | OOV exclusion bias | Fatal for OANC | G2P fallback |
| 4 | CMUdict first-entry bias | Scope limitation | Acknowledge only |
| 5 | Joint-η² not exhaustive | Fair | Bucket-N sensitivity |
| 6 | Brown 40% exclusion | Fair | Sensitivity analysis |
| 7 | Phonological content overclaimed | Fair | Title/abstract rescope |
| 8 | Causality vs Latinate suffix correlation | Fair but answered | Already in joint-η² math |

Original full review: preserved in conversation transcript.
