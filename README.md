# Schwa density as a phonological register classifier

**Source repository for:**
Townsend, K. (2026). *Schwa Density as a Phonological Stylistic Classifier: Primary Stylistic, Secondary Modality. A Four-Corpus Pre-Registered Replication.* Analysis and writing assistance provided by Claude (Anthropic).

- Paper (PDF, HTML, DOCX): <https://papers.letsharkness.com/schwa-density/>
- Pre-registration: [`prereg.md`](prereg.md) (locked before any confirmatory data look)
- Deviation log: [`deviations.md`](deviations.md)

## Summary

We test whether schwa density — the proportion of vowel phones in a text that are unstressed schwa (CMUdict `AH0`) — can serve as a single-feature register classifier in English text. Pre-registered confirmatory tests pass on NLTK multi-source (`N=164`) and the Standardized Project Gutenberg Corpus (`N=2,767`); sensitivity analyses on Brown (`N=313`) and OANC (`N=4,375`) surface the measure's operating regimes.

Principal findings:

- **T1 (minimum effect)** passes on all four corpora at every feasible bucket-size threshold (`N≥20`, `N≥30` locked, `N≥50`).
- **Function-word ablation.** Masking the 198 NLTK English stopwords before computing schwa density *preserves or amplifies* register discrimination (η² retention 0.93–1.27). Schwa density is not a stopword-frequency proxy.
- **Two operating regimes.** Schwa density is a *Primary Stylistic Feature* on within-prose variation (NLTK, SPGC, Brown) and a *Secondary Modality Feature* on speech-vs-writing variation (OANC).
- **Joint partial-η²** retains 46–53% of the register signal on the two pre-registered corpora after controlling for syllables per word, mean word length, and Latinate ratio.
- **OOV robustness.** G2P fallback via espeak-ng on OANC moves η² from 0.847 → 0.810 (retention 0.96); register ordering preserved.

## What's in this repo

```
paper_draft.tex            # source of truth for the paper
paper.pdf                  # shipped artifact

inline_cites.py            # preprocess \citep{key} → inline text for pandoc
html2pdf.js                # playwright HTML → PDF renderer

schwa_analyzer.py          # feature extraction: per-text schwa density + controls
confirmatory_tests.py      # locked T1/T2/T3 runner (do not modify)
sensitivity_analyses.py    # partial-η², classification accuracy, joint-η²
generate_artifacts.py      # builds results tables + figures fig1, fig2

function_word_ablation.py  # reruns T1 with NLTK stopwords masked (ablation)
g2p_oanc_rerun.py          # OANC re-run with espeak-ng G2P fallback for OOV
bucket_sensitivity.py      # T1 across N≥20/30/50 thresholds
validate_sentence_heuristic.py  # checks ℓ̄=20 vs NLTK Gutenberg ground truth
make_ablation_figure.py    # fig3 — ablation unmasking visualization

export_brown.py            # corpus prep for Brown (from NLTK)
export_nltk_multi.py       # corpus prep for NLTK multi-source
extract_oanc.py            # corpus prep for OANC-GrAF
extract_subsample.py       # SPGC subsample extraction
build_spgc_subsample.py    # SPGC sample builder
validate_brown.py          # reproduction check vs prior Brown result

fig1_eta_comparison.png           # register discrimination across corpora
fig2_schwa_by_register_spgc.png   # per-register SPGC distribution
fig3_ablation_unmasking.png       # before/after function-word masking

*_features.csv             # per-text feature tables
*_features_masked.csv      # same, with NLTK stopwords masked
oanc_features_g2p.csv      # OANC with G2P fallback for OOV
*_metadata.csv             # per-corpus register labels
results_table.csv          # main results
ablation_comparison.csv    # function-word ablation
g2p_comparison.csv         # OANC G2P sensitivity
bucket_sensitivity.csv     # bucket-N sensitivity
sentence_length_validation.csv   # empirical ℓ̄ on NLTK Gutenberg

SPGC_HANDOFF.md            # SPGC-specific gotchas
SUMMARY_MEMO.md            # one-page executive summary
REVISION_PLAN.md           # peer-review response log
```

## What's NOT in this repo

Raw third-party corpora (licensing + bulk):

- **SPGC** — [Zenodo record 2422561](https://zenodo.org/record/2422561) (CC-BY-4.0)
- **OANC-GrAF** — <https://www.anc.org/data/oanc/>
- **Brown, Gutenberg sample, Reuters, etc.** — via NLTK: `python -m nltk.downloader all`
- **CMUdict** — via NLTK or <http://www.speech.cs.cmu.edu/cgi-bin/cmudict>

## Reproducing the analysis

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. NLTK data
python -m nltk.downloader brown cmudict gutenberg punkt punkt_tab \
    reuters state_union webtext nps_chat inaugural movie_reviews \
    stopwords

# 3. Install espeak-ng (for the G2P fallback; only needed for Priority 3 rerun)
sudo apt-get install -y espeak-ng  # or brew install espeak-ng

# 4. Download corpora to their expected locations
#    (see individual extract_*.py / export_*.py scripts)

# 5. Run feature extraction (per corpus)
python schwa_analyzer.py --input nltk_multi_texts --output nltk_multi_features.csv \
    --metadata nltk_multi_metadata.csv --text-column id --register-column register
# ... and similarly for brown / spgc / oanc (SPGC adds --tokenized-input)

# 6. Locked confirmatory tests
python confirmatory_tests.py nltk_multi_features.csv NLTK_multi --out tests_summary.csv
python confirmatory_tests.py spgc_features.csv SPGC --out tests_summary.csv
python confirmatory_tests.py brown_features.csv Brown --out tests_summary.csv
python confirmatory_tests.py oanc_features.csv OANC --out tests_summary.csv

# 7. Sensitivity / robustness analyses
python sensitivity_analyses.py
python function_word_ablation.py
python bucket_sensitivity.py
python g2p_oanc_rerun.py
python validate_sentence_heuristic.py

# 8. Figures
python generate_artifacts.py
python make_ablation_figure.py

# 9. Paper build (no LaTeX required — pandoc + playwright)
python inline_cites.py
pandoc paper_inlined.tex -o paper.docx --standalone
pandoc paper_inlined.tex -o paper.html --standalone --mathjax
node html2pdf.js
```

Bootstrap CIs throughout use `random_state=42` and 1,000 within-group resamples; results are deterministic given the same corpus snapshots.

## Citation

```bibtex
@misc{townsend2026schwa,
  author       = {Townsend, Kyle},
  title        = {Schwa Density as a Phonological Stylistic Classifier:
                  Primary Stylistic, Secondary Modality.
                  A Four-Corpus Pre-Registered Replication},
  year         = {2026},
  howpublished = {\url{https://papers.letsharkness.com/schwa-density/}},
  note         = {Source: \url{https://github.com/kylegtownsend-collab/schwa-density-spgc}.
                  Analysis and writing assistance provided by Claude (Anthropic).},
}
```

## License

Dual-licensed — see [`LICENSE`](LICENSE).

- **Code** (Python + JS scripts): MIT
- **Paper, figures, data tables, documentation**: CC-BY-4.0
- **Third-party corpora**: each governed by its own upstream license (see `LICENSE`)
