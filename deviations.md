# Deviation log — Schwa Density Replication Study (SPGC + OANC)

**Date:** 2026-04-16
**Operator:** Claude Code
**Pre-registration:** prereg.md (locked 2026-04-15)
**Handoff:** SPGC_HANDOFF.md

This log records every place the executed analysis departed from the locked
pre-registration or the handoff document. Each entry tags the deviation as
either (a) inevitable given the data, or (b) a researcher decision.

---

## §1. Tokenization patch to schwa_analyzer.py
**Type:** (a) inevitable
**Where:** schwa_analyzer.py, `process_text()` and CLI

**What changed:** Added `--tokenized-input` flag. When set, the analyzer
bypasses `nltk.word_tokenize` / `nltk.sent_tokenize` and reads one token per
line. Sentence boundaries are unrecoverable, so `mean_sentence_length` is
set to a constant 20 (publication-typical) and `msl_estimated=True` is added
to each output row.

**Why:** SPGC ships pre-tokenized as one lowercased word per line, no
punctuation, no numbers (handoff §3 case 1). Verified on PG2701
(Moby-Dick) before patching.

**Impact:** On SPGC, `fk_grade` is effectively a linear transform of
`mean_syllables` because MSL is constant. Since η² is invariant under
linear transforms, η²(FK) on SPGC = η²(mean_syllables) exactly. The T2
non-inferiority comparison is fair (both schwa and "FK" suffer the same
sentence-handicap), but readers should understand SPGC's "FK" is a
syllables-per-word measure dressed up. Brown, NLTK, OANC all retain
real FK with intact sentence detection.

---

## §2. SPGC stratification: LCSH not LCC
**Type:** (b) researcher decision (forced by handoff bug)
**Where:** build_spgc_subsample.py

**What changed:** Handoff §2 specified stratification by Library of
Congress Classification (LCC) top-level letters from the metadata
`subjects` column. **No SPGC English book has any LCC code in `subjects`.**
The column contains Library of Congress *Subject Headings* (LCSH),
which are descriptive multi-segment strings (e.g. "United States --
History -- Revolution, 1775-1783 -- Sources").

**Resolution:** Mapped LCSH first-segments to register buckets via regex
patterns covering: fiction, poetry, drama, biography, history, science,
religion, philosophy, children, travel, essays, letters. Multi-match
priority order with `fiction` LAST as tiebreak (so "historical fiction" →
history, "Christian poetry" → poetry, "biographical drama" → drama).

**Why this is acceptable:** LCSH first-segments map more cleanly to
register than LCC top-level letters would have. The handoff's LCC scheme
would have lumped fiction, poetry, drama, and lit-crit all into one "P"
bucket. The LCSH approach yields 12 register-distinct buckets, all with
N≥30 after sampling.

**Sample composition:** 2,911 books selected from 27,084 bucketable
English texts, capped at 250 per bucket (random_state=42). After
analyzer filtering: N=2,767 (57 rejected for length/OOV).

---

## §3. Metadata language column format
**Type:** (a) inevitable
**Where:** build_spgc_subsample.py

**What changed:** Handoff §"Things that will trip you up" specified
`meta['language'] == "{'en'}"` (stringified Python set). Actual SPGC
metadata format is `"['en']"` (stringified Python list). Filter updated
accordingly. Without this fix the filter would have matched zero rows.

---

## §4. NLTK multi-source sample size
**Type:** (b) researcher decision
**Where:** export_nltk_multi.py

**What changed:** NLTK confirmatory rebuild yielded N=164 vs handoff
reference of N=223. The handoff's prior-session CSVs were not available.

**Why:** The pre-reg's 1,000-word minimum forced batching of short texts
(reuters articles ~200 words, movie reviews ~700 words) into multi-doc
groups to clear the threshold. Original session may have used
non-batched single docs and a lower min-words. Per pre-reg §9, code
preserved as-is and the min-words threshold was honored.

**Impact on results:** All three locked tests still pass on the rebuilt
NLTK sample (T1 PASS, T2 PASS, T3 PASS), confirming the publishable
claim survives the rebuild. r(schwa, cond_H) = −0.835 vs handoff
reference of −0.91; η²(schwa) = 0.616 vs ref 0.529.

---

## §5. Brown bucketing for comparison table
**Type:** (b) researcher decision
**Where:** confirmatory_tests.py output for Brown

**What changed:** Handoff reports Brown η²(schwa) = 0.558 across "5
Brown registers" but does not specify the 5-way grouping recipe. Brown
ships in NLTK with 15 fine categories. After applying the locked N≥30
bucket-exclusion rule, 6 Brown categories qualify (belles_lettres,
government, hobbies, learned, lore, news), giving η²(schwa) = 0.202.
Using a standard Francis-Kučera 5-way grouping (press / general / learned
/ fiction) gives η²(schwa) = 0.518 ≈ the handoff reference of 0.558.

**For the comparison table:** Brown is reported with the 6-bucket result
(honoring the locked N≥30 rule applied uniformly across corpora) and the
5-bucket result (matching the prior reference). Both are shown in the
table with the bucketing recipe noted.

**Why this matters:** Brown's small qualifying-N corpus and narrow
register space (all "informative prose" after N≥30 filter) explains why
the partial η²(schwa | syllables) drops to 0.032 on Brown — schwa and
syllables do similar work in narrow register spaces. NLTK and SPGC, with
broader register coverage, retain 76-83% of schwa's discrimination after
controlling for syllables.

---

## §6. OANC added as 5th replication corpus
**Type:** (b) researcher decision (with explicit non-pre-registered tag)
**Where:** extract_oanc.py, oanc_features.csv

**What changed:** OANC-GrAF (~625MB zip, 8,824 .txt files across 8
register subdirectories) added to the corpus comparison as a fifth
replication.

**Status:** **NOT PRE-REGISTERED.** The OANC results are reported as
sensitivity replication only, clearly labeled in all tables and figures.
T1/T2/T3 outcomes on OANC do not modify the publishable claim, which
rests on the three pre-registered corpora (NLTK, GITenberg, SPGC) per
the handoff §"Background".

**Register taxonomy from path structure:** spoken_conv (face-to-face),
spoken_phone (Switchboard), fiction (single-author Eggan, N=1, dropped),
journal (Slate + Verbatim), letters (ICIC), nonfiction (OUP), technical
(911 report + biomed + government + plos), travel (Berlitz).

**Test outcomes:** T1 PASS strongly (η²(schwa) = 0.847), T2 **FAIL**
(η²(schwa) − η²(FK) = −0.047, 90% CI [−0.054, −0.040]), T3 PASS (|r| =
0.899). The T2 failure on OANC is methodologically informative rather
than a contradiction of the publishable claim:

- OANC mixes speech registers (short sentences, simple vocab) with
  technical writing (long sentences, complex vocab). FK's two terms
  (sentence length AND syllables) align in the same direction,
  amplifying FK's discrimination.
- Schwa only captures the syllables side; by design it cannot match FK
  when sentence-length is also strongly register-discriminative.
- Partial η²(schwa, register | syllables) on OANC drops to 0.148 (17%
  retained), and r(schwa, syllables) = 0.938 — the strongest schwa-
  syllables coupling of any corpus tested. This is a "schwa is mostly a
  syllables proxy" regime.

This refines the scope of the publishable claim: schwa matches or beats
FK on **within-prose stylistic discrimination** (NLTK rhetorical types,
SPGC literary genres) where schwa carries register signal independent of
syllable count; schwa is essentially a syllables proxy for **cross-
modality discrimination** (speech vs technical, narrow informative
prose) where syllables and schwa carry the same information. The honest
positioning is "schwa is useful for within-prose stylistic
discrimination," not "schwa beats FK universally."

---

## §7. Figure / writeup framing reprioritized
**Type:** (b) researcher decision
**Where:** Final artifacts (fig1, fig2, abstract, results section)

**What changed:** Handoff §5 specified fig1 as the |r(schwa, cond_H)|
correlation bar chart (the entropy story) and fig2 as the η² comparison
(the register story). After mid-analysis methodological review, the
figures are reprioritized:

- **Fig 1 (headline):** η²(schwa) vs η²(FK) grouped bar across corpora
- **Fig 2:** Schwa density boxplots by register on most heterogeneous corpus
- **Fig S1 (supplementary):** correlation bars with 0.36 line, captioned as
  pipeline-consistency check
- **Fig S2 (supplementary):** schwa-vs-cond_H scatter with structural
  confound note

**Why:** Two methodological reviews flagged that the original handoff
framing over-weighted the entropy correlation. The line-31 caveat about
marginal/conditional collinearity (r ≈ 0.99) means the entropy
correlation is partly structural and not independent evidence for a
"transition structure" register effect. The pre-registered tests T1 and
T2 (η², the register discrimination) carry the publishable claim. T3
remains in the test record but is reframed in the writeup as a
pipeline-consistency check, not a substantive replication.

The pre-registration document is unchanged. Only the writeup framing
and figure priorities shift, which is fully under the writer's
discretion.

---

## §8. Sensitivity analyses added beyond pre-reg
**Type:** (b) researcher decision (transparency only)
**Where:** sensitivity_analyses.py

**What changed:** Added two checks beyond the pre-registered sensitivity
analyses:

1. **r(schwa, mean_syllables) and the partial η²(schwa, register | syllables).**
   Tests whether schwa is a syllables-per-word proxy or carries
   independent register signal. Brown: drops 0.202 → 0.032 (proxy story
   holds — but Brown's qualifying buckets are narrow). NLTK: 0.616 → 0.510
   (83% retained). SPGC: 0.365 → 0.276 (76% retained). Conclusion: schwa
   is mostly a syllables proxy on narrow-register corpora and carries
   substantial independent signal on heterogeneous corpora.

2. **Marg-vs-cond entropy divergence per corpus:** r(schwa, marg_H) vs
   r(schwa, cond_H). Brown 0.004, SPGC 0.012 (structural-dominance holds);
   NLTK qualifying subset 0.154, OANC 0.046 (conditional entropy carries
   genuinely different information on these corpora, possibly register-
   dependent transition structure beyond unigrams). Worth flagging in
   discussion; does not modify locked tests.

3. **Joint partial η²(schwa, register | syllables + mwl + latinate):**
   Tests whether schwa packages the joint signal of three surface
   lexical features or carries genuinely independent register signal.
   Results across corpora:

   | Corpus | Raw η² | Partial η² (joint 3-control) | % retained | Interpretation |
   |---|---|---|---|---|
   | Brown | 0.202 | 0.030 | 15% | Below crud floor — joint controls absorb schwa's signal |
   | NLTK_multi | 0.616 | 0.282 | 46% | Above crud floor — schwa carries independent signal |
   | SPGC | 0.365 | 0.192 | 53% | Above crud floor — schwa carries independent signal |
   | OANC | 0.847 | 0.109 | 13% | Joint controls absorb most of schwa's signal |

   The two pre-registered confirmatory corpora (NLTK, SPGC) support the
   strong phonological-content claim: schwa retains 46-53% of its
   register signal even after controlling for the three best surface
   lexical features. Brown and OANC fall into a "compression" regime
   where schwa is largely a one-feature compression of the joint
   surface-lexical signal.

Both are reported as descriptive sensitivity, with no decision rules
attached.

---

## §9. Prior-corpus CSVs not available
**Type:** (a) inevitable
**Where:** Comparison table

**What changed:** Handoff §"Inputs you should have" listed
brown_schwa_results.csv, confirmatory_results.csv (NLTK), and
gitenberg_with_registers.csv as inputs. None were available — they were
only present in the prior web-app instance and could not be transferred.

**Resolution:** Brown and NLTK rebuilt from scratch (with deviations §4
and §5 above). GITenberg corpus would require a fresh ~150-book download
+ analyzer run. Reported as "GITenberg (per handoff reference)" in the
comparison table using the values from the handoff (N=142, η²(schwa)=
0.318, η²(FK)=0.117, r(schwa, cond_H)=−0.94) with a footnote noting
those numbers were not re-verified in this session.

---
