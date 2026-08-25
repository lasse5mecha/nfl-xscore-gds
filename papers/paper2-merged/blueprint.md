# Merged Paper: Game Deserved Score — Decomposing NFL Team Quality and Testing the Defensive Dominance Hypothesis

**Supersedes:** `papers/paper2-gds/` (rejected by JSS, fit) and `papers/paper3-archetypes/`
(rejected by IJSF, unverifiable self-citation + apparent duplication of the SSRN working
paper). Source material for both is reused; neither source directory is modified or deleted.

**Status:** Outline (blueprint) — precedes `paper.tex` drafting.
**Target:** ~11,500–12,000 body words (see word-budget note below; a fully rigorous merge of
two full papers runs closer to ~13,000–13,800 words even after aggressive cutting — treat
11,500–12,000 as the post-trim target, not the first-draft target).
**Depends on:** Task #1 (extend `analyze_archetypes_v2.py` to 2018–2025, N=256) must complete
before any Results/Discussion numbers below are finalized — everything in this blueprint that
currently reads "224 team-seasons" / a specific stat becomes N=256 with recomputed values.

---

## Why this replaces two separate papers

- Both `paper2.tex` and `paper3.tex` (submission copies) cited the xScore/GDS methodology as
  `[Author, 2026]` for double-blind anonymity, but neither submission's `references.bib`
  actually contained the entry — literally unverifiable, which is what IJSF's editor flagged.
- The **master** (non-anonymized) `papers/paper2-gds/references.bib` already has a correct,
  real entry for the SSRN working paper (`mecha2026xscore`, with DOI/URL) — the bug was
  introduced specifically in the venue-anonymized copies (`ijsf/`, `jss/`), which dropped the
  entry when redacting the in-text citation. This merge starts from the master content, which
  doesn't have that defect, and — per the approved plan — cites the SSRN paper openly rather
  than redacting it at all, so the defect structurally cannot recur.
- Paper 2's "component decomposition" (46.4%/14.2%/73.6%) and paper 3's H3 evidence are the
  *same statistic*, reported as two different findings in two different submissions. Merging
  removes this duplication rather than just hiding it better.
- Genuine new content over both prior submissions and the SSRN paper: the sample extends from
  224 to 256 team-seasons with the newly-completed 2025 season (SEA won Super Bowl LX,
  offense-dominant, extending the streak to 8/8).

---

## Title

**"Game Deserved Score: Decomposing NFL Team Quality and Testing the Defensive Dominance
Hypothesis"** (drops paper 3's causal "Offense Wins Championships" framing, which IJSF flagged).

---

## Section Plan

### Abstract (~250w)
Source: paper2 abstract (lines 48–50) + paper3 abstract (lines 48–49 of paper3.tex).
One narrative: GDS as a transparent 3-component metric (validated against wins) → applied to
test "defense wins championships" via 5 methods → all rejected → extended through the 2025
season as an out-of-sample check. State the correlational caveat here, not just in Discussion
(direct fix for IJSF's causal-language critique). Update N=224→256 and all headline numbers
once Task #1 lands.

### 1. Introduction (~1,500w)
Source: paper2 §1 (full, ~600w usable) + paper3 §1 (full, ~1,500w, condense hard).
- Open with paper3's stakes framing (salary cap zero-sum, $301.2M cap, Bear Bryant origin —
  condense the cultural-embedding paragraph to ~150w, it's coloring not substance).
- Fold in paper2's point-differential-is-noisy motivation as the measurement-gap argument
  (~200w): existing metrics (EPA/DVOA/PFF/WP) can't decompose reproducibly → need GDS.
  Reference Related Work for the full critique instead of repeating it here.
  1500).
- State the three hypotheses (H1/H2/H3, paper3 lines 74–78) verbatim — this is precise,
  falsifiable, and shouldn't be paraphrased.
- Close with **the correlational caveat**, moved up from Discussion per the IJSF fix, and a
  one-sentence roadmap.
- Cut: rule-change history to one sentence (was a full paragraph in paper3 line 66); RQ1/RQ2
  thesis-structure framing (not applicable, this isn't the thesis).

### 2. Related Work (~1,700w)
Source: paper2 §2 (full, ~1,400w) + paper3 §2 (full, ~1,500w, heavily condensed) + paper2's
"Comparison with Existing Metrics" (currently paper2 §6.3, lines 527–529 — relocate here, it's
literature positioning not a Discussion finding).
- 2.1 Existing team-quality metrics (EPA/DVOA/PFF/WP) — paper2's subsections 74–92, condensed
  by ~30%: keep the four-property gap statement (line 90–92), it's the paper's clearest
  positioning claim.
- 2.2 The "Defense Wins Championships" debate — paper3 §2.1 (Robst et al. 2011 critique, lines
  90–99), condensed to ~500w.
- 2.3 Postseason randomness and resource allocation — paper3 §2.2+§2.3 (lines 101–117),
  condensed hard to ~500w combined; keep Mulholland (2019) and Massey & Thaler (2013), cut
  Moneyball/cross-sport analogy detail.
- Do not restate "Contribution of This Study" (paper3 §2.4) as a separate subsection — fold its
  content into the Introduction's closing paragraph instead, avoids saying the same thing twice.

### 3. Data, xScore Recap, and the GDS Framework (~2,600w)
Source: paper2 §3 (Data & Probability Baseline, full, lines 99–124) + §4 (The GDS Framework,
full, lines 129–319, includes worked example and descriptive-stats subsection).
- **This is the direct fix for IJSF's "unverifiable" complaint.** Expand the xScore recap
  (currently ~150w in paper2 §3.3) to ~300w — enough that a reviewer can assess the model's
  validity (features, Brier score, calibration/ECE, held-out test methodology) without needing
  to trust an external source. Cite `mecha2026xscore` (the SSRN working paper — real bib entry,
  see references.bib below) openly, third-person phrasing, plus a footnote noting a
  peer-reviewed treatment of the model is a separate submission if that's still true at
  drafting time.
- Keep the full GDS derivation (Off_xVOA, Def_xVOA, ST_Value, turnover attribution, formal
  properties) — this is the paper's core methodological contribution and shouldn't be thinned.
  Compress "Formal Properties" (paper2 lines 246–254) to the existing bullet list only (already
  compact, no further cut needed).
- Keep the worked example (DET vs DAL, lines 280–319) — concrete and short, aids reviewer
  comprehension of an unfamiliar framework.
- Trim: descriptive-statistics subsection (lines 256–278) from full prose to table + 2-sentence
  summary (~150w saved) — interesting but secondary to the paper's argument.
- Update N=224→256 throughout, including Table (descriptives) once Task #1 numbers land.

### 4. Archetype Classification and Hypothesis-Testing Methods (~1,300w)
Source: paper3 §3 (Data and Methods, lines 127–282), MINUS §3.1 Measurement Framework
(lines 130–137) — that content is now covered in full by §3 above; keep only a 1-sentence
bridge ("Using the GDS components defined in §3...").
- Sample and Variables (paper3 lines 139–214): IV/DV tables, `offense_share` formula (Eq. 1
  in paper3, keep verbatim), threshold definitions. Update N=224→256, playoff subsample
  94→(recompute), SB winners 7→8.
- Hypotheses (paper3 §3.3, lines 216–219): cross-reference §1 rather than restating H1/H2/H3.
- Statistical Procedures (paper3 §3.4, lines 221–264): keep all five methods' justifications
  and equations — this is methodologically load-bearing, don't cut.
- Robustness Design (paper3 §3.5, lines 266–281): keep all six checks' descriptions (2-3
  sentences each, already compact).

### 5. Results — Framework Validation & Descriptive Patterns (~1,600w)
Source: paper2 §5.1–5.3 (Game-Winner Prediction, Season-Level Correlation incl. Table 2
component split, Component Structure Top/Bottom, lines 329–426) + paper3 §4.1 (Descriptive
Patterns: quartile table + SB participants table, lines 289–382).
- **Report the 46.4%/14.1%(→recompute)/3.3:1 split exactly once, here** — Table 2 (paper2
  lines 348–361) serves double duty: evidence the metric is valid (this section) AND the
  primary evidence for H3 (cross-referenced from §6, not restated). This is the single change
  that removes the duplicate-reporting an editor would otherwise flag.
- Keep game-winner prediction (86.1%, lines 329–333) and quartile table + SB participants table
  (paper3 lines 292–382) — these are the paper's strongest descriptive evidence.
- Compress "Component Structure: Top and Bottom Teams" (paper2 lines 390–426) from full table +
  4 paragraphs to table + 1 paragraph (~400w saved).
- **Cut from body, move to online supplement:** Luck Analysis (paper2 §5.4, lines 428–453) —
  paper2's own blueprint already flagged this as tangential; keep a single sentence pointing to
  the supplement. Also move "Baseline Comparison: Point Differential" (paper2 lines 365–381)
  and "Temporal Stability" / "Reliability: Split-Half Convergence" (paper2 lines 455–500) to
  supplement — good psychometric rigor, not essential to the paper's central argument, and the
  first target for trimming if still over budget.

### 6. Results — Hypothesis Tests, Consolidated Robustness (~2,700w)
Source: paper3 §4.2–4.5 (Hypothesis Tests, Quadrant Analysis, Robustness and Sensitivity,
Hypothesis Verdict, lines 384–634).
- Five tests reported compactly (Spearman, Pearson/R², Cohen's d, logistic, OLS) — keep all,
  cross-reference §5's split rather than restating numbers.
- **New table required here (direct fix for IJSF's "four incompatible operationalizations"
  critique):** one consolidated table cross-referencing every operationalization used anywhere
  in the paper — continuous `offense_share` (Spearman/Pearson), the ±0.3 threshold split
  (Cohen's d, quadrant), the quartile split (Table 3/quartile table), and the within-elite
  median split (quadrant analysis) — against its sample definition, N, and result. Declare the
  continuous `offense_share` + ±0.3 threshold as the **primary specification** (drives the
  headline Cohen's d and quadrant findings); quartile and within-elite-median splits are
  explicit robustness, not parallel primary analyses.
- **Fix the logistic-regression/Table-6 inconsistency:** in the Hypothesis Testing Summary
  table (paper3 Table 6, lines 460–484), the logistic regression's p-values currently appear as
  unqualified "evidence" for H2 despite the EPV=3.5 caveat stated elsewhere. Rewrite so the
  logistic result is explicitly framed as a **supporting/robustness check only** — the primary
  verdict for H2 rests on the fact that all 8 (post-2025) SB winners are offense-dominant, not
  on the underpowered logistic coefficients. State the EPV caveat once, inline, where the
  logistic table appears; don't let unqualified p-values resurface in the summary table.
- Quadrant analysis (paper3 §4.3, lines 488–529): keep, compress slightly.
- **Move to supplement, keep only a summary table + short paragraph in body:** the detailed
  per-check robustness prose (paper3 §4.4, lines 531–630 — opponent-strength, field-position
  mediation, time-window table, threshold-sensitivity table, single-team exclusion, era
  comparison). Body keeps: one compact table with each check's key number, and a 3-sentence
  "robustness held across all checks" paragraph. This is the single largest cut available
  (~400w) and is consistent with what several journals expect for exploratory/secondary
  robustness detail anyway.
- Hypothesis Verdict (paper3 lines 632–634): keep, ~3 sentences.

### 7. Discussion (~2,000w)
Source: paper2's "Practical Meaning of the 3.3:1 Ratio" (§6.1, lines 512–514, ~150w usable) +
paper3 §5 (full, lines 639–708).
- Verdict (paper3 §5.1, condensed, ~300w)
- Competent Defense Threshold (paper3 §5.2, lines 651–660, ~450w) — keep close to full length,
  this is paper3's real theoretical contribution beyond hypothesis rejection.
- Structural Explanations (paper3 §5.3, lines 662–671, ~350w) — three mechanisms kept at
  reduced length (2-3 sentences each instead of 4-5).
- Why the Myth Persists (paper3 §5.4, lines 673–682, ~200w, trimmed hard from ~500w) — keep the
  availability-bias/denominator-problem core argument, cut extended romantic-appeal discussion
  to one sentence.
- Strategic Implications (paper3 §5.5, lines 684–691, ~200w) — **explicitly conditioned on
  causal-if language** (already present in source, "conditional on these associations...
  reflecting causal mechanisms" — keep this framing intact, don't let it drift toward
  unconditional prescription; this satisfies IJSF's specific ask).
- Limitations (~550w): merge paper3 §5.6 (lines 693–708, six limitations: rule-era window,
  no drive-level opponent adjustment, correlation-vs-causation, logistic EPV, era-subsample
  size, QB confound) with paper2's GDS-specific limitations (§6.5, lines 531–550: no drive-level
  opponent adjustment [already covered, dedupe], drive-start-vs-mid-drive mismatch, no
  personnel effects, special-teams construction) — combine into one limitations list, don't
  repeat the opponent-adjustment point twice.

### 8. Conclusion (~450w)
Source: paper2 §7 (lines 555–560) + paper3 §6 (lines 713–724).
One synthesis: metric contribution (86.1% game-winner accuracy, 73.6%/(recompute) variance
explained) + hypothesis-test contribution (all 3 hypotheses rejected, convergent across 5
methods) + the 2025 extension as an out-of-sample confirmation + future work (positional
decomposition, player-level attribution, salary-cap efficiency — both papers list overlapping
future-work items, merge into one list, don't repeat).

---

## Online Supplement (new — required to hit the word budget without losing rigor)

Move here, referenced but not reproduced in body:
- Luck Analysis (paper2 §5.4)
- Point-differential baseline comparison (paper2 §5.2.1)
- Temporal Stability / per-season correlation table (paper2 §5.5)
- Split-Half Reliability (paper2 §5.6)
- Full per-check robustness prose: opponent-strength, field-position mediation detail,
  time-window table, threshold-sensitivity table, single-team exclusion, era comparison
  (paper3 §4.4)

---

## Word Budget

| Section | Target |
|---|---|
| Abstract | 250 |
| 1. Introduction | 1,500 |
| 2. Related Work | 1,700 |
| 3. Data, xScore Recap & GDS Framework | 2,600 |
| 4. Archetype Classification & Methods | 1,300 |
| 5. Results — Validation & Descriptive | 1,600 |
| 6. Results — Hypothesis Tests & Robustness | 2,700 |
| 7. Discussion | 2,000 |
| 8. Conclusion | 450 |
| **Total (first-draft target)** | **~14,100** |

First-draft target is above the 11,500–12,000 final target. Apply cuts in this priority order
during the polish pass (stop as soon as target is hit):
1. Trim structural-explanations (§7) from 3 mechanisms at length to 2 tight ones (~150w)
2. Trim Related Work's DWC-literature/postseason-randomness framing further (~250w)
3. Cut why-the-myth-persists to a single paragraph (~100w)
4. Trim Introduction's remaining scene-setting (~150w)
5. Tighten §3's GDS-framework prose (equations/tables stay, surrounding prose gets leaner)
   (~250–400w)
6. If still over: cut the worked example (§3) entirely — nice-to-have, not load-bearing (~350w)

---

## Tables (carried over, updated to N=256)

1. Descriptive statistics for GDS components (paper2 Table `descriptives`)
2. Correlation between GDS components and win% (paper2 Table `components`) — reported once,
   serves §5 and is cross-referenced from §6
3. Top-5/bottom-5 teams by GDS/game, most recent complete season (paper2 Table `top-bottom`)
4. Independent variables (paper3 Table `ivs`)
5. Dependent variables (paper3 Table `dvs`)
6. Offense-share quartile outcomes (paper3 Table `quartile`)
7. Super Bowl participants with xVOA components (paper3 Table `sb-participants`) — extend to
   include 2025/SEA
8. Logistic regression output (paper3 Table `logistic`)
9. Quadrant analysis of elite teams (paper3 Table `quadrant`)
10. **New:** Consolidated operationalization table (see §6 above — the direct IJSF fix)
11. Hypothesis testing summary (paper3 Table `hypothesis-summary`) — revised per the
    logistic/EPV fix above

## Figures (carried over from thesis/figures, same as both source papers used)

1. Pipeline flowchart (fig9_pipeline_flowchart.pdf)
2. GDS/game vs. win% scatter (fig6_gds_vs_winpct.pdf)
3. Off/Def xVOA vs. win% dual panel (fig3_xvoa_vs_winpct.pdf)
4. Quartile bar chart (fig4_quartile_bars.pdf)
5. Offense share vs. playoff wins scatter (fig7_offense_share_pw.pdf)
6. Quadrant diagram (fig10_quadrant_diagram.pdf) — include if space permits after word-budget
   cuts land; cut first among figures if still tight.

---

## Second-pass review (completed)

Read `paper.tex` end-to-end after drafting, per your request to check the finished paper for
inconsistencies. Found and fixed:
- A paragraph got glued onto `\end{figure}` with no blank line during an earlier edit (§3.5) —
  cosmetic but would have rendered as a run-on.
- A real numerical error in the abstract: "eight of the 256 team-seasons... were entirely
  unseen" should have been 32 (one full season = 32 teams, not eight — "eight" was a slip from
  thinking about the eight-season/eight-championship count nearby).
- New England's 2025 entry in the SB-participants table was described as merely "a
  conference-level participant" when the data (playoff_wins=3, non-bye pattern) means they
  actually reached and lost Super Bowl LX — same signature as the 2021 Bengals row, which the
  original paper described accurately. Fixed to match.
- Minor: the consolidated-operationalizations table's within-elite row showed N=64 (the whole
  elite pool) where the other primary-spec row shows both group sizes; changed to "25 vs. 25
  (of 64 elite)" for consistency.

Re-ran the brace-balance / citation-key / label / figure-existence / table-column-count checks
after these fixes — all clean.

## Cross-check before final draft (Task #8 — do not attempt against this outline)

- Every number that appears in more than one section (the 3.3:1 ratio especially) must resolve
  to one source computation, not be independently retyped in two places.
- N and all dependent stats reflect 256 team-seasons (2018–2025), not 224.
- `references.bib` contains a real, resolvable entry for `mecha2026xscore` — this is the entry
  that was silently missing from both prior venue submissions.
