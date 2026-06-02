# Paper 1: xScore — A Calibrated Multinomial Model for Drive-Level Expected Points in the NFL

## Target: ~9,800 words | Venue: JQAS

---

## Abstract (~200 words)

We introduce xScore, an open-source multinomial XGBoost model that predicts four mutually exclusive drive outcomes (touchdown, field goal, turnover, punt/other) from seven situational features in professional American football. Unlike play-level Expected Points Added (EPA), which aggregates noisily and conflates heterogeneous outcomes, xScore operates at the natural unit of offensive possession—the drive—and produces a full probability distribution that converts to a single expected-points figure. Trained on 241,195 plays from seven NFL seasons (2018–2024) and evaluated on a held-out 2025 test set (34,415 plays), the model achieves a multiclass Brier score of 0.156 with per-class AUC-ROC ranging from 0.68 (turnover) to 0.81 (punt/other). Per-class isotonic calibration ensures predicted probabilities closely match observed frequencies across the full probability range. Temporal generalization to pre-rule-change seasons (2014–2017) yields an equivalent Brier score of 0.150, confirming stability across regulatory eras. SHAP analysis validates that the model's learned feature hierarchy—field position dominant, followed by down and distance—matches established domain knowledge. The model, code, and data are fully open-source, providing the NFL analytics community with a transparent, reproducible drive-level probability baseline for downstream team quality measurement.

---

## Section Structure

| # | Section | Words | Source |
|---|---------|-------|--------|
| 1 | Introduction | 1,200 | Ch1 RQ1 + research gap (EPA limitations) |
| 2 | Related Work | 1,500 | sec-ml-sports + sec-nfl-metrics (condensed) |
| 3 | Data | 800 | sec-data (condensed) |
| 4 | Model | 2,500 | sec-xscore-model (core) |
| 5 | Results | 2,200 | sec-model-performance (core) |
| 6 | Discussion & Limitations | 1,200 | Ch5 limitations (xScore-specific) + EPA comparison |
| 7 | Conclusion | 400 | Future work pointer |
| — | **Total** | **~9,800** | |

---

## 1. Introduction (1,200 words)

### Keep from thesis:
- EPA limitations argument (ch1 lines 21, 30-31) — this is the gap
- Three-property requirement: calibration, drive-level, open-source
- Brief motivation: salary cap → need reproducible team quality metrics → need baseline model

### Cut:
- Full "defense wins championships" narrative (Paper 3's intro)
- Detailed RQ2/RQ3 framing
- All thesis structure content

### Condense:
- Problem statement → 2 paragraphs max (cap constraint + need for metric)
- Research gap → focus only on EPA/DVOA/PFF limitations relevant to probability modeling
- Contribution statement → 1 paragraph: "We introduce xScore..."

### End with:
"In a companion paper [cite Paper 2], we demonstrate that xScore serves as the foundation for a team quality decomposition framework."

---

## 2. Related Work (1,500 words)

### From sec-ml-sports.tex:
- xG in soccer (closest antecedent) — ~400 words
- Calibration literature (Niculescu-Mizil) — ~200 words
- Gradient boosting + XGBoost background — ~200 words

### From sec-nfl-metrics.tex:
- EPA framework + its 3 limitations (noise, attribution, game-script) — ~500 words
- Win probability models (Lock & Nettleton) — ~100 words
- DVOA/PFF reproducibility gap — ~100 words (brief, saves detail for Paper 2)

### Cut:
- All playoff-factors and resource-allocation literature (Paper 3)
- Extended ML history (Friedman deep dive → 1 sentence)
- Cervone basketball paper
- Full DVOA/PFF critique (Paper 2)

---

## 3. Data (800 words)

### Keep from sec-data.tex:
- nflfastR source, 2018–2024 scope, 3 reasons for 2018 boundary (~150 words condensed)
- Play filtering criteria — table stays, explanation shrinks (~200 words)
- Target variable definition (4-class drive outcome) — equation + brief rationale (~200 words)
- Feature table — keep as-is (~150 words)
- Train/test split rationale (~100 words)

### Cut:
- Rolling EPA shrinkage formula (move to appendix)
- Extended play-filtering justification paragraphs
- Full temporal-split argument (1 sentence: "Random splits create leakage via rolling features")

---

## 4. Model (2,500 words)

### Keep from sec-xscore-model.tex:
- Model objective + xScore equation + xEP formula — ESSENTIAL (~600 words)
- Multinomial vs binary justification — ~200 words (condense from ~350)
- Calibration as primary objective — ~150 words
- Gradient boosted trees architecture — ~200 words
- Why trees (4 paragraphs → 2 paragraphs) — ~250 words
- Hyperparameters — table stays, cut rationale to 1-line each (~200 words)
- Isotonic calibration — full 3-step protocol ESSENTIAL (~500 words)
- Evaluation metrics (Brier, AUC, calibration curves) — ~300 words

### Cut:
- Phase 1 vs Phase 2 conceptual argument (→ 2 sentences)
- Intuition validation specification (move to Results)
- SHAP methodology specification (move to Results)
- Extended turnover penalty discussion

---

## 5. Results (2,200 words)

### Keep from sec-model-performance.tex:
- Dataset summary — 3 bullet points (~100 words)
- Phase comparison table + brief interpretation (~300 words)
- OOD generalization table + 1-paragraph interpretation (~250 words)
- Calibration results + figure reference (~300 words)
- Intuition validation table + brief assessment (~400 words)
- SHAP results — 4 findings (~500 words)
- Summary paragraph (~150 words)

### Cut:
- Extended goal-to-go discussion → 2 sentences
- Extended own-30 discussion → 1 sentence
- Extended 4th-and-22 discussion → 1 sentence
- Repeated GDS forward-references

### Figures: Fig 1 (calibration curves), Fig 2 (field heatmap), Fig 5 (SHAP beeswarm)

---

## 6. Discussion & Limitations (1,200 words)

### Write new:
- Why drive-level ≠ aggregated play-level EPA (~300 words)
- Calibration enables downstream use: cite Paper 2 (~200 words)
- Temporal stability across rule eras (~200 words)

### Keep from sec-limitations.tex (xScore-specific only):
- Data limitations: no tracking data, no personnel groupings (~200 words)
- Special teams as residual (~100 words, pointer to Paper 2)
- Rules environment caveat (~200 words)

### Cut:
- Correlation/causation (Paper 3)
- Small championship sample (Paper 3)
- Schedule strength (Paper 2)

---

## 7. Conclusion (400 words)

- Summarize: multinomial drive-level model, calibrated, open-source, temporally stable
- Key contribution: first public drive-level expected-points model for NFL
- Pointer to Paper 2: "The xScore baseline enables a three-component team quality decomposition..."
- Pointer to extensions: player-level attribution, positional decomposition

---

## Figures & Tables

| # | Type | Content | Source |
|---|------|---------|--------|
| T1 | Table | Feature set | sec-data Tab 2 |
| T2 | Table | Hyperparameters | sec-xscore-model Tab 3 |
| T3 | Table | Phase comparison | sec-model-performance Tab 4 |
| T4 | Table | Per-class AUC-ROC | sec-model-performance Tab 5 |
| T5 | Table | OOD results | sec-model-performance Tab 6 |
| T6 | Table | Intuition validation | sec-model-performance Tab 7 |
| F1 | Figure | Calibration curves (4 panels) | fig1_calibration.pdf |
| F2 | Figure | TD probability heatmap | fig2_field_heatmap.pdf |
| F3 | Figure | SHAP beeswarm | fig5_shap_beeswarm.pdf |

---

## Appendix Material
- Full grid search results
- Rolling EPA shrinkage derivation
- Pre- vs post-calibration ECE values
- Per-class Brier decomposition

---

## Source Files (thesis originals — DO NOT MODIFY)

| File | Usage |
|------|-------|
| thesis/chapters/ch1-introduction.tex | Intro (RQ1 framing, research gap) |
| thesis/chapters/ch2/sec-ml-sports.tex | Related Work |
| thesis/chapters/ch2/sec-nfl-metrics.tex | Related Work |
| thesis/chapters/ch3/sec-data.tex | Data section |
| thesis/chapters/ch3/sec-xscore-model.tex | Model section (CORE) |
| thesis/chapters/ch4/sec-model-performance.tex | Results section (CORE) |
| thesis/chapters/ch5/sec-limitations.tex | Limitations (partial) |

---

## Cross-References to Other Papers

- **Cites Paper 2:** "In a companion paper, we apply xScore to construct the Game Deserved Score framework [Paper 2 ref]"
- **Cites Paper 3:** Not directly; Paper 3 is downstream of Paper 2
- **Cites arXiv thesis:** "Full methodological details and extended results are available in [arXiv ref]"
- **Paper 2 depends on this:** Paper 2's method section says "Using the xScore model [cite Paper 1]..."
- **Paper 3 depends on this:** Indirectly through Paper 2

---

## Key Standalone Argument

The paper must convince reviewers that drive-level multinomial modeling is a genuine contribution, not just "EPA aggregated to drives." The three differentiators:

1. **Categorical outcomes preserve information** — EPA conflates FGs, TOs, and punts into a continuous value; xScore maintains the full distribution
2. **Drive-level is less noisy** — one 80-yard TD doesn't dominate like it does in play-level EPA sums
3. **Calibration is first-class** — EPA models optimize discrimination; xScore optimizes calibration because downstream use (GDS) treats probabilities as genuine expectations
