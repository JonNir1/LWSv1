# Results Comparison: April 2026 vs. August 2026

This compares the analysis conclusions from before and after the preprocessing pipeline refactor: the **April
2026** state (`main` @ `b7459cf`, pre-refactor, 12 subjects, `Results - Backup`) against the **August 2026**
state (`dev` @ `8475c45`, post-refactor, 27 subjects, `Results`). The goal is to determine which conclusions from
the April analyses — chiefly the VSS 2026 poster (`_publications_/2026_vss.ipynb`) — still hold, and for those
that moved, whether the code change or the added subjects caused it. The accompanying
[RESULTS_COMPARISON_2026-04_vs_2026-08.csv](RESULTS_COMPARISON_2026-04_vs_2026-08.csv) is the full claim-by-claim
ledger; this file is the summary. Old values are read from `b7459cf`'s saved notebook outputs only (no old code
was re-run); new values are computed live from `dev` against `Results`, on the 12 subjects common to both eras
and on the full 27.

Old: `main` @ `b7459cf` (2026-04-22), 12 subjects, `Results - Backup`, saved notebook outputs only.
New: `dev` @ `8475c45` (2026-08-19), 27 subjects, `Results`.
Pipeline arm = old vs. new-code-on-12-subjects. Sample arm = new-12 vs. new-27.

Detection is unchanged (59,047 dominant-eye fixations, 1,639 identifications, both eras) — every difference below
is stage 2/3 or sample.

## Verdicts

| Claim | Old | New (12) | New (27) | Verdict |
| --- | --- | --- | --- | --- |
| P0 overall LWS rate | 26.24% | 27.57% | 27.60% | STABLE |
| P1 animate > inanimate | +11.0pp, P=1.0000 | +10.4pp | +7.7pp | STABLE |
| P2 human face < other animate | -8.3pp, P=0.0005 | -8.3pp | -5.6pp | STABLE |
| P3 handmade vs natural | +1.1pp, null | +1.2pp | +0.3pp | STABLE |
| P4 grayscale > color | +7.4pp, P=0.0009 | +10.4pp | +9.7pp | **TRIGGERED** |
| P5 noise vs color | +1.1pp, null | +1.0pp | +2.4pp | STABLE |
| P6 grayscale > noise | +6.3pp, P=0.9964 | +9.4pp | +7.3pp | **TRIGGERED** |
| P7 angle 0 vs 20 | +1.8pp, null | +3.9pp | -2.4pp | **TRIGGERED** (sign flip) |
| P8 angle 0 vs 20, grayscale | +1.2pp, null | -2.2pp | -5.0pp | **TRIGGERED** (sign flip) |
| P9 funnel | 2341→614 (26.2%) | 2390→660 (27.6%) | 5337→1424 (26.7%) | STABLE |
| SF1-3 (freq., mirrors P1-3) | — | — | — | STABLE |
| SF4a/c (freq., mirrors P4/P6) | p=0.0039 / p=0.0094 | — | — | **TRIGGERED** |
| SF5 target-category omnibus | p<.001 | — | — | STABLE |
| SF6 target-angle shape | 0°/10°/20° monotonic ↓ | not monotonic | not monotonic | **TRIGGERED** |
| SF7 3-way interaction | p=0.03175 (sole hit among 4) | — | — | **TRIGGERED** |
| H1.5b human face, noise only | P=0.1105 | -6.3pp | -2.4pp | BORDERLINE |
| HR1 miss rate | 24.44% | 24.77% | 24.73% | STABLE |
| HR2 trial-cat. (hit rate) | p<.001, **did not converge** | — | — | CAUTION |
| HR3 target-cat. (hit rate) | p<.001 | order preserved | order preserved | STABLE |
| TE1 inclusion rate | 85.0% | 84.7% | 81.3% | SHIFTED (sample arm) |
| TE2a BW vs Color (inclusion) | p=0.0187 | — | — | **TRIGGERED** |
| TE3 per-criterion breakdown | — | — | — | NOT_COMPARABLE |
| TOT1 fatigue (GAM shape) | weak rise 25.8→27.1% | not computed | not computed | **TRIGGERED** |
| TIT1 within-trial (GAM shape) | flat then drop near trial end | not computed | not computed | **TRIGGERED** |
| SP1 spatial (GAM) | no valid old p-value | — | — | NOT_COMPARABLE |

## What changed and why

- **P4/P6/SF4a/SF4c — grayscale effect strengthens.** Grayscale-vs-color and grayscale-vs-noise contrasts move
  +3.0-3.1pp on the pipeline arm alone (code change, same 12 subjects), same direction, already significant
  before and after descriptively. Likely the `TOBII_MONITOR` / `px2deg` fix (`1e86a5e`) and/or visits now
  respecting outlier exclusion (`CODE_REVIEW.md` H2) interacting with BW-trial visit geometry specifically.
  Needs a refit to confirm, but directionally this looks like it gets stronger, not weaker.
- **P7/P8 — target-angle contrasts flip sign.** Both were already null (p=0.71, p=0.58) in April; the sign flip
  is consistent with noise around zero rather than a real reversal, but the rule fires regardless. SF6's
  descriptive means confirm this: the old monotonic 0°>10°>20° pattern does not hold in either new view.
- **SF7 — the one marginal finding (p=0.03175) sits inside the near-boundary trigger band**, and is the sole
  significant interaction among four tested. Highest-risk claim in the poster's supporting analysis.
- **HR2 — the old frequentist hit-rate model never converged.** Its own saved output says `Convergence status:
  FALSE`. Any comparison against it is comparing against a shaky baseline, independent of what changed since.
- **TE1 — trial inclusion drops on the sample arm only** (84.7% → 81.3%), not the pipeline arm (85.0% → 84.7%).
  The added subjects have a lower baseline inclusion rate; not a pipeline artifact.
- **TE2a — BW vs Color inclusion odds ratio's old p (0.0187) sits inside the near-boundary band**, flagged for
  refit regardless of how far it moved.
- **TE3 dropped.** The old notebook's per-criterion table reads *standalone* criterion flags
  (`check_trial_inclusion_criteria()` output directly); the new pipeline's `trial_funnel` only exposes
  *cumulative* `upto_` columns. These are different constructs — comparing them (as an earlier draft of the
  extraction script did) would silently compare non-equivalent numbers. No fix attempted; out of scope without
  calling `pipeline/stage3_classify/trial_inclusion.py` predicates directly.
- **TOT1/TIT1 — shape claims, no descriptive proxy available**, and unrefit as instructed. TIT1's saved curve
  additionally carries a likely structural confound: the terminal drop toward trial end may be mostly the
  `not_close_to_trial_end` funnel criterion mechanically zeroing out late visits, not a real "window of
  vulnerability" effect — worth re-examining once refit.
- **SP1 has no valid old p-value at all** (pyGAM's own documented bug + an in-notebook R error), so it is out
  of comparison scope entirely, not merely unrefit.

## Refits: not run (decision)

Ten claims triggered a refit rule (P4/P6/P7/P8, SF4a/SF4c/SF6/SF7, TE2a, HR2 family, TOT1/TIT1). Refitting them
means the full VSS Bayesian model plus four R model families, each on two subject views. Skipped: the claims that
matter most — the VSS poster's headline contrasts (P1, P2, animacy and human-face effects, the overall LWS rate) —
are already `STABLE` from the descriptive comparison alone, and that's the result that gets carried forward. The
ten flagged claims stay flagged (see table) rather than resolved; treat them as open if any of them individually
becomes load-bearing for a future analysis (SF7 and HR2 are the two worth remembering — SF7 is the one marginal
finding in the whole set, and HR2's old baseline never converged in the first place).
