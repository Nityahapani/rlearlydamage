# Fifth-round response — Phase 13

This is an active results draft. Pending cohorts are not treated as successful experiments.
The design is `prereg/phase13.json`; experimental choices, amendments before their target data,
fixed coefficients and actual completion status are logged separately.

## 1. Gap attribution versus harm beyond lost time

The abstract, contributions, results headings and conclusion now use two explicit tiers.
Tier I concerns the same-update gap, which includes the clean learner's additional progress.
Tier II requires a pause/onset intervention together with detected harm against the historical
pause benchmark. A negative recovery AUC is not automatically a late or irreversible deficit.
The main movement result +1.9 [−0.7,+4.4] is not called a persistent-damage demonstration, and its
90% actor share remains Tier I.

**New independent data:** CartPole seeds 600–607 completed all planned fork/pause and context
cells. Native pause-relative AUC cost is −155.0 [−220.1,−90.0], but the late-window difference
−22.5 [−78.9,+33.9] is inconclusive. Actor share changes from 96% of the clean-context same-update
gap to 75% of its state-pause gap; the corresponding pause critic share is 32%. The pause actor
main effect is −117.4 [−195.5,−39.3]. This is not a reason to generalize the 90% headline to all
forms of damage. The stronger compound primary claim includes paired dominance against the
critic/Adam and has unadjusted p=.0310; the final fixed-five-claim Holm value is reported separately.

Asterix pause-reference results: Independent Asterix seeds 800–807 now provide direct pause-reference evidence: natural AUC −1.57 [−1.84,−1.30]; state-pause gap −1.44 [−1.84,−1.03]; actor main effect −1.47 [−1.70,−1.24], point share 98%. The compound claim passes the fixed five-claim family (Holm <.001 even with incomplete studies set to1). The nominal late-window interval is −0.88 [−1.56,−0.19], a finite endpoint rather than irreversibility. Standardized-minus-natural D is +0.016 [−0.304,+0.336]; its secondary 5%-scale TOST passes marginally (p=.0437). State-pause versus historical-pause reference equivalence is not established at that scale, so the two controls are never called identical.

## 2. Procedural causation and context/optimizer dependence

New runner identity checks establish that the D-context all-damaged cell is literally the natural
D continuation, not merely that its parameters match. Its full score curve is reported alongside
the standardized-context DDD and historical/state-level pause baselines. CartPole includes full
factorials in both clean and damaged future contexts, reset-world actor corners, and a common
fresh-Adam 2x2. All checks are retained regardless of result.

CartPole's standardized-minus-natural D AUC difference is −12.5 [−44.2,+19.1]; the prespecified
5%-of-clean-score context-equivalence check does not pass (p=.211). Therefore the manuscript does
not assert that context is neutral. The actor effect remains negative in the checked natural and
reset contexts and with common fresh moments at nominal intervals; those are procedure-robustness
checks, not a unique decomposition of natural information storage. The common-fresh-Adam actor
effect is −126.9 [−236.8,−17.0].

The methods explicitly distinguish uniformly averaged main effects from Shapley's different
background weights, and explain why parameter-dependent moments can create transient mismatch.
A rescue restores a policy, not a measured microscopic encoding or general learning capacity.

## 3. Independent replications and conditional selection

The new task cohort uses genuinely unused IDs 600–607, not only a new contrast on IDs 0–7.
The Asterix cohort uses 800–807. An independent actuator-noise cohort uses 700–707.
P13_C_RESPONSE

Selected-positive-condition replication is still not a random sample of tasks. The manuscript
retains all unresolved, null and incomplete settings, and never interprets the eligible-setting
count as a probability that actor dominance will hold in a new task. The legacy Freeway n=4
remains descriptive. No old and new cohorts are pooled to rescue a failed primary decision.

## 4. Estimands, families and uneven strength

The headline map lists estimand, primary contrast, sample, multiplicity family and scoped status.
Legacy families remain local; they are not retrospectively assigned a paper-wide error rate.
Phase 13 prespecifies one fixed family of five NEW primary claims. A–C use maximum one-sided
p-values for the required intersection of harm, actor effect and paired Shapley dominance;
Holm is then applied over A,B,C,E,D. Unrun/incomplete studies receive p=1. Context intervals and
other secondary comparisons are labelled as such, not absorbed into a uniform “confirmed” grade.

The ±1 replay-identity bound remains marginal and margin-dependent; ±.75 and alpha=.01 sensitivity
remain visible. Long-restriction tests stay bounded/inconclusive, including the interim-look and
combined outlier/looks sensitivity. No irreversible no-return claim is made.

## 5. Controlled replay exposure rather than time-course inference alone

P13_E_RESPONSE

The new assay changes only reward labels on the same genuine transition tuples, while matching
ages, state/action identities, collection policy and replay/update schedules. A frozen clean
behavior policy prevents exposure from changing the collected trajectories. At K=25, the old
entries are fully overwritten and all non-learner state is asserted equal. Matched parameter/moment
continuations are primary; crossed and common-fresh-moment controls are secondary. This tests a
specified pathway within a controlled assay, not unique mediation of the earlier movement data.
The old overwrite time-course account remains explicitly an interpretation, and identity within
one practical point is never described as zero identity at every relevant scale.

## 6. Frozen prospective diagnostic comparison

All coefficients and scales for six simple models were frozen on the old 32 before any held-out
seed 900–915. Models include mean, acute gap, removal entropy, score-only, KL and acute+KL. The
primary metric is paired held-out squared-error improvement of acute+KL over acute gap; practical
utility requires a 10% point gain and the phase-wise Holm criterion. No validation refitting or
best-model selection is permitted. Feature rank and prediction rank are reported distinctly.
P13_D_RESPONSE

Even a successful cost-prediction result would not validate a probability of permanent failure,
causality of KL, or transfer to a new task. The old rho=−.48 result, small within-cohort gain and
combined p=.062 sensitivity remain visible rather than reinterpreted by the new data.
