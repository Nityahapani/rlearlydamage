# Response to the fourth-round review

Every change is logged in `LOG.md` (Phase 12). The pre-registration commits are adeb6b0, 92fd75b and bf9a946; the deviations are logged with timestamps.

## Main request: broader generalization (new experiments, Phase 12)

We pre-registered seven new tests before any of their data existed. The six PPO tests use the same $2^3$ actor × critic × Adam surgery, 8 seeds each, the same decision criterion, and onsets from the same meaning-matched rule. The rule is now measured from the untrained level so that it also works for tasks with negative returns.

| id | setting | new axis | outcome |
|---|---|---|---|
| G1 | CartPole-v1, reward flip | non-pixel task | **replicates**: actor 95% (8/8); damage beyond lost time −197* |
| G2 | Acrobot-v1, reward flip | non-pixel, negative-reward task | no damage to localize (wide CIs; weak null) |
| G3 | SpaceInvaders, actuator noise | new deficit family (dynamics) | **replicates**: actor 115% (8/8); no damage beyond lost time |
| G4 | Freeway-MinAtar, reward flip | third MinAtar game | incomplete: 4 of 8 seeds; actor share 79%, critic contributes substantially; descriptive only |
| G5 | SpaceInvaders, observation noise | new deficit family (sensory) | **replicates**: actor 107% (8/8); no damage beyond lost time |
| G6 | SpaceInvaders, reward extinction | new deficit family (passive) | **replicates**: actor 104% (8/8); no detected harm beyond lost time |
| G7 | PQN-R, reward flip (wrong rewards stored in replay) | second learner under a second deficit | registered primary: joint network block 92%; buffer mean −0.56 [−1.73,+0.61]. On the shorter matched horizon the buffer effect is −2.14 [−4.00,−0.28] (secondary) |

Pre-registered cross-setting summary: Among the ten completed PPO settings, eight have a detected same-update gap and all eight meet the actor criterion (shares 82–115%). The incomplete Freeway sample is excluded; Acrobot is unresolved.

Out-of-sample test of the earlier post-hoc gap–damage relation: No completed, untouched new setting lies in a prediction band. CartPole is consistent but its acute values had been seen before the addendum; Freeway is consistent only descriptively at n=4. This relation is not treated as validated.

## Concern 1: uneven evidence

- **Claims ledger (new Table 2, introduction).** Every headline claim is listed with its sample size, whether it is an independent new-seed replication or a new contrast on reused IDs, and an explicit strength label (Confirmed / Bounded / Exploratory / Post-hoc).
- **Abstract, contributions and conclusion** now use the label from the ledger for each claim. The timing pattern, for example, is called exploratory everywhere. The critic effect is significant in the pooled n=12, but its fresh-only n=8 interval includes zero; that distinction is also stated.
- **Evidence-status sentence** at the top of every results section.
- **Corrected sentence.** We removed the claim that "the abstract and conclusion rest on confirmatory results"; it was not accurate for the timing pattern.

## Concern 2: "carrier" is operational

- **New definition paragraph** in Section 3: a component carries damage iff transplanting it changes recovery under our surgery procedure. The paragraph states that this does not identify encoding, cause or plasticity loss.
- **Softened wording.** Mechanistic phrasing ("mediated rather than stored", "tests these mechanisms causally") was rewritten. The PQN-R section title no longer says "the buffer mediates".
- **Abstract** gives the operational definition and states that the results do not reveal encoding or plasticity. The DDD surgery cell has damaged learner components but a standardized clean future training context; it is not claimed to be the untouched natural damaged continuation.
- **Context audit:** corrected the claim that DDD is the natural damaged continuation. All surgery cells use the clean environment/RNG context; repeated DDD confirms repeatability of that procedure, not neutrality of the context change. The state-level pause reference is likewise distinguished from the historical delay-shifted control.

## Concern 3: scope

- **New "Scope of the conclusions" paragraph** at the start of the Discussion. It lists the learners, environments, deficits, network sizes and budgets actually tested, and states that "in our settings" refers to exactly this space.
- **Abstract and conclusion** end with the scope statement.
- **Phase 12** widens the tested space itself: CartPole, Acrobot (and Freeway); actuator noise, observation noise, extinction; PQN-R under the reward flip.

## Concern 4: the ±1 AUC margin

- **Motivation from the pre-registration** (a873dc0) now appears in the paper. The margin is the smallest identity effect that would justify calling data a carrier *alongside the network*: about 3% of the arms' AUC, about 30% of the targeted-purge benefit, the size of the rung-3 estimate, with context from both buffer regimes (22% of the 100k network-transfer estimate and 38% of the 1M estimate). This is an author-chosen practical threshold, not an established RL standard or a confidence bound on ratios.
- **What is claimed.** Only that effects of 1 AUC point or more are excluded. The fragility at ±0.75 and the positive point estimate are stated in the abstract, results and limitations.

## Concern 5: interim looks

- **New appendix paragraph and table (`tab:interim`).** It sets out what was fixed before the looks (seeds, code, configuration, estimands, margins, decision rules: 63a8f1d). The completion decision was budget-driven and logged with the reporting rule before any new run (ac63459). The seed-330 diagnosis and exclusion sensitivity were selected after data, labelled post-hoc and kept outside every decision. Completion followed the original seed plan but was decided with knowledge of the interim estimates; we do not describe it as blind.
- **Conservative check.** A two-look Bonferroni sensitivity check preserves the final primary decisions (the KL predictor goes from Holm p = 0.010 to 0.021). It does not retrospectively eliminate all possible adaptive research or reporting choices. Combining it with post-hoc exclusion of seed 330 is weaker (Holm 0.031 × 2 = 0.062); that sensitivity is reported rather than hidden.
- **Phase 12 withheld formal interim primary-AUC analyses**, but CartPole removal values (clean and damaged) were exposed during progress checks. This is disclosed, including a correction in the log (79c9c85). The main panel design was fixed before those values were seen. The later gap–damage addendum therefore reports CartPole separately rather than treating it as an untouched validation.

## Concern 6: point-of-no-return wording

- **Abstract and conclusion** now say "no *average* point of no return" and give the one extreme flat-score collapse among 44 long-restricted agents, with its exact conditional binomial 95% interval (0.06–12%), labelled as a post-hoc observation.
- **Rare failures, longer restrictions and other environments** are explicitly not excluded.

## Completion and archival status

Six new tests completed their planned n=8. Freeway completed all factorial arms for seeds 0–3; seeds 4–5 have partial arms and are preserved but excluded from the factorial aggregate. Optional fresh-seed replications H1 (600–607) and H2 (700–707) never ran after the sandbox reset. No new experiment was launched during PDF finalization.

Completed-setting claims do not count the incomplete Freeway sample as confirmatory, and no independent-seed generalization replication is claimed for the new panel.
