# Phase 12 authoritative timeline

UTC commit and manifest timestamps are authoritative. Some manually estimated times in LOG.md are inaccurate; do not use them to infer experiment order.

| commit | committed at (UTC) | event |
|---|---|---|
| ac63459 | 2026-10-14T14:36:55Z | Phase 11: resume unfinished pre-registered Phase 9 runs (E2 s4-7, E3 s316-331); interim-look disclosure |
| 7138fe2 | 2026-10-14T16:39:53Z | Phase 11: paper updated with completed Asterix-flip (n=8) and PONR replication (n=32) |
| adeb6b0 | 2026-10-14T17:25:46Z | Phase 12 pre-registration: generalization panel (3 envs, 3 new deficit families), new deficits action_noise/reward_scale, long controls and onsets |
| 92fd75b | 2026-10-14T17:34:13Z | Phase 12 pre-registration addendum: out-of-sample test of gap-damage hypothesis; queue fix note |
| bf9a946 | 2026-10-14T17:36:33Z | Phase 12 addendum 2: G7 PQN-R reward-flip network x buffer surgery (algorithm axis); pqn reward_flip deficit |
| 79c9c85 | 2026-10-14T20:14:31Z | Phase 12 audit: correct disclosure of CartPole acute-outcome progress checks |
| 3ddffe2 | 2026-10-14T20:24:14Z | Phase 12 G7: register secondary recovery grid matched to PPO in environment steps |
| 8adf203 | 2026-10-14T20:41:41Z | Phase 12 H1 pre-registration: independent fresh-seed CartPole replication, budget gated |

## Design-to-run checks

| test | first manifest timestamp | design commit | design timestamp precedes run? |
|---|---|---|---|
| cartpole-rflip | 2026-10-14T17:25:57.435124+00:00 | adeb6b0 | yes |
| acrobot-rflip | 2026-10-14T17:50:17.896444+00:00 | adeb6b0 | yes |
| si-actnoise | 2026-10-14T18:05:52.347443+00:00 | adeb6b0 | yes |
| si-obsnoise | 2026-10-14T19:14:03.127908+00:00 | adeb6b0 | yes |
| si-extinction | 2026-10-14T20:10:58.186526+00:00 | adeb6b0 | yes |

The gap–damage addendum followed visibility of CartPole acute values. Accordingly G1 is not treated as untouched validation. The main G1–G6 design preceded all their deficit runs.

G7 and H1 were selected after prior Phase 12 information but registered before their own data. This is not a blinded global research programme.
