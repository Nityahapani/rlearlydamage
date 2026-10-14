# Phase 10 onset-reference checks — p10_onsetref_rflip

Seeds [0, 1, 2, 3, 4, 5, 6, 7] (n = 8); window [100, 150); AUC over +10..+150.

- (a) DDD bit-identity vs Phase 8 fork-reference run: max |diff| = 0
- (b) state-level pause PPP AUC 67.4 vs time-shifted pause C(w1+k-L) AUC 66.2; PPP - pause: +1.21 [-0.59, +3.00]
- DDD - PPP (beyond lost time, state-level): -19.78 [-27.81, -11.76]*
- DDD - C(w1+k-L) (delay-adjusted AUC, time-shifted): -18.58 [-27.21, -9.95]*
- PPP at +0: 45.6; C(w0) (should match): 45.6
