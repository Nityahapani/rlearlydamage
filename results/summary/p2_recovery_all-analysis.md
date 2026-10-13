# PPO recovery test (Phase 2), corrected paired t-intervals

Regenerated 2026-10-13 from the committed summaries with scripts/analyze_recovery.py (t-intervals, n=4; gap = control - deficit, positive = damage; delay-adj gap = control(u - 50) - deficit(u)).

| config | quantity | update | deficit | control | gap = control - deficit [95% CI] (% of control) | delay-adj gap |
|---|---|---|---|---|---|---|
| move_early | Q1a during mask, unmasked eval | (0,50) | 5.6 | 15.7 | +10.2 [+8.7, +11.6] (+65%) | - |
| move_early | Q1b during mask, as-experienced (masked) | (0,50) | 3.0 | 15.6 | +12.6 [+11.4, +13.8] (+81%) | - |
| move_early | Q2 immediately after removal | 50 | 5.4 | 28.2 | +22.8 [+18.7, +26.8] (+81%) | -1.8 [-2.6, -0.9] |
| move_early | Q3 +50 after removal | 100 | 27.8 | 44.8 | +17.0 [+9.1, +24.8] (+38%) | -0.1 [-7.6, +7.5] |
| move_early | Q3 +100 after removal | 150 | 53.5 | 57.9 | +4.5 [-17.2, +26.1] (+8%) | -8.7 [-24.9, +7.5] |
| move_early | Q4 +250 after removal | 300 | 102.2 | 95.6 | -6.6 [-23.9, +10.7] (-7%) | -16.4 [-30.6, -2.1] |
| move_late | Q1a during mask, unmasked eval | (100,150) | 24.5 | 50.8 | +26.3 [+10.2, +42.4] (+52%) | - |
| move_late | Q1b during mask, as-experienced (masked) | (100,150) | 3.1 | 49.3 | +46.2 [+34.6, +57.7] (+94%) | - |
| move_late | Q2 immediately after removal | 150 | 18.5 | 58.7 | +40.1 [+21.6, +58.7] (+68%) | +26.9 [+8.1, +45.6] |
| move_late | Q3 +50 after removal | 200 | 54.7 | 68.3 | +13.6 [+2.3, +24.9] (+20%) | +3.3 [-8.5, +15.0] |
| move_late | Q3 +100 after removal | 250 | 74.9 | 85.8 | +10.9 [+5.5, +16.4] (+13%) | -6.6 [-19.2, +6.0] |
| move_late | Q4 +250 after removal | 400 | 103.6 | 110.8 | +7.2 [+1.3, +13.2] (+7%) | -2.0 [-4.1, +0.1] |
| fire_early | Q1a during mask, unmasked eval | (0,50) | 4.2 | 15.7 | +11.5 [+10.2, +12.9] (+73%) | - |
| fire_early | Q1b during mask, as-experienced (masked) | (0,50) | 0.0 | 15.6 | +15.6 [+14.4, +16.7] (+100%) | - |
| fire_early | Q2 immediately after removal | 50 | 4.5 | 28.2 | +23.7 [+19.3, +28.1] (+84%) | -0.8 [-1.7, +0.2] |
| fire_early | Q3 +50 after removal | 100 | 26.8 | 44.8 | +18.0 [+9.9, +26.1] (+40%) | +1.0 [-6.7, +8.7] |
| fire_early | Q3 +100 after removal | 150 | 45.0 | 57.9 | +12.9 [-0.1, +25.9] (+22%) | -0.3 [-4.8, +4.2] |
| fire_early | Q4 +250 after removal | 300 | 90.1 | 95.6 | +5.5 [-7.2, +18.1] (+6%) | -4.3 [-28.4, +19.8] |
| fire_late | Q1a during mask, unmasked eval | (100,150) | 22.6 | 50.8 | +28.3 [+11.7, +44.8] (+56%) | - |
| fire_late | Q1b during mask, as-experienced (masked) | (100,150) | 0.9 | 49.3 | +48.4 [+37.1, +59.8] (+98%) | - |
| fire_late | Q2 immediately after removal | 150 | 15.2 | 58.7 | +43.4 [+15.9, +70.9] (+74%) | +30.2 [+19.1, +41.3] |
| fire_late | Q3 +50 after removal | 200 | 50.8 | 68.3 | +17.4 [-0.6, +35.5] (+26%) | +7.1 [-10.0, +24.2] |
| fire_late | Q3 +100 after removal | 250 | 69.7 | 85.8 | +16.1 [+3.6, +28.7] (+19%) | -1.4 [-19.1, +16.2] |
| fire_late | Q4 +250 after removal | 400 | 103.9 | 110.8 | +6.9 [-4.1, +17.9] (+6%) | -2.4 [-8.3, +3.6] |
