# Phase 12 tests in pre-registered priority order (sourced by queue_p12.sh).
run cartpole-rflip  --env CartPole-v1      --deficit reward_flip  --flip-prob 1.0 --window 10 60  --recovery 150
run acrobot-rflip   --env Acrobot-v1       --deficit reward_flip  --flip-prob 1.0 --window 60 110 --recovery 150
run si-actnoise     --env SpaceInvaders-MinAtar --deficit action_noise --noise-prob 0.5 --window 100 150 --recovery 150
run freeway-rflip   --env Freeway-MinAtar  --deficit reward_flip  --flip-prob 1.0 --window 20 70 --recovery 150
run si-obsnoise     --env SpaceInvaders-MinAtar --deficit obs_noise --sigma 1.0 --window 100 150 --recovery 150
run si-extinction   --env SpaceInvaders-MinAtar --deficit reward_scale --scale 0.0 --window 100 150 --recovery 150
