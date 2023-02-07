# Reinforcement Learning with ManiSkill2

This contains single-file implementations that solve with LiftCube environment with rgbd or state observations. All scripts use mostly the same arguments, see the following sections on how to run the scripts for each RL library we have examples for

Currently provided examples:

- [Stable Baselines 3](#stable-baselines-3)
- [RLlib](#rllib)

## Stable Baselines 3

There are two scripts, sb3_ppo_liftcube_rgbd.py which trains using RGBD observations and sb3_ppo_liftcube_state.py which trains using state observations.

```
# Training
python sb3_ppo_liftcube_rgbd.py --log-dir=logs

# Evaluation
python sb3_ppo_liftcube_rgbd.py --eval --model-path=path/to/model
````

Pass in `--help` for more options (e.g. logging, number of parallel environmnets, whether to use ManiSkill2 Vectorized Environments or not etc.)

For Stable Baselines 3, you need to install `stable_baselines3` as well as `tensorboard`.

## RLlib

There are two scripts, rllib_ppo_liftcube_rgbd.py which trains using RGBD observations and rllib_ppo_liftcube_state.py which trains using state observations.


```
# Training
python rllib_ppo_liftcube_state.py --log-dir=logs --exp-name=test

# Evaluation
python rllib_ppo_liftcube_state.py --eval --model-path=path/to/model
```

Pass in `--help` for more options (e.g. logging, number of parallel environmnets, whether to use ManiSkill2 Vectorized Environments or not etc.)

By default the code uses a PyTorch backend, you can easily edit the script to try a different backend.