"""
Example showing how to wrap the iGibson class using ray for rllib.
Multiple environments are only supported on Linux. If issues arise, please ensure torch/numpy
are installed *without* MKL support.

This example requires ray to be installed with rllib support, and pytorch to be installed:
    `pip install torch "ray[rllib]"`

Note: rllib only supports a single observation modality:
"""
import gym
import argparse
import functools
import logging
import operator
import os
from sys import platform
import mani_skill2.envs

from mani_skill2.utils.wrappers import RecordEpisode

import ray
import yaml
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
# ray.init(local_mode=True)
ray.init(ignore_reinit_error=True)


# Defines a continuous, infinite horizon, task where done is always False
# unless a timelimit is reached.
class ContinuousTaskWrapper(gym.Wrapper):
    def __init__(self, env, max_episode_steps: int) -> None:
        super().__init__(env)
        self._elapsed_steps = 0
        self._max_episode_steps = max_episode_steps

    def reset(self):
        self._elapsed_steps = 0
        return super().reset()

    def step(self, action):
        ob, rew, done, info = super().step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info["TimeLimit.truncated"] = True
        else:
            done = False
            info["TimeLimit.truncated"] = False
        return ob, rew, done, info


# A simple wrapper that adds a is_success key which SB3 tracks
class SuccessInfoWrapper(gym.Wrapper):
    def step(self, action):
        ob, rew, done, info = super().step(action)
        info["is_success"] = info["success"]
        return ob, rew, done, info

def env_creator(env_config):
    env_id = env_config["env_id"]
    obs_mode = env_config["obs_mode"]
    max_episode_steps = env_config["max_episode_steps"]
    import mani_skill2.envs

    env = gym.make(
        env_id,
        obs_mode=obs_mode,
        reward_mode=env_config["reward_mode"],
        control_mode=env_config["control_mode"],
    )
    # For training, we regard the task as a continuous task with infinite horizon.
    # you can use the ContinuousTaskWrapper here for that
    if max_episode_steps is not None:
        env = ContinuousTaskWrapper(env, max_episode_steps)
    if env_config["record_dir"] is not None:
        env = SuccessInfoWrapper(env)
        env = RecordEpisode(
            env, env_config["record_dir"], info_on_video=True, render_mode="cameras"
        )
    return env



def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple script demonstrating how to use Stable Baselines 3 with ManiSkill2 and RGBD Observations"
    )
    parser.add_argument("-e", "--env-id", type=str, default="LiftCube-v0")
    parser.add_argument(
        "-n",
        "--n-envs",
        type=int,
        default=8,
        help="number of parallel envs to run. Note that increasing this does not increase rollout size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed to initialize training with",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=100,
        help="Max steps per episode before truncating them",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=300_000,
        help="Total timesteps for training",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="path for where logs, checkpoints, and videos are saved",
    )
    parser.add_argument(
        "--eval", action="store_true", help="whether to only evaluate policy"
    )
    parser.add_argument(
        "--model-path", type=str, help="path to sb3 model for evaluation"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    register_env("ManiSkill2Env", env_creator)
    obs_mode = "state"
    control_mode = "pd_ee_delta_pose"
    reward_mode = "dense"
    rollout_size = 3200
    num_envs = args.n_envs
    config = {
        "env": "ManiSkill2Env",
        "env_config": {
            "env_id": args.env_id,
            "max_episode_steps": args.max_episode_steps,
            "obs_mode": "state",
            "reward_mode": "dense",
            "control_mode": control_mode,
            "record_dir": None
        },
        "num_gpus": 1,
        # "remote_worker_envs": True,
        "num_workers": num_envs,
        "num_envs_per_worker": 1,
        "num_cpus_per_worker": 1,
        "rollout_fragment_length": 100,
        "train_batch_size": rollout_size,
        "sgd_minibatch_size": 400,
        "num_sgd_iter": 15,
        "kl_target": 0.05,
        "gamma": 0.85,
        "framework": "torch",
    }
    stop = {"training_iteration": args.total_timesteps}
    # if resume is not None:
    #     assert restore_checkpoint is not None, "Error: When resuming must provide explicit path to checkpoint"

    results = tune.run(
        "PPO",
        config=config,
        verbose=2,
        name="test",
        local_dir=args.log_dir,
        checkpoint_freq=100,
        stop=stop,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
