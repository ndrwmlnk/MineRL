# Simple env test.
import json
import select
import time
import logging
from logging import getLogger
import os

import gym
import minerl

import numpy as np

import chainer

import chainerrl
from chainerrl.wrappers import ContinuingTimeLimit
from chainerrl.wrappers.atari_wrappers import ScaledFloatFrame

import sys
sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir)))
from utility import utils
from utility.q_functions import NatureDQNHead, A3CFF
from utility.env_wrappers import (
    SerialDiscreteActionWrapper, CombineActionWrapper, SerialDiscreteCombineActionWrapper,
    ContinuingTimeLimitMonitor,
    MoveAxisWrapper, FrameSkip, FrameStack, ObtainPoVWrapper, FullObservationSpaceWrapper)

from utility.env_wrappers import ScaledFloatFrame as SFF

# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLTreechop-v0')
MINERL_MAX_EVALUATION_EPISODES = int(os.getenv('MINERL_MAX_EVALUATION_EPISODES', 5))

def wrap_env(env, test):
        # wrap env: time limit...
        if isinstance(env, gym.wrappers.TimeLimit):
            env = env.env
            max_episode_steps = env.spec.max_episode_steps
            env = ContinuingTimeLimit(env, max_episode_steps=max_episode_steps)
        # wrap env: observation...
        # NOTE: wrapping order matters!
        env = FrameSkip(env, skip=4)
        env = ObtainPoVWrapper(env)
        # convert hwc -> chw as Chainer requires.
        env = MoveAxisWrapper(env, source=-1, destination=0)
        # env = ScaledFloatFrame(env)
        env = SFF(env)
        env = FrameStack(env, 4, channel_order='chw')
        # wrap env: action...
        env = SerialDiscreteActionWrapper(
            env,
            always_keys=['attack'], reverse_keys=['forward'], exclude_keys=['back', 'left', 'right', 'sneak', 'sprint'], exclude_noop=False)

        return env

def scaleFrames(frames):
    for frame in frames:
        frame = np.array(frame).astype(np.float32) / 255.0

    return frames

def main():
    """
    This function will be called for training phase.
    """
    # Sample code for illustration, add your code below to run in test phase.
    # Load trained model from train/ directory
    core_env = gym.make(MINERL_GYM_ENV)
    wrapped_env = wrap_env(core_env, test=False)
    # for i in range(wrapped_env.action_space.n):
    #     print("Action {0}: {1}".format(i, wrapped_env.action(i)))

    head = NatureDQNHead(n_input_channels=12, n_output_channels=512)
    model = A3CFF(n_actions=5, head=head)

    opt = chainer.optimizers.Adam(alpha=2.5e-4, eps=1e-8)
    opt.setup(model)
    opt.add_hook(chainer.optimizer.GradientClipping(40))

    def phi(x):
        # observation -> NN input
        return np.asarray(x)

    CLIP_EPS = 0.1
    agent = chainerrl.agents.ppo.PPO(
        model, opt, gpu=-1, gamma=0.99, phi=phi, update_interval=1024,
        minibatch_size=32, epochs=3, clip_eps=CLIP_EPS, standardize_advantages=False)
    agent.load("models/ppo")

    for _ in range(MINERL_MAX_EVALUATION_EPISODES):
        obs = wrapped_env.reset()
        done = False
        netr = 0
        items_crafted = False
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = wrapped_env.step(action)
                
            netr += reward
            print("Net reward: ", netr)


    wrapped_env.close()

if __name__ == "__main__":
    main()
