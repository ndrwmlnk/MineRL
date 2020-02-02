# Simple env test.
# noinspection PyUnresolvedReferences
import minerl

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import chainer
import chainerrl
import gym
import numpy as np
from chainer import cuda
from chainerrl.wrappers import ContinuingTimeLimit
from chainerrl.wrappers.atari_wrappers import FrameStack
from chainerrl.wrappers.atari_wrappers import ScaledFloatFrame as SFF

sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir)))

from utility.config import CONFIG, SINGLE_FRAME_AGENT_ATTACK
from utility.env_wrappers import (SerialDiscreteActionWrapper, MoveAxisWrapper, FrameSkip, FrameStack, ObtainPoVWrapper)
from utility.q_functions import DistributionalDuelingDQN


# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
# MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamond-v0')
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLTreechop-v0')
MINERL_MAX_EVALUATION_EPISODES = int(os.getenv('MINERL_MAX_EVALUATION_EPISODES', 5))

TRAINING_EPISODES = 150

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
    env = SFF(env)
    env = FrameStack(env, CONFIG["RAINBOW_HISTORY"], channel_order='chw')
    # wrap env: action...
    env = SerialDiscreteActionWrapper(
        env,
        always_keys=CONFIG["ALWAYS_KEYS"], reverse_keys=CONFIG["REVERSE_KEYS"], exclude_keys=CONFIG["EXCLUDE_KEYS"], exclude_noop=False)

    return env


def parse_agent(agent):
    return {'DQN': chainerrl.agents.DQN,
            'DoubleDQN': chainerrl.agents.DoubleDQN,
            'PAL': chainerrl.agents.PAL,
            'CategoricalDoubleDQN': chainerrl.agents.CategoricalDoubleDQN}[agent]


def get_agent(
        n_actions, arch, n_input_channels,
        noisy_net_sigma, start_epsilon, final_epsilon, final_exploration_frames, explorer_sample_func,
        lr, adam_eps,
        prioritized, steps, update_interval, replay_capacity, num_step_return,
        agent_type, gpu, gamma, replay_start_size, target_update_interval, clip_delta, batch_accumulator
):
    # Q function
    n_atoms = 51
    v_min = -10
    v_max = 10
    q_func = DistributionalDuelingDQN(n_actions, n_atoms, v_min, v_max, n_input_channels=n_input_channels)

    # explorer
    if noisy_net_sigma is not None:
        chainerrl.links.to_factorized_noisy(
            q_func, sigma_scale=noisy_net_sigma)
        # Turn off explorer
        explorer = chainerrl.explorers.Greedy()
    else:
        explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon, final_epsilon, final_exploration_frames, explorer_sample_func)

    opt = chainer.optimizers.Adam(alpha=lr, eps=adam_eps)

    opt.setup(q_func)

    # Select a replay buffer to use
    if prioritized:
        # Anneal beta from beta0 to 1 throughout training
        betasteps = steps / update_interval
        rbuf = chainerrl.replay_buffer.PrioritizedReplayBuffer(replay_capacity, alpha=0.5, beta0=0.4, betasteps=betasteps, num_steps=num_step_return)
    else:
        rbuf = chainerrl.replay_buffer.ReplayBuffer(replay_capacity, num_step_return)

    # build agent
    def phi(x):
        # observation -> NN input
        return np.asarray(x)

    Agent = parse_agent(agent_type)
    agent = Agent(
        q_func, opt, rbuf, gpu=gpu, gamma=gamma, explorer=explorer, replay_start_size=replay_start_size,
        target_update_interval=target_update_interval, clip_delta=clip_delta, update_interval=update_interval,
        batch_accumulator=batch_accumulator, phi=phi)

    return agent


def train(agent, wrapped_env):
    """
    This function will be called for training phase.
    """
    out_dir = Path(os.environ["HOME"], f"rainbow_{CONFIG['RAINBOW_HISTORY']}")
    if not out_dir.exists():
        out_dir.mkdir()

    i = 0
    rewards = np.array([])
    ep = 0
    obs = wrapped_env.reset()
    done = False
    info = {}
    netr = 0
    reward = 0

    maximum_frames = 8000000
    steps = maximum_frames // 4

    # record number of steps for each training episode and the cumulative number of step (total)
    try:
        while i < steps:
            if done or "error" in info:
                agent.stop_episode_and_train(obs, reward, done)
                rewards = np.array([*rewards, (i, netr)])
                ep += 1
                print(f"===================== END {i} - REWARD {netr} =====================")
                np.savetxt(Path(out_dir, "rewards.txt"), rewards)
                agent.save(out_dir)

                obs = wrapped_env.reset()
                done = False
                netr = 0
                reward = 0
                print(f"===================== EPISODE {i} =====================")

            action = agent.act_and_train(obs, reward)
            obs, reward, done, info = wrapped_env.step(action)
            netr += reward

            if agent.t % 1000 == 0:
                rewards = np.array([*rewards, (i, netr)])
                print("Net reward: ", netr)
            i += 1

    except KeyboardInterrupt:
        agent.stop_episode_and_train(obs, reward, done)
        print("Saving agent")

    np.savetxt(Path(out_dir, "rewards.txt"), rewards)
    agent.save(out_dir)

    print("Done!")


def main(args):
    """
    This function will be called for training phase.
    """
    chainerrl.misc.set_random_seed(0)
    CONFIG.apply(SINGLE_FRAME_AGENT_ATTACK)

    core_env = gym.make(MINERL_GYM_ENV)
    wrapped_env = wrap_env(core_env, test=False)

    maximum_frames = 8000000
    steps = maximum_frames // CONFIG["HISTORY"]
    agent = get_agent(
        n_actions=wrapped_env.action_space.n, arch='distributed_dueling', n_input_channels=wrapped_env.observation_space.shape[0], noisy_net_sigma=None,
        start_epsilon=CONFIG["START_EPSILON"], final_epsilon=CONFIG["FINAL_EPSILON"], final_exploration_frames=CONFIG["DECAY_STEPS"],
        explorer_sample_func=wrapped_env.action_space.sample,
        lr=0.0000625, adam_eps=0.00015, prioritized=True, steps=steps, update_interval=4,
        replay_capacity=30000, num_step_return=10, agent_type='CategoricalDoubleDQN', gpu=args.gpu, gamma=0.99, replay_start_size=5000,
        target_update_interval=10000, clip_delta=True, batch_accumulator='mean')

    if args.load:
        agent.load(args.load)

    train(agent, wrapped_env)
    wrapped_env.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpu", default=-1, action="store_const", const=0)
    parser.add_argument("--load", "-l", help="Path to model weights")
    main(parser.parse_args())
