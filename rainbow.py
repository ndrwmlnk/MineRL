import chainer
import chainerrl
import gym
import numpy as np

from chainerrl.wrappers import ContinuingTimeLimit
from chainerrl.wrappers.atari_wrappers import FrameStack
from chainerrl.wrappers.atari_wrappers import ScaledFloatFrame as SFF

from utility.env_wrappers import (SerialDiscreteActionWrapper, MoveAxisWrapper, FrameSkip, FrameStack, ObtainPoVWrapper)
from utility.q_functions import DistributionalDuelingDQN

from utility.config import CONFIG


def wrap_env(env):
    # wrap env: time limit...
    if isinstance(env, gym.wrappers.TimeLimit):
        env = env.env
        max_episode_steps = env.spec.max_episode_steps
        env = ContinuingTimeLimit(env, max_episode_steps=max_episode_steps)
    # wrap env: observation...
    # NOTE: wrapping order matters!
    env = FrameSkip(env, skip=CONFIG["FRAME_SKIP"])
    env = ObtainPoVWrapper(env)
    # convert hwc -> chw as Chainer requires.
    env = MoveAxisWrapper(env, source=-1, destination=0)
    env = SFF(env)
    env = FrameStack(env, CONFIG["RAINBOW_HISTORY"], channel_order='chw')
    # wrap env: action...
    env = SerialDiscreteActionWrapper(
        env,
        always_keys=CONFIG["ALWAYS_KEYS"], reverse_keys=CONFIG["REVERSE_KEYS"], exclude_keys=CONFIG["EXCLUDE_KEYS"],
        exclude_noop=CONFIG["EXCLUDE_NOOP"], max_camera_range=CONFIG["MAX_CAMERA_RANGE"])

    return env


def parse_agent(agent):
    return {'DQN': chainerrl.agents.DQN,
            'DoubleDQN': chainerrl.agents.DoubleDQN,
            'PAL': chainerrl.agents.PAL,
            'CategoricalDoubleDQN': chainerrl.agents.CategoricalDoubleDQN}[agent]


def get_agent(
        n_actions, n_input_channels, explorer_sample_func, gpu, steps, test=False,
        lr=0.0000625, adam_eps=0.00015, prioritized=True, update_interval=4,
        replay_capacity=30000, num_step_return=10, agent_type='CategoricalDoubleDQN',  gamma=0.99, replay_start_size=5000,
        target_update_interval=10000, clip_delta=True, batch_accumulator='mean'
):
    # Q function
    n_atoms = 51
    v_min = -10
    v_max = 10
    q_func = DistributionalDuelingDQN(n_actions, n_atoms, v_min, v_max, n_input_channels=n_input_channels)

    # explorer
    if test:
        chainerrl.links.to_factorized_noisy(
            q_func, sigma_scale=CONFIG["NOISY_NET_SIGMA"])
        # Turn off explorer
        explorer = chainerrl.explorers.Greedy()
    else:
        explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(CONFIG["START_EPSILON"],
                                                                CONFIG["FINAL_EPSILON"],
                                                                CONFIG["DECAY_STEPS"],
                                                                explorer_sample_func)

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
