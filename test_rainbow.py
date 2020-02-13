# Simple env test.
# noinspection PyUnresolvedReferences
import minerl

import os
import sys
import time

import chainer
import chainerrl
import gym
import numpy as np
from chainer import cuda
from chainerrl.wrappers import ContinuingTimeLimit
from chainerrl.wrappers.atari_wrappers import FrameStack
from chainerrl.wrappers.atari_wrappers import ScaledFloatFrame as SFF

sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir)))

from utility.config import CONFIG, SINGLE_FRAME_AGENT_ATTACK_ALWAYS_FWD, SINGLE_FRAME_AGENT
from utility.env_wrappers import (SerialDiscreteActionWrapper, MoveAxisWrapper, FrameSkip, FrameStack, ObtainPoVWrapper)
from utility.q_functions import DistributionalDuelingDQN

from saliency import create_and_save_saliency_image, make_barplot, ACTIONS

# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
# MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamond-v0')
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
        noisy_net_sigma, final_epsilon, final_exploration_frames, explorer_sample_func,
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
            1.0, final_epsilon, final_exploration_frames, explorer_sample_func)

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


def act(agent, obs, last_state, chop_start, reward, ):
    action_value = agent.model(
        agent.batch_states([obs], agent.xp, agent.phi))
    state = cuda.to_cpu(agent.model.state)
    # print("Q-Values: ", action_value.q_values.data[0])
    # print("Advantage: ", agent.model.advantage)
    print("State: ", state)
    q = float(action_value.max.array)
    action = cuda.to_cpu(action_value.greedy_actions.array)[0]

    # Update stats
    # agent.average_q *= agent.average_q_decay
    # agent.average_q += (1 - agent.average_q_decay) * q
    #
    # agent.logger.debug('t:%s q:%s action_value:%s', agent.t, q, action_value)

    STATE_THRESHOLD = 450

    # if the state value is high enough, the agent is standing in front of a tree
    if state >= STATE_THRESHOLD and not last_state:
        print(f"Starting Chopping")
        action = 1
        last_state = True
    elif last_state and reward > 0:
        print("Stopped Chopping")
        last_state = False
    return action, last_state


def main(args):
    """
    This function will be called for training phase.
    """
    # if args.gpu == 0:
    #     sleeping_amount = 0.0
    # else:
    sleeping_amount = 0.5

    chainerrl.misc.set_random_seed(0)
    CONFIG.apply(SINGLE_FRAME_AGENT)
    CONFIG.test()

    core_env = gym.make(MINERL_GYM_ENV)
    wrapped_env = wrap_env(core_env, test=False)
    for i in range(wrapped_env.action_space.n):
        print("Action {0}: {1}".format(i, wrapped_env.action(i)))

    maximum_frames = 8000000
    steps = maximum_frames // 4
    agent = get_agent(
        n_actions=wrapped_env.action_space.n, arch='distributed_dueling', n_input_channels=wrapped_env.observation_space.shape[0],
        noisy_net_sigma=CONFIG["NOISY_NET_SIGMA"], final_epsilon=CONFIG["FINAL_EPSILON"], final_exploration_frames=CONFIG["DECAY_STEPS"],
        explorer_sample_func=wrapped_env.action_space.sample,
        lr=0.0000625, adam_eps=0.00015, prioritized=True, steps=steps, update_interval=4,
        replay_capacity=30000, num_step_return=10, agent_type='CategoricalDoubleDQN', gpu=args.gpu, gamma=0.99, replay_start_size=5000,
        target_update_interval=10000, clip_delta=True, batch_accumulator='mean'
    )

    print(f"Loading agent from {args.load}")
    agent.load(args.load)

    for _ in range(MINERL_MAX_EVALUATION_EPISODES):
        obs = wrapped_env.reset()
        # input("Press ENTER to continue")
        done = False
        netr = 0
        step = 0
        reward = 0
        last_action = None
        last_state = False
        chop_start = 0

        while not done:
            start = time.time()
            action, last_state = act(agent, obs, last_state, chop_start, reward)
            make_barplot(cuda.to_cpu(agent.model.advantage), step)
            create_and_save_saliency_image(agent, obs, step, reward, netr, ACTIONS[action], last_action)
            last_action = ACTIONS[action]
            print("Action: ", ACTIONS[action])
            obs, reward, done, info = wrapped_env.step(action)
            netr += reward
            # print("Net reward: ", netr)
            step += 1
            sleep_time = sleeping_amount - (time.time() - start)
            if sleep_time >= 0:
                time.sleep(sleep_time)

    wrapped_env.close()
