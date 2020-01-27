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

from utility.q_functions import DistributionalDuelingDQN
from utility.env_wrappers import (SerialDiscreteActionWrapper, MoveAxisWrapper, FrameSkip, FrameStack, ObtainPoVWrapper)

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
    env = FrameStack(env, 4, channel_order='chw')
    # wrap env: action...
    env = SerialDiscreteActionWrapper(
        env,
        always_keys=['attack'], reverse_keys=['forward'], exclude_keys=['back', 'left', 'right', 'sneak', 'sprint'], exclude_noop=False)

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


def act(agent, obs):
    action_value = agent.model(
        agent.batch_states([obs], agent.xp, agent.phi))
    state = agent.model.state
    print("Q-Values: ", action_value.q_values.data[0])
    print("Advantage: ", agent.model.advantage)
    print("State: ", state)
    q = float(action_value.max.array)
    action = cuda.to_cpu(action_value.greedy_actions.array)[0]

    # Update stats
    agent.average_q *= agent.average_q_decay
    agent.average_q += (1 - agent.average_q_decay) * q

    agent.logger.debug('t:%s q:%s action_value:%s', agent.t, q, action_value)

    # if the state value is high enough, the agent is standing in front of a tree
    if state >= 450:
        print("CHOPPING TREE")
        # execute action 1 to attack and toggle off 'forward'
        action = 1
    return action


def main(gpu=-1, sleeping_amount=0.5):
    """
    This function will be called for training phase.
    """
    chainerrl.misc.set_random_seed(0)

    core_env = gym.make(MINERL_GYM_ENV)
    wrapped_env = wrap_env(core_env, test=False)
    # for i in range(wrapped_env.action_space.n):
    #     print("Action {0}: {1}".format(i, wrapped_env.action(i)))

    maximum_frames = 8000000
    steps = maximum_frames // 4
    agent = get_agent(
        n_actions=wrapped_env.action_space.n, arch='distributed_dueling', n_input_channels=wrapped_env.observation_space.shape[0],
        noisy_net_sigma=0.5, final_epsilon=0.01,
        final_exploration_frames=10 ** 6, explorer_sample_func=wrapped_env.action_space.sample,
        lr=0.0000625, adam_eps=0.00015, prioritized=True, steps=steps, update_interval=4,
        replay_capacity=30000, num_step_return=10, agent_type='CategoricalDoubleDQN', gpu=gpu, gamma=0.99, replay_start_size=5000,
        target_update_interval=10000, clip_delta=True, batch_accumulator='mean'
    )

    agent.load("models/rainbow")

    for _ in range(MINERL_MAX_EVALUATION_EPISODES):
        obs = wrapped_env.reset()
        input("Press ENTER to continue")
        done = False
        netr = 0
        step = 0
        reward = 0
        last_action = None

        while not done:
            start = time.time()
            action = act(agent, obs)
            make_barplot(agent.model.advantage, step)
            create_and_save_saliency_image(agent, obs, step, reward, netr, action, last_action)
            last_action = action
            print("Action: ", ACTIONS[action])
            obs, reward, done, info = wrapped_env.step(action)
            netr += reward
            print("Net reward: ", netr)
            step += 1
            sleep_time = sleeping_amount - (time.time() - start)
            if sleep_time >= 0:
                print(f"Sleeping {round(sleep_time)}s.")
                time.sleep(sleep_time)

    wrapped_env.close()


if __name__ == "__main__":
    main()
