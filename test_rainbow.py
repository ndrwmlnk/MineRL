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

from utility.config import CONFIG
from utility.env_wrappers import (SerialDiscreteActionWrapper, MoveAxisWrapper, FrameSkip, FrameStack, ObtainPoVWrapper)
from utility.q_functions import DistributionalDuelingDQN

from saliency import create_and_save_saliency_image, make_barplot, ACTIONS, save_image

# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
# MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamond-v0')
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLTreechop-v0')
MINERL_MAX_EVALUATION_EPISODES = int(os.getenv('MINERL_MAX_EVALUATION_EPISODES', 5))


def act(agent, obs, last_state, reward, step):
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
        print(f"Start Chopping")
        from pathlib import Path
        out = Path("saliency", "chopping-states")
        out.mkdir(exist_ok=True)
        for i, o in enumerate(np.split(np.array(obs), CONFIG["RAINBOW_HISTORY"])):
            save_image(o, Path(out, str(step).zfill(5) + '_' + str(i) + '_' + str(round(state)) + '.jpg'))

        action = 1
        last_state = True
    elif last_state and reward > 0:
        print("Stop Chopping")
        last_state = False
    return action, last_state


def main(args):
    """
    This function will be called for training phase.
    """
    if args.gpu == 0:
        sleeping_amount = 0.0
    else:
        sleeping_amount = 0.5

    chainerrl.misc.set_random_seed(0)
    CONFIG.test()

    core_env = gym.make(MINERL_GYM_ENV)
    wrapped_env = wrap_env(core_env, test=False)

    # wrong way to adjust the action space
    # wrapped_env._actions[1]['attack'] = 1

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
