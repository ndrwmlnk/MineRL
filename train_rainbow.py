# Simple env test.
# noinspection PyUnresolvedReferences
import minerl
import os
import sys
import time
import pickle
from argparse import ArgumentParser
from pathlib import Path

import chainerrl
import gym
import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir)))

from rainbow import wrap_env, get_agent
from utility.config import CONFIG, SINGLE_FRAME_AGENT_ATTACK_AND_FORWARD

# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLTreechop-v0')

TRAINING_EPISODES = 1000
ACTION_PER_EPISODE = 1000

EXPORT_DIR = Path(os.environ["HOME"], f"rainbow_{CONFIG['RAINBOW_HISTORY']}")


def save_agent_and_stats(ep, agent, forest, stats, netr):
    out_dir = Path(EXPORT_DIR, "train", f"ep_{ep}_forest{forest}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open("stats.pickle", "wb") as fp:
        pickle.dump(stats, fp)
    np.savetxt(Path(out_dir, "reward.txt"), netr)
    agent.save(out_dir)


def run_episode(agent, wrapped_env, forest, test=False):
    wrapped_env.seed(forest)
    obs = wrapped_env.reset()
    done = False
    info = {}
    netr = 0
    reward = 0
    stats = []

    for i in range(ACTION_PER_EPISODE):
        if i % 100 == 0:
            print(f"Net reward: {netr}")

        if done or ("error" in info):
            break

        if test:
            action = agent.act(obs)
        else:
            action = agent.act_and_train(obs, reward)

        stats.append((agent.model.state, agent.model.q_value, agent.model.advantage))

        if action == 1:
            wrapped_env.env.env.env.env.env._skip = 24
        else:
            wrapped_env.env.env.env.env.env._skip = CONFIG["FRAME_SKIP"]

        obs, reward, done, info = wrapped_env.step(action)
        netr += reward

    if test:
        agent.stop_episode()
    else:
        agent.stop_episode_and_train(obs, reward, done)
    return netr, stats


def validate_agent(ep, agent, wrapped_env, forests):
    rewards = np.array([])
    for forest in forests:
        netr, stats = run_episode(agent, wrapped_env, forest, test=True)
        rewards = np.array([*rewards, (forest, netr)])
    out_dir = Path(EXPORT_DIR, "validation", f"val{ep}_mean{round(rewards.mean(axis=0)[1])}")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(Path(out_dir, "rewards.txt"), rewards)
    agent.save(out_dir)


def train(wrapped_env):
    """
    This function will be called for training phase.
    """
    agent = get_agent(n_actions=wrapped_env.action_space.n,
                      n_input_channels=wrapped_env.observation_space.shape[0],
                      explorer_sample_func=wrapped_env.action_space.sample,
                      gpu=-1,
                      steps=CONFIG["DECAY_STEPS"])

    forests = [7, 45, 100, 200, 300, 420, 3456, 5000, 300000, 600506]
    mean_len = 0
    for ep in range(1, TRAINING_EPISODES + 1):
        if ep % 100 == 0:
            # Validation
            print(f"===================== VALIDATION {ep % 100} =====================")
            CONFIG.test()
            agent = get_agent(n_actions=wrapped_env.action_space.n,
                              n_input_channels=wrapped_env.observation_space.shape[0],
                              explorer_sample_func=wrapped_env.action_space.sample,
                              gpu=-1,
                              steps=10000)
            agent.load(Path(EXPORT_DIR, "train", f"ep_{ep - 1}_forest{forests[(ep - 1) % 10]}"))
            validate_agent(ep % 100, agent, wrapped_env, forests)
            CONFIG.train()
            agent = get_agent(n_actions=wrapped_env.action_space.n,
                              n_input_channels=wrapped_env.observation_space.shape[0],
                              explorer_sample_func=wrapped_env.action_space.sample,
                              gpu=-1,
                              steps=(TRAINING_EPISODES - (ep % 100) - 1) * 1000)
            print(f"===================== END VALIDATION {ep % 100} =====================")
        else:
            # Train
            print(f"===================== EPISODE {ep} =====================")
            ep_start = time.time()
            netr, stats = run_episode(agent, wrapped_env, forests[ep % 10])
            save_agent_and_stats(ep, agent, forests[ep % 10], stats, netr)
            mean_len = (time.time() - ep_start + mean_len) / 2
            print(f"===================== END {ep} - REWARD {netr} - MEAN LENGTH {round(mean_len * 60)}m =====================")


def main(args):
    """
    This function will be called for training phase.
    """
    # 0
    chainerrl.misc.set_random_seed(0)
    CONFIG.apply(SINGLE_FRAME_AGENT_ATTACK_AND_FORWARD)

    core_env = gym.make(MINERL_GYM_ENV)
    wrapped_env = wrap_env(core_env)

    # wrong way to adjust the action space
    wrapped_env._actions[1]['forward'] = 0
    wrapped_env._actions[2]['forward'] = 0
    wrapped_env._actions[3]['forward'] = 0

    for i in range(wrapped_env.action_space.n):
        print("Action {0}: {1}".format(i, wrapped_env.action(i)))

    train(wrapped_env)
    wrapped_env.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpu", default=-1, action="store_const", const=0)
    parser.add_argument("--load", "-l", help="Path to model weights")
    main(parser.parse_args())
