# Simple env test.
# noinspection PyUnresolvedReferences
import minerl
import logging
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

EXPORT_DIR = Path(os.environ["HOME"], f"rainbow_{CONFIG['RAINBOW_HISTORY']}")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=Path(EXPORT_DIR, "train.log"),
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger('rainbow-train')


def save_agent_and_stats(ep, agent, forest, stats, netr):
    out_dir = Path(EXPORT_DIR, "train", f"ep_{ep}_forest{forest}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir, "stats.pickle"), "wb") as fp:
        pickle.dump(stats, fp)
    np.savetxt(Path(out_dir, "reward.txt"), np.array([netr]))
    if (ep % 5) == 0:
        agent.save(out_dir)


def run_episode(agent, wrapped_env, forest, actions, test=False):
    wrapped_env.seed(forest)
    obs = wrapped_env.reset()
    done = False
    info = {}
    netr = 0
    reward = 0
    stats = []

    for i in range(actions):
        print(f"Step {i}, netr: {netr}")
        if (i % (actions / 10)) == 0:
            logger.info(f"Net reward: {netr}")

        if done or ("error" in info):
            break

        if test:
            action = agent.act(obs)
        else:
            action = agent.act_and_train(obs, reward)

        stats.append((agent.model.state, agent.model.q_values, agent.model.advantage))

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


def validate_agent(ep, agent, wrapped_env, actions, forests):
    rewards = np.array([])
    for forest in forests:
        netr, stats = run_episode(agent, wrapped_env, forest, actions, test=True)
        rewards = np.array([*rewards, (forest, netr)])
    out_dir = Path(EXPORT_DIR, "validation", f"val{ep}_mean{round(rewards.mean(axis=0)[1])}")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(Path(out_dir, "rewards.txt"), rewards)
    agent.save(out_dir)


def train(wrapped_env, args):
    """
    This function will be called for training phase.
    """
    agent = get_agent(n_actions=wrapped_env.action_space.n,
                      n_input_channels=wrapped_env.observation_space.shape[0],
                      explorer_sample_func=wrapped_env.action_space.sample,
                      gpu=-1,
                      steps=args.episodes * args.steps)

    forests = [7, 45, 100, 200, 300, 420, 3456, 5000, 300000, 600506]
    mean_len = 0
    vals = 0
    for ep in range(1, args.episodes + 1):
        if args.seed:
            forest = args.seed
        else:
            forest = forests[ep % 10]
        if (ep % (args.episodes / 10)) == 0 and args.validate:
            if not args.seed:
                forest = forests[(ep - 1) % 10]
            last_eps = agent.explorer.epsilon
            # Validation
            logger.info(f"===================== VALIDATION {vals} =====================")
            CONFIG.test()
            agent = get_agent(n_actions=wrapped_env.action_space.n,
                              n_input_channels=wrapped_env.observation_space.shape[0],
                              explorer_sample_func=wrapped_env.action_space.sample,
                              gpu=-1,
                              steps=args.steps * 10,
                              test=True)
            agent.load(Path(EXPORT_DIR, "train", f"ep_{ep - 1}_forest{forest}"))
            validate_agent(vals, agent, wrapped_env, args.steps, forests)
            CONFIG.train()
            agent.load(Path(EXPORT_DIR, "train", f"ep_{ep - 1}_forest{forest}"))
            agent = get_agent(n_actions=wrapped_env.action_space.n,
                              n_input_channels=wrapped_env.observation_space.shape[0],
                              explorer_sample_func=wrapped_env.action_space.sample,
                              gpu=-1,
                              start_eps=last_eps,
                              steps=(args.episodes - vals - 1) * args.steps)
            logger.info(f"===================== END VALIDATION {vals} =====================")
            vals += 1
        else:
            # Train
            logger.info(f"===================== EPISODE {ep} =====================")
            ep_start = time.time()
            netr, stats = run_episode(agent, wrapped_env, forest, args.steps)
            save_agent_and_stats(ep, agent, forest, stats, netr)
            if ep == 1:
                mean_len = (time.time() - ep_start) / 2
            else:
                mean_len = (time.time() - ep_start + mean_len) / 2
            logger.info(f"===================== END {ep} - REWARD {netr} - MEAN LENGTH {round(mean_len)}s =====================")


def main(args):
    """
    This function will be called for training phase.
    """
    # 0
    chainerrl.misc.set_random_seed(0)
    CONFIG.apply(SINGLE_FRAME_AGENT_ATTACK_AND_FORWARD)
    logger.info(f"Started loading {MINERL_GYM_ENV}")
    start = time.time()
    core_env = gym.make(MINERL_GYM_ENV)
    logger.info(f"Loaded {MINERL_GYM_ENV} in {round(time.time() - start)}s")
    wrapped_env = wrap_env(core_env)

    # wrong way to adjust the action space
    wrapped_env._actions[1]['forward'] = 0
    wrapped_env._actions[2]['forward'] = 0
    wrapped_env._actions[3]['forward'] = 0

    for i in range(wrapped_env.action_space.n):
        logger.info("Action {0}: {1}".format(i, wrapped_env.action(i)))

    train(wrapped_env, args)
    wrapped_env.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpu", default=-1, action="store_const", const=0)
    parser.add_argument("--load", "-l", help="Path to model weights")
    parser.add_argument("--seed", type=int, help="Seed for MineRL environment")
    parser.add_argument("--episodes", "-e", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--steps", "-s", type=int, default=1000, help="Number of actions taken per episode")
    parser.add_argument("--validate", default=False, action="store_true", help="Perform validation after every 10% of total episodes")
    main(parser.parse_args())
