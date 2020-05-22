# Simple env test.
# noinspection PyUnresolvedReferences
import minerl
import logging
import os
import re
import shutil
import sys
import time
import pickle
from argparse import ArgumentParser
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import chainerrl
import gym
import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir)))

from rainbow import wrap_env, get_agent
from rollout import save_obs
from rainbow_action_sequence import get_actions
from utility.config import CONFIG

# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLTreechop-v0')

now_obj = datetime.now()
EXPORT_DIR = Path(os.environ["HOME"], f"rainbow_{now_obj.strftime('%d-%b-%Y_%H:%M:%S')}")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = Path(EXPORT_DIR, "logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

handler = RotatingFileHandler(Path(LOG_DIR, "train.log"), maxBytes=50 * 1024 * 1024, backupCount=500)
logging.basicConfig(handlers=[handler],
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger('rainbow-train')


def safe_log(msg, level="INFO"):
    if level == "DEBUG":
        n_level = 10
    elif level == "INFO":
        n_level = 20
    else:
        # ERROR
        n_level = 40
    try:
        logger.log(n_level, msg)
    except:
        print(f"Error! Couldn't log '{msg}'.")


def save_agent_and_stats(ep, agent, forest, stats, netr, total_episodes):
    out_dir = Path(EXPORT_DIR, "train", f"ep_{ep}_forest{forest}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir, "stats.pickle"), "wb") as fp:
        pickle.dump(stats, fp)
    np.savetxt(Path(out_dir, "reward.txt"), np.array([netr]))
    if ((ep % 5) == 0) or ((ep % round(total_episodes / 10)) == 0):
        agent.save(out_dir)


def run_curriculum_episode(agent, wrapped_env, forest, actions):
    wrapped_env.seed(forest)
    obs = wrapped_env.reset()
    done = False
    info = {}
    netr = 0
    reward = 0
    stats = []
    steps = 0

    for i in range(len(actions)):
        if done or ("error" in info):
            break
        if (i % round(len(actions) / 10)) == 0:
            safe_log(f"Net reward: {netr}")

        safe_log(f"Epsilon: {agent.explorer.epsilon}", "DEBUG")
        steps += 1
        agent.act_and_train(obs, reward)
        action = actions[i]

        stats.append((agent.model.state, agent.model.q_values, agent.model.advantage))

        if action == 1:
            wrapped_env.env.env.env.env.env._skip = 24
        else:
            wrapped_env.env.env.env.env.env._skip = CONFIG["FRAME_SKIP"]

        obs, reward, done, info = wrapped_env.step(action)
        netr += reward

    agent.stop_episode_and_train(obs, reward, done)
    return netr, stats, steps


def run_episode(agent, wrapped_env, forest, actions, out_dir=None, test=False):
    wrapped_env.seed(forest)
    obs = wrapped_env.reset()
    done = False
    info = {}
    netr = 0
    reward = 0
    stats = []

    steps = 0

    for i in range(actions):
        if done or ("error" in info):
            break
        if (i % round(actions / 10)) == 0:
            safe_log(f"Net reward: {netr}")
        if not test:
            safe_log(f"Epsilon: {agent.explorer.epsilon}", "DEBUG")
        steps += 1

        if test:
            if not out_dir:
                raise ValueError(f"Export directory for validation observations is None.")
            action = agent.act(obs)
            save_obs(agent, obs, i, Path(out_dir))
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
    return netr, stats, steps


def validate_agent(ep, agent, wrapped_env, args, forests):
    rewards = np.array([])
    out_dir = Path(EXPORT_DIR, "validation", f"val{ep}")
    for i, forest in enumerate(forests):
        if args.seed:
            forest = args.seed
        val_ep_dir = Path(out_dir, str(i).zfill(2))
        netr, stats, steps = run_episode(agent, wrapped_env, forest, args.steps, val_ep_dir, test=True)
        sal_dir = Path(".", "saliency")
        try:
            shutil.move(Path(sal_dir, "saliency-traces.txt"), Path(out_dir, "saliency-traces.txt"))
            shutil.rmtree(sal_dir)
        except OSError as e:
            safe_log(f"Error: {sal_dir}: {e.strerror}", "ERROR")
        rewards = np.array([*rewards, (forest, netr)])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(Path(out_dir, "rewards.txt"), rewards)


def load_agent_at_episode(path, idx):
    p = re.compile(f"_{idx}_")
    for f in path.iterdir():
        if p.search(str(f)):
            return f
    raise ValueError(f"Episode {idx} not found at {path}")


def train(wrapped_env, args):
    """
    This function will be called for training phase.
    """
    action_sequence = get_actions(Path("action-sequences", "420_4_64.txt"))

    agent = get_agent(n_actions=wrapped_env.action_space.n,
                      n_input_channels=wrapped_env.observation_space.shape[0],
                      explorer_sample_func=wrapped_env.action_space.sample,
                      gpu=args.gpu,
                      steps=args.episodes * args.steps,
                      gamma=CONFIG["GAMMA"])

    forests = [7, 45, 100, 200, 300, 420, 3456, 5000, 300000, 600506]
    mean_len = 0

    def is_curriculum_episode(ep):
        return (ep % round(ep / ((ep / 100) * args.curriculum))) == 0

    # Train
    for ep in range(1, args.episodes + 1):
        if args.seed:
            forest = args.seed
        else:
            forest = forests[ep % 10]

        safe_log(f"===================== EPISODE {ep} =====================")
        ep_start = time.time()
        if args.curriculum > 0 and is_curriculum_episode(ep):
            netr, stats, steps = run_curriculum_episode(agent,
                                                        wrapped_env,
                                                        forest,
                                                        action_sequence)
        else:
            netr, stats, steps = run_episode(agent, wrapped_env, forest, args.steps)
        save_agent_and_stats(ep, agent, forest, stats, netr, args.episodes)
        if ep == 1:
            mean_len = (time.time() - ep_start)
        else:
            mean_len = (time.time() - ep_start + mean_len) / 2
        safe_log(f"===================== END {ep} - REWARD {netr} - MEAN LENGTH {round(mean_len)}s =====================")

    # Validate
    agent = get_agent(n_actions=wrapped_env.action_space.n,
                      n_input_channels=wrapped_env.observation_space.shape[0],
                      explorer_sample_func=wrapped_env.action_space.sample,
                      gpu=args.gpu,
                      steps=args.steps * 10 * 10,
                      test=True,
                      gamma=CONFIG["GAMMA"])

    if args.validate:
        for val in range(1, args.episodes + 1):
            if (val % round(args.episodes / 10)) == 0:
                safe_log(f"===================== VALIDATION {val} =====================")
                agent.load(load_agent_at_episode(Path(EXPORT_DIR, "train"), val))
                validate_agent(val, agent, wrapped_env, args, forests)
                safe_log(f"===================== END VALIDATION {val} =====================")


def main(args):
    """
    This function will be called for training phase.
    """
    # 0
    chainerrl.misc.set_random_seed(0)
    CONFIG.load(Path(args.conf))
    CONFIG.dump(Path(EXPORT_DIR, "config.json"))
    safe_log(f"Started loading {MINERL_GYM_ENV}")
    start = time.time()
    core_env = gym.make(MINERL_GYM_ENV)
    safe_log(f"Loaded {MINERL_GYM_ENV} in {round(time.time() - start)}s")
    wrapped_env = wrap_env(core_env)

    # wrong way to adjust the action space
    wrapped_env._actions[1]['forward'] = 0
    wrapped_env._actions[2]['forward'] = 0
    wrapped_env._actions[3]['forward'] = 0

    for i in range(wrapped_env.action_space.n):
        safe_log("Action {0}: {1}".format(i, wrapped_env.action(i)))

    train(wrapped_env, args)
    wrapped_env.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpu", default=-1, action="store_const", const=0)
    parser.add_argument("--conf", "-f", help="Path to configuration file")
    parser.add_argument("--seed", type=int, help="Seed for MineRL environment")
    parser.add_argument("--episodes", "-e", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--steps", "-s", type=int, default=1000, help="Number of actions taken per episode")
    parser.add_argument("--curriculum", "-c", type=int, default=0, help="Percentage of curriculum learning episodes")
    parser.add_argument("--validate", default=False, action="store_true", help="Perform validation after every 10% of total episodes")
    main(parser.parse_args())
