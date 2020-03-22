# Generate observations rollout
# noinspection PyUnresolvedReferences
import minerl

import os
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

import chainerrl
import gym

sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir)))

from rainbow import wrap_env, get_agent
from rollout import save_obs, OUT_DIR
from utility.config import CONFIG, SINGLE_FRAME_AGENT_ATTACK_AND_FORWARD

MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLTreechop-v0')


def parse_line(line):
    if "x" in line:
        action, times = line.split("x")
        return [int(action)] * int(times)
    else:
        try:
            return int(line)
        except ValueError:
            return []


def flatten(l):
    new = []
    for el in l:
        if type(el) == list:
            for sub_el in el:
                new.append(sub_el)
        else:
            new.append(el)
    return new


def get_actions(path):
    with open(path) as fp:
        lines = fp.readlines()
    return flatten(map(parse_line, lines))


def main(args):
    """
    This function will be called for training phase.
    """
    chainerrl.misc.set_random_seed(0)
    CONFIG.apply(SINGLE_FRAME_AGENT_ATTACK_AND_FORWARD)
    CONFIG.test()

    actions_file = Path(args.actions)
    if not actions_file.is_file():
        raise ValueError(f"Actions file {actions_file} is not a file!")

    actions = get_actions(actions_file)
    if len(actions) < args.steps:
        raise ValueError(f"{len(actions)} actions specified for {args.steps} steps in {actions_file}")
    print(f"{len(actions)} actions: {actions}")

    core_env = gym.make(MINERL_GYM_ENV)
    wrapped_env = wrap_env(core_env)

    # wrong way to adjust the action space
    wrapped_env._actions[1]['forward'] = 0
    wrapped_env._actions[2]['forward'] = 0
    wrapped_env._actions[3]['forward'] = 0

    for i in range(wrapped_env.action_space.n):
        print("Action {0}: {1}".format(i, wrapped_env.action(i)))

    forest = 420
    wrapped_env.seed(forest)

    agent = get_agent(n_actions=wrapped_env.action_space.n,
                      n_input_channels=wrapped_env.observation_space.shape[0],
                      explorer_sample_func=wrapped_env.action_space.sample,
                      gpu=args.gpu,
                      steps=len(actions),
                      test=True,
                      gamma=0.95)

    out_dir = Path(OUT_DIR, "sequences", Path(args.actions).name.split(".")[0])
    out_dir.mkdir(parents=True, exist_ok=True)

    obs = wrapped_env.reset()
    done = False
    info = {}
    netr = 0
    import time
    for i in range(args.steps):
        action = actions[i]
        if done or ("error" in info):
            break

        agent.act(obs)
        last_action = CONFIG["ACTION_SPACE"][action]

        if action == 1:
            wrapped_env.env.env.env.env.env._skip = 24
        else:
            wrapped_env.env.env.env.env.env._skip = CONFIG["FRAME_SKIP"]

        obs, reward, done, info = wrapped_env.step(action)
        if args.rollout:
            save_obs(agent, obs, i,reward, netr, action, last_action, out_dir, 4.5)
        else:
            wrapped_env.render(mode="human")
        time.sleep(1.0)
        if reward > 0:
            print(f"Received reward +{reward}")
        netr += reward

    sal_dir = Path(".", "saliency")
    try:
        shutil.rmtree(sal_dir)
    except OSError as e:
        print(f"Error: {sal_dir} : {e.strerror}")

    agent.stop_episode()
    wrapped_env.close()
    print(f"Chopped blocks: {netr}")


if __name__ == "__main__":
    ARCH_JAVA_PATH = Path("/usr/lib/jvm/java-8-openjdk")
    if ARCH_JAVA_PATH.is_dir():
        os.environ["JAVA_HOME"] = str(ARCH_JAVA_PATH)
    parser = ArgumentParser()
    parser.add_argument("--gpu", default=-1, action="store_const", const=0)
    parser.add_argument("--rollout", default=False, action="store_true")
    parser.add_argument("--steps", "-s", type=int, default=32, help="Number of actions taken per episode")
    parser.add_argument("--actions")

    main(parser.parse_args())
