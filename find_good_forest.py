# noinspection PyUnresolvedReferences
import minerl

import os
import shutil
import sys
from pathlib import Path
from random import randint

import chainerrl
import gym

sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir)))

from rainbow import wrap_env, get_agent
from train_rainbow import run_episode, save_agent_and_stats
from utility.config import CONFIG, SINGLE_FRAME_AGENT_ATTACK_AND_FORWARD
OUT = Path("forest_testing")


MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLTreechop-v0')


def verify_reward(wrapped_env, forest):
    agent = get_agent(n_actions=wrapped_env.action_space.n,
                      n_input_channels=wrapped_env.observation_space.shape[0],
                      explorer_sample_func=wrapped_env.action_space.sample,
                      gpu=-1,
                      steps=32,
                      gamma=0.95)

    return map(lambda i: run_episode(agent, wrapped_env, forest, 32),
               range(30))


def main():
    """
    This function will be called for training phase.
    """
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

    agent = get_agent(n_actions=wrapped_env.action_space.n,
                      n_input_channels=wrapped_env.observation_space.shape[0],
                      explorer_sample_func=wrapped_env.action_space.sample,
                      gpu=-1,
                      steps=32,
                      gamma=0.95)
    bad_forests = []
    i = 0
    while i < 1000:
        forest = randint(0, 999999)
        if forest in bad_forests:
            continue
        netr, stats = run_episode(agent, wrapped_env, forest, 32)
        if netr >= 4:
            out_dir = Path(OUT, f"{forest}")
            for j, stats in enumerate(verify_reward(wrapped_env, forest)):
                save_agent_and_stats(j, agent, forest, stats[0], stats[1])
            shutil.move(Path(os.environ["HOME"], f"rainbow_{CONFIG['RAINBOW_HISTORY']}"),
                        out_dir,
                        copy_function=shutil.copytree)
        else:
            bad_forests.append(forest)
        i += 1

    wrapped_env.close()


if __name__ == "__main__":
    ARCH_JAVA_PATH = Path("/usr/lib/jvm/java-8-openjdk")
    COMPUTE_JAVA_PATH = Path(os.environ["HOME"], "jdk8u222-b10")
    UBUNTU_JAVA_PATH = Path("/usr/lib/jvm/java-8-openjdk-amd64")

    if ARCH_JAVA_PATH.is_dir():
        os.environ["JAVA_HOME"] = str(ARCH_JAVA_PATH)
    elif COMPUTE_JAVA_PATH.is_dir():
        os.environ["JAVA_HOME"] = str(COMPUTE_JAVA_PATH)
    elif UBUNTU_JAVA_PATH.is_dir():
        os.environ["JAVA_HOME"] = str(UBUNTU_JAVA_PATH)
    else:
        print(f"No Java 8 instance was found on this machine!")
        sys.exit(1)

    os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
    main()