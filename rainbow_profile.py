# Measure rainbow performance
# noinspection PyUnresolvedReferences
import minerl
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import chainerrl
import gym
import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir)))

from rainbow import wrap_env, get_agent
from utility.config import CONFIG, SINGLE_FRAME_AGENT_ATTACK_AND_FORWARD

MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLTreechop-v0')


def profile_episode(agent, wrapped_env, forest, actions, test=False):
    wrapped_env.seed(forest)
    done = False
    info = {}
    reward = 0
    action_lengths = []
    start = time.time()
    obs = wrapped_env.reset()

    for i in range(actions):
        if done or ("error" in info):
            break

        if test:
            action = agent.act(obs)
        else:
            action = agent.act_and_train(obs, reward)

        action_lengths.append(time.time() - start)
        start = time.time()

        if action == 1:
            wrapped_env.env.env.env.env.env._skip = 24
        else:
            wrapped_env.env.env.env.env.env._skip = CONFIG["FRAME_SKIP"]

        obs, reward, done, info = wrapped_env.step(action)

    if test:
        agent.stop_episode()
    else:
        agent.stop_episode_and_train(obs, reward, done)
    return action_lengths


def main(args):
    """
    This function will be called for training phase.
    """
    chainerrl.misc.set_random_seed(0)
    CONFIG.apply(SINGLE_FRAME_AGENT_ATTACK_AND_FORWARD)
    core_env = gym.make(MINERL_GYM_ENV)
    wrapped_env = wrap_env(core_env)

    forest = 420

    # wrong way to adjust the action space
    wrapped_env._actions[1]['forward'] = 0
    wrapped_env._actions[2]['forward'] = 0
    wrapped_env._actions[3]['forward'] = 0

    for i in range(wrapped_env.action_space.n):
        print("Action {0}: {1}".format(i, wrapped_env.action(i)))

    steps = int(args.steps)

    agent = get_agent(n_actions=wrapped_env.action_space.n,
                      n_input_channels=wrapped_env.observation_space.shape[0],
                      explorer_sample_func=wrapped_env.action_space.sample,
                      gpu=args.gpu,
                      steps=steps,
                      test=args.test,
                      gamma=0.95)
    action_lengths = profile_episode(agent, wrapped_env, forest, steps, args.test)
    np.savetxt(Path(".", f"{str(steps).zfill(4)}_steps_profiling.txt"),
               np.array(action_lengths, dtype='float32'))
    wrapped_env.close()


if __name__ == "__main__":
    ARCH_JAVA_PATH = Path("/usr/lib/jvm/java-8-openjdk")
    UBUNTU_JAVA_PATH = Path(os.environ["HOME"], "jdk8u222-b10")

    if ARCH_JAVA_PATH.is_dir():
        os.environ["JAVA_HOME"] = str(ARCH_JAVA_PATH)
    elif UBUNTU_JAVA_PATH.is_dir():
        os.environ["JAVA_HOME"] = str(UBUNTU_JAVA_PATH)

    os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]

    parser = ArgumentParser()
    parser.add_argument("--gpu", default=-1, action="store_const", const=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--test", default=False, action="store_true")

    main(parser.parse_args())
