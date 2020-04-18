# Simple env test.
# noinspection PyUnresolvedReferences
import minerl

import os
import sys

import chainerrl
import gym
from chainer import cuda

sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir)))

from rainbow import wrap_env, get_agent
from utility.config import CONFIG, DOUBLE_FRAME_AGENT_ATTACK_AND_FORWARD


# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLTreechop-v0')
MINERL_MAX_EVALUATION_EPISODES = int(os.getenv('MINERL_MAX_EVALUATION_EPISODES', 5))


def main(args):
    """
    This function will be called for training phase.
    """

    chainerrl.misc.set_random_seed(0)
    CONFIG.apply(DOUBLE_FRAME_AGENT_ATTACK_AND_FORWARD)

    core_env = gym.make(MINERL_GYM_ENV)
    wrapped_env = wrap_env(core_env)

    # wrong way to adjust the action space
    wrapped_env._actions[1]['forward'] = 0
    wrapped_env._actions[2]['forward'] = 0
    wrapped_env._actions[3]['forward'] = 0

    for i in range(wrapped_env.action_space.n):
        print("Action {0}: {1}".format(i, wrapped_env.action(i)))

    maximum_frames = 8000000
    steps = maximum_frames // 4
    agent = get_agent(n_actions=wrapped_env.action_space.n,
                      n_input_channels=wrapped_env.observation_space.shape[0],
                      explorer_sample_func=wrapped_env.action_space.sample,
                      gpu=args.gpu,
                      steps=steps,
                      test=not args.train)

    print(f"Loading agent from {args.load}")
    agent.load(args.load)

    if args.seed:
        wrapped_env.seed(args.seed)

    for _ in range(MINERL_MAX_EVALUATION_EPISODES):
        obs = wrapped_env.reset()
        done = False
        netr = 0
        step = 0

        while not done:
            action = agent.act(obs)
            state = cuda.to_cpu(agent.model.state)
            print("State: ", state)
            print("Action: ", CONFIG["ACTION_SPACE"][action])
            #wrapped_env.render(mode="human")
            obs, reward, done, info = wrapped_env.step(action)
            netr += reward
            print("Net reward: ", netr)
            step += 1

    wrapped_env.close()
