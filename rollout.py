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
import matplotlib.pyplot as plt
import numpy as np
import PIL
from chainer import cuda
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir)))

from rainbow import wrap_env, get_agent
from utility.imgutils import save_image, observation_to_rgb, get_depth
from utility.config import CONFIG

MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLTreechop-v0')
OUT_DIR = Path(os.environ["HOME"], "minerl_rollouts")


def concat_h(img1, img2):
    if not isinstance(img1, PIL.Image.Image):
        img1 = Image.fromarray(img1, mode="RGB")
    if not isinstance(img2, PIL.Image.Image):
        img2 = Image.fromarray(img2, mode="RGB")
    dst = Image.new('RGB', (img1.width + img2.width, img1.height))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (img1.width, 0))
    return dst


def merge_images(imgs):
    if len(imgs) >= 1:
        last = None
        for i in imgs:
            if last is None:
                last = i
            else:
                last = concat_h(last, i)
        return last
    else:
        raise ValueError(f"{imgs} is supposed to be a list")


def overlay_text(obs, text, pos=(0, 0), fill="black"):
    img = Image.fromarray(obs, mode="RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('./FiraMono-Regular.ttf', 8)
    w, h = font.getsize(text)
    if fill:
        draw.rectangle((*pos, pos[0] + w, pos[1] + h), fill=fill)
    draw.text(pos, text, (255, 255, 255), font=font)
    return img


def overlay_state_value(obs, value, pos=(0, 0)):
    text = f"State: {round(float(value), ndigits=2)}"
    return overlay_text(obs, text, pos)


def overlay_q_values(obs, values, pos=(0, 0)):
    text = "\n".join([f"Q{i}: {round(float(v), ndigits=2)}" for i, v in enumerate(values)])
    return overlay_text(obs, text, pos, fill="")


def save_obs(agent, obs, step, out_dir, size=(256, 256)):
    if not isinstance(obs, np.ndarray):
        obs = np.array(obs)
    rgb_dir = Path(out_dir, "rgb")
    depth_dir = Path(out_dir, "depth")
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    q_values = cuda.to_cpu(agent.model.q_values)

    def export_obs(o, name):
        save_image(overlay_q_values(observation_to_rgb(o), q_values),
                   Path(rgb_dir, name), size)
        if o.shape[0] == 4:
            save_image(get_depth(o), Path(depth_dir, name), size)

    o = np.split(obs, CONFIG["RAINBOW_HISTORY"])[-1]
    name = f"{str(step).zfill(4)}.png"
    export_obs(o, name)


def main(args):
    """
    This function will be called for training phase.
    """
    chainerrl.misc.set_random_seed(0)
    if not args.conf:
        CONFIG.load(Path(args.load, "config.json"))
    else:
        CONFIG.load(Path(args.conf))
    core_env = gym.make(MINERL_GYM_ENV)
    wrapped_env = wrap_env(core_env)

    # wrong way to adjust the action space
    wrapped_env._actions[1]['forward'] = 0
    wrapped_env._actions[2]['forward'] = 0
    wrapped_env._actions[3]['forward'] = 0

    for i in range(wrapped_env.action_space.n):
        print("Action {0}: {1}".format(i, wrapped_env.action(i)))

    steps = args.steps
    load_dir = Path(args.load)

    if args.seed:
        wrapped_env.seed(args.seed)
        out_dir = Path(OUT_DIR, f"{load_dir.name}_{args.seed}")
    else:
        out_dir = Path(OUT_DIR, load_dir.name)

    agent = get_agent(n_actions=wrapped_env.action_space.n,
                      n_input_channels=wrapped_env.observation_space.shape[0],
                      explorer_sample_func=wrapped_env.action_space.sample,
                      gpu=args.gpu,
                      steps=steps,
                      test=True,
                      gamma=CONFIG["GAMMA"])

    print(f"Loading agent from {load_dir}")
    agent.load(load_dir)

    def run_episode():
        if args.seed:
            wrapped_env.seed(args.seed)
        obs = wrapped_env.reset()
        done = False
        info = {}
        netr = 0

        states = []
        advantages = []

        for i in range(0, steps):
            if done or ("error" in info):
                break

            obs_array = np.array(obs)
            if np.split(np.array(obs), CONFIG["RAINBOW_HISTORY"])[-1].shape[0] == 4:
                for j, frame in enumerate(obs._frames):
                    obs._frames[j] = frame[:-1]

            action = agent.act(obs)

            save_obs(agent, obs_array, i, out_dir)

            states.append(cuda.to_cpu(agent.model.state))
            advantages.append(list(cuda.to_cpu(agent.model.advantage)))

            if action == 1:
                wrapped_env.env.env.env.env.env._skip = 24
            else:
                wrapped_env.env.env.env.env.env._skip = CONFIG["FRAME_SKIP"]

            obs, reward, done, info = wrapped_env.step(action)
            netr += reward

        agent.stop_episode()

        return np.array(states), np.array(advantages).T

    states, advantages = run_episode()

    def plot_trace(trace, label, path):
        np.savetxt(Path(path.parent, f"{label}-trace.txt"), trace)
        fig, ax = plt.subplots()
        ax.plot(trace, label=label)
        ax.legend()
        fig.savefig(path, dpi=600)
        plt.close(fig)

    for i, action in enumerate(advantages):
        plot_trace(action, CONFIG["ACTION_SPACE"][i], Path(out_dir, f"advantage_{CONFIG['ACTION_SPACE'][i]}.png"))

    plot_trace(states, "State", Path(out_dir, f"state_values.png"))

    wrapped_env.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpu", default=-1, action="store_const", const=0)
    parser.add_argument("--conf", "-f", help="Path to configuration file")
    parser.add_argument("--seed", type=int, help="Seed for MineRL environment")
    parser.add_argument("--steps", "-s", type=int, default=1000)
    parser.add_argument("--load", default="models/rainbow")

    main(parser.parse_args())
