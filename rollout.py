# Generate observations rollout
# noinspection PyUnresolvedReferences
import minerl

import os
import re
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
from saliency import save_image, make_barplot, create_and_save_saliency_image, observation_to_rgb
from utility.config import CONFIG, DOUBLE_FRAME_AGENT_ATTACK_AND_FORWARD

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


def save_obs(agent, obs, step, reward, netr, action, last_action, out_dir, sal_std=None, size=(256, 256), export=True, sal_export=True):
    adv_dir = Path(out_dir, "advantage")
    tot_adv_dir = Path(out_dir, "total_advantage")
    state_dir = Path(out_dir, "state")

    out_dir.mkdir(parents=True, exist_ok=True)

    value = cuda.to_cpu(agent.model.state)
    q_values = cuda.to_cpu(agent.model.q_values)
    if sal_export:
        make_barplot(cuda.to_cpu(agent.model.advantage), step)

    rollout = create_and_save_saliency_image(agent, obs, step, reward, netr, action, last_action, sal_std=sal_std, export=sal_export)

    def export_obs(o, name, adv, state, tot_adv):
        save_image(overlay_q_values(observation_to_rgb(o), q_values),
                   Path(out_dir, name), size)
        if sal_export:
            adv_dir.mkdir(parents=True, exist_ok=True)
            tot_adv_dir.mkdir(parents=True, exist_ok=True)
            state_dir.mkdir(parents=True, exist_ok=True)

            save_image(overlay_state_value(observation_to_rgb(state), value),
                       Path(state_dir, name), size)
            save_image(merge_images([a for a in adv]),
                       Path(adv_dir, name), size)
            save_image(tot_adv,
                       Path(tot_adv_dir, name), size)

    if export:
        if step == 0:
            for i, o in enumerate(np.split(np.array(obs), CONFIG["RAINBOW_HISTORY"])):
                name = f"{str(i).zfill(4)}.png"
                export_obs(o, name, *rollout[i])

        else:
            o = np.split(np.array(obs), CONFIG["RAINBOW_HISTORY"])[-1]
            name = f"{str(step + CONFIG['RAINBOW_HISTORY'] - 1).zfill(4)}.png"
            export_obs(o, name, *rollout[-1])


def main(args):
    """
    This function will be called for training phase.
    """
    chainerrl.misc.set_random_seed(0)
    CONFIG.apply(DOUBLE_FRAME_AGENT_ATTACK_AND_FORWARD)
    start_epsilon = None
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
    sal_std = 4.5

    if args.test:
        pattern = re.compile(r"_(\d+)_")
        out_dir = Path(OUT_DIR, "test", pattern.search(load_dir.name).group(1))
    else:
        pattern = re.compile(r"^(ep_\d+)_")
        out_dir = Path(OUT_DIR, "train", pattern.search(load_dir.name).group(1))
        # Compute START_EPSILON
        ep_name = load_dir.name
        pattern = re.compile(r"^ep_(\d+)_forest(\d+)")
        m = pattern.search(ep_name)
        ep_idx = int(m.group(1))
        forest = int(m.group(2))
        expl = chainerrl.explorers.LinearDecayEpsilonGreedy(CONFIG["START_EPSILON"],
                                                            CONFIG["FINAL_EPSILON"],
                                                            steps,
                                                            wrapped_env.action_space.sample)
        start_epsilon = expl.compute_epsilon((ep_idx * args.steps) + 1)
        print(f"Initial epsilon: {start_epsilon}, Saliency std: {sal_std}, Forest seed: {forest}")
        wrapped_env.seed(forest)

    agent = get_agent(n_actions=wrapped_env.action_space.n,
                      n_input_channels=wrapped_env.observation_space.shape[0],
                      explorer_sample_func=wrapped_env.action_space.sample,
                      gpu=args.gpu,
                      steps=steps,
                      start_eps=start_epsilon,
                      test=args.test,
                      gamma=0.95)

    print(f"Loading agent from {load_dir}")
    agent.load(load_dir)

    def run_episode(export):
        wrapped_env.seed(420)
        obs = wrapped_env.reset()
        done = False
        last_action = None
        info = {}
        reward = 0
        netr = 0

        states = []
        advantages = []

        for i in range(steps):
            if done or ("error" in info):
                break

            if args.test:
                action = agent.act(obs)
            else:
                action = agent.act_and_train(obs, reward)

            save_obs(agent, obs, i, reward, netr, CONFIG["ACTION_SPACE"][action], last_action, out_dir, sal_std, export=export)
            last_action = CONFIG["ACTION_SPACE"][action]

            states.append(cuda.to_cpu(agent.model.state))
            advantages.append(list(cuda.to_cpu(agent.model.advantage)))

            if action == 1:
                wrapped_env.env.env.env.env.env._skip = 24
            else:
                wrapped_env.env.env.env.env.env._skip = CONFIG["FRAME_SKIP"]

            obs, reward, done, info = wrapped_env.step(action)
            netr += reward

        if args.test:
            agent.stop_episode()
        else:
            agent.stop_episode_and_train(obs, reward, done)

        return np.array(states), np.array(advantages).T

    run_episode(export=False)
    sal_dir = Path(".", "saliency")
    p = Path(sal_dir, "std.txt")
    sal_std = float(np.loadtxt(p))
    try:
        shutil.rmtree(sal_dir)
    except OSError as e:
        print(f"Error: {sal_dir} : {e.strerror}")
        sys.exit(1)

    states, advantages = run_episode(export=True)

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
    ARCH_JAVA_PATH = Path("/usr/lib/jvm/java-8-openjdk")
    if ARCH_JAVA_PATH.is_dir():
        os.environ["JAVA_HOME"] = str(ARCH_JAVA_PATH)
    parser = ArgumentParser()
    parser.add_argument("--gpu", default=-1, action="store_const", const=0)
    parser.add_argument("--steps", "-s", default=1000)
    parser.add_argument("--test", default=False, action="store_true")
    parser.add_argument("--load", default="models/rainbow")

    main(parser.parse_args())
