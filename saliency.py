# Original implementation
# https://github.com/chainer/chainerrl-visualizer/blob/master/chainerrl_visualizer/worker_jobs/saliency_job.py
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import color, img_as_float, data
from skimage.filters import gaussian

ACTIONS = {
    0: "FWD_ATT",
    1: "STOP",
    2: "JUMP",
    3: "LOOK_L",
    4: "LOOK_R"
}

SALIENCY_STD = 4.842388453060134
FUDGE_FACTOR = 255 / (1.5 * SALIENCY_STD)

LAST_ACTION = None

def mkdir_p(path):
    if not path.exists():
        path.mkdir()


def save_image(array, path):
    img = Image.fromarray(array)
    img.save(path)


OUT_PATH = Path(".", "saliency")


def chw_to_hwc(array):
    shape = array.shape
    if shape[0] == 3 and shape[2] == 64:  # CHW
        return array.transpose(1, 2, 0)
    else:
        return array


def observation_to_rgb(obs):
    obs = chw_to_hwc(obs)

    # Rescale observation
    obs_min = obs.min()
    obs_max = obs.max()
    obs_rgb = 255.0 * (obs - obs_min) / (obs_max - obs_min)

    return obs_rgb.astype('uint8')


def make_barplot(score, step, name="advantage"):
    barplts_dir = Path(OUT_PATH, f"{name}_barplots")
    mkdir_p(barplts_dir)

    # Thanks to https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
    labels = list(ACTIONS.values())
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    ax.bar(x, score, label=name)

    ax.set_title(f'{name} by action')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([-score.max(), score.max() + (score.max() / 10)])
    ax.legend()

    fig.tight_layout()
    fig.savefig(Path(barplts_dir, f'step{step}_{name}_by_action.png'), dpi=600)
    plt.close(fig)


def create_and_save_saliency_image(agent, obs, step, s_reward, reward, next_action, last_action):
    mkdir_p(OUT_PATH)
    obs_dir = Path(OUT_PATH, "observations")
    mkdir_p(obs_dir)

    obs = np.array(obs)
    RAINBOW_HISTORY = int(obs.shape[0] / 3)

    if not last_action:
        last_action = "None"

    suptitle = f"Step: {step} - Step reward: {s_reward} - Tot reward: {reward} - Last action: {last_action} - Next action {next_action}"

    squared_fig, squared_axs = plt.subplots(RAINBOW_HISTORY, len(ACTIONS.keys()))
    squared_fig.suptitle(suptitle)

    n_imgs = (len(ACTIONS.keys()) * 2)
    fig, axs = plt.subplots(RAINBOW_HISTORY, n_imgs + 1)
    fig.suptitle(suptitle)

    for i, img in enumerate(np.split(obs, RAINBOW_HISTORY)):
        base_img = observation_to_rgb(img)
        save_image(base_img, Path(obs_dir, f"step{step}_obs{i}.png"))
        axs[i][0].imshow(base_img)
        axs[i][0].axis('off')

        score, squared_score = score_frame(agent, obs, i, OUT_PATH, step, RAINBOW_HISTORY)
        s = score.sum(axis=2)
        make_barplot(s, step, "saliency")
        imgs = list(map(lambda a: advantages_on_base_image(a, base_img), score))

        for j, adv in enumerate(squared_score):
            squared_axs[0][j].set_title(ACTIONS[j])
            squared_axs[i][j].imshow(_saliency_on_base_image(adv, base_img))
            squared_axs[i][j].axis('off')

        for j in range(1, n_imgs, 2):
            k = int((j - 1) / 2)
            if ((j - 1) % 2) == 0:
                axs[0][j].set_title(ACTIONS[k])
            axs[i][j].imshow(imgs[k][1])
            axs[i][j].axis('off')
            axs[i][j + 1].imshow(imgs[k][0], cmap='gray', vmin=0, vmax=255)
            axs[i][j + 1].axis('off')

    scores_dir = Path(OUT_PATH, "saliency")
    mkdir_p(scores_dir)
    fig.savefig(Path(scores_dir, f'step{step}_saliency.png'), dpi=600)
    plt.close(fig)

    squared_dir = Path(OUT_PATH, "squared_saliency")
    mkdir_p(squared_dir)
    squared_fig.savefig(Path(squared_dir, f'step{step}_squared_saliency.png'), dpi=600)
    plt.close(squared_fig)


def _saliency_on_base_image(saliency, base_img, channel=0, sigma=0):
    # base_img shape is intended to be (height, width, channel)
    size = base_img.shape[0:2]  # height, width
    saliency_max = saliency.max()
    saliency = np.array(Image.fromarray(saliency).resize(size, resample=Image.BILINEAR)).astype(np.float32)

    if sigma != 0:
        saliency = gaussian(saliency, sigma=sigma)

    saliency -= saliency.min()
    saliency = FUDGE_FACTOR * saliency_max * saliency / saliency.max()

    img = base_img.astype("uint16")
    img[:, :, channel] += saliency.astype("uint16")
    return img.astype("uint8")


def paste_saliency_on_base_image(saliency, base_img):
    # https://stackoverflow.com/questions/9193603/applying-a-coloured-overlay-to-an-image-in-either-pil-or-imagemagik/9204506#9204506
    base_img = np.array(Image.fromarray(base_img, mode="RGB").convert("L")).astype(np.float32)
    alpha = 1
    # Construct RGB version of grey-level image
    img_color = np.dstack((base_img, base_img, base_img))

    # Convert the input image and color mask to Hue Saturation Value (HSV)
    # colorspace
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(saliency)

    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    return color.hsv2rgb(img_hsv)

# GRAYSCALE
def advantages_on_base_image(saliency, base_img, sigma=0):
    export_saliency_trace(saliency)
    red = 0
    blue = 2
    size = base_img.shape[0:2]  # height, width
    saliency = np.array(Image.fromarray(saliency).resize(size, resample=Image.BILINEAR)).astype(np.float32)
    if sigma != 0:
        saliency = gaussian(saliency, sigma=sigma)

    saliency = FUDGE_FACTOR * saliency
    map = np.zeros(base_img.shape).astype("uint16")

    for h in range(size[0]):
        for w in range(size[1]):
            pixel_saliency = saliency[h, w]
            if pixel_saliency > 0:
                map[h, w, red] = pixel_saliency
            elif pixel_saliency < 0:
                map[h, w, blue] = abs(pixel_saliency)

    return np.array(paste_saliency_on_base_image(map, base_img)).astype("uint8"), map.astype("uint8")


# RBG
def advantages_on_base_image_RGB(saliency, base_img, sigma=0):
    export_saliency_trace(saliency)
    red = 0
    blue = 2
    size = base_img.shape[0:2]  # height, width
    saliency = np.array(Image.fromarray(saliency).resize(size, resample=Image.BILINEAR)).astype(np.float32)
    if sigma != 0:
        saliency = gaussian(saliency, sigma=sigma)

    saliency = FUDGE_FACTOR * saliency
    base_img = base_img.astype("uint16")
    img = base_img.astype("uint16")
    img2 = np.ones(base_img.shape).astype("uint16")

    for h in range(size[0]):
        for w in range(size[1]):
            pixel_saliency = saliency[h, w]
            if pixel_saliency > 0:
                img[h, w, red] += pixel_saliency
                img2[h, w, red] = pixel_saliency
            elif pixel_saliency < 0:
                img[h, w, blue] += abs(pixel_saliency)
                img2[h, w, blue] = abs(pixel_saliency)

    return img.astype("uint8"), img2.astype("uint8")


def export_saliency_trace(saliency):
    saliency_npy = Path(OUT_PATH, "saliency-traces.txt")
    std_npy = Path(OUT_PATH, "std.txt")
    if saliency_npy.exists():
        saliency_traces = np.loadtxt(saliency_npy)
    else:
        saliency_traces = np.array([])
    try:
        max_saliency = saliency.max()
        min_saliency = saliency.min()
        print(f"saliency_max/saliency_min = {round(max_saliency)} / {round(min_saliency)}")
        print(f"Saliency std: {saliency_traces.std()}")
        saliency_traces = np.array([*saliency_traces, max_saliency, min_saliency])
        np.savetxt(saliency_npy, saliency_traces)
        np.savetxt(std_npy, [saliency_traces.std()])
    except:
        print("Saving saliency stddev...")
        np.savetxt(saliency_npy, saliency_traces)
        np.savetxt(std_npy, [saliency_traces.std()])
        print("Done")
        raise


def score_frame(agent, obs, idx, out_dir, step, n_frames, radius=5, density=10):
    size = obs.shape
    height = size[1]
    width = size[2]

    agent.model(agent.batch_states([obs], agent.xp, agent.phi))
    advantage_values = agent.model.advantage

    advantages_squared = np.zeros(
        (int(width / density) + 1, int(height / density) + 1, advantage_values.shape[0]))
    advantages_scores = np.zeros(
        (int(width / density) + 1, int(height / density) + 1, advantage_values.shape[0]))

    for i in range(0, height, density):
        for j in range(0, width, density):
            pix_dir = Path(out_dir, f"{i}-{j}")
            mask_path = Path(pix_dir, f"mask.png")
            mkdir_p(pix_dir)

            mask = _get_mask([i, j], size=[height, width], radius=radius)

            if not mask_path.exists():
                save_image(observation_to_rgb(mask), mask_path)
                save_image(observation_to_rgb(1 - mask), Path(pix_dir, f"one_minus_mask.png"))

            perturbed_imgs = occlude_ith_frame(obs, idx, mask, pix_dir, step, n_frames)

            agent.model(agent.batch_states([perturbed_imgs.astype(np.float32)], agent.xp, agent.phi))
            perturbed_advantage = agent.model.advantage
            diff = advantage_values - perturbed_advantage
            advantages_squared[int(j / density), int(i / density)] = np.power(diff, 2)
            advantages_scores[int(j / density), int(i / density)] = diff

    def resize_and_interpolate(scores):
        max_score = scores.max()
        scores = np.array(Image.fromarray(scores).resize([height, width], resample=Image.BILINEAR)).astype(np.float32)
        return max_score * scores / scores.max()

    # Index scores by action
    advantages_scores = np.swapaxes(advantages_scores, 0, 2)
    advantages_squared = np.swapaxes(advantages_squared, 0, 2)
    return map(resize_and_interpolate, advantages_scores), map(resize_and_interpolate, advantages_squared)


def _get_mask(center, size, radius=5):
    y, x = np.ogrid[-center[0]:size[0] - center[0], -center[1]:size[1] - center[1]]
    keep = (x * x + y * y) / (x * x + y * y).max() >= 0.01
    msk = np.zeros(size)
    msk[keep] = 1  # select a circle of pixels
    msk = gaussian(msk, sigma=radius)  # blur circle of pixels
    return msk


def occlude_single_frame(frame, msk, out_dir, step, radius=3):
    def export_image_for_obs(img, name):
        act_dir = Path(out_dir, name)
        mkdir_p(act_dir)
        save_image(observation_to_rgb(img),
                   Path(act_dir, f"step{step}_{name}.png"))

    img_msk = frame * msk
    export_image_for_obs(img_msk, "obs_dot_mask")

    gf = gaussian(chw_to_hwc(frame), sigma=radius, multichannel=True)
    export_image_for_obs(gf, "gaussian_filter")

    gf_one_mmask = gf.transpose(2, 0, 1) * (1 - msk)
    export_image_for_obs(gf_one_mmask, "gaussian_filter_times_one_minus_mask")

    p_img = img_msk + gf_one_mmask
    export_image_for_obs(p_img, "perturbed_obs")
    return p_img


def occlude_ith_frame(frames, i, msk, out_dir, step, n_frames):
    l_frames = np.split(frames, n_frames)
    before = l_frames[:i]
    frame = l_frames[i]
    after = l_frames[i + 1:]
    return np.concatenate([*before,
                           occlude_single_frame(frame, msk, out_dir, f"{step}_obs{i}"),
                           *after])
