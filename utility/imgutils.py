import numpy as np
from PIL import Image


def save_image(array, path, size=None):
    if isinstance(array, Image.Image):
        img = array
    else:
        img = Image.fromarray(array)
    if size:
        img.resize(size, resample=Image.BILINEAR)
    img.save(path)


def chw_to_hwc(array):
    shape = array.shape
    if shape[0] == 3 and shape[2] == 64:  # CHW
        return array.transpose(1, 2, 0)
    else:
        return array


def remove_depth(obs, history):
    if np.split(np.array(obs), history)[-1].shape[0] == 4:
        for j, frame in enumerate(obs._frames):
            obs._frames[j] = frame[:-1]
    return obs


def get_depth(obs):
    if obs.shape[0] == 4:
        obs = obs[-1]
        obs_min = obs.min()
        obs_max = obs.max()
        obs = 255.0 * (obs - obs_min) / (obs_max - obs_min)

        return obs.astype('uint8')
    else:
        raise ValueError(f"Observation has no depth value. Shape: {obs.shape}")


def observation_to_rgb(obs):
    if obs.shape[0] == 3:
        obs = chw_to_hwc(obs)
    elif obs.shape[0] == 4:
        obs = chw_to_hwc(obs[:-1])

    # Rescale observation
    obs_min = obs.min()
    obs_max = obs.max()
    obs_rgb = 255.0 * (obs - obs_min) / (obs_max - obs_min)

    return obs_rgb.astype('uint8')
