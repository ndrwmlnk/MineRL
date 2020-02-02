import json
import copy

FOUR_FRAMES_AGENT = {
    "train": {
        "RAINBOW_HISTORY": 4,
        "START_EPSILON": 1.0,
        "FINAL_EPSILON": 0.01,
        "DECAY_STEPS": 10 ** 6,
        "ALWAYS_KEYS": ['attack'],
        "REVERSE_KEYS": ['forward'],
        "EXCLUDE_KEYS": ['back', 'left', 'right', 'sneak', 'sprint']
    },
    "test": {"NOISY_NET_SIGMA": 0.5}
}

SINGLE_FRAME_AGENT = {
    "train": {
        "RAINBOW_HISTORY": 1,
        "START_EPSILON": 1.0,
        "FINAL_EPSILON": 0.25,
        "DECAY_STEPS": 10 ** 6,
        "ALWAYS_KEYS": ['attack'],
        "REVERSE_KEYS": ['forward'],
        "EXCLUDE_KEYS": ['back', 'left', 'right', 'sneak', 'sprint']
    },
    "test": {"NOISY_NET_SIGMA": 0.5}
}
SINGLE_FRAME_AGENT_ATTACK = {
    "train": {
        "RAINBOW_HISTORY": 1,
        "START_EPSILON": 1.0,
        "FINAL_EPSILON": 0.25,
        "DECAY_STEPS": 10 ** 6,
        "ALWAYS_KEYS": [],
        "REVERSE_KEYS": ['forward'],
        "EXCLUDE_KEYS": ['back', 'left', 'right', 'sneak', 'sprint']},
    "test": {"NOISY_NET_SIGMA": 0.5}
}


class Configuration(dict):
    def __init__(self, conf):
        Configuration.is_valid(conf)
        dict.__init__(self, {})
        self.apply(conf)

    @staticmethod
    def is_valid(conf):
        if "train" in conf:
            for k in ["RAINBOW_HISTORY",
                      "START_EPSILON",
                      "FINAL_EPSILON",
                      "DECAY_STEPS",
                      "ALWAYS_KEYS",
                      "REVERSE_KEYS",
                      "EXCLUDE_KEYS"]:
                if k not in conf["train"]:
                    raise ValueError(f"Training configuration doesn't contain a '{k}' key!")
            return True
        else:
            raise ValueError(f"Configuration doesn't contain a 'train' key!")

    def load(self, path):
        if path.is_file():
            with open(path) as fp:
                conf = json.load(fp)
            if Configuration.is_valid(conf):
                self.apply(conf)
        else:
            raise ValueError(f"{path} is not a file!")

    def apply(self, conf):
        self.trainc = copy.deepcopy(conf["train"])
        test = copy.deepcopy(conf["train"])
        if "test" in conf:
            test.update(conf["test"])
        self.testc = test
        self.update(self.trainc)

    def test(self):
        self.apply(self.testc)


CONFIG = Configuration(FOUR_FRAMES_AGENT)
