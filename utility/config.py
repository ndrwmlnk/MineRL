import json
import copy

# Action 0: OrderedDict([('forward', 1), ('back', 0), ('left', 0), ('right', 0), ('jump', 0), ('sneak', 0), ('sprint', 0), ('attack', 1), ('camera', array([0., 0.], dtype=float32))])
# Action 1: OrderedDict([('forward', 0), ('back', 0), ('left', 0), ('right', 0), ('jump', 0), ('sneak', 0), ('sprint', 0), ('attack', 1), ('camera', array([0., 0.], dtype=float32))])
# Action 2: OrderedDict([('forward', 1), ('back', 0), ('left', 0), ('right', 0), ('jump', 1), ('sneak', 0), ('sprint', 0), ('attack', 1), ('camera', array([0., 0.], dtype=float32))])
# Action 3: OrderedDict([('forward', 1), ('back', 0), ('left', 0), ('right', 0), ('jump', 0), ('sneak', 0), ('sprint', 0), ('attack', 1), ('camera', array([  0., -10.], dtype=float32))])
# Action 4: OrderedDict([('forward', 1), ('back', 0), ('left', 0), ('right', 0), ('jump', 0), ('sneak', 0), ('sprint', 0), ('attack', 1), ('camera', array([ 0., 10.], dtype=float32))])
FOUR_FRAMES_AGENT = {
    "train": {
        "RAINBOW_HISTORY": 4,
        "START_EPSILON": 1.0,
        "FINAL_EPSILON": 0.01,
        "ALWAYS_KEYS": ['attack'],
        "REVERSE_KEYS": ['forward'],
        "EXCLUDE_KEYS": ['back', 'left', 'right', 'sneak', 'sprint'],
        "FRAME_SKIP": 4,
        "EXCLUDE_NOOP": False,
        "MAX_CAMERA_RANGE": 10.0,
        "ACTION_SPACE": {
            0: "FWD_ATTACK",
            1: "STOP",
            2: "FWD_JUMP",
            3: "FWD_LEFT",
            4: "FWD_RIGHT",
        }
    },
    "test": {"NOISY_NET_SIGMA": 0.5}
}

DOUBLE_FRAME_AGENT_ATTACK_AND_FORWARD = {
    "train": {
        "RAINBOW_HISTORY": 2,
        "GAMMA": 0.93,
        "START_EPSILON": 0.99,
        "FINAL_EPSILON": 0.0,
        "ALWAYS_KEYS": ['forward'],
        "REVERSE_KEYS": [],
        "EXCLUDE_KEYS": ['back', 'left', 'right', 'sneak', 'sprint'],
        "FRAME_SKIP": 12,
        "EXCLUDE_NOOP": True,
        "MAX_CAMERA_RANGE": 1.875,
        "ACTION_SPACE": {
            0: "FWD_JUMP",
            1: "ATT",
            2: "LEFT",
            3: "RIGHT",
        }
    },
    "test": {"NOISY_NET_SIGMA": 0.5}
}


SINGLE_FRAME_AGENT_ATTACK_AND_FORWARD = {
    "train": {
        "RAINBOW_HISTORY": 1,
        "START_EPSILON": 0.99,
        "FINAL_EPSILON": 0.0,
        "ALWAYS_KEYS": ['forward'],
        "REVERSE_KEYS": [],
        "EXCLUDE_KEYS": ['back', 'left', 'right', 'sneak', 'sprint'],
        "FRAME_SKIP": 12,
        "EXCLUDE_NOOP": True,
        "MAX_CAMERA_RANGE": 1.875,
        "ACTION_SPACE": {
            0: "FWD_JUMP",
            1: "ATT",
            2: "LEFT",
            3: "RIGHT",
        }
    },
    "test": {"NOISY_NET_SIGMA": 0.5}
}


class Configuration(dict):
    def __init__(self, conf):
        Configuration.is_valid(conf)
        dict.__init__(self, {})
        self.trainc = {}
        self.testc = {}
        self.apply(conf)

    @staticmethod
    def is_valid(conf):
        if "train" in conf:
            for k in ["RAINBOW_HISTORY",
                      "GAMMA",
                      "START_EPSILON",
                      "FINAL_EPSILON",
                      "ALWAYS_KEYS",
                      "REVERSE_KEYS",
                      "EXCLUDE_KEYS",
                      "FRAME_SKIP",
                      "EXCLUDE_NOOP",
                      "MAX_CAMERA_RANGE"]:
                if k not in conf["train"]:
                    raise ValueError(f"Current configuration doesn't contain a '{k}' key!")
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
        self.trainc = {}
        self.testc = {}
        self.trainc = copy.deepcopy(conf["train"])
        test = copy.deepcopy(conf["train"])
        if "test" in conf:
            test.update(conf["test"])
        self.testc = test
        self.update(self.trainc)

    def test(self):
        self.update(self.testc)

    def train(self):
        self.update(self.trainc)


CONFIG = Configuration(DOUBLE_FRAME_AGENT_ATTACK_AND_FORWARD)
