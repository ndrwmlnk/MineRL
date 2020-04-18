import json
from pathlib import Path

# Action 0: OrderedDict([('forward', 1), ('back', 0), ('left', 0), ('right', 0), ('jump', 0), ('sneak', 0), ('sprint', 0), ('attack', 1), ('camera', array([0., 0.], dtype=float32))])
# Action 1: OrderedDict([('forward', 0), ('back', 0), ('left', 0), ('right', 0), ('jump', 0), ('sneak', 0), ('sprint', 0), ('attack', 1), ('camera', array([0., 0.], dtype=float32))])
# Action 2: OrderedDict([('forward', 1), ('back', 0), ('left', 0), ('right', 0), ('jump', 1), ('sneak', 0), ('sprint', 0), ('attack', 1), ('camera', array([0., 0.], dtype=float32))])
# Action 3: OrderedDict([('forward', 1), ('back', 0), ('left', 0), ('right', 0), ('jump', 0), ('sneak', 0), ('sprint', 0), ('attack', 1), ('camera', array([  0., -10.], dtype=float32))])
# Action 4: OrderedDict([('forward', 1), ('back', 0), ('left', 0), ('right', 0), ('jump', 0), ('sneak', 0), ('sprint', 0), ('attack', 1), ('camera', array([ 0., 10.], dtype=float32))])
FOUR_FRAMES_AGENT = {
    "RAINBOW_HISTORY": 4,
    "GAMMA": 0.93,
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
}


# Action 0: OrderedDict([('forward', 1), ('back', 0), ('left', 0), ('right', 0), ('jump', 1), ('sneak', 0), ('sprint', 0), ('attack', 0), ('camera', array([0., 0.], dtype=float32))])
# Action 1: OrderedDict([('forward', 0), ('back', 0), ('left', 0), ('right', 0), ('jump', 0), ('sneak', 0), ('sprint', 0), ('attack', 1), ('camera', array([0., 0.], dtype=float32))])
# Action 2: OrderedDict([('forward', 0), ('back', 0), ('left', 0), ('right', 0), ('jump', 0), ('sneak', 0), ('sprint', 0), ('attack', 0), ('camera', array([ 0.   , -1.875], dtype=float32))])
# Action 3: OrderedDict([('forward', 0), ('back', 0), ('left', 0), ('right', 0), ('jump', 0), ('sneak', 0), ('sprint', 0), ('attack', 0), ('camera', array([0.   , 1.875], dtype=float32))])
DOUBLE_FRAME_AGENT_ATTACK_AND_FORWARD = {
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
}


class Configuration(dict):
    def __init__(self, conf=None):
        if conf is not None and Configuration.is_valid(conf):
            dict.__init__(self, conf)
        else:
            dict.__init__(self, {})

    @staticmethod
    def is_valid(conf):
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
            if k not in conf:
                raise ValueError(f"Current configuration doesn't contain a '{k}' key!")
        return True

    def load(self, path: Path):
        if path.is_file():
            with open(path) as fp:
                conf = json.load(fp)
            if Configuration.is_valid(conf):
                self.apply(conf)
        else:
            raise ValueError(f"{path} is not a file!")

    def dump(self, path: Path):
        if path.parent.is_dir():
            with open(path, "w") as fp:
                json.dump(self, fp, indent=2)
        else:
            raise ValueError(f"{path.parent} is not a directory!")

    def apply(self, conf):
        self.update(conf)


CONFIG = Configuration()