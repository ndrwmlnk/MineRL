import json
from pathlib import Path


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
