import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


class Config:
    def __init__(self):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            self._cfg = json.load(f)

    @property
    def camera(self):
        return self._cfg["camera"]

    @property
    def pushup_counter(self):
        return self._cfg["pushup_counter"]

    @property
    def utility_weights(self):
        return self._cfg["utility_function"]["weights"]

    @property
    def logging(self):
        return self._cfg["logging"]

    @property
    def pose_model(self):
        return self._cfg["models"]["pose"]
