"""
gets an image, returns all extracted features
"""

import os

import cv2
import numpy as np
from .features import features

DEBUG = bool(os.getenv("DEBUG"))


class FeatureExtractor:
    """
    Given an image as input, it does a little preprocessing to make
    it compatible.

    It can calculate features after breaking the image into segments
    """

    def __init__(
        self,
        img,
        final_size=(300, 300),
        preprocess_config={"do_scale": True, "do_fix_channels": True},
        preprocess_enabled=True,
        window_size={"h": 4, "w": 4},
        window_stride={"h": 4, "w": 4},
    ):
        self.img = img
        self.window_size = window_size
        self.window_stride = window_stride
        self.final_size = final_size

        if preprocess_enabled:
            self.preprocess(**preprocess_config)

        self.features = []

    def preprocess(self, do_scale=True, do_fix_channels=True):
        """
        Make all images look mostly the same
        """
        self.log(">> preprocess")

        if do_scale:
            self.img = cv2.resize(
                self.img, self.final_size, interpolation=cv2.INTER_CUBIC
            )
            self.log(f"Resized image to {self.img.shape}")

        if do_fix_channels:
            self.log(f"Checking and fixing channels")
            if len(self.img.shape) == 2:
                # got a grayscale image
                self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
            else:
                # image is "3D", might have an extra channel
                n_channels = self.img.shape[-1]

                if n_channels < 3:
                    # if shape is (x, y, 1) or something like this
                    self.img = np.dstack([self.img * (3 - n_channels)])
                else:
                    # if it has a transparency channel as well
                    self.img = self.img[:, :, :3]

        self.log("<< preprocess")

    def break_blocks(self):
        """
        Breaks an image into blocks defined by window_size and window_stride
        """
        self.log(">> break_blocks")
        # a turn-off switch of sorts
        if self.window_size is None or self.window_stride is None:
            return [self.img]

        blocks = []

        h, w = self.img.shape[:2]

        for i in range(0, h - self.window_size["h"], self.window_stride["h"]):
            for j in range(0, w - self.window_size["w"], self.window_stride["w"]):
                blocks.append(
                    self.img[
                        i : i + self.window_size["h"], j : j + self.window_size["w"], :
                    ]
                )

        self.log("<< break_blocks")
        return blocks

    def get_features(self):
        """
        Extracts all the features
        """
        self.log(">> get_features")
        blocks = self.break_blocks()

        self.log(f"Image broken into {len(blocks)} blocks")
        self.log()

        feature_vector = []

        for b, block in enumerate(blocks):
            self.log(f"\033[FBlock {b}")
            for feature in features:
                self.log(f"> {feature.__name__}" + " " * 10, end="\r")
                feature_vector.append(feature(block).ravel())

        self.log("\n<< get_features")
        return np.hstack(feature_vector)

    @staticmethod
    def log(*args, **kwargs):
        """
        Prints stuff on the screen if logging is enabled
        Logging can be enabled by setting DEBUG environment variable
        """
        if DEBUG:
            print(*args, **kwargs)
