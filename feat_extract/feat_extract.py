"""
gets an image, returns all extracted features
"""

import cv2
import numpy as np
from .features import features


class FeatureExtractor:
    """
    Given an image as input, it does a little preprocessing to make
    it compatible.

    It can calculate features after breaking the image into segments
    """

    def __init__(
        self,
        img,
        segments=2,
        final_size=(300, 300),
        self.window = {
            "h" : 4,
            "w" : 4,
            }
        preprocess_config={"do_scale": True, "do_fix_channels": True},
        preprocess_enabled=True,
    ):
        self.img = img
        self.segments = segments
        self.final_size = final_size

        if preprocess_enabled:
            self.preprocess(**preprocess_config)

        self.features = []

    def preprocess(self, do_scale=True, do_fix_channels=True):
        """
        Make all images look mostly the same
        """
        if do_scale:
            self.img = cv2.resize(
                self.img, self.final_size, interpolation=cv2.INTER_CUBIC
            )

        if do_fix_channels:
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

    def break_blocks(self):
        """
        Breaks an image into the specified number of segments
        """
        # a turn-off switch of sorts
        if self.segments <= 1:
            return [self.img]

        x = self.img.shape[0] // self.segments
        y = self.img.shape[1] // self.segments

        blocks = []

        for i in range(0, self.segments - 1):
            for j in range(0, self.segments - 1):
                blocks.append(self.img[i * x : (i + 1) * x, j * y : (j + 1) * y, :])

            blocks.append(self.img[i * x : (i + 1) * x, (self.segments - 1) * y :, :])

        for j in range(0, self.segments - 1):
            blocks.append(self.img[(self.segments - 1) * x :, j * y : (j + 1) * y, :])

        blocks.append(self.img[(self.segments - 1) * x :, (self.segments - 1) * y :, :])

        return blocks

    def get_features(self):
        """
        Extracts all the features
        """
        # blocks = self.break_blocks()
        feature_vector = np.zeros((0))

        x = self.img.shape[0] // self.segments
        y = self.img.shape[1] // self.segments
        
        for i in range(0, x - self.window["h"]):
            for j in range(0, y - self.window["w"]):
                block = self.img[i: i+self, window["h"],j:j+self.window["w"], :]
                for feature in features:
                    feature_vector = np.hstack([feature_vector, feature(block).ravel()])

        return feature_vector

    def get_features2(self):
        """
        Extracts all the features
        """
        blocks = self.break_blocks()
        feature_vector = np.zeros((0))

        x = self.img.shape[0] // self.segments
        y = self.img.shape[1] // self.segments
        
        for block in blocks:
            for feature in features:
                feature_vector = np.hstack([feature_vector, feature(block).ravel()])

        return feature_vector
