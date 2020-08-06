"""
Makes changes to an image
"""

from copy import deepcopy
from random import randint
import os

import numpy as np
import cv2

from .errors import ImageNotFoundError, InvalidImageError


class Alter:
    """
    Used to introduce some variations in an image
    """

    def __init__(self, img_path):
        self.img_path = img_path

        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)

        self.edits = [name]
        self.ext = ext

        self.img = self.imread(img_path)

    @staticmethod
    def imread(img_path):
        """
        Raises exception if file doesn't exist or is invalid
        Returns the image if valid
        """
        if not os.path.exists(img_path):
            raise ImageNotFoundError(f"Image {img_path} could'nt be located")

        img = cv2.imread(img_path)

        if img is None:
            raise InvalidImageError(f"Image {img_path} could'nt be loaded")

        return img

    def rotate(self, angle):
        """
        Rotates image by angle
        """
        image_center = np.array(self.img.shape[1::-1]) / 2
        rot_mat = cv2.getRotationMatrix2D(tuple(image_center), angle, 1.0)

        self.img = cv2.warpAffine(
            self.img, rot_mat, self.img.shape[1::-1], flags=cv2.INTER_LINEAR
        )

        self.edits.append(f"rotate:{angle}")
        return self

    def scale(self, fx=0.5, fy=0.5):
        """
        Scales the image
        """
        self.img = cv2.resize(
            self.img, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC
        )

        self.edits.append(f"scale:{fx}x{fy}")
        return self

    def affine_trans(self):
        """
        Applies some affine transformation to the image
        """
        h, w, c = self.img.shape

        # pts1 = np.float32([[randint(0,rows),randint(0,cols)],[randint(0,rows),randint(0,cols)],[randint(0,rows),randint(0,cols)]])
        # pts2 = np.float32([[randint(0,rows),randint(0,cols)],[randint(0,rows),randint(0,cols)],[randint(0,rows),randint(0,cols)]])
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

        M = cv2.getAffineTransform(pts1, pts2)

        self.img = cv2.warpAffine(self.img, M, (w, h))

        self.edits.append("affine")
        return self

    @staticmethod
    def overlay_transparent(bg_img, overlay_img, x, y):
        """
        Puts an image on top of another
        used by overlay function
        """
        bg_h, bg_w = bg_img.shape[:2]

        if x >= bg_w or y >= bg_h:
            return bg_img

        h, w = overlay_img.shape[:2]

        if x + w > bg_w:
            w = bg_w - x
            overlay_img = overlay_img[:, :w]

        if y + h > bg_h:
            h = bg_h - y
            overlay_img = overlay_img[:h]

        if overlay_img.shape[2] < 4:
            overlay_img = np.concatenate(
                [
                    overlay_img,
                    np.ones((*overlay_img.shape[:2], 1), dtype=overlay_img.dtype) * 255,
                ],
                axis=2,
            )

        overlay = overlay_img[..., :3]
        mask = overlay_img[..., 3:] / 255.0

        ret = bg_img.copy()

        ret[y : y + h, x : x + w] = (1.0 - mask) * ret[
            y : y + h, x : x + w
        ] + mask * overlay

        return ret

    def overlay(self, img2_path="./hurr.png"):
        """
        Puts img2 on top of current image
        """
        img2 = self.imread(img2_path)

        self.img = self.overlay_transparent(self.img, img2, 0, 0)

        self.edits.append(f"overlay:{os.path.basename(img2_path)}")
        return self

    def copy(self):
        """
        Returns a copy of this
        """
        return deepcopy(self)

    def write(self, name=None, ext=None):
        """
        Writes the image to disk
        """
        if name is None:
            name = "-".join(self.edits)
        if ext is None:
            ext = self.ext

        cv2.imwrite(name + ext, self.img)


if __name__ == "__main__":
    Alter("./inp.jpg").rotate(25).overlay().scale().affine_trans().write()
