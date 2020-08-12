"""
Returns the edges detected in an image
"""

import cv2
import numpy as np


def edge(img):
    """
    Returns the canny edges detected in the image
    Expects an RGB image (3 channels)
    """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return cv2.Canny(img, 100, 200)


if __name__ == "__main__":
    img = cv2.imread("./test.png")
    print("Loaded image")

    h = edge(img)
    print("Calculation done")

    cv2.imshow("EDGE DETECTION | Press q to quit", h)

    while 1:
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
