"""
Make all the features accessible
"""

from .histogram import histogram
from .black import black
from .sift import sift
from .surf import surf
from .edge import edge
from .hog import hog

features = [
    black,
    histogram,
]

experimental_features = [
    sift,
    surf,
    edge,
    hog,  # hog returns HUGE number of features, have to use it wisely
]
