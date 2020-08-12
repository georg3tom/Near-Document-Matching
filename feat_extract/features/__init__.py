"""
Make all the features accessible
"""

from .black import black
from .edge import edge
from .histogram import histogram
from .hog import hog
from .sift import sift
from .stats import stats
from .surf import surf

features = [
    black,
    histogram,
    stats,
]

experimental_features = [
    edge,
    hog,  # hog returns HUGE number of features, have to use it wisely
    sift,
    stats,
    surf,
]
