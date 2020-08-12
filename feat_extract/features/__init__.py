"""
Make all the features accessible
"""

from .histogram import histogram
from .black import black
from .sift import sift
from .surf import surf
from .edge import edge

features = [
    black,
    histogram,
]

experimental_features = [
    sift,
    surf,
    edge,
]
