from os import listdir
from os.path import isfile, join, dirname, realpath
from random import randint
import sys

path = dirname(realpath(__file__)) + "/../"
sys.path.insert(0, path)
from alter import Alter


if __name__ == "__main__":
    src = "./data/img/"
    out = "./output/"

    files = [f for f in listdir(src) if isfile(join(src, f))]

    for f in files:
        altr = Alter(join(src, f))
        altr.salt().write(directory=out)

        altr = Alter(join(src, f))
        altr.pepper().write(directory=out)

        # altr = Alter(join(src, f))
        # altr.saltAndPepper().write(directory=out)
