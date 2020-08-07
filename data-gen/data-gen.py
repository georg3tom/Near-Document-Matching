from os import listdir
from os.path import isfile, join, dirname, realpath, splitext
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
        altr = Alter(join(src,f))
        altr.rotate(randint(5,25)).write(directory=out)
        altr = Alter(join(src,f))
        altr.overlay().write(directory=out)
        altr = Alter(join(src,f))
        altr.scale().write(directory=out)
        altr = Alter(join(src,f))
        altr.affine_trans().write(directory=out)

        altr = Alter(join(src,f))
        altr.rotate(randint(5,25))
        altr.scale().write(directory=out)
        altr.overlay().write(directory=out)
        altr.affine_trans().write(directory=out)



