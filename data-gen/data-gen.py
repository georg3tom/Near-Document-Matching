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
        altr.rotate(randint(5,25)).write(join(out,splitext(f)[0]+"-r"))
        altr = Alter(join(src,f))
        altr.overlay().write(join(out,splitext(f)[0]+"-o"))
        altr = Alter(join(src,f))
        altr.scale().write(join(out,splitext(f)[0]+"-s"))
        altr = Alter(join(src,f))
        altr.affine_trans().write(join(out,splitext(f)[0]+"-at"))

        altr = Alter(join(src,f))
        altr.rotate(randint(5,25))
        altr.scale().write(join(out,splitext(f)[0]+"-r-s"))
        altr.overlay().write(join(out,splitext(f)[0]+"-r-s-o"))
        altr.affine_trans().write(join(out,splitext(f)[0]+"r-s-o-at"))



