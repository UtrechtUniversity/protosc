#!/usr/bin/env python

from pathlib import Path
from protosc.io import read_image


stim_data_dir = Path("..", "data", "Nimstim faces")
grayscale = True

if __name__ == "__main__":
    open_files = stim_data_dir.glob("*_O.bmp")
    closed_files = stim_data_dir.glob("*_C.bmp")

    

    for x in open_files:
        print(read_image(x).shape)
#     print([x for x in open_files])
#     print([x for x in closed_files])
