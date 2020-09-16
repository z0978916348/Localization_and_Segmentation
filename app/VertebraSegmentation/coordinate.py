import os
from os import mkdir, listdir
from os.path import splitext, exists, join

import torch
import torch.nn as nn
def get_information(filename, dir):

    name = splitext(filename)[0]

    f = open(f"{dir}/{name}.txt", 'r')
    
    lines = f.readlines()
    
    coordinate = []
    
    # x1, y1, x2, y2, box_w, box_h

    for id, line in enumerate(lines):
        line = line.replace('\n', '')
        line = line.split(' ')
        coordinate.append(line)
        

    return coordinate



