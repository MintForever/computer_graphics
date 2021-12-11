import hw3
import os
from PIL import Image
import os
import math
import numpy as np
import sys

with open ('implemented.txt', 'r') as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
    for f in lines:
        # hw3(f)
        print('generating image for file '+f+'...')
        os.system('python hw3.py '+f)
