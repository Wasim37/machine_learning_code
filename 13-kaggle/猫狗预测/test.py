import glob
import os, sys
import random
from tqdm import tqdm
import gc

import numpy as np

import seaborn as sns
import pandas as pd

def gen_image_label(directory):
    ''' A generator that yields (label, id, jpg_filename) tuple.'''
    for root, dirs, files in os.walk(directory):
        for f in files:
            _, ext = os.path.splitext(f)
            if ext != '.jpg':
                continue
            basename = os.path.basename(f)
            splits = basename.split('.')
            if len(splits) == 3:
                label, id_, ext = splits
            else:
                label = None
                id_, ext = splits
            fullname = os.path.join(root, f)
            yield label, int(id_), fullname
            

train_data_dir = '../input_dog_cat/train'         
lst = list(gen_image_label(train_data_dir))
print(list)
print(gc.collect())
print("assa")