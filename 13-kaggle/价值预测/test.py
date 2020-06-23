import numpy as np
import pandas as pd
import matplotlib as plt
import os, sys, time

import warnings
warnings.filterwarnings('ignore')

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


def main(debug = False):
    num_rows = 1000 if debug else None
    train = pd.read_csv('../input_Santander_Value/train.csv')
    test = pd.read_csv('../input_Santander_Value/test.csv')    
    

if __name__ == "__main__":
    submission_file_name = "submission_kerne_wasim.csv"
    with timer('Full model run'):
        main(debug=True)