# -*- coding: utf-8 -*-

# import standard libraries
import os

# import third-party libraries
import numpy as np

# import my libraries

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    base_luminance = 203
    peak_luminance_list = [203, 400, 600, 1000, 2000, 4000, 10000]
    for peak_luminance in peak_luminance_list:
        print(f"{np.log2(peak_luminance / base_luminance):.3f}")
