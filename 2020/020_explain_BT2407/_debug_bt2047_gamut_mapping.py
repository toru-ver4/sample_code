# -*- coding: utf-8 -*-
"""
BT2407 実装用の各種LUTを作成する
===============================

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from multiprocessing import Pool, cpu_count, Array

# import my libraries


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def lch_to_lab(lch):
    shape_bak = lch.shape
    aa = lch[..., 1] * np.cos(lch[..., 2])
    bb = lch[..., 1] * np.sin(lch[..., 2])
    return np.dstack((lch[..., 0], aa, bb)).reshape(shape_bak)


def print_blog_param():
    """
    ブログ記載用のパラメータを吐く
    """
    lch_40 = np.array([68, 100, np.deg2rad(40)])
    lch_270 = np.array([40, 60, np.deg2rad(270)])
    lab_40 = lch_to_lab(lch_40)
    lab_270 = lch_to_lab(lch_270)
    print(lab_40)
    print(lab_270)


def main_func():
    print_blog_param()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
