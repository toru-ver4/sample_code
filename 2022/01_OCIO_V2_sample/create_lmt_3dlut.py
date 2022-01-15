# -*- coding: utf-8 -*-
"""
create 3dlut for LMT
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour import write_LUT, LUT3D

# import my libraries
import transfer_functions as tf


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def apply_matrix(src, mtx):
    shape_bak = src.shape
    a = src[..., 0]*mtx[0][0] + src[..., 1]*mtx[0][1] + src[..., 2]*mtx[0][2]
    b = src[..., 0]*mtx[1][0] + src[..., 1]*mtx[1][1] + src[..., 2]*mtx[1][2]
    c = src[..., 0]*mtx[2][0] + src[..., 1]*mtx[2][1] + src[..., 2]*mtx[2][2]

    return np.dstack([a, b, c]).reshape(shape_bak)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    grid_num = 33
    mtx = np.array(
        [[0.6, 0.4, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.4, 0.6]]
    )
    work_shape = (grid_num ** 3, 3)
    lut_3d_shape = (grid_num, grid_num, grid_num, 3)
    st2084_rgb = LUT3D.linear_table(grid_num).reshape(work_shape)
    linear_rgb = tf.eotf(st2084_rgb, tf.ST2084)
    linear_rgb = apply_matrix(linear_rgb, mtx)
    linear_rgb = linear_rgb ** 1.2
    st2084_rgb = tf.oetf(
        np.clip(linear_rgb, 0.0, 1.0), tf.ST2084)

    lut3d = LUT3D(
        table=st2084_rgb.reshape(lut_3d_shape), name="LMT sample")
    file_name = "./luts/greenish_gamma1.2_rev8.cube"
    write_LUT(lut3d, file_name)
