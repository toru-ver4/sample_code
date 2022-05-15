# -*- coding: utf-8 -*-
"""
"""

# import standard libraries
import os
from multiprocessing import Pool, cpu_count
from re import U

# import third-party libraries
import numpy as np
from colour.io import write_image, read_image

# import my libraries
from common import MeasureExecTime

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def get_src_fname(idx):
    src_dir = "/work/overuse/2022/Rayearth/src/"
    fname_base = f"Rayearth_1440x1080_{idx:08d}.tif"
    src_fname = src_dir + fname_base

    return src_fname


def get_dst_fname(idx):
    dst_dir = "/work/overuse/2022/Rayearth/dst/"
    fname_base = f"Interlaced_Rayearth_{idx:08d}.tif"
    dst_fname = dst_dir + fname_base

    return dst_fname


def create_interlace_mask(output_idx):
    width = 1440
    height = 1080

    zero_2px_line = np.zeros((width * 2))
    one_2px_line = np.ones((width * 2))
    if (output_idx % 4) == 0:
        unit = np.vstack([zero_2px_line, one_2px_line]).reshape(1, -1)
        mono_mask = np.repeat(
            unit, height//4, axis=0).reshape(height, width, 1)
        mask_img = np.dstack([mono_mask, mono_mask, mono_mask])
    elif (output_idx % 4) == 2:
        unit = np.vstack([one_2px_line, zero_2px_line]).reshape(1, -1)
        mono_mask = np.repeat(
            unit, height//4, axis=0).reshape(height, width, 1)
        mask_img = np.dstack([mono_mask, mono_mask, mono_mask])
    else:
        mask_img = np.zeros((height, width, 3))

    return mask_img


def add_interlace(idx=0):
    output_idx_list = [idx*2, idx*2 + 1]
    src_img = read_image(get_src_fname(idx))

    for output_idx in output_idx_list:
        mask_img = create_interlace_mask(output_idx)
        dst_img = src_img * mask_img
        dst_fname = get_dst_fname(output_idx)
        print(dst_fname)
        write_image(dst_img, dst_fname, 'uint8')


def debug_func():
    # even_img = create_interlace_mask(output_idx=0)
    # write_image(even_img, './even.tif', 'uint8')
    # odd_img = create_interlace_mask(output_idx=2)
    # write_image(odd_img, './odd.tif', 'uint8')
    total_frame = 5515

    for idx in range(total_frame):
        add_interlace(idx)
        # if idx > 4:
        #     break


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug_func()
    # main_func()
