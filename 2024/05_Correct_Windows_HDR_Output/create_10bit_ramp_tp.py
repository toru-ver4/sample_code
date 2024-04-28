# -*- coding: utf-8 -*-
"""

==========

"""

# import standard libraries
import os
from pathlib import Path
import subprocess

# import third-party libraries
import numpy as np

# import my libraries
import test_pattern_generator2 as tpg
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

BIT_DEPTH = 10
NUM_OF_CODE_VALUE = (2 ** BIT_DEPTH)
MAXIMUM_CODE_VALUE = NUM_OF_CODE_VALUE - 1
COLOR_LIST = np.array([
    [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [1, 0, 1], [1, 1, 0], [0, 1, 1]
], dtype=np.uint16)


def calc_block_num_h(width=1920, block_size=64):
    return width // block_size


def calc_ramp_pattern_block_st_pos_with_color_idx(
        code_value=0, width=1920, block_size=64, color_kind_idx=0):
    color_block_height = calc_ramp_pattern_block_st_pos(
        code_value=MAXIMUM_CODE_VALUE, width=width, block_size=block_size)[1]\
        + block_size
    st_pos = calc_ramp_pattern_block_st_pos(
        code_value=code_value, width=width, block_size=block_size)
    st_pos = [st_pos[0], st_pos[1] + color_block_height * color_kind_idx]

    return st_pos


def calc_ramp_pattern_block_st_pos(
        code_value=0, width=1920, block_size=64):
    block_num_h = calc_block_num_h(width=width, block_size=block_size)
    st_pos_h = (code_value % block_num_h) * block_size
    st_pos_v = (code_value // block_num_h) * block_size
    st_pos = (st_pos_h, st_pos_v)

    return st_pos


def create_10bit_cms_test_pattern(
        width=3840, height=2160, block_size=32):
    img = np.zeros((height, width, 3), dtype=np.uint16)
    block_img_base = np.ones((block_size, block_size, 3), dtype=np.uint16)

    for color_kind_idx in range(len(COLOR_LIST)):
        color = COLOR_LIST[color_kind_idx]
        for code_value in range(NUM_OF_CODE_VALUE):
            block_img = block_img_base * code_value * color
            st_pos = calc_ramp_pattern_block_st_pos_with_color_idx(
                code_value=code_value, width=width, block_size=block_size,
                color_kind_idx=color_kind_idx
            )
            tpg.merge(img, block_img, st_pos)

    img = img / NUM_OF_CODE_VALUE
    fname = "./tp_img/tp_10bit_ramp_wrgbmyc.png"
    print(fname)
    tpg.img_wirte_float_as_16bit_int(fname, img)

    img_xyz = cs.rgb_to_large_xyz(
        rgb=img, color_space_name=cs.BT709)
    img_709_on_2020 = cs.large_xyz_to_rgb(
        xyz=img_xyz, color_space_name=cs.BT2020)
    fname = "./tp_img/tp_10bit_ramp_wrgbmyc_709_on_2020.png"
    print(fname)
    tpg.img_wirte_float_as_16bit_int(fname, img_709_on_2020)


def main_func():
    # width = 3840
    # height = 2160
    # block_size = 32
    # create_10bit_cms_test_pattern(
    #     width=width, height=height, block_size=block_size
    # )
    conv_to_avif()


def conv_to_avif():
    file_list = [
        "./tp_img/tp_10bit_ramp_wrgbmyc.png",
        "./tp_img/tp_10bit_ramp_wrgbmyc_709_on_2020.png"
    ]
    for png_fname in file_list:
        pp = Path(png_fname)
        ext = pp.suffix
        parent = str(pp.parent)
        avif_fname = "./" + parent + "/" + pp.stem + ".avif"

        cmd = [
            "avifenc", png_fname, "-d", "10", "-y", "444", "--cicp", "9/16/9",
            "-r", "full", "--min", "0", "--max", "0", "--ignore-exif", avif_fname
        ]
        subprocess.run(cmd)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
