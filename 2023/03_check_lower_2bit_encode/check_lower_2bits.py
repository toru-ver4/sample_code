# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np

# import my libraries
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2023 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def calc_st_pos_rectangle_patch(
        width, height, block_length, idx):
    h_num = width // block_length
    # v_num = height // block_length
    h_idx = idx % h_num
    v_idx = idx // h_num

    st_pos_h = h_idx * block_length
    st_pos_v = v_idx * block_length

    return st_pos_h, st_pos_v


def create_base_ramp_pattern(width, height, block_length):
    """
    Create a ramp test pattern for investigation.
    """
    src_bit_depth = 10
    src_num_of_cv = 2 ** src_bit_depth

    img = np.zeros((height, width, 3))
    block_img_base = np.ones((block_length, block_length, 3))
    for idx in range(src_num_of_cv):
        st_pos_h, st_pos_v = calc_st_pos_rectangle_patch(
            width=width, height=height, block_length=block_length, idx=idx)
        ed_pos_h = st_pos_h + block_length
        ed_pos_v = st_pos_v + block_length
        block_img = block_img_base * idx
        img[st_pos_v:ed_pos_v, st_pos_h:ed_pos_h] = block_img

    img = img / (src_num_of_cv - 1)

    return img


def make_test_ramp_img_name():
    return "./img/test_block_ramp_imgage.png"


def save_test_ramp_image(img):
    fname = make_test_ramp_img_name()
    tpg.img_wirte_float_as_16bit_int(fname, img)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    width = 1920
    height = 1080
    block_length = 16
    ramp_img = create_base_ramp_pattern(
        width=width, height=height, block_length=block_length)
    save_test_ramp_image(ramp_img)
