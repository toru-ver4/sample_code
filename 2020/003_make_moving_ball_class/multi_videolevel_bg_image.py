# -*- coding: utf-8 -*-
"""
様々な Code Value をもつ背景画像の作成
=====================================

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
import cv2

# import my libraries
import test_pattern_generator2 as tpg
from font_control import TextDrawer
from font_control import NOTO_SANS_MONO_REGULAR
import transfer_functions as tf
# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def make_multi_video_level_background(
        width=1920, height=1080, h_sample_num=5, v_sample_num=3,
        max_rate=0.6, font_size=20):
    img = np.zeros((height, width, 3))
    block_num = h_sample_num * v_sample_num
    video_levels = np.linspace(0, 1, block_num) * max_rate
    h_block_size_list = tpg.equal_devision(width, h_sample_num)
    v_block_size_list = tpg.equal_devision(height, v_sample_num)

    for v_idx in range(v_sample_num):
        block_height = v_block_size_list[v_idx]
        for h_idx in range(h_sample_num):
            eval_num = v_idx % 2
            should_zero = ((h_idx % 2) == eval_num)
            vl_idx = v_idx * h_sample_num + h_idx
            block_width = h_block_size_list[h_idx]
            block_img = np.ones((block_height, block_width, 3))\
                * video_levels[vl_idx]
            st_pos_h = np.sum(h_block_size_list[:h_idx]) if h_idx > 0 else 0
            st_pos_v = np.sum(v_block_size_list[:v_idx]) if v_idx > 0 else 0
            st_pos = (st_pos_h, st_pos_v)
            block_img = block_img if should_zero else block_img * 0.0
            nits = tf.eotf_to_luminance(video_levels[vl_idx], tf.ST2084)
            text = f"{nits:5.0f} nits" if nits >= 1000 else f"{nits:3.1f} nits"
            text_drawer = TextDrawer(
                block_img, text=text, pos=(0, 0), font_color=(0, 0, 0),
                font_size=font_size, font_path=NOTO_SANS_MONO_REGULAR)
            text_drawer.draw()
            tpg.merge(img, block_img, st_pos)

    fname = f"./img/bg_img_{h_sample_num}x{v_sample_num}.png"

    cv2.imwrite(fname, np.uint16(np.round(img[..., ::-1] * 0xFFFF)))


def main_func():
    make_multi_video_level_background()
    make_multi_video_level_background(
        width=1920, height=1080, h_sample_num=16, v_sample_num=9,
        max_rate=0.7, font_size=20)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
