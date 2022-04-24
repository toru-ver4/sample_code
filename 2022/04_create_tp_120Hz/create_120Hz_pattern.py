# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
from multiprocessing import Pool, cpu_count

# import third-party libraries
import numpy as np
from colour.io import write_image

# import my libraries
import font_control2 as fc2
import test_pattern_generator2 as tpg
import transfer_functions as tf
import cv2

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


FFMPEG_NORMALIZE_COEF = 65340
FFMPEG_NORMALIZE_COEF_INV = 65535/65340

REVISION = 1


def main_func():
    pass


class Square():
    def __init__(
            self, size=200, color=[0.1, 0.1, 0.0],
            base_period=2, num_of_period=4, global_st_pos=[0, 0], h_len=512,
            accel_rate=1.0, fps=60):
        self.size = size
        self.global_st_pos = global_st_pos
        self.color = np.array(color)
        self.base_period = base_period
        self.num_of_period = num_of_period
        self.accel_rate = accel_rate
        self.num_of_frame = fps * self.base_period
        self.h_len = h_len
        self.debug_y = []

    def calc_pos_h(self, idx):
        y = np.sin(
                2*np.pi/(self.num_of_frame)*idx*self.accel_rate - np.pi/2)\
            * 0.5 + 0.5
        self.pos_h = int(y * self.h_len + 0.5)
        self.st_pos = [
            self.global_st_pos[0] + self.pos_h - (self.size//2),
            self.global_st_pos[1] - (self.size//2)]
        self.ed_pos = [self.st_pos[0] + self.size, self.st_pos[1] + self.size]

        self.debug_y.append(y)

    def get_st_pos_ed_pos(self):
        return self.st_pos, self.ed_pos

    def get_size(self):
        return self.size

    def debug_plot(self):
        # import plot_utility as pu
        # x = np.arange(len(self.debug_y))
        # y = self.debug_y

        # fig, ax1 = pu.plot_1_graph()
        # ax1.plot(x, y)
        # pu.show_and_save(
        #     fig, legend_loc=None, save_fname="fuga.png", show=False)
        pass


def debug_dot_pattern():
    fname = "./img/org_dot.png"
    img = tpg.complex_dot_pattern(
        kind_num=4, whole_repeat=1)
    write_image(img, fname)

    nn = 8
    fg_color = np.array([1, 1, 1])
    bg_color = np.array([0, 0, 0])
    fname = f"./img/coplex_dot_n-{nn}.png"
    img = tpg.complex_dot_pattern2(
        nn=nn, fg_color=fg_color, bg_color=bg_color)
    write_image(img, fname)


def debug_line_cross_pattern():
    fg_color = np.array([0, 0, 0])
    bg_color = np.array([1, 1, 1])
    img = tpg.line_cross_pattern(
        nn=3, num_of_min_line=1, fg_color=fg_color, bg_color=bg_color,
        mag_rate=32)

    write_image(img, "./img/line_cross.png")


def debug_multi_border_pattern():
    fg_color = np.array([1, 1, 1])
    bg_color = np.array([0.1, 0.1, 0.1])
    length = 12
    thickness = 2
    img = np.zeros((length, length, 3))
    tpg.draw_border_line(img, (0, 0), length, thickness, fg_color)
    write_image(img, "./img/border_test.png", 'uint8')
    ll = tpg.calc_l_for_block(
        block_idx=2, num_of_block=3, num_of_line=3)
    print(ll)
    tpg.create_multi_border_tp(
        num_of_line=2, num_of_block=5,
        fg_color=fg_color, bg_color=bg_color, mag_rate=6)


def debug_func():
    # debug_dot_pattern()
    debug_line_cross_pattern()
    # debug_multi_border_pattern()
    pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug_func()
    # main_func()
