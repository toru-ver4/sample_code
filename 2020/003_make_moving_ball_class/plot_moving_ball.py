# -*- coding: utf-8 -*-
"""
動くタマをプロットする
=====================

要は ReflectiveMovingObject の動作確認

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from numpy.random import rand, seed
import cv2

# import my libraries
from reflective_moving_object import ReflectiveMovingObject

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def main_func():
    radius = 30
    bg_level = 96
    velocity_rate = 20
    seed(0)
    velocity_list = np.array([rand(2), rand(2), rand(2), rand(2)])
    velocity_list = np.int16(np.round(velocity_list * velocity_rate))
    color_list = [(192, 192, 192), (192, 0, 0), (0, 192, 0), (0, 0, 192)]

    rmo_list = [ReflectiveMovingObject(
        velocity_init=velocity_list[idx], radius=radius)
        for idx in range(len(velocity_list))]
    trial_num = 720
    img_base = np.ones((1080, 1920, 3), dtype=np.uint8) * bg_level
    for idx in range(trial_num):
        fname = "./img/test_ball_bg_{:04d}_lv_{:04d}.png".format(bg_level, idx)
        pos_list = [
            rmo_list[idx].get_pos() for idx in range(len(velocity_list))]
        img = img_base.copy()
        for idx, pos in enumerate(pos_list):
            img = cv2.circle(
                img, pos_list[idx], radius,
                color=color_list[idx], thickness=-1)
            rmo_list[idx].calc_next_pos()
        cv2.imwrite(fname, img[:, :, ::-1])


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
