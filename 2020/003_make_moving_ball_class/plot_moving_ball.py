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
import test_pattern_generator2 as tpg
from multiprocessing import Pool, cpu_count

# import my libraries
from reflective_moving_object import ReflectiveMovingObject

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
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


def theread_wrapper_draw_main(kwargs):
    draw_main(**kwargs)


def draw_main(idx, radius, pos_list_total, bg_image, color_list):
    fname = "./img_with_bg_image/ball_size_{:03d}_lv_{:04d}.png".format(
        radius, idx)
    # pos_list = [
    #     rmo_list[idx].get_pos() for idx in range(len(velocity_list))]
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for c_idx, pos in enumerate(pos_list_total[idx]):
        img = cv2.circle(
            img, pos, radius,
            color=color_list[c_idx], thickness=-1)
        # rmo_list[idx].calc_next_pos()  # マルチスレッド化にともなじ事前に計算
    # alpha channel は正規化する。そうしないと中間調合成時に透けてしまう
    alpha = np.max(img, axis=-1)
    alpha = alpha / np.max(alpha)
    img = np.dstack((img / 0xFF, alpha))

    bg_temp = bg_image.copy()
    img = tpg.merge_with_alpha(bg_temp, img)
    print(fname)
    cv2.imwrite(fname, np.uint16(np.round(img[:, :, ::-1] * 0xFFFF)))


def plot_with_bg_image(radius=20):
    velocity_rate = 20
    seed(0)
    velocity_list = np.array([rand(2), rand(2), rand(2), rand(2)])
    velocity_list = np.int16(np.round(velocity_list * velocity_rate))
    color_list = [(192, 192, 192), (192, 0, 0),
                  (0, 192, 0), (0, 0, 192)]
    bg_image = cv2.imread("./img/bg_img_5x3.png",
                          cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR) / 0xFFFF

    rmo_list = [ReflectiveMovingObject(
        velocity_init=velocity_list[idx], radius=radius)
        for idx in range(len(velocity_list))]
    trial_num = 900

    # 先に全部座標を計算してしまう
    pos_list_total = []
    for idx in range(trial_num):
        pos_list = [
            rmo_list[idx].get_pos() for idx in range(len(velocity_list))]
        pos_list_total.append(pos_list)
        for idx, pos in enumerate(pos_list):
            rmo_list[idx].calc_next_pos()

    args = []
    for idx in range(trial_num):
        kwargs = dict(
            idx=idx, radius=radius, pos_list_total=pos_list_total,
            bg_image=bg_image, color_list=color_list)
        args.append(kwargs)
        # draw_main(**kwargs)
    with Pool(cpu_count()) as pool:
        pool.map(theread_wrapper_draw_main, args)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # main_func()
    plot_with_bg_image(radius=5)
    plot_with_bg_image(radius=10)
    plot_with_bg_image(radius=20)
    plot_with_bg_image(radius=40)
