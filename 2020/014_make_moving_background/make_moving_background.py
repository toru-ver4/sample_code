# -*- coding: utf-8 -*-
"""
動く背景動画を作成する
======================

Description.

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour import write_image, read_image
from multiprocessing import Pool, cpu_count

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def thread_wrapper_cut_and_save(kwargs):
    cut_and_save(**kwargs)


def cut_and_save(
        frame_idx, base_img, st_pos_h_list, st_pos_v_list,
        width, height, fps=60, sec=3, h_px=5, v_px=3,
        bg_file="./bg_image/bg_img_16x9.png"):
    base = os.path.basename(os.path.splitext(bg_file)[0])
    out_name = f"./bg_image_seq/{base}_{fps}fps_{sec}s_{frame_idx:04d}.png"

    st_pov_h = st_pos_h_list[frame_idx]
    ed_pos_h = st_pov_h + width
    st_pos_v = st_pos_v_list[frame_idx]
    ed_pos_v = st_pos_v + height

    out_img = base_img[st_pos_v:ed_pos_v, st_pov_h:ed_pos_h]
    print(out_name)

    write_image(out_img, out_name, bit_depth='uint16')


def moving_background(
        fps=60, sec=7, h_px=6, v_px=3, bg_file="./bg_image/bg_img_16x9.png"):
    frame = fps * sec

    # 切り取り用の大きい画像を生成
    img = read_image(bg_file)
    width, height = (img.shape[1], img.shape[0])
    img_temp = np.hstack([img, img])
    img_4x = np.vstack([img_temp, img_temp])

    # 切り出し位置を計算しておく
    idx = np.arange(frame)
    st_pos_h_list = (idx * h_px) % width
    st_pos_v_list = (idx * v_px) % height

    args = []
    for frame_idx in idx:
        kwargs = dict(
            frame_idx=frame_idx, base_img=img_4x,
            st_pos_h_list=st_pos_h_list, st_pos_v_list=st_pos_v_list,
            width=width, height=height,
            fps=fps, sec=sec, h_px=h_px, v_px=v_px, bg_file=bg_file)

        args.append(kwargs)
        # cut_and_save(**kwargs)

    with Pool(cpu_count()) as pool:
        pool.map(thread_wrapper_cut_and_save, args)


def main_func():
    moving_background(
        fps=60, sec=7, h_px=6, v_px=3, bg_file="./bg_image/bg_img_16x9.png")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
