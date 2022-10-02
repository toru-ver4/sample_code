# -*- coding: utf-8 -*-
"""
"""

# import standard libraries
import os
from multiprocessing import Pool, cpu_count

# import third-party libraries
import numpy as np

# import my libraries
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def debug_func_scroll_image():
    base_img = tpg.img_read("./img/scale_tp_1920x1080.png")
    height, width = base_img.shape[:2]
    num_of_step = 8
    step_h = -width // num_of_step
    step_v = -height // num_of_step
    for idx in range(num_of_step * 2):
        offset_h = step_h * idx
        offset_v = step_v * idx
        out_img = tpg.scroll_image(
            img=base_img, offset_h=offset_h, offset_v=offset_v)
        fname = f"./debug/scale_{idx:08d}.png"
        print(fname)
        tpg.img_write(fname, out_img)


def thread_wrapper_create_scroll_image_seq_core(args):
    create_scroll_image_seq_core(**args)


def create_scroll_image_seq_core(
        frame_idx, offset_h, offset_v, in_fname, step_h, step_v):
    base_img = tpg.img_read(in_fname)
    out_img = tpg.scroll_image(
        img=base_img, offset_h=offset_h, offset_v=offset_v)
    fname = f"./debug/scale_step-hv_{step_h}_{step_v}_{frame_idx:08d}.png"
    print(fname)
    tpg.img_write(fname, out_img)


def create_scroll_image_seq():
    in_fname = "./img/scale_tp_3840x2160.png"
    num_of_frame = 480
    step_h = 7
    step_v = -5

    total_process_num = num_of_frame
    block_process_num = int(cpu_count() * 0.8)
    block_num = int(round(total_process_num / block_process_num + 0.5))

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            l_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={l_idx}")  # User
            if l_idx >= total_process_num:                         # User
                break
            offset_h = step_h * l_idx
            offset_v = step_v * l_idx
            d = dict(
                frame_idx=l_idx, offset_h=offset_h, offset_v=offset_v,
                in_fname=in_fname, step_h=step_h, step_v=step_v)
            args.append(d)
        #     create_scroll_image_seq_core(**d)
        #     break
        # break
        with Pool(block_process_num) as pool:
            pool.map(thread_wrapper_create_scroll_image_seq_core, args)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # debug_func_scroll_image()
    create_scroll_image_seq()
