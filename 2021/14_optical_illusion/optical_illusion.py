# -*- coding: utf-8 -*-
"""

```
"""

# import standard libraries
import os
from multiprocessing import Pool, cpu_count

# import third-party libraries
import numpy as np
import cv2
from colour.utilities import tstack

# import my libraries
import test_pattern_generator2 as tpg
import transfer_functions as tf
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

lightness = 70
bbb = 50
bg_lab = np.array([lightness, 42, 18])
fg_lab = np.array([lightness, 0, -50])

bg_color = cs.lab_to_rgb(bg_lab, cs.BT709)
fg_color = cs.lab_to_rgb(fg_lab, cs.BT709)

bg_color[bg_color <= 0] = 0.01
fg_color[fg_color <= 0] = 0.01


def create_bg_dot_pattern(
        width=640, height=640, dot_pattern_rate=16,
        radius_out=250, radius_in=150):
    dot_img_width = width // dot_pattern_rate
    dot_img_height = height // dot_pattern_rate

    mask = np.random.randint(0, 2, (dot_img_height, dot_img_width, 1))
    inv_mask = np.uint8(1 - mask)
    temp_img = mask
    dot_img = np.dstack([temp_img, temp_img, temp_img])
    img = cv2.resize(
        dot_img, (width, height), interpolation=cv2.INTER_NEAREST)

    center = (width//2, height//2)
    radius = radius_out

    img = img * bg_color
    # img[img <= 0] = 1.0
    img[inv_mask] = np.array([[[1.0, 1.0, 1.0]]])
    img = cv2.circle(img, center, radius, bg_color, -1)

    radius2 = radius_in
    img = cv2.circle(img, center, radius2, fg_color, -1)

    img_non_linear = np.uint16(tf.oetf(np.clip(img, 0, 1), tf.SRGB) * 0xFFFF)

    return img_non_linear


def create_horizontal_movement(
        fps=60, cycle_num=5, cycle_sec=0.6, cycle_sec_offset=0,
        amp=300, amp_offset=550):

    sec = cycle_num * cycle_sec
    x = np.arange(int(round(sec*fps)))
    cycle_frame = int(round(fps * cycle_sec))
    offset_frame = int(round(fps * cycle_sec_offset))
    y = np.sin(2*np.pi*((x-offset_frame)/cycle_frame))

    win_x = np.arange(int(round(cycle_sec*fps)))
    len_win = len(win_x)
    win_y = np.sin(
        2*np.pi*((win_x-offset_frame)/cycle_frame)-np.pi/2)
    # for data in win_y:
    #     print(data)
    win_y = (win_y + 1) / 2

    y[:len_win//2] = y[:len_win//2] * win_y[:len_win//2]
    y[-len_win//2:] = y[-len_win//2:] * win_y[len_win//2:]
    y = y * amp + amp_offset

    return y


def create_move_seq_core(base_img, dst_width, dst_height, pos_list, idx):
    base_dir = "/work/overuse/2021/14_optical_illusion/test_movement/"
    fname_base = base_dir + "test_seq_{idx:04d}.png"
    fname = fname_base.format(idx=idx)
    v_st = pos_list[idx, 1]
    v_ed = v_st + dst_height
    h_st = pos_list[idx, 0]
    h_ed = h_st + dst_width

    img = base_img[v_st:v_ed, h_st:h_ed]

    print(f"writing {fname}")
    tpg.img_write(fname, img)


def thread_wrapper_create_move_seq_core(args):
    create_move_seq_core(**args)


def create_move_seq():
    amp = 200
    dot_pattern_rate = 16
    radius_out = 250
    radius_in = 150
    dst_width = radius_out * 4
    dst_height = dst_width
    width = dst_width + int(amp * 2.05)
    height = width

    fps = 60
    cycle_num = 6
    cycle_sec = 0.8
    cycle_sec_offset = 0
    amp_offset = 0

    base_img = create_bg_dot_pattern(
        width=width, height=height, dot_pattern_rate=dot_pattern_rate,
        radius_out=radius_out, radius_in=radius_in)

    offset_list = create_horizontal_movement(
        fps=fps, cycle_num=cycle_num, cycle_sec=cycle_sec,
        cycle_sec_offset=cycle_sec_offset, amp=amp, amp_offset=amp_offset)
    h_pos_list = np.uint16(
        np.round((width // 2) - offset_list - (dst_width // 2)))
    v_pos_list = np.ones_like(h_pos_list) * ((height // 2) - (dst_height // 2))
    pos_list = np.uint16(tstack([h_pos_list, v_pos_list]))

    j_num = len(h_pos_list)

    total_process_num = j_num
    block_process_num = int(cpu_count() / 2 + 0.999)
    block_num = int(round(total_process_num / block_process_num + 0.5))

    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            j_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={j_idx}")  # User
            if j_idx >= total_process_num:                         # User
                break
            d = dict(
                base_img=base_img,
                dst_width=dst_width, dst_height=dst_height,
                pos_list=pos_list, idx=j_idx)
            # create_move_seq_core(**d)
            # print(d)
            args.append(d)
        #     break
        # break
        with Pool(block_process_num) as pool:
            pool.map(thread_wrapper_create_move_seq_core, args)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    create_move_seq()
