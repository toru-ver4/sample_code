# -*- coding: utf-8 -*-
"""

```
"""

# import standard libraries
import os
from multiprocessing import Pool, cpu_count
from typing import NamedTuple

# import third-party libraries
import numpy as np
import cv2
from colour import LCHab_to_Lab
from colour.utilities import tstack
from numpy.core.shape_base import stack
from numpy.random import sample

# import my libraries
import test_pattern_generator2 as tpg
import transfer_functions as tf
import color_space as cs
from common import MeasureExecTime

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

lightness = 70
base_chroma = 50
base_hue = 270

edge_lch = np.array([lightness, base_chroma, base_hue])
edge_lab = LCHab_to_Lab(edge_lch)
fg_lab = np.array([lightness, 0, 50])

base_value = 0.0
fg_color = cs.lab_to_rgb(fg_lab, cs.BT709)
edge_color = cs.lab_to_rgb(edge_lab, cs.BT709)

fg_color[fg_color <= 0] = 0.001
edge_color[edge_color <= 0] = 0.001


class BaseParam(NamedTuple):
    amp: int = 200
    dot_pattern_rate: int = 16
    radius_out: int = 250
    radius_in: int = 150
    dst_width: int = 1000
    dst_height: int = 1000
    width: int = 2000
    height: int = 2000
    fps: int = 60
    cycle_num: int = 6
    cycle_sec: float = 0.8
    cycle_sec_offset: float = 0.0
    amp_offset: int = 0
    tile_v_num: int = 4
    tile_h_num: int = 2


def create_fg_color_list(sample_num):
    st_hue = base_hue + 90
    ed_hue = st_hue + 180
    hh = np.linspace(st_hue, ed_hue, sample_num)
    ll = np.ones_like(hh) * lightness
    cc = np.ones_like(hh) * base_chroma
    lch = tstack([ll, cc, hh])
    lab = LCHab_to_Lab(lch)
    rgb = cs.lab_to_rgb(lab, cs.BT709)
    rgb = np.clip(rgb, 0.0, 1.0)

    return rgb


def create_bg_dot_pattern(
        width=640, height=640, dot_pattern_rate=16,
        radius_out=250, radius_in=150,
        fg_color=fg_color, edge_color=edge_color):
    dot_img_width = width // dot_pattern_rate
    dot_img_height = height // dot_pattern_rate
    np.random.seed(100)

    mask = np.random.randint(0, 2, (dot_img_height, dot_img_width, 1))
    inv_mask = np.uint8(1 - mask)
    temp_img = mask
    dot_img = np.dstack([temp_img, temp_img, temp_img])
    img = cv2.resize(
        dot_img, (width, height), interpolation=cv2.INTER_NEAREST)

    center = (width//2, height//2)

    img = img * fg_color
    img[img <= 0] = base_value
    img[inv_mask] = np.array([[[1.0, 1.0, 1.0]]])
    img = cv2.circle(img, center, radius_out, fg_color, -1)

    img = cv2.circle(img, center, radius_in, edge_color, -1)

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


def create_move_seq_1st_sample():
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

    mt = MeasureExecTime()

    mt.start()
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
            create_move_seq_core(**d)
            mt.lap()
            # print(d)
            args.append(d)
            # break
        # break
        # with Pool(block_process_num) as pool:
        #     pool.map(thread_wrapper_create_move_seq_core, args)
    mt.end()


def trim_image(base_img, idx, pos_list, bp: BaseParam):
    v_st = pos_list[idx, 1]
    v_ed = v_st + bp.dst_height
    h_st = pos_list[idx, 0]
    h_ed = h_st + bp.dst_width

    dst_img = base_img[v_st:v_ed, h_st:h_ed]

    return dst_img


def create_multi_move_pattern_core(
        f_idx, bp: BaseParam, fg_color_list, pos_list):
    v_img_buf = []
    for v_idx in range(bp.tile_v_num):
        h_img_buf = []
        for h_idx in range(bp.tile_h_num):
            c_idx = v_idx * bp.tile_h_num + h_idx
            base_img = create_bg_dot_pattern(
                width=bp.width, height=bp.height,
                dot_pattern_rate=bp.dot_pattern_rate,
                radius_out=bp.radius_out, radius_in=bp.radius_in,
                fg_color=fg_color_list[c_idx], edge_color=edge_color)
            img = trim_image(
                base_img=base_img, idx=f_idx, pos_list=pos_list, bp=bp)

            h_img_buf.append(img)
        v_img_buf.append(np.hstack(h_img_buf))

    out_img = np.vstack(v_img_buf)

    base_dir = "/work/overuse/2021/14_optical_illusion/multi_move_sample/"
    fname_base = base_dir + "multi_test_seq_{idx:04d}.png"
    fname = fname_base.format(idx=f_idx)
    print(fname)
    tpg.img_write(fname, out_img)


def calc_pos_list(bp: BaseParam):
    offset_list = create_horizontal_movement(
        fps=bp.fps, cycle_num=bp.cycle_num, cycle_sec=bp.cycle_sec,
        cycle_sec_offset=bp.cycle_sec_offset, amp=bp.amp,
        amp_offset=bp.amp_offset)
    h_pos_list = np.uint16(
        np.round((bp.width // 2) - offset_list - (bp.dst_width // 2)))
    v_pos_list = np.ones_like(h_pos_list)\
        * ((bp.height // 2) - (bp.dst_height // 2))
    pos_list = np.uint16(tstack([h_pos_list, v_pos_list]))

    return pos_list


def create_multi_color_movie_sample():
    g_bg_width = 3840
    g_bg_height = 2160

    bp = BaseParam(
        amp=g_bg_width//26,
        dot_pattern_rate=16,
        radius_out=100,
        radius_in=70,
        dst_width=g_bg_width//8,
        dst_height=g_bg_height//4,
        width=g_bg_width//4,
        height=g_bg_height//2,
        fps=60,
        cycle_num=6,
        cycle_sec=0.8,
        cycle_sec_offset=0.0,
        amp_offset=0,
        tile_v_num=4,
        tile_h_num=8,
    )
    sample_color_num = bp.tile_v_num * bp.tile_h_num
    fg_color_list = create_fg_color_list(sample_num=sample_color_num)
    pos_list = calc_pos_list(bp=bp)

    frame_num = len(pos_list)

    total_process_num = frame_num
    block_process_num = int(cpu_count())
    block_num = int(round(total_process_num / block_process_num + 0.5))

    mt = MeasureExecTime()
    mt.start()
    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            f_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, f_idx={f_idx}")  # User
            if f_idx >= total_process_num:                         # User
                break
            d = dict(
                f_idx=f_idx, bp=bp,
                fg_color_list=fg_color_list, pos_list=pos_list)
            # print(d)
            # create_multi_move_pattern_core(**d)
            args.append(d)
        #     break
        # break
        with Pool(block_process_num) as pool:
            pool.map(thread_wrapper_create_multi_move_pattern_core, args)
        mt.lap()
    mt.end()


def thread_wrapper_create_multi_move_pattern_core(args):
    create_multi_move_pattern_core(**args)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_move_seq_1st_sample()
    # create_fg_color_list()
    create_multi_color_movie_sample()
