# -*- coding: utf-8 -*-
"""
create hue-chroma pattern
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np

# import my libraries
import plot_utility as pu
import test_pattern_generator2 as tpg
import transfer_functions as tf

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def calc_pos(t0, t1, t2, t3, fps):
    sample = int((t3 - t0) * fps)
    t = np.linspace(t0, t3, sample, endpoint=False)
    y = np.zeros_like(t)

    idx_0 = t < t1
    idx_1 = (t >= t1) & (t < t2)
    idx_2 = (t >= t2) & (t <= t3)
    y[idx_0] = (np.sin(np.pi/(2*t1)*t[idx_0]+6*np.pi/4)+1)\
        / (np.sin(2*np.pi)+1)
    y[idx_1] = 1/t1 * (t[idx_1] - t1) + 1
    y[idx_2] = np.sin(np.pi/(2*(t3-t2))*(t[idx_2]-t2))\
        + 1/t1 * (t2 - t1) + 1

    y = y / np.max(y)

    return t, y


def debug_pos_plot_with_sin():
    t0 = 0
    t1 = 0.2
    t2 = 1.8
    t3 = 2.0

    x, y = calc_pos(t0, t1, t2, t3, fps=60)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Title",
        graph_title_size=None,
        xlabel="X Axis Label", ylabel="Y Axis Label",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(x, y)
    pu.show_and_save(
        fig=fig, legend_loc=None, save_fname="./img/debug_circle_pos.png",
        show=False)


def debug_draw_accel():
    width = 1920
    height = 1080
    fps = 60
    base_dir = "/work/overuse/2021/12_local_dimming_checker/img_seq/"
    fname_base = base_dir + "debug_accel_{idx:04d}.png"

    t0 = 0.5
    t1 = t0 + 0.1
    t2 = t1 + 0.3
    t3 = t2 + 0.1
    t4 = t3 + 0.5
    block_width = height // 4
    block_height = block_width
    margin = block_width // 2
    st_pos = [margin, margin]
    ed_pos = (width - margin * 3, margin)

    block_img = np.ones((block_height, block_width, 3))
    img = np.zeros((height, width, 3))

    cnt = 0

    t0_2 = t0 - t0
    t1_2 = t1 - t0
    t2_2 = t2 - t0
    t3_2 = t3 - t0
    x, y = calc_pos(t0_2, t1_2, t2_2, t3_2, fps)
    x = x + t0
    y = y * (ed_pos[0] - st_pos[0]) + st_pos[0]

    for idx in range(int(fps * t0)):
        img_dst = img.copy()
        pos = [int(y[0]), margin]
        tpg.merge(img_dst, block_img, pos)
        fname = fname_base.format(idx=cnt)
        print(fname)
        tpg.img_wirte_float_as_16bit_int(fname, img_dst)
        cnt += 1

    for idx in range(0, int((t3-t0) * fps)):
        img_dst = img.copy()
        pos = [int(y[idx]), margin]
        tpg.merge(img_dst, block_img, pos)
        fname = fname_base.format(idx=cnt)
        print(fname)
        tpg.img_wirte_float_as_16bit_int(fname, img_dst)
        cnt += 1

    for idx in range(0, int((t4-t0)*fps)):
        img_dst = img.copy()
        pos = [int(y[-1]), margin]
        tpg.merge(img_dst, block_img, pos)
        fname = fname_base.format(idx=cnt)
        print(fname)
        tpg.img_wirte_float_as_16bit_int(fname, img_dst)
        cnt += 1


def create_slide_pattern():
    width = 1920 * 2
    height = 1080 * 2
    fps = 60
    base_dir = "/work/overuse/2021/12_local_dimming_checker/img_seq/"
    fname_base = base_dir + "slide_pattern_{idx:04d}.png"

    t0 = 0.7
    t1 = t0 + 0.1
    t2 = t1 + 0.3
    t3 = t2 + 0.1
    t4 = t3 + 0.7
    block_width = height // 4
    block_height = block_width
    margin = block_width // 2
    st_pos = [margin, margin]
    ed_pos = (width - margin * 3, margin)

    block_img = np.ones((block_height, block_width, 3))
    upper_img = np.zeros((height//2, width, 3))
    lower_img = np.ones((height//2, width, 3))\
        * tf.oetf_from_luminance(10, tf.ST2084)

    cnt = 0

    t0_2 = t0 - t0
    t1_2 = t1 - t0
    t2_2 = t2 - t0
    t3_2 = t3 - t0
    x, y = calc_pos(t0_2, t1_2, t2_2, t3_2, fps)
    x = x + t0
    y = y * (ed_pos[0] - st_pos[0]) + st_pos[0]

    for idx in range(int(fps * t0)):
        upper_img_dst = upper_img.copy()
        lower_img_dst = lower_img.copy()
        pos = [int(y[0]), margin]
        tpg.merge(upper_img_dst, block_img, pos)
        tpg.merge(lower_img_dst, block_img, pos)
        fname = fname_base.format(idx=cnt)
        print(fname)
        img_dst = np.vstack([upper_img_dst, lower_img_dst])
        tpg.img_wirte_float_as_16bit_int(fname, img_dst)
        cnt += 1

    for idx in range(0, int((t3-t0) * fps)):
        upper_img_dst = upper_img.copy()
        lower_img_dst = lower_img.copy()
        pos = [int(y[idx]), margin]
        tpg.merge(upper_img_dst, block_img, pos)
        tpg.merge(lower_img_dst, block_img, pos)
        fname = fname_base.format(idx=cnt)
        print(fname)
        img_dst = np.vstack([upper_img_dst, lower_img_dst])
        tpg.img_wirte_float_as_16bit_int(fname, img_dst)
        cnt += 1

    for idx in range(0, int((t4-t3)*fps)):
        upper_img_dst = upper_img.copy()
        lower_img_dst = lower_img.copy()
        pos = [int(y[-1]), margin]
        tpg.merge(upper_img_dst, block_img, pos)
        tpg.merge(lower_img_dst, block_img, pos)
        fname = fname_base.format(idx=cnt)
        print(fname)
        img_dst = np.vstack([upper_img_dst, lower_img_dst])
        tpg.img_wirte_float_as_16bit_int(fname, img_dst)
        cnt += 1


def create_blink_pattern():
    width = 1920 * 2
    height = 1080 * 2
    fps = 60
    base_dir = "/work/overuse/2021/12_local_dimming_checker/img_seq/"
    fname_base = base_dir + "blink_pattern_{idx:04d}.png"

    t0 = 0.5
    t1 = t0 + 0.1
    t2 = t1 + 0.7
    t3 = t2 + 0.1
    t4 = t3 + 0.5

    block_width = width // 4
    block_height = block_width
    margin = block_width // 2

    block_img = np.ones((block_height, block_width, 3))
    upper_img = np.zeros((height, width//2, 3))
    lower_img = np.ones((height, width//2, 3))\
        * tf.oetf_from_luminance(10, tf.ST2084)
    dst_img = np.hstack([upper_img, lower_img])

    upper_img_with_block = upper_img.copy()
    tpg.merge(upper_img_with_block, block_img, (margin, margin))
    lower_img_with_block = lower_img.copy()
    tpg.merge(lower_img_with_block, block_img, (margin, margin))
    dst_img_with_block = np.hstack(
        [upper_img_with_block, lower_img_with_block])

    for idx in range(int(t4*fps)):
        fname = fname_base.format(idx=idx)
        print(fname)
        if ((idx >= t0*fps) and (idx <= t3*fps)):
            tpg.img_wirte_float_as_16bit_int(fname, dst_img_with_block)
        else:
            tpg.img_wirte_float_as_16bit_int(fname, dst_img)


def main_func():
    create_slide_pattern()
    # create_blink_pattern()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
    # debug_pos_plot_with_sin()
    # debug_draw_accel()
