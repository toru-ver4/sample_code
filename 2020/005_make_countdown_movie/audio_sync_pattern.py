# -*- coding: utf-8 -*-
"""
create hue-chroma pattern
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from numpy.core.shape_base import block

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


class FrameMarker():
    def __init__(
            self, st_pos_list, block_width, block_height,
            fg_color, bg_color, center_color):
        self.st_pos_list = st_pos_list
        self.block_width = block_width
        self.block_height = block_height
        self.fg_color = fg_color
        self.bg_color = bg_color
        self.center_color = center_color

    def fill(self, target_img, color='bg'):
        if color == 'fg':
            color_val = self.fg_color
        elif color == 'bg':
            color_val = self.bg_color
        elif color == 'center':
            color_val = self.center_color
        for st_pos in self.st_pos_list:
            st_h = st_pos[0]
            ed_h = st_h + self.block_width
            st_v = st_pos[1]
            ed_v = st_v + self.block_height
            img = np.ones((self.block_height, self.block_width, 3))\
                * np.array(color_val)
            target_img[st_v:ed_v, st_h:ed_h] = img


def create_pos_list(
        width=1920, height=1080, fps=60,
        block_width=100, block_height=100, margin=100,
        marker_st_pos_v=800,
        fg_color=None, bg_color=None, center_color=None):

    # output buffer
    frame_marker_list = [None] * fps

    # calc idx=0 position
    st_pos_h = (width // 2) - block_width // 2
    st_pos_v = marker_st_pos_v
    base_st_pos = np.array([st_pos_h, st_pos_v], dtype=np.uint16)
    st_pos_list = np.array([base_st_pos], dtype=np.uint16)
    frame_maker_temp = FrameMarker(
        st_pos_list=st_pos_list,
        block_width=block_width, block_height=block_height,
        fg_color=fg_color, bg_color=bg_color, center_color=center_color)
    frame_marker_list[0] = frame_maker_temp

    # calc idx from 1 to (fps/2)
    for idx in range(1, fps//2 + 1):
        # foward
        offset = np.array(
            [(block_width + margin) * idx, 0], dtype=np.uint16)
        st_pos_temp = base_st_pos + offset
        st_pos_list = np.array([st_pos_temp], dtype=np.uint16)
        frame_maker_temp = FrameMarker(
            st_pos_list=st_pos_list,
            block_width=block_width, block_height=block_height,
            fg_color=fg_color, bg_color=bg_color, center_color=center_color)
        frame_marker_list[idx] = frame_maker_temp

        # backward

        # special treatment for idx == fps//2
        if idx == (fps//2):
            st_pos_temp2 = base_st_pos - offset
            st_pos_list = np.array(
                [st_pos_temp, st_pos_temp2], dtype=np.uint16)
        else:
            st_pos_temp = base_st_pos - offset
            st_pos_list = np.array([st_pos_temp], dtype=np.uint16)
        frame_maker_temp = FrameMarker(
            st_pos_list=st_pos_list,
            block_width=block_width, block_height=block_height,
            fg_color=fg_color, bg_color=bg_color, center_color=center_color)
        frame_marker_list[fps - idx] = frame_maker_temp

    # for idx in range(fps):
    #     fm = frame_marker_list[idx]
    #     if fm is not None:
    #         print(f"idx={idx}, st_pos={fm.st_pos_list[0]}")

    return frame_marker_list


def create_fill_block_sequence(
        width=1920, height=1080, fps=60, frame_marker_list=None,
        block_width=100, block_height=100):

    base_img = np.zeros((height, width, 3))
    fg_color = tf.eotf(np.array([192, 192, 192]) / 255, tf.GAMMA24)
    center_color = tf.eotf(np.array([174, 194, 238]) / 255, tf.GAMMA24)
    bg_color = tf.eotf(np.array([36, 36, 36]) / 255, tf.GAMMA24)
    base_dir = "/work/overuse/2021/13_audio_sync_pattern/img_seq/"
    fname_base = base_dir + "sync_{width}x{height}_{fps}p_{idx:04d}.png"

    for idx in range(fps):
        if idx == 0:
            color = center_color
        else:
            color = bg_color
        frame_marker_list[idx].fill(
            target_img=base_img, width=block_width, height=block_height,
            color=color)

    sec = 3
    frame = fps * sec
    for idx in range(frame):
        img = base_img.copy()
        frame_marker_list[idx % fps].fill(
            target_img=img, width=block_width, height=block_height,
            color=fg_color)
        out_img = tf.oetf(np.clip(img, 0, 1), tf.GAMMA24)
        alpha = create_alpha_channel(out_img)
        out_img = np.dstack([out_img, alpha])
        fname = fname_base.format(
            width=width, height=height, fps=fps, idx=idx)
        print(fname)
        tpg.img_wirte_float_as_16bit_int(fname, out_img)


def create_frame_marker_list(
        width=1920, height=1080, fps=60, padding=96, block_height=52,
        marker_st_pos_v=800,
        fg_color=None, bg_color=None, center_color=None):
    block_width = int(((width - padding * 2) / fps)) // 2
    margin = block_width

    frame_marker_list = create_pos_list(
        width=width, height=height, fps=fps,
        block_width=block_width, block_height=block_height,
        marker_st_pos_v=marker_st_pos_v,
        margin=margin, fg_color=fg_color,
        bg_color=bg_color, center_color=center_color)

    return frame_marker_list
    # create_fill_block_sequence(
    #     width=width, height=height, fps=fps,
    #     frame_marker_list=frame_marker_list,
    #     block_width=block_width, block_height=block_height)


def create_alpha_channel(img):
    alpha_img = np.zeros((img.shape[0], img.shape[1], 1))
    non_zero_idx = ((img[..., 0] > 0) | (img[..., 1] > 0)) | (img[..., 2] > 0)
    alpha_img[non_zero_idx] = 1

    return alpha_img


def debug_alpha_blend():
    img = np.zeros((1080, 1920, 3))
    color = np.array([0.5, 0.5, 0.5])
    img[300:600, 300:600] = np.ones((300, 300, 3)) * color

    alpha_img = create_alpha_channel(img)

    img_with_alpha = np.dstack([img, alpha_img])

    tpg.img_wirte_float_as_16bit_int("./img/debug_alpha.png", img_with_alpha)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # main_func(fps=24)
    # main_func(fps=30)
    # main_func(fps=50)
    # main_func(fps=60)
    # debug_alpha_blend()

    aa = tf.eotf(np.array([192, 192, 192]) / 255, tf.GAMMA24)
    print(type(aa))
