# -*- coding: utf-8 -*-
"""
Title
==============

Description.

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np

# import my libraries
import transfer_functions as tf
import test_pattern_generator2 as tpg
import font_control as fc

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


SCALE_LUMINANCE_DEF = np.array([
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 20, 30, 40, 50, 60, 70, 80, 90,
    100, 200, 300, 400, 500, 600, 700, 800, 900,
    1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])

TEXT_LUMINANCE_DEF = np.array([
    0.1,
    1,
    10,
    100, 500,
    1000, 5000, 10000])


def create_h_scale_img(
        scale_pos, scale_width, scale_height,
        line_color_low=[0.5, 0.5, 0.5], line_color_high=[0.1, 0.1, 0.1],
        color_inv_point_rate=0.5):
    width = scale_width
    height = scale_height
    color_inv_point = int(round(width * color_inv_point_rate + 0.5))
    img = np.zeros((height, width, 4))
    pos_low = scale_pos[scale_pos <= color_inv_point]
    pos_high = scale_pos[scale_pos > color_inv_point]
    img[:, pos_low] = np.array(list(line_color_low) + [1])
    img[:, pos_high] = np.array(list(line_color_high) + [1])

    img[0, :color_inv_point+1] = np.array(list(line_color_low) + [1])
    img[0, color_inv_point+1:] = np.array(list(line_color_high) + [1])

    return img


def composite_scale_text(
        img, scale_img, text_luminance, text_pos,
        font_color_low=[0.5, 0.5, 0.5], font_color_high=[0.1, 0.1, 0.1],
        color_inv_point_rate=0.5, inverse=False, size_rate=1):
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Bold.otf"
    height = img.shape[0]
    width = img.shape[1]
    scale_img_height = scale_img.shape[0]
    v_offset = int(round(6 * size_rate))
    font_size = int(round(10 * size_rate))
    text_pos_v = scale_img_height + v_offset

    color_inv_point = int(round(width * color_inv_point_rate + 0.5))

    for idx in range(len(text_luminance)):
        luminance = text_luminance[idx]
        scale_pos_h = text_pos[idx]
        if luminance < 1:
            text = f"{luminance:.1f} nits"
        else:
            text = f"{int(luminance):d} nits"
        text_width, text_height = fc.get_text_width_height(
            text=text, font_path=font_path, font_size=font_size)
        text_pos_h = scale_pos_h - text_width // 2
        if scale_pos_h <= color_inv_point:
            font_color_temp = font_color_low
        else:
            font_color_temp = font_color_high
        if inverse is True:
            text_pos_v_temp = height - text_pos_v - text_height
        else:
            text_pos_v_temp = text_pos_v
        text_drawer = fc.TextDrawer(
            img, text=text,
            pos=(text_pos_h, text_pos_v_temp),
            font_color=font_color_temp,
            font_size=font_size, font_path=font_path)
        text_drawer.draw()


def h_incremental_gradation_with_scale(
        width=1920, height=1080, max_luminance=1000, text_max_luminance=500,
        inverse=False, size_rate=1):
    # define parameters
    color_80_nits = np.ones(3) * tf.oetf_from_luminance(80, tf.ST2084)
    color_1_nits = np.ones(3) * tf.oetf_from_luminance(1, tf.ST2084)
    if inverse is True:
        line_color_low = color_1_nits
        line_color_high = color_80_nits
        font_color_low = color_1_nits
        font_color_high = color_80_nits
    else:
        line_color_low = color_80_nits
        line_color_high = color_1_nits
        font_color_low = color_80_nits
        font_color_high = color_1_nits
    scale_luminamce = SCALE_LUMINANCE_DEF[SCALE_LUMINANCE_DEF <= max_luminance]
    scale_pos = tf.oetf_from_luminance(scale_luminamce, tf.ST2084)\
        / tf.oetf_from_luminance(max_luminance, tf.ST2084)
    scale_pos = np.uint32(np.round(scale_pos * (width - 1)))

    max_cv = tf.oetf_from_luminance(max_luminance, tf.ST2084)
    line_ramp = np.linspace(0, max_cv, width)
    img = tpg.h_mono_line_to_img(line_ramp, height)

    # composite scale image
    scale_pos = width - 1 - scale_pos if inverse else scale_pos
    scale_img = create_h_scale_img(
        scale_pos=scale_pos,
        scale_width=width, scale_height=int(round(10 * size_rate)),
        line_color_low=line_color_low, line_color_high=line_color_high)

    if inverse is True:
        img = img[:, ::-1, :]
        scale_img = scale_img[::-1, :, :]
        scale_merge_pos = (0, height - scale_img.shape[0])
    else:
        img = img
        scale_img = scale_img
        scale_merge_pos = (0, 0)
    tpg.merge_with_alpha(img, scale_img, pos=scale_merge_pos)

    # composite luminance text
    text_luminance\
        = TEXT_LUMINANCE_DEF[TEXT_LUMINANCE_DEF <= text_max_luminance]
    text_pos = tf.oetf_from_luminance(text_luminance, tf.ST2084)\
        / tf.oetf_from_luminance(max_luminance, tf.ST2084)
    text_pos = np.uint32(np.round(text_pos * (width - 1)))
    text_pos = width - 1 - text_pos if inverse else text_pos
    composite_scale_text(
        img=img, scale_img=scale_img, text_luminance=text_luminance,
        text_pos=text_pos,
        font_color_low=font_color_low, font_color_high=font_color_high,
        inverse=inverse, size_rate=size_rate)

    return img


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    img_1 = h_incremental_gradation_with_scale(
        width=3840, height=1080, max_luminance=1000, size_rate=2)
    img_2 = h_incremental_gradation_with_scale(
        width=3840, height=1080, max_luminance=1000, inverse=True, size_rate=2)
    # h_incremental_gradation_with_scale(
    #     width=3840, height=1080, max_luminance=1000)

    img = np.vstack([img_1, img_2])
    tpg.img_wirte_float_as_16bit_int(
        "./img/h_inc_grad_3840x2160_1000-nits.png", img)

    img_1 = h_incremental_gradation_with_scale(
        width=2160, height=1920, max_luminance=1000,
        text_max_luminance=500, size_rate=2)
    img_2 = h_incremental_gradation_with_scale(
        width=2160, height=1920, max_luminance=1000,
        text_max_luminance=500, inverse=True, size_rate=2)
    img = np.hstack([np.rot90(img_1), np.rot90(img_2)])
    tpg.img_wirte_float_as_16bit_int(
        "./img/v_inc_grad_3840x2160_1000_nits.png", img)
