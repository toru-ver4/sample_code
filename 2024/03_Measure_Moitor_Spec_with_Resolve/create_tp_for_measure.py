# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour import xyY_to_XYZ

# import my libraries
import test_pattern_generator2 as tpg
import font_control2 as fc2
import transfer_functions as tf
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def create_tp_base_name(
        color_mask=[1, 1, 0], cv=1023/1023, patch_area_ratio=0.03):
    patch_area_ratio_int = int(patch_area_ratio*100)
    cv_int = int(cv * 1023 + 0.5)
    color_str = f"{color_mask[0]}_{color_mask[1]}_{color_mask[2]}"
    base_name = f"color-{color_str}_cv-{cv_int:04d}_"
    base_name += f"ratio-{patch_area_ratio_int:03d}"

    return base_name


def draw_cv_info(img, cv, font_size, fg_color):
    text = f"{int(cv * 1023 + 0.5):04d} CV"

    # create instance
    text_draw_ctrl = fc2.TextDrawControl(
        text=text, font_color=fg_color,
        font_size=font_size, font_path=fc2.NOTO_SANS_CJKJP_MEDIUM,
        stroke_width=0, stroke_fill=None)

    # calc position
    pos = (10, 10)

    text_draw_ctrl.draw(img=img, pos=pos)


def calc_patch_size(img, patch_area_ratio):
    height, width = img.shape[:2]
    areta_all = width * height
    areta_patch = areta_all * patch_area_ratio

    width = int((areta_patch * 16/9) ** 0.5)
    height = int((areta_patch * 9/16) ** 0.5)

    return width, height


def create_patch(color_mask=[1, 1, 0], cv=1023/1023, patch_area_ratio=0.03):
    # width = 3840
    # height = 2160
    width = 1920
    height = 1080
    linear_cv = tf.eotf(cv, tf.GAMMA24)  # linearization for text rendering
    font_size = int(18 * height / 1080 + 0.5)
    font_fg_color = tf.eotf(np.array([0.1, 0.1, 0.1]), tf.GAMMA24)
    fname_base = create_tp_base_name(
        color_mask=color_mask, cv=cv, patch_area_ratio=patch_area_ratio)
    fname = f"./tp_img/{fname_base}.png"

    img = np.zeros((height, width, 3))
    patch_width, patch_height = calc_patch_size(
        img=img, patch_area_ratio=patch_area_ratio)
    patch = np.ones((patch_height, patch_width, 3))\
        * linear_cv * np.array(color_mask)
    pos = (
        (width // 2) - (patch_width // 2), (height // 2) - (patch_height // 2))
    print(f"patch size = {patch.shape}")
    tpg.merge(img, patch, pos)

    draw_cv_info(img=img, cv=cv, font_size=font_size, fg_color=font_fg_color)

    print(fname)
    tpg.img_wirte_float_as_16bit_int(
        filename=fname, img_float=tf.oetf(img, tf.GAMMA24))


def create_black():
    width = 1920
    height = 2160
    img = np.zeros((height, width, 3))
    fname = "./tp_img/black.png"
    print(fname)
    tpg.img_wirte_float_as_16bit_int(filename=fname, img_float=img)


def create_cv_list(num_of_block=64):
    step = 1024 // num_of_block
    cv_list = np.arange(0, 1024, step)
    cv_list[-1] = 1023

    return cv_list


def create_white_patch_all():
    cv_list = create_cv_list(num_of_block=64)
    # print(cv_list)
    for cv in cv_list:
        create_patch(color_mask=[1, 1, 1], cv=cv/1023, patch_area_ratio=0.03)
        create_patch(color_mask=[1, 1, 1], cv=cv/1023, patch_area_ratio=0.05)
        create_patch(color_mask=[1, 1, 1], cv=cv/1023, patch_area_ratio=0.10)
        create_patch(color_mask=[1, 1, 1], cv=cv/1023, patch_area_ratio=0.20)
        create_patch(color_mask=[1, 1, 1], cv=cv/1023, patch_area_ratio=0.25)
        create_patch(color_mask=[1, 1, 1], cv=cv/1023, patch_area_ratio=0.30)
        create_patch(color_mask=[1, 1, 1], cv=cv/1023, patch_area_ratio=0.40)
        create_patch(color_mask=[1, 1, 1], cv=cv/1023, patch_area_ratio=0.50)
        create_patch(color_mask=[1, 1, 1], cv=cv/1023, patch_area_ratio=0.60)
        create_patch(color_mask=[1, 1, 1], cv=cv/1023, patch_area_ratio=0.70)
        create_patch(color_mask=[1, 1, 1], cv=cv/1023, patch_area_ratio=0.80)
        create_patch(color_mask=[1, 1, 1], cv=cv/1023, patch_area_ratio=0.90)
        create_patch(color_mask=[1, 1, 1], cv=cv/1023, patch_area_ratio=1.00)


def create_cc_rgb_value(luminance=100):
    xyY = tpg.generate_color_checker_xyY_value()
    xyY[..., 2] = xyY[..., 2] * luminance / 100
    large_xyz = xyY_to_XYZ(xyY)
    rgb = cs.large_xyz_to_rgb(xyz=large_xyz, color_space_name=cs.BT2020)

    return rgb


def create_cc_patch(rgb, window_size):
    width = 1920
    height = 1080
    img = np.zeros((height, width, 3))
    patch_width, patch_height = calc_patch_size(
        img=img, patch_area_ratio=window_size)
    patch = np.ones((patch_height, patch_width, 3)) * rgb
    pos = (
        (width // 2) - (patch_width // 2), (height // 2) - (patch_height // 2))
    tpg.merge(img, patch, pos)

    return img


def create_cc_tp_fname(cc_idx, luminance, window_size):
    window_size_str = int(window_size * 100)
    fname = f"./img_cctp/idx-{cc_idx:02d}_lumi-{luminance:04d}_"
    fname += f"win-{window_size_str:03d}.png"

    return fname


def create_color_checker_measure_pattern():
    lumiannce_list = [100, 200, 400, 600, 1000]
    window_size_list = [0.03, 0.10, 0.20, 0.50, 1.00]

    for luminance in lumiannce_list:
        cc_rgb = create_cc_rgb_value(luminance=luminance)
        for window_size in window_size_list:
            for cc_idx in range(len(cc_rgb)):
                img = create_cc_patch(
                    rgb=cc_rgb[cc_idx], window_size=window_size)
                fname = create_cc_tp_fname(
                    cc_idx=cc_idx, luminance=luminance,
                    window_size=window_size)
                img_non_linear = tf.oetf_from_luminance(img * 100, tf.ST2084)
                print(fname)
                tpg.img_wirte_float_as_16bit_int(fname, img_non_linear)


def verify_patch():
    fname = "./img_cctp/idx-17_lumi-1000_win-003.png"
    img = tpg.img_read_as_float(fname)
    rgb = img[1080//2, 1920//2]

    rgb_linear = tf.eotf_to_luminance(rgb, tf.ST2084)
    large_xyz = cs.rgb_to_large_xyz(
        rgb=rgb_linear, color_space_name=cs.BT2020)
    print(large_xyz)
    from colour import XYZ_to_xyY
    print(XYZ_to_xyY(large_xyz))


def create_colorchecker_xyY_for_blog():
    xyY = tpg.generate_color_checker_xyY_value()
    sdr_xyY = xyY.copy()
    sdr_xyY[..., 2] = sdr_xyY[..., 2] * 100
    hdr_xyY = xyY.copy()
    hdr_xyY[..., 2] = hdr_xyY[..., 2] * 1000

    print("| Index | x (SDR) | y (SDR) | Y (SDR) | x (HDR) | y (HDR) | Y (HDR) |")
    print("|----:|-----:|----:|----:|-----:|----:|----:|")
    for idx in range(24):
        buf = f"| {idx+1} "
        buf += f"| {sdr_xyY[idx, 0]:.4f} | {sdr_xyY[idx, 1]:.4f} | {sdr_xyY[idx, 2]:.3f} "
        buf += f"| {hdr_xyY[idx, 0]:.4f} | {hdr_xyY[idx, 1]:.4f} | {hdr_xyY[idx, 2]:.2f} |"
        print(buf)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_black()
    # create_white_patch_all()
    # create_color_checker_measure_pattern()
    # verify_patch()
    # create_colorchecker_xyY_for_blog()
