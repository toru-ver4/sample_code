# -*- coding: utf-8 -*-

# import standard libraries
import subprocess
import sys
from pathlib import Path

# import third-party libraries
import numpy as np
from colour import read_image

# import my libraries
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
TY_LIB_DIR = PROJECT_DIR.parents[1] / "ty_lib"
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(TY_LIB_DIR))

from create_src_test_pattern2 import (
    decode_n_bit_yuv420_to_rgb444,
    get_10bit_ramp_from_img,
    COLOR_LIST
)


def float_to_n_bit(xx, max_cv):
    return np.round(xx * max_cv).astype(np.uint16)


def mask_color(xx, color_list):
    for ii, color in enumerate(color_list):
        for jj in range(len(color)):
            xx[ii, :, jj] = xx[ii, :, jj] * color[jj]


def create_ref_ramp_data(bit_depth):
    color_list = COLOR_LIST
    num_of_color = len(color_list)
    num_of_patch = 2 ** bit_depth

    x = np.arange(num_of_patch, dtype=np.uint16)
    ref_data = np.repeat(x[..., np.newaxis], 3, axis=-1)
    ref_data = np.repeat(ref_data[np.newaxis, ...], num_of_color, axis=0)
    mask_color(ref_data, color_list)

    return ref_data


def check_gray_diff(tolerance=1, gray_diff_list=None):
    ret_value = None
    gray_total_max_diff = np.max(gray_diff_list)
    if gray_total_max_diff < tolerance + 1:
        print(f"Gray maximum difference is {gray_total_max_diff}")
        print("OK")
        ret_value = True
    else:
        print(f"Gray maximum difference is {gray_total_max_diff}")
        print("NG")
        ret_value = False

    return ret_value


def check_color_diff(tolerance=2, color_diff_list=None):
    ret_value = None
    color_total_max_diff = np.max(color_diff_list)
    if color_total_max_diff < tolerance + 1:
        print(f"Color maximum difference is {color_total_max_diff}")
        print("OK")
        ret_value = True
    else:
        print(f"Color maximum difference is {color_total_max_diff}")
        print("NG")
        ret_value = False

    return ret_value


if __name__ == '__main__':
    condition_list = [
        [8, "bt.709"],
        [8, "bt.2020"],
        [10, "bt.709"],
        [10, "bt.2020"],
        [12, "bt.709"],
        [12, "bt.2020"],
    ]

    diff_buf = np.zeros((6, 2), dtype=np.uint16)  # diff_buf[:, 0] -> gray ramp, diff_buf[:, 1] -> color ramp

    for iii, condition in enumerate(condition_list):
        bit_depth = condition[0]
        gamut_str = condition[1]
        max_cv = (2 ** bit_depth) - 1

        yuv420_fname = f"./raw/ffmpeg_3840x2160_yuv420p{bit_depth}le_{gamut_str}.yuv"
        rgb444_fname = f"./img/ffmpeg_dst_img_v2_{bit_depth:02}-bit_{gamut_str}.dpx"

        decode_n_bit_yuv420_to_rgb444(
            yuv420_fnmae=yuv420_fname, out_fname=rgb444_fname, bit_depth=bit_depth, gamut=gamut_str
        )
        read_data_float = get_10bit_ramp_from_img(read_image(rgb444_fname), bit_depth)
        read_data = float_to_n_bit(read_data_float, max_cv)

        ref_data = create_ref_ramp_data(bit_depth=bit_depth)

        diff = np.abs(read_data.astype(np.int16) - ref_data.astype(np.int16))

        gray_max_diff = np.max(diff[0])
        color_max_diff = np.max(diff[1:])

        print(f"[Debug] {bit_depth}-bit, {gamut_str}: gray_diff = {gray_max_diff}, color_diff = {color_max_diff}")

        diff_buf[iii, 0] = gray_max_diff
        diff_buf[iii, 1] = color_max_diff

    # ------------------------------------------------
    # check diff
    # ------------------------------------------------
    ret_gray = check_gray_diff(tolerance=1, gray_diff_list=diff_buf[..., 0])

    # ret_color = True
    ret_color = check_color_diff(tolerance=2, color_diff_list=diff_buf[..., 1])

    if ret_gray and ret_color:
        sys.exit(0)
    else:
        sys.exit(1)
