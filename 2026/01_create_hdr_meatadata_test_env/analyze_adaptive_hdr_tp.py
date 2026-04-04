import os
import sys
from pathlib import Path
from itertools import product

import numpy as np
from colour.difference import delta_E_CIE2000
from colour import XYZ_to_Lab

from test_pattern_generator2 import img_read_as_float, get_colorchecker_ref_lab
import color_space as cs


STEP_RAMP_ST_POS_H_FHD = 60
STEP_RAMP_ED_POS_H_FHD = 1856
STEP_RAMP_ST_POS_V_FHD = 860
STEP_RAMP_ED_POS_V_FHD = 990

CC_ST_POS_H_FHD = 1248
CC_ED_POS_H_FHD = 1818
CC_ST_POS_V_FHD = 170
CC_ED_POS_V_FHD = 512


def get_step_ramp_pos_list(img_width, img_height):
    """
    pos_list = (num_of_color, num_of_step, num_of_coordinate)
    num_of_coordinate = (pos_v, pos_h)
    """
    st_pos_h = int(STEP_RAMP_ST_POS_H_FHD / 1920 * img_width)
    ed_pos_h = int(STEP_RAMP_ED_POS_H_FHD / 1920 * img_width)
    pos_h_list = np.linspace(st_pos_h, ed_pos_h, 65, dtype=np.uint16)

    st_pos_v = int(STEP_RAMP_ST_POS_V_FHD / 1080 * img_height)
    ed_pos_v = int(STEP_RAMP_ED_POS_V_FHD / 1080 * img_height)
    pos_v_list = np.linspace(st_pos_v, ed_pos_v, 7, dtype=np.uint16)

    pos_list = np.zeros((7, 65, 2), dtype=np.uint16)

    for v_idx in range(7):
        pos_list[v_idx, :, 1] = pos_h_list.reshape(1, 1, -1)
    for h_idx in range(65):
        pos_list[:, h_idx, 0] = pos_v_list.reshape(1, 1, -1)

    return pos_list


def get_colorchecker_pos_list(img_width, img_height):
    """
    pos_list = (24, num_of_coordinate)
    num_of_coordinate = (pos_v, pos_h)
    """
    st_pos_h = int(CC_ST_POS_H_FHD / 1920 * img_width)
    ed_pos_h = int(CC_ED_POS_H_FHD / 1920 * img_width)
    pos_h_list = np.linspace(st_pos_h, ed_pos_h, 6, dtype=np.uint16)

    st_pos_v = int(CC_ST_POS_V_FHD / 1080 * img_height)
    ed_pos_v = int(CC_ED_POS_V_FHD / 1080 * img_height)
    pos_v_list = np.linspace(st_pos_v, ed_pos_v, 4, dtype=np.uint16)

    pos_list = np.zeros((4, 6, 2), dtype=np.uint16)

    for v_idx in range(4):
        pos_list[v_idx, :, 1] = pos_h_list.reshape(1, 1, -1)
    for h_idx in range(6):
        pos_list[:, h_idx, 0] = pos_v_list.reshape(1, 1, -1)

    return pos_list.reshape(-1, 2)


def read_jxr_as_bt2020_linear(fname: str):
    scrgb_img = img_read_as_float(fname)
    bt2020_img = cs.rgb_to_rgb(scrgb_img, cs.BT709, cs.BT2020)

    return bt2020_img


def get_step_ramp_7colors(img: np.ndarray, pos_list: list):
    pos_list = np.array(pos_list, dtype=np.uint16)
    step_ramp_data = img[pos_list[:, :, 0], pos_list[:, :, 1]]
    
    return step_ramp_data


def get_colorchecker_colors(img: np.ndarray, pos_list: list):
    pos_list = np.array(pos_list, dtype=np.uint16)
    cc_color = img[pos_list[:, 0], pos_list[:, 1]]
    
    return cc_color


def conv_bt2020_rgb_to_lab(rgb_linear):
    large_xyz = cs.rgb_to_large_xyz(rgb_linear, cs.BT2020)
    lab = XYZ_to_Lab(large_xyz)
    
    return lab


def calc_bt2020_colorchecker_de2000(rgb_linear):
    ref_lab = get_colorchecker_ref_lab()
    target_lab = conv_bt2020_rgb_to_lab(rgb_linear=rgb_linear)
    de2000 = delta_E_CIE2000(ref_lab, target_lab)

    return de2000


def calc_bt2020_colorchecker_delta_ab(rgb_linear):
    ref_lab = get_colorchecker_ref_lab()
    target_lab = conv_bt2020_rgb_to_lab(rgb_linear=rgb_linear)
    for ref, target in zip(ref_lab, target_lab):
        print(ref, target)
    delta_a = target_lab[:, 1] - ref_lab[:, 1]
    delta_b = target_lab[:, 2] - ref_lab[:, 2]

    delta_ab = np.sqrt((delta_a ** 2) + (delta_b ** 2))

    return delta_ab


def debug_each_function():
    step_ramp_pos_list = get_step_ramp_pos_list(img_width=1920, img_height=1080)
    cc_pos_list = get_colorchecker_pos_list(img_width=1920, img_height=1080)
    debug_img_fname = "./capture_img/av1_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-None.jxr"
    img = read_jxr_as_bt2020_linear(debug_img_fname)
    # step_ramp_7colors = get_step_ramp_7colors(img=img, pos_list=step_ramp_pos_list)
    cc_colors = get_colorchecker_colors(img=img, pos_list=cc_pos_list)
    # print(cc_colors)
    calc_bt2020_colorchecker_de2000(rgb_linear=cc_colors)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug_each_function()
