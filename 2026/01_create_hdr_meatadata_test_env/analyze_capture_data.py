import os

import numpy as np
from screeninfo import get_monitors

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
    pos_list = (num_of_color, num_of_step, num_of_pos)
    pos = (pos_v, pos_h)
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


def get_cc_pos_list(img_width, img_height):
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

    return pos_list


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    step_ramp_pos_list = get_step_ramp_pos_list(img_width=1920, img_height=1080)
    cc_pos_list = get_cc_pos_list(img_width=1920, img_height=1080)
    print(cc_pos_list)