# -*- coding: utf-8 -*-
"""

==========

"""

# import standard libraries
import os
from pathlib import Path
import subprocess

# import third-party libraries
import numpy as np

# import my libraries
import test_pattern_generator2 as tpg
import color_space as cs
import font_control2 as fc
import transfer_functions as tf

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

BIT_DEPTH = 10
NUM_OF_CODE_VALUE = (2 ** BIT_DEPTH)
MAXIMUM_CODE_VALUE = NUM_OF_CODE_VALUE - 1
COLOR_LIST = np.array([
    [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [1, 0, 1], [1, 1, 0], [0, 1, 1]
], dtype=np.uint16)


def calc_block_num_h(width=1920, block_size=64):
    return width // block_size


def calc_ramp_pattern_block_st_pos_with_color_idx(
        code_value=0, width=1920, block_size=64, color_kind_idx=0):
    color_block_height = calc_ramp_pattern_block_st_pos(
        code_value=MAXIMUM_CODE_VALUE, width=width, block_size=block_size)[1]\
        + block_size
    st_pos = calc_ramp_pattern_block_st_pos(
        code_value=code_value, width=width, block_size=block_size)
    st_pos = [st_pos[0], st_pos[1] + color_block_height * color_kind_idx]

    return st_pos


def calc_ramp_pattern_block_st_pos(
        code_value=0, width=1920, block_size=64):
    block_num_h = calc_block_num_h(width=width, block_size=block_size)
    st_pos_h = (code_value % block_num_h) * block_size
    st_pos_v = (code_value // block_num_h) * block_size
    st_pos = (st_pos_h, st_pos_v)

    return st_pos


def create_info_text_img(
        width=3840, height=160, font_size=80, text="hoge"):
    bg_color = np.array([0.002, 0.002, 0.002])
    img = np.ones((height, width, 3)) * bg_color
    # create instance
    text_draw_ctrl = fc.TextDrawControl(
        text=f" {text}", font_color=[0.2, 0.2, 0.2],
        font_size=font_size, font_path=fc.NOTO_SANS_CJKJP_MEDIUM,
        stroke_width=0, stroke_fill=None)

    # calc position
    text_width, text_height = text_draw_ctrl.get_text_width_height()
    pos_h = 0
    pos_v = (height // 2) - (text_height // 2)
    pos = (pos_h, pos_v)

    text_draw_ctrl.draw(img=img, pos=pos)

    img_st2084 = tf.oetf_from_luminance(img * 100, tf.ST2084)

    return img_st2084


def create_10bit_cms_test_pattern_img(
        width=3840, height=2160, block_size=32):
    img = np.zeros((height, width, 3), dtype=np.uint16)
    block_img_base = np.ones((block_size, block_size, 3), dtype=np.uint16)

    for color_kind_idx in range(len(COLOR_LIST)):
        color = COLOR_LIST[color_kind_idx]
        for code_value in range(NUM_OF_CODE_VALUE):
            block_img = block_img_base * code_value * color
            st_pos = calc_ramp_pattern_block_st_pos_with_color_idx(
                code_value=code_value, width=width, block_size=block_size,
                color_kind_idx=color_kind_idx
            )
            tpg.merge(img, block_img, st_pos)

    return img / MAXIMUM_CODE_VALUE


def create_10bit_cms_test_pattern_img_with_text_info(
        width=3840, height=2160, block_size=32,
        text="sample"):
    img_src_cs = create_10bit_cms_test_pattern_img(
        width=width, height=height, block_size=block_size
    )
    img_text = create_info_text_img(width=width, height=160, text=text)

    img = np.vstack([img_src_cs, img_text])

    return img


def conv_rec709_pq_to_rec2020_pq(img):
    rec2020_img_linear = tf.eotf(img, tf.ST2084)
    large_xyz = cs.rgb_to_large_xyz(
        rgb=rec2020_img_linear, color_space_name=cs.BT709
    )
    rec709_linear = cs.large_xyz_to_rgb(
        xyz=large_xyz, color_space_name=cs.BT2020
    )
    rec709_st2084 = tf.oetf(np.clip(rec709_linear, 0.0, 1.0), tf.ST2084)

    return rec709_st2084


def main_func():
    width = 3840
    block_height = 720
    text_area_height = 160
    tp_area_height = block_height - text_area_height
    block_size = 16

    img_rec2020_pq = create_10bit_cms_test_pattern_img_with_text_info(
        width=width, height=tp_area_height, block_size=block_size,
        text="Rec.2020 WRGBMYC Gradient Pattern (0 CV - 1023 CV)"
    )

    img_for_rec709 = create_10bit_cms_test_pattern_img_with_text_info(
        width=width, height=tp_area_height, block_size=block_size,
        text="Rec.709 WRGBMYC Gradient Pattern (0 CV - 1023 CV)"
    )
    img_rec709_pq = conv_rec709_pq_to_rec2020_pq(img=img_for_rec709)
    
    img_bt2020_name = "./debug/src_tp/tp_10bit_ramp_wrgbmyc_rec2020.png"
    tpg.img_wirte_float_as_16bit_int(img_bt2020_name, img_rec2020_pq)
    img_bt709_name = "./debug/src_tp/tp_10bit_ramp_wrgbmyc_rec709.png"
    tpg.img_wirte_float_as_16bit_int(img_bt709_name, img_rec709_pq)


def conv_to_avif():
    file_list = [
        "./tp_img/tp_10bit_ramp_wrgbmyc.png",
        "./tp_img/tp_10bit_ramp_wrgbmyc_709_on_2020.png"
    ]
    for png_fname in file_list:
        pp = Path(png_fname)
        ext = pp.suffix
        parent = str(pp.parent)
        avif_fname = "./" + parent + "/" + pp.stem + ".avif"

        cmd = [
            "avifenc", png_fname, "-d", "10", "-y", "444", "--cicp", "9/16/9",
            "-r", "full", "--min", "0", "--max", "0", "--ignore-exif", avif_fname
        ]
        subprocess.run(cmd)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
