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
from colour import normalised_primary_matrix
from colour.algebra import vector_dot
from scipy import linalg
from colour.io import write_image

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

TP_WIDTH = 3840
TP_BLOCK_HEIGHT = 720
TP_TEXT_AREA_HEIGHT = 60
TP_BLOCK_SIZE = 16


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


def calc_ramp_pattern_block_center_pos_with_color_idx(
        code_value=0, width=1920, block_size=64, color_kind_idx=0):
    st_pos = calc_ramp_pattern_block_st_pos_with_color_idx(
        code_value=code_value, width=width,
        block_size=block_size, color_kind_idx=color_kind_idx)
    center_pos = [st_pos[0] + block_size//2, st_pos[1] + block_size//2]

    return center_pos


def create_info_text_img(
        width=3840, height=160, font_size=40, text="hoge"):
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
        width=3840, height=2160, block_size=32, text_area_height=80,
        text="sample"):
    img_src_cs = create_10bit_cms_test_pattern_img(
        width=width, height=height, block_size=block_size
    )
    img_text = create_info_text_img(
        width=width, height=text_area_height, text=text)

    img = np.vstack([img_src_cs, img_text])

    return img


def calc_rgb_to_rgb_matrix(src_cs_name, dst_cs_name):
    """
    Examples
    --------
    >>> calc_rgb_to_rgb_matrix(src_cs_name=cs.BT709, dst_cs_name=cs.BT2020)
    [[ 0.6274039   0.32928304  0.04331307]
     [ 0.06909729  0.9195404   0.01136232]
     [ 0.01639144  0.08801331  0.89559525]]    
    """
    src_gamut_xy = cs.get_primaries(color_space_name=src_cs_name).flatten()
    dst_gamut_xy = cs.get_primaries(color_space_name=dst_cs_name).flatten()
    w = cs.D65

    npm_src = normalised_primary_matrix(primaries=src_gamut_xy, whitepoint=w)
    npm_dst = normalised_primary_matrix(primaries=dst_gamut_xy, whitepoint=w)
    npm_dst_inv = linalg.inv(npm_dst)

    conv_mtx = npm_dst_inv.dot(npm_src)

    return conv_mtx


def conv_rec709_pq_to_rec2020_pq(img):
    rec2020_img_linear = tf.eotf(img, tf.ST2084)

    # large_xyz = cs.rgb_to_large_xyz(
    #     rgb=rec2020_img_linear, color_space_name=cs.BT709
    # )
    # rec709_linear = cs.large_xyz_to_rgb(
    #     xyz=large_xyz, color_space_name=cs.BT2020
    # )

    conv_mtx = calc_rgb_to_rgb_matrix(
        src_cs_name=cs.BT709, dst_cs_name=cs.BT2020)
    rec709_linear = vector_dot(conv_mtx, rec2020_img_linear)

    rec709_st2084 = tf.oetf(np.clip(rec709_linear, 0.0, 1.0), tf.ST2084)

    return rec709_st2084


def main_func():
    width = TP_WIDTH
    block_height = TP_BLOCK_HEIGHT
    text_area_height = TP_TEXT_AREA_HEIGHT
    tp_area_height = block_height - text_area_height
    block_size = TP_BLOCK_SIZE

    img_rec2020_pq = create_10bit_cms_test_pattern_img_with_text_info(
        width=width, height=tp_area_height, block_size=block_size,
        text_area_height=text_area_height,
        text="Rec.2020 WRGBMYC Gradient Pattern (0 CV - 1023 CV)"
    )

    img_for_rec709 = create_10bit_cms_test_pattern_img_with_text_info(
        width=width, height=tp_area_height, block_size=block_size,
        text_area_height=text_area_height,
        text="Rec.709 WRGBMYC Gradient Pattern (0 CV - 1023 CV)"
    )
    img_rec709_pq = conv_rec709_pq_to_rec2020_pq(img=img_for_rec709)

    eval_img = np.vstack([img_rec709_pq, img_rec2020_pq])
    
    fname_bt2020_png = "./debug/src_tp/tp_10bit_ramp_wrgbmyc_rec2020.png"
    tpg.img_wirte_float_as_16bit_int(fname_bt2020_png, img_rec2020_pq)
    fname_bt709_png = "./debug/src_tp/tp_10bit_ramp_wrgbmyc_rec709.png"
    fname_bt709_exr = "./debug/src_tp/tp_10bit_ramp_wrgbmyc_rec709.exr"
    tpg.img_wirte_float_as_16bit_int(fname_bt709_png, img_rec709_pq)
    write_image(image=img_rec709_pq, path=fname_bt709_exr, bit_depth='float32')
    np.save("./debug/src_tp/tp_10bit_ramp_wrgbmyc_rec709.npy", img_rec709_pq)
    eval_img_name = "./debug/src_tp/10bit_gradient_tp_709_2020_xxx_2.png"
    tpg.img_wirte_float_as_16bit_int(eval_img_name, eval_img)


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
    # calc_rgb_to_rgb_matrix(src_cs_name=cs.BT709, dst_cs_name=cs.BT2020)
    # calc_rgb_to_rgb_matrix(src_cs_name=cs.BT2020, dst_cs_name=cs.BT709)
