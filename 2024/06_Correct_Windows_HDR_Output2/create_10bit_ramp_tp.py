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
from colour import LUT3D

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

TP_FILE_NAME = "./debug/src_tp/10bit_gradient_tp_709_2020_17x17x17.png"


def calc_block_num_h(width=1920, block_size=64):
    return width // block_size


def calc_ramp_pattern_block_st_pos_with_color_idx(
        code_value=0, width=1920, block_size=64, color_kind_idx=0 ):
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


def create_10bit_17x17x17_test_patch_img(
        width=3840, height=2160, block_size=32):
    img = np.zeros((height, width, 3), dtype=np.uint16)
    block_img_base = np.ones((block_size, block_size, 3), dtype=np.uint16)
    rgb = np.round(LUT3D.linear_table(size=17) * 1023).astype(np.uint16)
    rgb = rgb.reshape(-1, 3)

    for idx, yyy in enumerate(rgb):
        block_img = block_img_base * yyy
        st_pos = calc_ramp_pattern_block_st_pos(
            code_value=idx, width=width, block_size=block_size)
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
    
    merge_pos = calc_ramp_pattern_block_st_pos_with_color_idx(
        code_value=0, width=width, block_size=block_size, color_kind_idx=7)

    tpg.merge(img_src_cs, img_text, pos=merge_pos)

    return img_src_cs


def create_10bit_17x17x17_test_pattern_img_with_text_info(
        width=3840, height=2160, block_size=32, text_area_height=80,
        text="sample"):
    img_src_cs = create_10bit_17x17x17_test_patch_img(
        width=width, height=height, block_size=block_size
    )
    img_text = create_info_text_img(
        width=width, height=text_area_height, text=text)
    
    num_of_v_block = int(round( (17 ** 3) / (width // block_size) + 0.5 ))
    st_pos_v = num_of_v_block * block_size

    tpg.merge(img_src_cs, img_text, (0, st_pos_v))
    # img = np.vstack([img_src_cs, img_text])

    return img_src_cs


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


def get_gradient_tp_ref_value(tp_img, width, block_size):
    """
    return data[color_idx][code_value][3]
    """
    num_of_color = 7
    num_of_code_value = 1024
    data = np.zeros((num_of_color, num_of_code_value, 3))
    for color_idx in range(num_of_color):
        for cv in range(num_of_code_value):
            base_pos = calc_ramp_pattern_block_st_pos_with_color_idx(
                code_value=cv, width=width, block_size=block_size,
                color_kind_idx=color_idx
            )
            center_pos = (
                base_pos[0] + (block_size//2), base_pos[1] + (block_size//2)
            )
            rgb = tp_img[center_pos[1], center_pos[0]]
            data[color_idx][cv] = rgb

    return data

def get_17x17x17_tp_ref_value(tp_img, width, block_size):
    num_of_grid = 17
    num_of_patch = num_of_grid ** 3
    data = np.zeros((num_of_patch, 3))
    for idx in range(num_of_patch):
        base_pos = calc_ramp_pattern_block_st_pos(
            code_value=idx, width=width, block_size=block_size)
        center_pos = (
            base_pos[0] + (block_size//2),
            base_pos[1] + (block_size//2),
        )
        rgb = tp_img[center_pos[1], center_pos[0]]
        data[idx] = rgb

    return data


def main_func():
    width = TP_WIDTH
    block_height = TP_BLOCK_HEIGHT
    text_area_height = TP_TEXT_AREA_HEIGHT
    tp_area_height = block_height
    block_size = TP_BLOCK_SIZE

    img_rec2020_pq = create_10bit_cms_test_pattern_img_with_text_info(
        width=width, height=tp_area_height, block_size=block_size,
        text_area_height=text_area_height,
        text="Rec.2020 WRGBMYC Gradient Pattern (0 CV - 1023 CV)"
    )

    img_for_rec709 = create_10bit_cms_test_pattern_img_with_text_info(
        width=width, height=tp_area_height, block_size=block_size,
        text_area_height=text_area_height,
        text="Rec.709 on Rec.2020 WRGBMYC Gradient Pattern (0 CV - 1023 CV)"
    )
    img_rec709_pq = conv_rec709_pq_to_rec2020_pq(img=img_for_rec709)

    img_rec2020_patch = create_10bit_17x17x17_test_pattern_img_with_text_info(
        width=width, height=tp_area_height, block_size=block_size,
        text_area_height=text_area_height,
        text="Rec.2020 17x17x17 Test Patch")

    eval_img = np.vstack([img_rec709_pq, img_rec2020_pq, img_rec2020_patch])
    
    fname_bt2020_png = "./debug/src_tp/tp_10bit_ramp_wrgbmyc_rec2020.png"
    tpg.img_wirte_float_as_16bit_int(fname_bt2020_png, img_rec2020_pq)

    fname_bt709_png = "./debug/src_tp/tp_10bit_ramp_wrgbmyc_rec709.png"
    fname_bt709_exr = "./debug/src_tp/tp_10bit_ramp_wrgbmyc_rec709.exr"
    tpg.img_wirte_float_as_16bit_int(fname_bt709_png, img_rec709_pq)
    write_image(image=img_rec709_pq, path=fname_bt709_exr, bit_depth='float32')
    np.save("./debug/src_tp/tp_10bit_ramp_wrgbmyc_rec709.npy", img_rec709_pq)

    fname_17x17x17_patch_png\
        = "./debug/src_tp/tp_10bit_ramp_17x17x17_rec2020.png"
    tpg.img_wirte_float_as_16bit_int(fname_17x17x17_patch_png, img_rec2020_patch)

    eval_img_name = TP_FILE_NAME
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


def debug_get_ref_value():
    block_height = TP_BLOCK_HEIGHT
    block_size = TP_BLOCK_SIZE
    width = TP_WIDTH
    tp_fname = TP_FILE_NAME
    tp_img = tpg.img_read_as_float(tp_fname)
    tp_img_rec709 = tp_img[0:block_height]
    tp_img_rec2020 = tp_img[block_height:block_height*2]
    tp_img_17x17x17 = tp_img[block_height*2:block_height*3]
    rgb_2020 = get_gradient_tp_ref_value(
        tp_img=tp_img_rec2020, width=width, block_size=block_size)
    rgb_709 = get_gradient_tp_ref_value(
        tp_img=tp_img_rec709, width=width, block_size=block_size)
    rgb_17x17x17 = get_17x17x17_tp_ref_value(
        tp_img=tp_img_17x17x17, width=width, block_size=block_size)

    # rgb = np.round(rgb_709 * 1023).astype(np.uint16)
    # for c_idx in range(7):
    #     for cv in range(1024):
    #         print(c_idx, cv, rgb[c_idx, cv])

    rgb_17x17x17 = np.round(rgb_17x17x17 * 1023).astype(np.uint16)
    for idx in range(17**3):
        print(idx, rgb_17x17x17[idx])    


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # main_func()
    debug_get_ref_value()
    # calc_rgb_to_rgb_matrix(src_cs_name=cs.BT709, dst_cs_name=cs.BT2020)
    # calc_rgb_to_rgb_matrix(src_cs_name=cs.BT2020, dst_cs_name=cs.BT709)
