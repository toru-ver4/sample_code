# -*- coding: utf-8 -*-

# import standard libraries
import sys
import os
from pathlib import Path

# import third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from colour import write_image, read_image
from colour.io.image import Image_Specification_Attribute
from colour.algebra import vecmul

# import my libraries
import test_pattern_generator2 as tpg

GRAY_PATCH_SIZE = 32
COLOR_PATCH_SIZE = 64

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

def calc_color_checker_pattern_block_st_pos(
        color_idx=1, width=1920, grey_block_size=32, color_block_size=64):
    color_checker_h_num = 6
    block_num_h = calc_block_num_h(width=width, block_size=grey_block_size)
    st_pos_v_offset = ((1023 // block_num_h) + 8) * grey_block_size
    st_pos_v = (color_idx // color_checker_h_num) * color_block_size\
        + st_pos_v_offset
    st_pos_h = (color_idx % color_checker_h_num) * color_block_size
    st_pos = (st_pos_h, st_pos_v)

    return st_pos


def calc_rgbmyc_pattern_block_st_pos(
        color_idx=1, width=1920, grey_block_size=32, color_block_size=64):
    block_num_h = calc_block_num_h(width=width, block_size=grey_block_size)
    st_pos_v = ((1023 // block_num_h) + 4) * grey_block_size
    st_pos_h = (color_idx % block_num_h) * color_block_size
    st_pos = (st_pos_h, st_pos_v)

    return st_pos


def calc_block_num_h(width=1920, block_size=64):
    return width // block_size


def calc_gradation_pattern_block_st_pos(code_value, width, block_size):
    block_num_h = calc_block_num_h(width=width, block_size=block_size)
    st_pos_h = (code_value % block_num_h) * block_size
    st_pos_v = (code_value // block_num_h) * block_size
    st_pos = (st_pos_h, st_pos_v)

    return st_pos


def calc_10bit_ramp_center_pos(width, block_size):
    pos_h_buf = []
    pos_v_buf = []
    
    for cv in range(1024):
        st_pos = calc_gradation_pattern_block_st_pos(
            code_value=cv, width=width, block_size=block_size
        )
        offset = block_size//2
        pos_h = st_pos[0] + offset
        pos_v = st_pos[1] + offset
        pos_h_buf.append(pos_h)
        pos_v_buf.append(pos_v)

    pos_h = np.array(pos_h_buf, dtype=np.uint16)
    pos_v = np.array(pos_v_buf, dtype=np.uint16)

    return pos_h, pos_v


def get_10bit_ramp_from_img(img: np.ndarray) -> np.ndarray:
    pos_h, pos_v = calc_10bit_ramp_center_pos(width=img.shape[1], block_size=GRAY_PATCH_SIZE)
    ramp_10bit = img[pos_v, pos_h]

    return ramp_10bit


def create_10bit_pattern():
    output_fname_png = "./img/src_img.png"
    output_fname_tif = "./img/src_img.tif"
    output_fname_dpx = "./img/src_img.dpx"
    output_fname_exr = "./img/src_img.exr"
    width = IMAGE_WIDTH
    height = IMAGE_HEIGHT
    cv_max = 1023
    gray_patch_size = GRAY_PATCH_SIZE
    color_patch_size = COLOR_PATCH_SIZE
    img = np.zeros((height, width, 3), dtype=np.uint16)
    grey_block_img_base = np.ones((gray_patch_size, gray_patch_size, 3), dtype=np.uint16)
    color_block_img_base = np.ones((color_patch_size, color_patch_size, 3), dtype=np.uint16)

    for cv in range(cv_max+1):
        block_img = grey_block_img_base * cv
        st_pos = calc_gradation_pattern_block_st_pos(
            code_value=cv, width=width, block_size=gray_patch_size
        )
        tpg.merge(img, block_img, st_pos)

    # RGBMYC
    color_list = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1]],
        dtype=np.uint8
    )
    for color_idx in range(len(color_list)):
        block_img = color_block_img_base * color_list[color_idx] * cv_max
        st_pos = calc_rgbmyc_pattern_block_st_pos(
            color_idx=color_idx, width=width,
            grey_block_size=gray_patch_size, color_block_size=color_patch_size
        )
        tpg.merge(img, block_img, st_pos)

    # Color Checker
    color_checker_linear = tpg.generate_color_checker_rgb_value()
    rgb_value_gm24 = np.clip(color_checker_linear, 0.0, 1.0) ** (1/2.4)
    rgb_value_10bit = np.uint16(np.round(rgb_value_gm24 * cv_max))

    for color_idx in range(len(rgb_value_10bit)):
        block_img = color_block_img_base * rgb_value_10bit[color_idx]
        st_pos = calc_color_checker_pattern_block_st_pos(
            color_idx=color_idx, width=width,
            grey_block_size=gray_patch_size, color_block_size=color_patch_size
        )
        tpg.merge(img, block_img, st_pos)

    tpg.img_wirte_float_as_16bit_int(
        filename=output_fname_png, img_float=img/1023
    )
    tpg.img_wirte_float_as_16bit_int(
        filename=output_fname_tif, img_float=img/1023
    )
    bit_option = Image_Specification_Attribute("oiio:BitsPerSample", 10)
    write_image(
        img/1023, output_fname_dpx, bit_depth='uint16', attributes=[bit_option]
    )
    compression_option = Image_Specification_Attribute("compression", 'zip')
    write_image(
        img/1023, output_fname_exr, bit_depth='float32', attributes=[compression_option]
    )


def test_10bit_pattern():
    def float_to_10bit(x):
        return np.round(x * 1023).astype(np.uint16)
    x = np.arange(1024, dtype=np.uint16)
    ref_data = np.repeat(x[..., np.newaxis], 3, axis=-1)

    dpx_10bit = float_to_10bit(get_10bit_ramp_from_img(read_image("./img/src_img.dpx")))
    png_10bit = float_to_10bit(get_10bit_ramp_from_img(read_image("./img/src_img.png")))
    tif_10bit = float_to_10bit(get_10bit_ramp_from_img(read_image("./img/src_img.tif")))
    exr_10bit = float_to_10bit(get_10bit_ramp_from_img(read_image("./img/src_img.exr")))

    np.testing.assert_array_equal(dpx_10bit, ref_data)
    np.testing.assert_array_equal(png_10bit, ref_data)
    np.testing.assert_array_equal(tif_10bit, ref_data)
    np.testing.assert_array_equal(exr_10bit, ref_data)


def calc_rgb_to_ycbcr_matrix(gamut="bt.709"):
    if gamut == "bt.709":
        coef_y = np.array([0.2126, 0.7152, 0.0722])
    elif gamut == "bt.2020":
        coef_y = np.array([0.2627, 0.6780, 0.0593])
    else:
        raise ValueError("Invalid `gamut` parameter")
    div_cb = (coef_y[0] + coef_y[1]) * 2
    div_cr = (coef_y[1] + coef_y[2]) * 2
    coef_cb = (np.array([0.0, 0.0, 1.0]) - coef_y) / div_cb
    coef_cr = (np.array([1.0, 0.0, 0.0]) - coef_y) / div_cr

    mtx = np.vstack([coef_y, coef_cb, coef_cr])

    return mtx


def dpx10_bit_to_i010(gamut="bt.709"):
    dpx_fname = "./img/src_img.dpx"
    img_10bit = read_image(dpx_fname)
    rgb_to_ycbcr_mtx = calc_rgb_to_ycbcr_matrix(gamut=gamut)
    ycbcr = vecmul(rgb_to_ycbcr_mtx, img_10bit)
    y = (ycbcr[:, :, 0].ravel() * 219 + 16) * 4
    cb = (ycbcr[::2, ::2, 1].ravel() * 224 + 128) * 4
    cr = (ycbcr[::2, ::2, 2].ravel() * 224 + 128) * 4

    img_array = np.round(np.concatenate([y, cb, cr])).astype(np.uint16)

    return img_array


def create_10bit_pattern_i010_format(fps=24, length_sec=5):
    yuv_fname = "./raw/src_1920x1080_I010.yuv"
    img_array = dpx10_bit_to_i010(gamut='bt.709')
    total_frames = int(fps * length_sec)

    frame = np.ascontiguousarray(img_array.astype('<u2', copy=False))
    with open(yuv_fname, 'wb') as f:
        for _ in range(total_frames):
            frame.tofile(f)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_10bit_pattern()
    create_10bit_pattern_i010_format()
    # test_10bit_pattern()
