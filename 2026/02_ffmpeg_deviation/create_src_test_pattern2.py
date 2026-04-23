# -*- coding: utf-8 -*-

# import standard libraries
import sys
import os
from pathlib import Path

# import third-party libraries
import numpy as np
from colour import write_image, read_image
from colour.io.image import Image_Specification_Attribute
from colour.algebra import vecmul

# import my libraries
import test_pattern_generator2 as tpg
from ffmpeg_analyze_common import (
    make_raw_yuv_name,
    make_n_bit_yuv420_name
)

GRAY_PATCH_SIZE = 8

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

COLOR_LIST = [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1]]


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


def calc_block_num_h(width, block_size):
    return width // block_size


def calc_gradation_pattern_block_st_pos(code_value, width, block_size):
    block_num_h = calc_block_num_h(width=width, block_size=block_size)
    st_pos_h = (code_value % block_num_h) * block_size
    st_pos_v = (code_value // block_num_h) * block_size
    st_pos = [st_pos_h, st_pos_v]

    return st_pos


def calc_one_color_height(max_cv, width, block_size):
    st_v = calc_gradation_pattern_block_st_pos(
        code_value=max_cv, width=width, block_size=block_size
    )[1]
    one_color_height = st_v + block_size

    return one_color_height


def calc_gradation_pattern_block_st_pos_with_color_idx(
        code_value, width, block_size, color_idx, one_color_height):
    st_pos = calc_gradation_pattern_block_st_pos(code_value, width, block_size)
    st_pos[1] += one_color_height * color_idx

    return st_pos


def calc_n_bit_ramp_center_pos(width, block_size, max_cv):
    pos_h_buf = []
    pos_v_buf = []
    
    for cv in range(max_cv+1):
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


def calc_n_bit_ramp_center_pos_with_color_idx(
        width, block_size, color_idx, one_color_height, max_cv):
    pos_h, pos_v = calc_n_bit_ramp_center_pos(width, block_size, max_cv)
    pos_v += (color_idx * one_color_height)

    return pos_h, pos_v


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
    yuv_fname = make_raw_yuv_name()
    img_array = dpx10_bit_to_i010(gamut='bt.709')
    total_frames = int(fps * length_sec)

    frame = np.ascontiguousarray(img_array.astype('<u2', copy=False))
    with open(yuv_fname, 'wb') as f:
        for _ in range(total_frames):
            frame.tofile(f)


def make_n_bit_test_pattern_fname(bit_depth):
    fname = f"./img/src_img_v2_{bit_depth:02d}-bit.dpx"

    return fname


def create_n_bit_rgb444_test_patten(bit_depth=8) -> np.ndarray:
    width = IMAGE_WIDTH
    height = IMAGE_HEIGHT
    gray_patch_size = GRAY_PATCH_SIZE
    num_of_one_color_patch = 2 ** bit_depth
    max_cv = num_of_one_color_patch - 1
    img = np.zeros((height, width, 3), dtype=np.uint16)
    grey_block_img_base = np.ones((gray_patch_size, gray_patch_size, 3), dtype=np.uint16)

    one_color_height = calc_one_color_height(max_cv=max_cv, width=width, block_size=gray_patch_size)
    color_list = np.array(COLOR_LIST, dtype=np.uint8)
    for c_idx, color in enumerate(color_list):
        for cv in range(max_cv+1):
            block_img = grey_block_img_base * cv * color
            st_pos = calc_gradation_pattern_block_st_pos_with_color_idx(
                code_value=cv, width=width, block_size=gray_patch_size,
                color_idx=c_idx, one_color_height=one_color_height
            )
            tpg.merge(img, block_img, st_pos)

    # tpg.img_wirte_float_as_16bit_int("./aaa.png", img/255)

    output_fname = make_n_bit_test_pattern_fname(bit_depth=bit_depth)
    bit_depth_str = "uint8" if bit_depth == 8 else 'uint16'
    bit_option = Image_Specification_Attribute("oiio:BitsPerSample", bit_depth)
    write_image(
        img/max_cv, output_fname, bit_depth=bit_depth_str, attributes=[bit_option]
    )


def get_10bit_ramp_from_img(img: np.ndarray, bit_depth: int) -> np.ndarray:
    num_of_color = len(COLOR_LIST)
    num_of_patch = (2 ** bit_depth)
    max_cv = num_of_patch - 1
    width = img.shape[1]
    gray_patch_size = GRAY_PATCH_SIZE
    one_color_height = calc_one_color_height(
        max_cv=max_cv, width=width, block_size=gray_patch_size
    )
    ramp = np.zeros((num_of_color, num_of_patch, 3))
    for c_idx in range(num_of_color):
        pos_h, pos_v = calc_n_bit_ramp_center_pos_with_color_idx(
            width=width, block_size=GRAY_PATCH_SIZE,
            color_idx=c_idx, one_color_height=one_color_height,
            max_cv=max_cv
        )
        ramp[c_idx] = img[pos_v, pos_h]

    return ramp


def test_n_bit_pattern(bit_depth: int):
    img_fname = make_n_bit_test_pattern_fname(bit_depth=bit_depth)
    num_of_patch = 2 ** bit_depth
    max_cv = (2 ** bit_depth) - 1
    color_list = COLOR_LIST
    def float_to_n_bit(xx, max_cv):
        return np.round(xx * max_cv).astype(np.uint16)

    def mask_color(xx, color_list):
        for ii, color in enumerate(color_list):
            for jj in range(len(color)):
                xx[ii, :, jj] = xx[ii, :, jj] * color[jj]

    x = np.arange(num_of_patch, dtype=np.uint16)
    ref_data = np.repeat(x[..., np.newaxis], 3, axis=-1)
    ref_data = np.repeat(ref_data[np.newaxis, ...], 7, axis=0)
    mask_color(ref_data, color_list)

    read_data_float = get_10bit_ramp_from_img(read_image(img_fname), bit_depth)
    read_data = float_to_n_bit(read_data_float, max_cv)

    np.testing.assert_array_equal(read_data, ref_data)


def rgb444_to_yuv420_n_bit(rgb_float: np.ndarray, gamut: str, bit_depth: int):
    rgb_to_ycbcr_mtx = calc_rgb_to_ycbcr_matrix(gamut=gamut)
    ycbcr = vecmul(rgb_to_ycbcr_mtx, rgb_float)
    to_int_coef = 2 ** (bit_depth - 8)

    y = (ycbcr[:, :, 0].ravel() * 219 + 16) * to_int_coef
    cb = (ycbcr[::2, ::2, 1].ravel() * 224 + 128) * to_int_coef
    cr = (ycbcr[::2, ::2, 2].ravel() * 224 + 128) * to_int_coef

    img_array = np.round(np.concatenate([y, cb, cr])).astype(np.uint16)

    return img_array


def create_n_bit_yuv420_pattern(fps=24, bit_depth=None, gamut="bt.709", length_sec=2):
    rgb_fname = make_n_bit_test_pattern_fname(bit_depth=bit_depth)
    rgb_float = read_image(rgb_fname)
    yuv_fname = make_n_bit_yuv420_name(bit_depth=bit_depth)
    img_array = rgb444_to_yuv420_n_bit(rgb_float=rgb_float, gamut=gamut, bit_depth=bit_depth)
    total_frames = int(fps * length_sec)

    frame = np.ascontiguousarray(img_array.astype('<u2', copy=False))
    with open(yuv_fname, 'wb') as f:
        for _ in range(total_frames):
            frame.tofile(f)


def create_test_pattern_all():
    bit_depth_list = [8, 10, 12]
    gamut_list = ["bt.709", 'bt.2020']
    for bit_depth in bit_depth_list:
        create_n_bit_rgb444_test_patten(bit_depth=bit_depth)
        for gamut in gamut_list:
            create_n_bit_yuv420_pattern(bit_depth=bit_depth, gamut=gamut)


def test_test_pattern_all():
    bit_depth_list = [8, 10, 12]
    gamut_list = ["bt.709", 'bt.2020']
    for bit_depth in bit_depth_list:
        test_n_bit_pattern(bit_depth=bit_depth)
        for gamut in gamut_list:
            pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_10bit_pattern()
    # test_10bit_pattern()
    # create_10bit_pattern_i010_format()

    create_test_pattern_all()
    test_test_pattern_all()
