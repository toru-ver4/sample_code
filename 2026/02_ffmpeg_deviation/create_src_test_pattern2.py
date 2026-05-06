# -*- coding: utf-8 -*-

# import standard libraries
import sys
import os
import shlex
import subprocess
from pathlib import Path

# import third-party libraries
import numpy as np
from colour import write_image, read_image
from colour.io.image import Image_Specification_Attribute
from colour.algebra import vecmul
from scipy import linalg

# import my libraries
import test_pattern_generator2 as tpg
from ffmpeg_analyze_common import (
    make_raw_yuv_name,
    make_n_bit_yuv420_name,
    make_n_bit_yuv422_name,
    make_n_bit_yuv444_name
)

GRAY_PATCH_SIZE = 16

IMAGE_WIDTH = 3840
IMAGE_HEIGHT = 2160

COLOR_LIST = [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1]]


def calc_color_checker_pattern_block_st_pos(
        color_idx=1, width=None, grey_block_size=32, color_block_size=64):
    color_checker_h_num = 6
    block_num_h = calc_block_num_h(width=width, block_size=grey_block_size)
    st_pos_v_offset = ((1023 // block_num_h) + 8) * grey_block_size
    st_pos_v = (color_idx // color_checker_h_num) * color_block_size\
        + st_pos_v_offset
    st_pos_h = (color_idx % color_checker_h_num) * color_block_size
    st_pos = (st_pos_h, st_pos_v)

    return st_pos


def calc_rgbmyc_pattern_block_st_pos(
        color_idx=1, width=None, grey_block_size=32, color_block_size=64):
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


def make_decoded_n_bit_yuv420_test_pattern_fname(bit_depth, gamut):
    fname = f"./img/ref_dst_img_v2_420-to-444_{bit_depth:02d}-bit_{gamut}.dpx"

    return fname


def make_decoded_n_bit_yuv422_test_pattern_fname(bit_depth, gamut):
    fname = f"./img/ref_dst_img_v2_422-to-444_{bit_depth:02d}-bit_{gamut}.dpx"

    return fname


def make_decoded_n_bit_yuv444_test_pattern_fname(bit_depth, gamut):
    fname = f"./img/ref_dst_img_v2_444-to-444_{bit_depth:02d}-bit_{gamut}.dpx"

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
    print(f"Output {output_fname}")
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


def test_n_bit_rgb444_test_pattern(bit_depth: int):
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


def _test_n_bit_decoded_rgb444_test_pattern(img_fname: str, bit_depth: int):
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

    # gray ramp test (tolerance is 1)
    np.testing.assert_allclose(read_data[0], ref_data[0], atol=1, rtol=0)

    # color ramp test (tolerance is 2)
    np.testing.assert_allclose(read_data[1:], ref_data[1:], atol=2, rtol=0)


def test_n_bit_decoded_yuv420_to_rgb444_test_pattern(bit_depth: int, gamut: str):
    img_fname = make_decoded_n_bit_yuv420_test_pattern_fname(bit_depth=bit_depth, gamut=gamut)
    _test_n_bit_decoded_rgb444_test_pattern(img_fname=img_fname, bit_depth=bit_depth)


def test_n_bit_decoded_yuv422_to_rgb444_test_pattern(bit_depth: int, gamut: str):
    img_fname = make_decoded_n_bit_yuv422_test_pattern_fname(bit_depth=bit_depth, gamut=gamut)
    _test_n_bit_decoded_rgb444_test_pattern(img_fname=img_fname, bit_depth=bit_depth)


def test_n_bit_decoded_yuv444_to_rgb444_test_pattern(bit_depth: int, gamut: str):
    img_fname = make_decoded_n_bit_yuv444_test_pattern_fname(bit_depth=bit_depth, gamut=gamut)
    _test_n_bit_decoded_rgb444_test_pattern(img_fname=img_fname, bit_depth=bit_depth)


def _n_bit_yuv_dtype(bit_depth: int):
    return np.uint8 if bit_depth == 8 else np.uint16


def _n_bit_yuv_dtype_str(bit_depth: int):
    return '<u1' if bit_depth == 8 else '<u2'


def _rgb444_to_yuv_n_bit(
        rgb_float: np.ndarray, gamut: str, bit_depth: int,
        chroma_v_step: int, chroma_h_step: int):
    rgb_to_ycbcr_mtx = calc_rgb_to_ycbcr_matrix(gamut=gamut)
    ycbcr = vecmul(rgb_to_ycbcr_mtx, rgb_float)
    to_int_coef = 2 ** (bit_depth - 8)

    y = (ycbcr[:, :, 0].ravel() * 219 + 16) * to_int_coef
    cb = (ycbcr[::chroma_v_step, ::chroma_h_step, 1].ravel() * 224 + 128) * to_int_coef
    cr = (ycbcr[::chroma_v_step, ::chroma_h_step, 2].ravel() * 224 + 128) * to_int_coef

    after_dtype = _n_bit_yuv_dtype(bit_depth=bit_depth)
    img_array = np.round(np.concatenate([y, cb, cr])).astype(after_dtype)

    return img_array


def rgb444_to_yuv420_n_bit(rgb_float: np.ndarray, gamut: str, bit_depth: int):
    return _rgb444_to_yuv_n_bit(
        rgb_float=rgb_float, gamut=gamut, bit_depth=bit_depth,
        chroma_v_step=2, chroma_h_step=2
    )


def rgb444_to_yuv422_n_bit(rgb_float: np.ndarray, gamut: str, bit_depth: int):
    return _rgb444_to_yuv_n_bit(
        rgb_float=rgb_float, gamut=gamut, bit_depth=bit_depth,
        chroma_v_step=1, chroma_h_step=2
    )


def rgb444_to_yuv444_n_bit(rgb_float: np.ndarray, gamut: str, bit_depth: int):
    return _rgb444_to_yuv_n_bit(
        rgb_float=rgb_float, gamut=gamut, bit_depth=bit_depth,
        chroma_v_step=1, chroma_h_step=1
    )


def _read_yuv_n_bit_data_as_yuv444(
        fname: str, bit_depth: int, width: int, height: int,
        chroma_v_step: int, chroma_h_step: int) -> np.ndarray:
    y_size = width * height
    uv_width = width // chroma_h_step
    uv_height = height // chroma_v_step
    uv_size = uv_width * uv_height
    total_samples = y_size + uv_size * 2
    dtype_str = _n_bit_yuv_dtype_str(bit_depth=bit_depth)

    frame = np.fromfile(fname, dtype=dtype_str, count=total_samples)
    if frame.size != total_samples:
        raise ValueError(
            f"Invalid frame size. expected={total_samples} samples, "
            f"actual={frame.size} samples."
        )
    
    y = frame[:y_size].reshape((height, width))
    u_src = frame[y_size:y_size + uv_size].reshape((uv_height, uv_width))
    v_src = frame[y_size + uv_size:].reshape((uv_height, uv_width))

    u = np.repeat(np.repeat(u_src, chroma_v_step, axis=0), chroma_h_step, axis=1)
    v = np.repeat(np.repeat(v_src, chroma_v_step, axis=0), chroma_h_step, axis=1)

    yuv444 = np.dstack([y, u, v])
    
    return yuv444


def read_yuv420_n_bit_data_as_yuv444(fname: str, bit_depth: int, width: int, height: int) -> np.ndarray:
    return _read_yuv_n_bit_data_as_yuv444(
        fname=fname, bit_depth=bit_depth, width=width, height=height,
        chroma_v_step=2, chroma_h_step=2
    )


def read_yuv422_n_bit_data_as_yuv444(fname: str, bit_depth: int, width: int, height: int) -> np.ndarray:
    return _read_yuv_n_bit_data_as_yuv444(
        fname=fname, bit_depth=bit_depth, width=width, height=height,
        chroma_v_step=1, chroma_h_step=2
    )


def read_yuv444_n_bit_data_as_yuv444(fname: str, bit_depth: int, width: int, height: int) -> np.ndarray:
    return _read_yuv_n_bit_data_as_yuv444(
        fname=fname, bit_depth=bit_depth, width=width, height=height,
        chroma_v_step=1, chroma_h_step=1
    )


def _create_n_bit_yuv_pattern(
        fps, bit_depth, gamut, length_sec, make_yuv_fname_func,
        rgb444_to_yuv_func):
    rgb_fname = make_n_bit_test_pattern_fname(bit_depth=bit_depth)
    rgb_float = read_image(rgb_fname)
    yuv_fname = make_yuv_fname_func(bit_depth=bit_depth, gamut=gamut)
    img_array = rgb444_to_yuv_func(rgb_float=rgb_float, gamut=gamut, bit_depth=bit_depth)
    total_frames = int(fps * length_sec)

    print(f"{rgb_fname} -> {yuv_fname}")

    dtype_str = _n_bit_yuv_dtype_str(bit_depth=bit_depth)
    frame = np.ascontiguousarray(img_array.astype(dtype_str, copy=False))
    with open(yuv_fname, 'wb') as f:
        for _ in range(total_frames):
            frame.tofile(f)


def create_n_bit_yuv420_pattern(fps=24, bit_depth=None, gamut="bt.709", length_sec=2):
    _create_n_bit_yuv_pattern(
        fps=fps, bit_depth=bit_depth, gamut=gamut, length_sec=length_sec,
        make_yuv_fname_func=make_n_bit_yuv420_name,
        rgb444_to_yuv_func=rgb444_to_yuv420_n_bit
    )


def create_n_bit_yuv422_pattern(fps=24, bit_depth=None, gamut="bt.709", length_sec=2):
    _create_n_bit_yuv_pattern(
        fps=fps, bit_depth=bit_depth, gamut=gamut, length_sec=length_sec,
        make_yuv_fname_func=make_n_bit_yuv422_name,
        rgb444_to_yuv_func=rgb444_to_yuv422_n_bit
    )


def create_n_bit_yuv444_pattern(fps=24, bit_depth=None, gamut="bt.709", length_sec=2):
    _create_n_bit_yuv_pattern(
        fps=fps, bit_depth=bit_depth, gamut=gamut, length_sec=length_sec,
        make_yuv_fname_func=make_n_bit_yuv444_name,
        rgb444_to_yuv_func=rgb444_to_yuv444_n_bit
    )


def yuv444_to_rgb444_float(yuv444_int: np.ndarray, gamut: str, bit_depth: int) -> np.ndarray:

    y_int = yuv444_int[:, :, 0]
    u_int = yuv444_int[:, :, 1]
    v_int = yuv444_int[:, :, 2]

    ratio = 2 ** (bit_depth - 8)

    y_float = (y_int.astype(np.int16) - 16 * ratio) / (219 * ratio)
    u_float = (u_int.astype(np.int16) - 128 * ratio) / (224 * ratio)
    v_float = (v_int.astype(np.int16) - 128 * ratio) / (224 * ratio)

    mtx = linalg.inv(calc_rgb_to_ycbcr_matrix(gamut=gamut))
    yuv = np.dstack([y_float, u_float, v_float])
    rgb_float = vecmul(mtx, yuv)

    return rgb_float


def decode_n_bit_yuv420_to_rgb444_with_default_fname(bit_depth, gamut):
    yuv420_fnmae = make_n_bit_yuv420_name(bit_depth=bit_depth, gamut=gamut)
    out_fname = make_decoded_n_bit_yuv420_test_pattern_fname(bit_depth=bit_depth, gamut=gamut)

    decode_n_bit_yuv420_to_rgb444(
        yuv420_fnmae=yuv420_fnmae, out_fname=out_fname, bit_depth=bit_depth, gamut=gamut
    )


def decode_n_bit_yuv422_to_rgb444_with_default_fname(bit_depth, gamut):
    yuv422_fnmae = make_n_bit_yuv422_name(bit_depth=bit_depth, gamut=gamut)
    out_fname = make_decoded_n_bit_yuv422_test_pattern_fname(bit_depth=bit_depth, gamut=gamut)

    decode_n_bit_yuv422_to_rgb444(
        yuv422_fnmae=yuv422_fnmae, out_fname=out_fname, bit_depth=bit_depth, gamut=gamut
    )


def decode_n_bit_yuv444_to_rgb444_with_default_fname(bit_depth, gamut):
    yuv444_fnmae = make_n_bit_yuv444_name(bit_depth=bit_depth, gamut=gamut)
    out_fname = make_decoded_n_bit_yuv444_test_pattern_fname(bit_depth=bit_depth, gamut=gamut)

    decode_n_bit_yuv444_to_rgb444(
        yuv444_fnmae=yuv444_fnmae, out_fname=out_fname, bit_depth=bit_depth, gamut=gamut
    )


def _decode_n_bit_yuv_to_rgb444(yuv_fname, out_fname, bit_depth, gamut, read_yuv_func):
    yuv444_int = read_yuv_func(
        fname=yuv_fname, bit_depth=bit_depth, width=IMAGE_WIDTH, height=IMAGE_HEIGHT
    )
    rgb_int = yuv444_to_rgb444_float(yuv444_int=yuv444_int, gamut=gamut, bit_depth=bit_depth)

    output_bit_depth = 'uint8' if bit_depth == 8 else 'uint16'
    write_image(rgb_int, out_fname, bit_depth=output_bit_depth)


def decode_n_bit_yuv420_to_rgb444(yuv420_fnmae, out_fname, bit_depth, gamut):
    _decode_n_bit_yuv_to_rgb444(
        yuv_fname=yuv420_fnmae, out_fname=out_fname, bit_depth=bit_depth, gamut=gamut,
        read_yuv_func=read_yuv420_n_bit_data_as_yuv444
    )


def decode_n_bit_yuv422_to_rgb444(yuv422_fnmae, out_fname, bit_depth, gamut):
    _decode_n_bit_yuv_to_rgb444(
        yuv_fname=yuv422_fnmae, out_fname=out_fname, bit_depth=bit_depth, gamut=gamut,
        read_yuv_func=read_yuv422_n_bit_data_as_yuv444
    )


def decode_n_bit_yuv444_to_rgb444(yuv444_fnmae, out_fname, bit_depth, gamut):
    _decode_n_bit_yuv_to_rgb444(
        yuv_fname=yuv444_fnmae, out_fname=out_fname, bit_depth=bit_depth, gamut=gamut,
        read_yuv_func=read_yuv444_n_bit_data_as_yuv444
    )


def _run_subprocess(args):
    print(shlex.join(args))
    subprocess.run(args, check=True)


def _make_ffmpeg_debug_mp4_fname(pix_fmt, bit_depth, gamut):
    target_dir = Path("./debug")
    target_dir.mkdir(parents=True, exist_ok=True)
    fname = target_dir / f"ref_3840x2160_{pix_fmt}{bit_depth}le_{gamut}_ffmpeg_check.mp4"

    return str(fname)


def _get_ffmpeg_color_options(gamut):
    if gamut == "bt.709":
        return {
            "primaries": "bt709",
            "trc": "bt709",
            "colorspace": "bt709",
        }
    elif gamut == "bt.2020":
        return {
            "primaries": "bt2020",
            "trc": "bt709",
            "colorspace": "bt2020nc",
        }
    else:
        raise ValueError("Invalid `gamut` parameter")


def _debug_yuv_using_ffmpeg(yuv_fname, mp4_fname, pix_fmt, bit_depth, gamut, fps, ffmpeg_bin):
    if not Path(yuv_fname).exists():
        raise FileNotFoundError(yuv_fname)

    color_options = _get_ffmpeg_color_options(gamut=gamut)
    pix_fmt = f"{pix_fmt}{bit_depth}le" if bit_depth != 8 else f"{pix_fmt}"
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", pix_fmt,
        "-video_size", f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}",
        "-framerate", str(fps),
        "-color_primaries", color_options["primaries"],
        "-color_trc", color_options["trc"],
        "-colorspace", color_options["colorspace"],
        "-color_range", "limited",
        "-i", yuv_fname,
        "-c:v", "libx265",
        "-x265-params", "lossless=1",
        "-pix_fmt", pix_fmt,
        "-color_primaries", color_options["primaries"],
        "-color_trc", color_options["trc"],
        "-colorspace", color_options["colorspace"],
        "-color_range", "limited",
        mp4_fname,
    ]
    _run_subprocess(cmd)


def debug_yuv422_yuv444_using_ffmpeg(
        bit_depth=10, gamut="bt.709", fps=24, ffmpeg_bin="ffmpeg"):
    yuv422_fname = make_n_bit_yuv422_name(bit_depth=bit_depth, gamut=gamut)
    yuv444_fname = make_n_bit_yuv444_name(bit_depth=bit_depth, gamut=gamut)
    mp4_422_fname = _make_ffmpeg_debug_mp4_fname(
        pix_fmt="yuv422p", bit_depth=bit_depth, gamut=gamut
    )
    mp4_444_fname = _make_ffmpeg_debug_mp4_fname(
        pix_fmt="yuv444p", bit_depth=bit_depth, gamut=gamut
    )

    _debug_yuv_using_ffmpeg(
        yuv_fname=yuv422_fname, mp4_fname=mp4_422_fname, pix_fmt="yuv422p",
        bit_depth=bit_depth, gamut=gamut, fps=fps, ffmpeg_bin=ffmpeg_bin
    )
    _debug_yuv_using_ffmpeg(
        yuv_fname=yuv444_fname, mp4_fname=mp4_444_fname, pix_fmt="yuv444p",
        bit_depth=bit_depth, gamut=gamut, fps=fps, ffmpeg_bin=ffmpeg_bin
    )


def create_test_pattern_all():
    bit_depth_list = [8, 10, 12]
    gamut_list = ["bt.709", 'bt.2020']
    for bit_depth in bit_depth_list:
        create_n_bit_rgb444_test_patten(bit_depth=bit_depth)
        for gamut in gamut_list:
            create_n_bit_yuv420_pattern(bit_depth=bit_depth, gamut=gamut, length_sec=0.5)
            create_n_bit_yuv422_pattern(bit_depth=bit_depth, gamut=gamut, length_sec=0.5)
            create_n_bit_yuv444_pattern(bit_depth=bit_depth, gamut=gamut, length_sec=0.5)


def test_test_pattern_all():
    bit_depth_list = [8, 10, 12]
    gamut_list = ["bt.709", 'bt.2020']
    for bit_depth in bit_depth_list:
        test_n_bit_rgb444_test_pattern(bit_depth=bit_depth)
        for gamut in gamut_list:
            decode_n_bit_yuv420_to_rgb444_with_default_fname(bit_depth=bit_depth, gamut=gamut)
            test_n_bit_decoded_yuv420_to_rgb444_test_pattern(bit_depth=bit_depth, gamut=gamut)

            decode_n_bit_yuv422_to_rgb444_with_default_fname(bit_depth=bit_depth, gamut=gamut)
            test_n_bit_decoded_yuv422_to_rgb444_test_pattern(bit_depth=bit_depth, gamut=gamut)

            decode_n_bit_yuv444_to_rgb444_with_default_fname(bit_depth=bit_depth, gamut=gamut)
            test_n_bit_decoded_yuv444_to_rgb444_test_pattern(bit_depth=bit_depth, gamut=gamut)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_10bit_pattern()
    # test_10bit_pattern()
    # create_10bit_pattern_i010_format()

    # create_test_pattern_all()
    # test_test_pattern_all()

    debug_yuv422_yuv444_using_ffmpeg(bit_depth=8)
    debug_yuv422_yuv444_using_ffmpeg(bit_depth=10)
    debug_yuv422_yuv444_using_ffmpeg(bit_depth=12)
