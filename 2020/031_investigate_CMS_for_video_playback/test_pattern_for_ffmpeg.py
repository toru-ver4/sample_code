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
import cv2
from colour.models import BT709_COLOURSPACE
from numpy.core.defchararray import center
import matplotlib.pyplot as plt

# import my libraries
import test_pattern_generator2 as tpg
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

FPS = 24
BIT_DEPTH = 8
CODE_VALUE_NUM = (2 ** BIT_DEPTH)
MAX_CODE_VALUE = CODE_VALUE_NUM - 1
TOTAL_SEC = 1
TOTAL_FRAME = FPS * TOTAL_SEC
COLOR_CHECKER_H_HUM = 6
RGBMYC_COLOR_LIST = np.array(
    [[1, 0, 0], [0, 1, 0], [0, 0, 1],
     [1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=np.uint8)
COLOR_CHECKER_LINEAR = tpg.generate_color_checker_rgb_value(
    color_space=BT709_COLOURSPACE)
SRC_PNG_DIR = "/work/overuse/2020/031_cms_for_video_playback/img_seq/"
DST_MP4_DIR = "/work/overuse/2020/031_cms_for_video_playback/mp4/"
DST_PNG_DIR = "/work/overuse/2020/031_cms_for_video_playback/mp4_to_png/"


def calc_block_num_h(width=1920, block_size=64):
    return width // block_size


def calc_gradation_pattern_block_st_pos(
        code_value=MAX_CODE_VALUE, width=1920, height=1080, block_size=64):
    block_num_h = calc_block_num_h(width=width, block_size=block_size)
    st_pos_h = (code_value % block_num_h) * block_size
    st_pos_v = (code_value // block_num_h) * block_size
    st_pos = (st_pos_h, st_pos_v)

    return st_pos


def calc_rgbmyc_pattern_block_st_pos(
        color_idx=1, width=1920, height=1080, block_size=64):
    block_num_h = calc_block_num_h(width=width, block_size=block_size)
    st_pos_v = ((MAX_CODE_VALUE // block_num_h) + 2) * block_size
    st_pos_h = (color_idx % block_num_h) * block_size
    st_pos = (st_pos_h, st_pos_v)

    return st_pos


def calc_color_checker_pattern_block_st_pos(
        color_idx=1, width=1920, height=1080, block_size=64):
    block_num_h = calc_block_num_h(width=width, block_size=block_size)
    st_pos_v_offset = ((MAX_CODE_VALUE // block_num_h) + 4) * block_size
    st_pos_v = (color_idx // COLOR_CHECKER_H_HUM) * block_size\
        + st_pos_v_offset
    st_pos_h = (color_idx % COLOR_CHECKER_H_HUM) * block_size
    st_pos = (st_pos_h, st_pos_v)

    return st_pos


def create_8bit_cms_test_pattern(width=1920, height=1080, block_size=64):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    block_img_base = np.ones((block_size, block_size, 3), dtype=np.uint8)

    # gradation
    for code_value in range(CODE_VALUE_NUM):
        block_img = block_img_base * code_value
        st_pos = calc_gradation_pattern_block_st_pos(
            code_value=code_value, width=width, height=height,
            block_size=block_size)
        tpg.merge(img, block_img, st_pos)

    # RGBMYC
    color_list = RGBMYC_COLOR_LIST
    for color_idx in range(len(color_list)):
        block_img = block_img_base * color_list[color_idx] * MAX_CODE_VALUE
        st_pos = calc_rgbmyc_pattern_block_st_pos(
            color_idx=color_idx, width=width, height=height,
            block_size=block_size)
        tpg.merge(img, block_img, st_pos)

    # Color Checker
    color_checker_linear_value = COLOR_CHECKER_LINEAR
    rgb_value_gm24 = np.clip(color_checker_linear_value, 0.0, 1.0) ** (1/2.4)
    rgb_value_8bit = np.uint8(np.round(rgb_value_gm24 * MAX_CODE_VALUE))

    for color_idx in range(len(rgb_value_8bit)):
        block_img = block_img_base * rgb_value_8bit[color_idx]
        st_pos = calc_color_checker_pattern_block_st_pos(
            color_idx=color_idx, width=width, height=height,
            block_size=block_size)
        tpg.merge(img, block_img, st_pos)

    return img


def make_src_tp_base_name():
    fname = "{src_png_dir}/src_grad_tp_{width}x{height}"
    fname += "_b-size_{block_size}_{frame_idx:04d}.png"

    return fname


def make_dst_png_tp_base_name():
    fname = "{dst_png_dir}/dst_grad_tp_{width}x{height}"
    fname += "_b-size_{block_size}_{frame_idx:04d}.png"

    return fname


def make_dst_dv17_tif_tp_base_name():
    fname = "{dst_png_dir}/dst_grad_tp_{width}x{height}"
    fname += "_b-size_{block_size}_dv17_decode_{frame_idx:04d}.tif"

    return fname


def make_dst_mp4_tp_base_name():
    fname = "{dst_mp4_dir}/src_grad_tp_{width}x{height}"
    fname += "_b-size_{block_size}_ffmpeg.mp4"

    return fname


def make_dst_hdr10_mp4_tp_base_name():
    fname = "{dst_mp4_dir}/src_grad_tp_{width}x{height}"
    fname += "_b-size_{block_size}_ffmpeg_HDR10.mp4"

    return fname


def create_gradation_pattern_sequence(
        width=1920, height=1080, block_size=64):
    for frame_idx in range(TOTAL_FRAME):
        img = create_8bit_cms_test_pattern(
            width=width, height=height, block_size=block_size)
        fname = make_src_tp_base_name().format(
            src_png_dir=SRC_PNG_DIR, width=width, height=height,
            block_size=block_size, frame_idx=frame_idx)
        # fname = f"{SRC_PNG_DIR}/src_grad_tp_{width}x{height}"
        # fname += f"_b-size_{block_size}_{frame_idx:04d}.png"
        cv2.imwrite(fname, img[..., ::-1])


def get_specific_pos_value(img, pos):
    """
    Parameters
    ----------
    img : ndarray
        image data.
    pos : list
        pos[0] is horizontal coordinate, pos[1] is verical coordinate.
    """
    return img[pos[1], pos[0]]


def read_code_value_from_gradation_pattern(
        fname=None, width=1920, height=1080, block_size=64):
    """
    Example
    -------
    >>> read_code_value_from_gradation_pattern(
    ...     fname="./data.png, width=1920, height=1080, block_size=64)
    {'ramp': array(
          [[[  0,   0,   0],
            [  1,   1,   1],
            [  2,   2,   2],
            [  3,   3,   3],
            ...
            [252, 252, 252],
            [253, 253, 253],
            [254, 254, 254],
            [255, 255, 255]]], dtype=uint8),
     'rgbmyc': array(
           [[255,   0,   0],
            [  0, 255,   0],
            [  0,   0, 255],
            [255,   0, 255],
            [255, 255,   0],
            [  0, 255, 255]], dtype=uint8),
     'colorchecker': array(
           [[123,  90,  77],
            [201, 153, 136],
            [ 99, 129, 161],
            [ 98, 115,  75],
            ...
            [166, 168, 168],
            [128, 128, 129],
            [ 91,  93,  95],
            [ 59,  60,  61]], dtype=uint8)}
    """
    # Gradation
    print(f"reading {fname}")
    img = cv2.imread(fname, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[:, :, ::-1]
    block_offset = block_size // 2
    ramp_value = np.zeros((1, CODE_VALUE_NUM, 3), dtype=np.uint8)
    for code_value in range(CODE_VALUE_NUM):
        st_pos = calc_gradation_pattern_block_st_pos(
            code_value=code_value, width=width, height=height,
            block_size=block_size)
        center_pos = (st_pos[0] + block_offset, st_pos[1] + block_offset)
        ramp_value[0, code_value] = get_specific_pos_value(img, center_pos)

    # RGBMYC
    rgbmyc_value = np.zeros_like(RGBMYC_COLOR_LIST)
    for color_idx in range(len(RGBMYC_COLOR_LIST)):
        st_pos = calc_rgbmyc_pattern_block_st_pos(
            color_idx=color_idx, width=width, height=height,
            block_size=block_size)
        center_pos = (st_pos[0] + block_offset, st_pos[1] + block_offset)
        rgbmyc_value[color_idx] = get_specific_pos_value(img, center_pos)
    rgbmyc_value.reshape((1, rgbmyc_value.shape[0], rgbmyc_value.shape[1]))

    # ColorChecker
    color_checker_value = np.zeros_like(COLOR_CHECKER_LINEAR, dtype=np.uint8)
    for color_idx in range(len(COLOR_CHECKER_LINEAR)):
        st_pos = calc_color_checker_pattern_block_st_pos(
            color_idx=color_idx, width=width, height=height,
            block_size=block_size)
        center_pos = (st_pos[0] + block_offset, st_pos[1] + block_offset)
        color_checker_value[color_idx] = get_specific_pos_value(
            img, center_pos)
    color_checker_value.reshape(
        (1, color_checker_value.shape[0], color_checker_value.shape[1]))

    return dict(
        ramp=ramp_value, rgbmyc=rgbmyc_value,
        colorchecker=color_checker_value)


def check_src_tp_code_value(width=1920, height=1080, block_size=64):
    fname = make_src_tp_base_name().format(
        src_png_dir=SRC_PNG_DIR, width=width, height=height,
        block_size=block_size, frame_idx=TOTAL_FRAME//2)
    code_value_data = read_code_value_from_gradation_pattern(
        fname=fname, width=width, height=height, block_size=block_size)

    ramp = code_value_data['ramp']
    rgbmyc = code_value_data['rgbmyc']
    color_checker = code_value_data['colorchecker']

    # Ramp
    x = np.arange(CODE_VALUE_NUM).astype(np.uint8)
    ramp_expected = np.dstack((x, x, x))
    if np.array_equal(ramp, ramp_expected):
        print("read data matched")
    else:
        raise ValueError("read data did not match!")


def debug_func():
    width = 1920
    height = 1080
    block_size = 64
    check_src_tp_code_value(
        width=width, height=height, block_size=block_size)


def encode_8bit_tp_src_with_ffmpeg(width=1920, height=1080, block_size=64):
    out_fname = make_dst_mp4_tp_base_name().format(
        dst_mp4_dir=DST_MP4_DIR, width=width, height=height,
        block_size=block_size)
    in_fname = make_src_tp_base_name().format(
        src_png_dir=SRC_PNG_DIR, width=width, height=height,
        block_size=block_size, frame_idx=0)
    in_fname_ffmpeg = in_fname.replace("0000", r"%4d")
    cmd = "ffmpeg"
    ops = [
        '-color_primaries', 'bt709', '-color_trc', 'bt709',
        '-colorspace', 'bt709',
        '-r', '24', '-i', in_fname_ffmpeg, '-c:v', 'libx264',
        '-movflags', 'write_colr',
        '-pix_fmt', 'yuv444p', '-qp', '0',
        '-color_primaries', 'bt709', '-color_trc', 'bt709',
        '-colorspace', 'bt709',
        str(out_fname), '-y'
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


def encode_8bit_tp_src_with_ffmpeg_hdr10(
        width=1920, height=1080, block_size=64):
    out_fname = make_dst_hdr10_mp4_tp_base_name().format(
        dst_mp4_dir=DST_MP4_DIR, width=width, height=height,
        block_size=block_size)
    in_fname = make_src_tp_base_name().format(
        src_png_dir=SRC_PNG_DIR, width=width, height=height,
        block_size=block_size, frame_idx=0)
    in_fname_ffmpeg = in_fname.replace("0000", r"%4d")
    cmd = "ffmpeg"
    ops = [
        '-color_primaries', 'bt2020', '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        '-r', '24', '-i', in_fname_ffmpeg,
        '-movflags', 'write_colr',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv444p', '-qp', '0',
        '-pix_fmt', 'yuv444p',
        '-color_primaries', 'bt2020', '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        str(out_fname), '-y'
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


def encode_8bit_tp_src_with_ffmpeg_hdr10_raw(
        width=1920, height=1080, block_size=64):
    out_fname = make_dst_hdr10_mp4_tp_base_name().format(
        dst_mp4_dir=DST_MP4_DIR, width=width, height=height,
        block_size=block_size)
    out_fname_raw = out_fname + ".h264"
    in_fname = make_src_tp_base_name().format(
        src_png_dir=SRC_PNG_DIR, width=width, height=height,
        block_size=block_size, frame_idx=0)
    in_fname_ffmpeg = in_fname.replace("0000", r"%4d")
    cmd = "ffmpeg"
    ops = [
        '-color_primaries', 'bt2020', '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        '-r', '24', '-i', in_fname_ffmpeg, '-c:v', 'libx264',
        '-pix_fmt', 'yuv444p', '-qp', '0',
        '-pix_fmt', 'yuv444p',
        '-color_primaries', 'bt2020', '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        str(out_fname_raw), '-y'
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


def encode_8bit_tp_src_with_ffmpeg_vp9(
        width=1920, height=1080, block_size=64):
    out_fname = make_dst_mp4_tp_base_name().format(
        dst_mp4_dir=DST_MP4_DIR, width=width, height=height,
        block_size=block_size)
    out_fname = out_fname.replace(".mp4", "_vp9.mp4")
    in_fname = make_src_tp_base_name().format(
        src_png_dir=SRC_PNG_DIR, width=width, height=height,
        block_size=block_size, frame_idx=0)
    in_fname_ffmpeg = in_fname.replace("0000", r"%4d")
    cmd = "ffmpeg"
    ops = [
        '-color_primaries', 'bt709', '-color_trc', 'bt709',
        '-colorspace', 'bt709',
        '-r', '24', '-i', in_fname_ffmpeg,
        '-movflags', 'write_colr',
        '-c:v', 'libvpx-vp9',
        '-pix_fmt', 'yuv444p', '-lossless', '1',
        '-bsf:v', 'vp9_metadata=color_space=bt2020',
        '-color_primaries', 'bt2020', '-color_trc', 'bt709',
        str(out_fname), '-y'
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


def decode_8bit_tp_src_with_ffmpeg(width=1920, height=1080, block_size=64):
    in_fname = make_dst_mp4_tp_base_name().format(
        dst_mp4_dir=DST_MP4_DIR, width=width, height=height,
        block_size=block_size)
    out_fname = make_dst_png_tp_base_name().format(
        dst_png_dir=DST_PNG_DIR, width=width, height=height,
        block_size=block_size, frame_idx=0)
    out_fname_ffmpeg = out_fname.replace("0000", r"%4d")
    cmd = "ffmpeg"
    ops = [
        '-i', in_fname, '-vsync', '0',
        str(out_fname_ffmpeg), '-y'
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


def compare_dv17_decode_data(width=1920, height=1080, block_size=64):
    fname = make_dst_dv17_tif_tp_base_name().format(
        dst_png_dir=DST_PNG_DIR, width=width, height=height,
        block_size=block_size, frame_idx=TOTAL_FRAME//2)
    code_value_data = read_code_value_from_gradation_pattern(
        fname=fname, width=width, height=height, block_size=block_size)

    ramp = code_value_data['ramp']

    # Ramp
    expected = np.arange(CODE_VALUE_NUM).astype(np.uint8)
    observed = ramp[..., 0].reshape((CODE_VALUE_NUM))

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="Decoded data by DaVinci 17 Beta 4",
        graph_title_size=None,
        xlabel="Input code value (8 bit)",
        ylabel="Output code value (8 bit)",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=[x * 32 for x in range(8)] + [255],
        ytick=[x * 32 for x in range(8)] + [255],
        xtick_size=None, ytick_size=None,
        linewidth=5,
        minor_xtick_num=None,
        minor_ytick_num=None,
        return_figure=True)
    # ax1.plot(expected, expected, label="Theoretical value")
    ax1.plot(expected, observed, label="Observed value")
    plt.legend(loc='upper left')
    plt.savefig("./img/dv17_decoded_data.png", bbox_inches='tight')
    plt.close(fig)


def compare_chrome_decode_data(width=1920, height=1080, block_size=64):
    fname = "./capture/chrome_sdr.png"
    code_value_data = read_code_value_from_gradation_pattern(
        fname=fname, width=width, height=height, block_size=block_size)

    ramp = code_value_data['ramp']

    # Ramp
    expected = np.arange(CODE_VALUE_NUM).astype(np.uint8)
    observed = ramp[..., 0].reshape((CODE_VALUE_NUM))

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="Decoded data by DaVinci 17 Beta 4",
        graph_title_size=None,
        xlabel="Input code value (8 bit)",
        ylabel="Output code value (8 bit)",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=[x * 32 for x in range(8)] + [255],
        ytick=[x * 32 for x in range(8)] + [255],
        xtick_size=None, ytick_size=None,
        linewidth=5,
        minor_xtick_num=None,
        minor_ytick_num=None,
        return_figure=True)
    # ax1.plot(expected, expected, label="Theoretical value")
    ax1.plot(expected, observed, label="Observed value")
    plt.legend(loc='upper left')
    plt.savefig("./img/chrome_decoded_data.png", bbox_inches='tight')
    plt.close(fig)


def main_func():
    width = 1920
    height = 1080
    block_size = 64
    # create_gradation_pattern_sequence(
    #     width=width, height=height, block_size=block_size)
    encode_8bit_tp_src_with_ffmpeg(
        width=width, height=height, block_size=block_size)
    # encode_8bit_tp_src_with_ffmpeg_hdr10(
    #     width=width, height=height, block_size=block_size)
    # encode_8bit_tp_src_with_ffmpeg_hdr10_raw(
    #     width=width, height=height, block_size=block_size)
    # decode_8bit_tp_src_with_ffmpeg(
    #     width=width, height=height, block_size=block_size)
    # compare_dv17_decode_data(
    #     width=width, height=height, block_size=block_size)
    # encode_8bit_tp_src_with_ffmpeg_vp9(
    #     width=width, height=height, block_size=block_size)
    # compare_chrome_decode_data(
    #     width=width, height=height, block_size=block_size)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
    # debug_func()
