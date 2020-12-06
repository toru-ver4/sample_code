# -*- coding: utf-8 -*-
"""

==========

"""

# import standard libraries
import os
import cv2
import subprocess
from pathlib import Path

# import third-party libraries
import numpy as np
import matplotlib.pyplot as plt

# import my libraries
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

FPS = 24
TOTAL_SEC = 1
TOTAL_FRAME = FPS * TOTAL_SEC
MIN_16BIT_VAL = 0xFC00
# MAX_16BIT_VAL = 0xFC10
MAX_16BIT_VAL = 0x10000
STEP_16BIT_VAL = 0x0004
SRC_PNG_DIR = "/work/overuse/2020/030_10bit_ramp_from_12bit_HEVC/img_seq/"
DST_MP4_DIR = "/work/overuse/2020/030_10bit_ramp_from_12bit_HEVC/mp4/"
DST_TIFF_DIR = "/work/overuse/2020/030_10bit_ramp_from_12bit_HEVC/tiff/"


def create_bg_image(
        width=1920, height=1080, seed=0, min_value=0x0000, max_value=0xFFFF):
    """
    width : int
        width of the image
    height : int
        height of the image
    seed : int
        seed
    min_vallue : int
        minimum value of the image
    max_vallue : int
        maximum value of the image

    Examples
    --------
    >>> for idx in range(24):
    >>>     bg_img = create_bg_image(
    ...         width=1920, height=1080, seed=idx,
    ...         min_value=0x0000, max_value=0xFFFF)
    >>>     fname = f"./img/bg_img_{idx:04d}.png"
    >>>     write_image(image=bg_image, path=fname, bit_depth='uint16')
    """
    # np.random.seed(seed=seed)
    bg_img = np.random.randint(
        min_value, max_value, (height, width, 3)).astype(np.uint16)
    return bg_img


def create_bg_image_with_ramp(
        width=1920, height=1080, seed=0, min_value=0x0000, max_value=0xFFFF):
    bg_img = create_bg_image(
        width=width, height=height, seed=seed,
        min_value=min_value, max_value=max_value)
    ramp = np.uint16(np.round(np.arange(1024) / 1023 * max_value))
    ramp = np.dstack((ramp, ramp, ramp))
    bg_img[0, :1024] = ramp

    return bg_img


def debug_create_uniform_distribution_frame(
        width=1920, height=1080, min_value=0x0000, max_value=0xFFFF):
    """
    width : int
        width of the image
    height : int
        height of the image
    min_vallue : int
        minimum value of the image
    max_vallue : int
        maximum value of the image
    """

    for idx in range(TOTAL_FRAME):
        bg_img = create_bg_image(
            width=width, height=height, seed=idx,
            min_value=min_value, max_value=max_value)
        fname = f"./img_seq/bg_img_{idx:04d}.png"
        cv2.imwrite(fname, bg_img[..., ::-1])


def make_base_fname():
    """
    Example
    -------
    >>> make_base_fname()
    "/work/overuse/2020/030_10bit_ramp_from_12bit_HEVC/img_seq/bg_img_max_{:04X}_{:04d}.png"
    """
    fname_base = SRC_PNG_DIR + "bg_img_max_{:04X}_{:04d}.png"

    return str(fname_base)


def save_src_png_as_sequence(
        width=1920, height=1080, min_value=0x0000, max_value=0xFFFF):
    """
    width : int
        width of the image
    height : int
        height of the image
    min_vallue : int
        minimum value of the image
    max_vallue : int
        maximum value of the image
    """
    for idx in range(TOTAL_FRAME):
        bg_img = create_bg_image_with_ramp(
            width=width, height=height, seed=idx,
            min_value=min_value, max_value=max_value)
        fname = make_base_fname().format(max_value, idx)
        print(fname)

        cv2.imwrite(fname, bg_img[..., ::-1])


def create_source_png_sequence(width=1920, height=1080):
    """
    width : int
        width of the image
    height : int
        height of the image
    """
    min_value = 0x0000
    for max_value in range(MIN_16BIT_VAL, MAX_16BIT_VAL, STEP_16BIT_VAL):
        print(f"{max_value:04X}")
        save_src_png_as_sequence(
            width=width, height=height,
            min_value=min_value, max_value=max_value)


def encode_src_sequence():
    for max_value in range(MIN_16BIT_VAL, MAX_16BIT_VAL, STEP_16BIT_VAL):
        in_fname_actual = make_base_fname().format(max_value, 0)
        in_fname_ffmpeg = in_fname_actual.replace("0000", "%04d")
        out_fname = DST_MP4_DIR\
            + f"HEVC_yuv444p12le_max_0x{max_value:04X}.mp4"
        cmd = "ffmpeg"
        ops = [
            '-color_primaries', 'bt709', '-color_trc', 'bt709',
            '-colorspace', 'bt709',
            '-r', '24', '-i', in_fname_ffmpeg, '-c:v', 'libx265',
            '-profile:v', 'main444-12', '-pix_fmt', 'yuv444p12le',
            '-x265-params', 'lossless=1',
            '-color_primaries', 'bt709', '-color_trc', 'bt709',
            '-colorspace', 'bt709',
            str(out_fname), '-y'
        ]
        args = [cmd] + ops
        print(" ".join(args))
        subprocess.run(args)


def im_read_16bit_tiff(filename):
    # DaVinci の仕様？で "_" が抜けることがあるので
    # filename を補正する処理を追加
    file_path = Path(filename)
    if not file_path.exists():
        parent = file_path.parent
        name = file_path.name
        name = name.replace("_", "")
        filename = str(parent.joinpath(name))
    print(filename)
    img_16bit_int = cv2.imread(
        filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[..., ::-1]

    return img_16bit_int


def reproduce_ramp(max_value):
    """
    Parameters
    ----------
    max_value : int
        max value of the source png data
    """
    tiff_file = f"{DST_TIFF_DIR}{max_value:04X}_00086400.tif"
    img_16bit_int = im_read_16bit_tiff(tiff_file)
    img_10bit_int = np.uint16(np.round(img_16bit_int / 0xFFFF * 1023))
    ramp_reproduced_r = img_10bit_int[0, :1024, 0]  # green data
    ramp_reproduced_g = img_10bit_int[0, :1024, 1]  # green data
    ramp_reproduced_b = img_10bit_int[0, :1024, 2]  # green data

    is_rgb_same = np.array_equal(ramp_reproduced_r, ramp_reproduced_g)\
        & np.array_equal(ramp_reproduced_g, ramp_reproduced_b)

    if not is_rgb_same:
        raise BufferError("invalid ramp data") 

    return ramp_reproduced_g


def plot_reproduced_ramp_error(max_value_list, sum_diff_list):
    """
    Parameters
    ----------
    max_value_list : list
        list of "max value of the source png data".
    sum_diff_list : list
        list of "total error between source ramp and reproduced ramp".
    """
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="Relationship between maximum value and error",
        graph_title_size=None,
        xlabel="Maximum value",
        ylabel="Total error",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None,
        return_figure=True)
    ax1.plot(max_value_list, sum_diff_list)
    # plt.legend(loc='upper left')
    plt.savefig("./img/error_all_range.png", bbox_inches='tight')
    # plt.show()
    plt.close(fig)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="Relationship between maximum value and error",
        graph_title_size=None,
        xlabel="Maximum value",
        ylabel="Total error",
        axis_label_size=None,
        legend_size=17,
        xlim=[65270, 65430],
        ylim=[-10, 600],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None,
        return_figure=True)
    ax1.plot(max_value_list, sum_diff_list, '-o')
    ok_idx = (sum_diff_list == 0)
    ax1.plot(
        max_value_list[ok_idx], sum_diff_list[ok_idx],
        'o', ms=10, color=pu.RED, label="誤差が0となる16bitの最大値")
    plt.legend(loc='upper left')
    plt.savefig("./img/error_narrow_range.png", bbox_inches='tight')
    # plt.show()
    plt.close(fig)


def evaluate_reproduced_ramp():
    # ramp_src = np.arange(1024)
    # max_value_list = []
    # sum_diff_list = []
    # for max_value in range(MIN_16BIT_VAL, MAX_16BIT_VAL, STEP_16BIT_VAL):
    #     ramp_dst = reproduce_ramp(max_value)
    #     abs_sum_diff = np.sum(np.abs(np.int32(ramp_dst) - np.int32(ramp_src)))
    #     max_value_list.append(max_value)
    #     sum_diff_list.append(abs_sum_diff)

    # np.save('./max_value_list.npy', np.array(max_value_list))
    # np.save('./sum_diff_list.npy', np.array(sum_diff_list))

    max_value_list = np.load('./max_value_list.npy')
    sum_diff_list = np.load('./sum_diff_list.npy')

    plot_reproduced_ramp_error(
        max_value_list=max_value_list, sum_diff_list=sum_diff_list)


def debug_evaluate_ramp_from_ffff_normalized_hevc():
    img_16bit_int = im_read_16bit_tiff("./img/FFFF_normalized.tif")
    img_10bit_int = np.uint16(np.round(img_16bit_int / 0xFFFF * 1023))
    ramp_dst = img_10bit_int[972, 448:448+1024, 0]
    ramp_src = np.arange(1024)

    for sss, ddd in zip(ramp_src, ramp_dst):
        print(sss, ddd)


def main_func():
    # create_source_png_sequence(width=1280, height=720)
    # encode_src_sequence()
    # evaluate_reproduced_ramp()
    pass


def debug_func():
    # debug_create_uniform_distribution_frame(
    #     width=1920, height=1080, min_value=0x0000, max_value=0xFFFF)
    debug_evaluate_ramp_from_ffff_normalized_hevc()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
    # debug_func()
