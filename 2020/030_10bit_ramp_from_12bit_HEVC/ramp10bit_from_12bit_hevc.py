# -*- coding: utf-8 -*-
"""

==========

"""

# import standard libraries
import os
import cv2
import subprocess

# import third-party libraries
import numpy as np


# import my libraries

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
MAX_16BIT_VAL = 0x10000
STEP_16BIT_VAL = 0x0004
SRC_PNG_DIR = "/work/overuse/2020/030_10bit_ramp_from_12bit_HEVC/img_seq/"
DST_MP4_DIR = "/work/overuse/2020/030_10bit_ramp_from_12bit_HEVC/mp4/"
DST_TIFF_DIR = "../../../overuse/2020/030_10bit_ramp_from_12bit_HEVC/tiff/"


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


def reproduce_ramp():
    for max_value in range(MIN_16BIT_VAL, MAX_16BIT_VAL, STEP_16BIT_VAL):
        tiff_file = f"{DST_TIFF_DIR}{max_value:04X}_00086400.tif"
        print(tiff_file)
        img_16bit_int = cv2.imread(
            tiff_file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[..., ::-1]
        img_10bit_int = np.uint16(np.round(img_16bit_int / 0xFFFF * 1023))
        ramp_src = np.arange(1024)
        ramp_dst = img_10bit_int[0, :1024, 1]  # green data
        for idx, data in enumerate(ramp_dst):
            print(idx, data)

        # print(ramp_dst - ramp_src)
        break


def main_func():
    # create_source_png_sequence(width=1280, height=720)
    encode_src_sequence()
    # reproduce_ramp()


def debug_func():
    debug_create_uniform_distribution_frame(
        width=1920, height=1080, min_value=0x0000, max_value=0xFFFF)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
    # debug_func()
