# -*- coding: utf-8 -*-
"""
Check Image-Data module of colour-science.
==========================================

Check the behavior of the follwing functions.

* write/read 8bit png files.
* write/read 16bit png files.
* write/read 8bit tiff files.
* write/read 16bit tiff files.
* write/read 10bit dpx files.
* write/read 12bit dpx files.
* write/read 16bit exr files.
* write/read 32bit exr files.

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour import write_image, read_image
from colour.io.image import ImageAttribute_Specification

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def check_int_8bit(filename):
    """
    8it 整数型データ保存の確認。
    """
    one_color = np.arange(0x100).reshape((16, 16, 1))
    img = np.uint8(np.dstack((np.zeros_like(one_color), one_color, one_color)))
    write_image(img/0xFF, filename, bit_depth='uint8')

    after_img = read_image(filename, bit_depth='uint8')
    abs_diff = abs(np.int64(img) - np.int64(after_img))
    print(np.all(abs_diff == 0))


def check_int_10bit(filename):
    """
    10it 整数型データ保存の確認。
    """
    one_color = np.arange(0x400).reshape((32, 32, 1))
    img = np.uint16(
        np.dstack((one_color, np.zeros_like(one_color), one_color)))
    bit_option = ImageAttribute_Specification("oiio:BitsPerSample", 10)
    write_image(
        img/0x3FF, filename, bit_depth='uint16', attributes=[bit_option])

    after_img = read_image(filename, bit_depth='float32')
    after_img = np.uint16(np.round(after_img * 0x3FF))
    abs_diff = abs(np.int64(img) - np.int64(after_img))
    print(np.all(abs_diff == 0))


def check_int_12bit(filename):
    """
    12it 整数型データ保存の確認。
    """
    one_color = np.arange(0x1000).reshape((64, 64, 1))
    img = np.uint16(
        np.dstack((one_color, np.zeros_like(one_color), one_color)))
    bit_option = ImageAttribute_Specification("oiio:BitsPerSample", 12)
    write_image(
        img/0xFFF, filename, bit_depth='uint16', attributes=[bit_option])

    after_img = read_image(filename, bit_depth='float32')
    after_img = np.uint16(np.round(after_img * 0xFFF))
    abs_diff = abs(np.int64(img) - np.int64(after_img))
    print(np.all(abs_diff == 0))


def check_int_16bit(filename):
    """
    16bit 整数型データ保存の確認。
    """
    one_color = np.arange(0x10000).reshape((256, 256, 1))
    img = np.uint16(
        np.dstack((one_color, one_color, np.zeros_like(one_color))))
    write_image(img/0xFFFF, filename, bit_depth='uint16')

    after_img = read_image(filename, bit_depth='uint16')
    abs_diff = abs(np.int64(img) - np.int64(after_img))
    print(np.all(abs_diff == 0))


def check_half_float(filename):
    """
    16bit 浮動小数点データ保存の確認。
    """
    one_color = np.linspace(0, 2, 0x10000, dtype=np.float16)
    one_color = one_color.reshape((256, 256, 1))
    img = np.dstack(
        (np.zeros_like(one_color), one_color, np.zeros_like(one_color)))
    write_image(img, filename, bit_depth='float16')

    """
    本当は read_image も bit_depth='float16' で確認したかったが、
    謎の不具合に遭遇したので 'float32' で read している。
    この不具合は colour-science は無関係っぽい。
    自作の OpenImageIO ラッパーでも同様の症状が発生したため。
    """
    after_img = read_image(filename, bit_depth='float32')

    abs_diff = abs(img - after_img)
    print(np.all(abs_diff == 0))  # 浮動小数点だが理論的には1bitも変化しないはずなので…


def check_float(filename):
    """
    32bit 浮動小数点データ保存の確認。
    """
    one_color = np.linspace(0, 3, 0x10000, dtype=np.float32)
    one_color = one_color.reshape((256, 256, 1))
    img = np.dstack(
        (np.zeros_like(one_color), one_color, np.zeros_like(one_color)))
    write_image(img, filename, bit_depth='float32')

    after_img = read_image(filename, bit_depth='float32')
    abs_diff = abs(img - after_img)
    print(np.all(abs_diff == 0))  # 浮動小数点だが理論的には1bitも変化しないはずなので…


def main_func():
    check_int_8bit(filename="./blog_img/8bit_tiff_sample.tiff")
    check_int_16bit(filename="./blog_img/16bit_tiff_sample.tiff")
    check_int_8bit(filename="./blog_img/8bit_tiff_sample.tif")
    check_int_16bit(filename="./blog_img/16bit_tiff_sample.tif")
    check_int_8bit(filename="./blog_img/8bit_png_sample.png")
    check_int_16bit(filename="./blog_img/16bit_png_sample.png")

    check_int_10bit(filename="./blog_img/10bit_dpx_sample.dpx")
    check_int_12bit(filename="./blog_img/12bit_dpx_sample.dpx")

    check_half_float(filename="./blog_img/16bit_float_exr_sample.exr")
    check_float(filename="./blog_img/32bit_float_exr_sample.exr")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
