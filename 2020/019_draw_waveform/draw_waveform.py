# -*- coding: utf-8 -*-
"""
波形モニターっぽい画像を作る
===========================


"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
import cv2

# import my libraries
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


COLOR_BAR_FILE = "./HDTV_COLOR_BAR_RGB.tiff"
# COLOR_BAR_FILE = "./riku_boost.tif"

IMREAD_16BIT_FLAG = (cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)


def draw_waveform(src_img, intensity=1.0, bit_depth=10):
    """
    dst_img に waveform を書く。0.0-1.0 のバッファ

    Parameters
    ----------
    src_img : array_like
        描画対象の画像
    intensity : float
        waveform の濃度。
        1.0 なら 10% が重なると 1.0 の明るさになる。
        0.1 なら 100% が重なると 1.0 の明るさになる。
        10.0 なら 1% が重なると 1.0の明るさになる。
    """
    max_value = (2 ** bit_depth) - 1
    dst_height = 2 ** bit_depth
    dst_width = src_img.shape[1]
    src_height = src_img.shape[0]
    src_width = src_img.shape[1]
    dst_img = np.zeros((dst_height, dst_width, 3))
    alpha = intensity / src_height * 10

    for v_idx in range(src_height):
        print(v_idx)
        for c_idx in range(3):
            h_idx = np.arange(src_width)
            dst_v_idx = max_value - src_img[v_idx, :, c_idx]
            dst_img[dst_v_idx, h_idx, c_idx] += alpha

    dst_img = np.clip(dst_img, 0, 1)

    r_rate = dst_img[..., 0]
    g_rate = dst_img[..., 1]
    b_rate = dst_img[..., 2]

    r_img = np.ones_like((dst_img)) * np.array([255, 75, 0]) / 255
    g_img = np.ones_like((dst_img)) * np.array([3, 175, 122]) / 255
    b_img = np.ones_like((dst_img)) * np.array([77, 196, 255]) / 255

    r_img = cv2.resize(r_img * r_rate[:, :, np.newaxis], (1024//3, 768))
    g_img = cv2.resize(g_img * g_rate[:, :, np.newaxis], (1024//3, 768))
    b_img = cv2.resize(b_img * b_rate[:, :, np.newaxis], (1024//3, 768))
    separate = np.hstack((r_img, g_img, b_img))

    dst_img_all = cv2.resize(dst_img, (1024, 768))

    tpg.preview_image(dst_img)
    tpg.preview_image(dst_img_all)
    tpg.preview_image(separate)
    cv2.imwrite("hoge.png", np.uint8(np.round(separate[..., ::-1] * 0xFF)))
    cv2.imwrite("huga.png", np.uint8(np.round(dst_img_all[..., ::-1] * 0xFF)))


def main_func():
    # 前処理
    img_16bit = cv2.imread(COLOR_BAR_FILE, IMREAD_16BIT_FLAG)
    img_10bit = np.uint16(np.round(img_16bit / 64))
    img_10bit = np.clip(img_10bit, 0, 1023)

    # 本処理
    draw_waveform(img_10bit, intensity=20)

    # 後処理


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
