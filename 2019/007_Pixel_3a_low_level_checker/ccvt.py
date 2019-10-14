#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Turbo を使ってHDR10信号の輝度マップを作成する
"""

# 外部ライブラリのインポート
import os
import numpy as np
import cv2
from colour.models import RGB_to_RGB, BT2020_COLOURSPACE, BT709_COLOURSPACE

# 自作ライブラリのインポート
import transfer_functions as tf


def main_func():
    fig_img = cv2.imread('./test.tiff',
                         cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[..., ::-1]
    fig_img = fig_img / 0xFFFF
    linear = tf.eotf(fig_img, tf.SRGB)
    bt709 = RGB_to_RGB(linear, BT2020_COLOURSPACE, BT709_COLOURSPACE)
    sRGB = tf.oetf(bt709, tf.SRGB)
    sRGB = np.clip(sRGB, 0.0, 1.0)
    cv2.imwrite("./bt709.tiff", np.uint16(np.round(sRGB * 0xFFFF))[:, :, ::-1])


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
