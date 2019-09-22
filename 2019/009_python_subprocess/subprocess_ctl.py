#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Matplotlib の Animation 機能を試す
"""

# 外部ライブラリのインポート
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from colour.models import eotf_ST2084


REF_WHITE_LUMINANCE = 100


def img_file_read(filename):
    """
    OpenCV の BGR 配列が怖いので並べ替えるwrapperを用意。
    今回はついでに正規化も済ませている
    """
    img = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)

    if img is not None:
        if img.dtype == np.uint8:
            return img[:, :, ::-1] / 0xFF
        elif img.dtype == np.uint16:
            return img[:, :, ::-1] / 0xFFFF
        else:
            print("warning: loaded file is not normalized.")
            return img[:, :, ::-1]
    else:
        return img


def plot_init():
    """
    プロットに必要な諸々の初期化をする。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    x = get_log10_x_scale(
        sample_num=1024, ref_val=1.0, min_exposure=-2.0, max_exposure=2.0)

    return fig, ax1, ax2, x


def update(frame_idx, ax1, ax2, x, img):
    ax1.clear()
    ax2.clear()
    ax1.grid(True, which='both')
    ax1.set_xscale('log', basex=10.0)
    x_val = [1.0 * (10 ** (x - 3)) for x in range(7)]
    x_caption = [r"$10^{{{}}}$".format(x - 3) for x in range(7)]
    ax1.set_xticks(x_val, x_caption)

    peak_luminance = (frame_idx + 1) * 10
    out_img = img / peak_luminance
    out_img = out_img.clip(0.0, 1.0)
    x2 = x / (peak_luminance / REF_WHITE_LUMINANCE)
    y2 = x2 ** (1 / 2.4)
    y2 = y2.clip(0.0, 1.0)

    ax1.plot(x, y2)
    ax2.imshow(out_img ** (1.0 / 2.4))
    plt.title("frame={}".format(frame_idx))


def main_func():
    """
    画像の OETF のかけかたを連続的に変化させたアニメーションを作る。
    """
    # 画像準備
    img = img_file_read("./img/src_hdr10.tiff")
    img_linear = eotf_ST2084(img)

    # plot の初期化
    fig, ax1, ax2, x = plot_init()

    # plot の実行
    ani = animation.FuncAnimation(
        fig, update, interval=100, frames=30, fargs=[ax1, ax2, x, img_linear])
    # plt.show()
    ani.save("hoge.mp4", writer='ffmpeg')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
