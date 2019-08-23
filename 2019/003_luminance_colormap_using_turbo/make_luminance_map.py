#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Turbo を使ってHDR10信号の輝度マップを作成する
"""

# 外部ライブラリのインポート
import os
import cv2
import numpy as np
from colour.models import eotf_ST2084, oetf_ST2084
from colour.models import BT2020_COLOURSPACE
from colour import RGB_luminance
from scipy import interpolate
import turbo_colormap  # https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f

# 自作ライブラリのインポート


def preview_image(img, order='rgb', over_disp=False):
    if order == 'rgb':
        cv2.imshow('preview', img[:, :, ::-1])
    elif order == 'bgr':
        cv2.imshow('preview', img)
    else:
        raise ValueError("order parameter is invalid")

    if over_disp:
        cv2.resizeWindow('preview', )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_turbo_colormap():
    """
    Turbo の Colormap データを Numpy形式で取得する
    """
    return np.array(turbo_colormap.turbo_colormap_data)


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


def img_file_write(filename, img):
    """
    OpenCV の BGR 配列が怖いので並べ替えるwrapperを用意。
    """
    cv2.imwrite(filename, np.uint16(np.round(img[:, :, ::-1] * 0xFFFF)))


def calc_luminance_from_hdr10_still_image(img_name):
    """
    HDR10相当の静止画データから輝度を計算する。

    Parameters
    ----------
    filename : strings
        filename of the image.

    Returns
    -------
    ndarray
        Luminance data.

    """
    img_rgb = img_file_read(img_name)
    img_rgb_linear = eotf_ST2084(img_rgb)
    img_y_linear = RGB_luminance(
        RGB=img_rgb_linear, primaries=BT2020_COLOURSPACE.primaries,
        whitepoint=BT2020_COLOURSPACE.whitepoint)

    return img_y_linear


def log_y_to_turbo(log_y):
    """
    輝度データを Turbo で色付けする。
    輝度データは Non-Linear かつ [0:1] の範囲とする。

    scipy.interpolate.interp1d を使って、R, G, B の
    各種値を線形補間して使う。

    Parameters
    ----------
    log_y : ndarray
        A 1D-array luminance data.

    Returns
    -------
    ndarray
        Luminance data. The shape is (height, width, 3).
    """
    # Turbo データ準備
    turbo = get_turbo_colormap()
    if len(turbo.shape) != 2:
        print("warning: turbo shape is invalid.")

    # scipy.interpolate.interp1d を使って補間する準備
    x = np.linspace(0, 1, turbo.shape[0])
    func_rgb = [interpolate.interp1d(x, turbo[:, idx]) for idx in range(3)]

    out_rgb = [func(log_y) for func in func_rgb]

    return np.dstack(out_rgb)


def get_output_filename(in_filename):
    root, _ = os.path.splitext(in_filename)
    return root + "_turbo.tiff"


def main_func(in_img_fname="./img/pq_bt2020_d65_tp.tiff"):
    # HDR10 ファイルの Y成分を計算
    y = calc_luminance_from_hdr10_still_image(
        img_name=in_img_fname)

    # Y成分を Log2 or Log10 or PQ でエンコード
    log_y = oetf_ST2084(y)

    # Non-Linear 空間で Trubo 適用
    turbo_img = log_y_to_turbo(log_y=log_y)

    # プロット
    preview_image(turbo_img)

    # 保存
    out_fname = get_output_filename(in_img_fname)
    img_file_write(out_fname, turbo_img)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func(in_img_fname="./img/pq_bt2020_d65_tp.tiff")
    main_func(in_img_fname="./img/sample_hdr_000.tif")
