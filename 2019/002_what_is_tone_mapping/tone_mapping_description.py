#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
トーンマッピングの説明用の図を作るなど。
"""

# 外部ライブラリのインポート
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from colour.models import eotf_ST2084, eotf_BT1886, oetf_ST2084
from colour.models import BT2020_COLOURSPACE, BT709_COLOURSPACE
from colour import RGB_luminance

# 自作ライブラリのインポート
import plot_utility as pu


SRC_FILE_NAME = "./img/src_st2084_bt2020_d65.tif"
DST_W_TM_FILE_NAME = "./img/dst_gamma2.4_bt709_d65_with_tone_mapping.tif"
DST_WO_TM_FILE_NAME = "./img/dst_gamma2.4_bt709_d65_without_tone_mapping.tif"
ST2084_RAMP_FILE_NAME = "./img/st2084_ramp.tif"
ST2084_TO_BT709_W_TM_FILE_NAME =\
    "./img/st2084_ramp_to_slog3_to_709_with_tone_mapping.tif"
NOMINAL_WHITE_LUMINANCE = 100
RGB_COLOUR_LIST = ["#FF4800", "#03AF7A", "#005AFF"]


def img_file_read(filename):
    """
    OpenCV の BGR 配列が怖いので並べ替えるwrapperを用意。
    """
    img = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)

    if img is not None:
        return img[:, :, ::-1]
    else:
        return img


def img_file_write(filename, img):
    """
    OpenCV の BGR 配列が怖いので並べ替えるwrapperを用意。
    """
    cv2.imwrite(filename, img[:, :, ::-1])


def get_log10_x_scale(
        sample_num=32, ref_val=1.0, min_exposure=-4.0, max_exposure=4.0):
    """
    Log10スケールのx軸データを作る。

    Examples
    --------
    >>> get_log10_x_scale(sample_num=10, min_exposure=-4.0, max_exposure=4.0)
    array([  1.00000000e-04   7.74263683e-04   5.99484250e-03   4.64158883e-02
             3.59381366e-01   2.78255940e+00   2.15443469e+01   1.66810054e+02
             1.29154967e+03   1.00000000e+04])
    """
    x_min = np.log10(ref_val * (10 ** min_exposure))
    x_max = np.log10(ref_val * (10 ** max_exposure))
    x = np.linspace(x_min, x_max, sample_num)

    return 10.0 ** x


def make_st2084_h_ramp():
    """
    トーンマッピングの特性をプロットするための、
    基準となるRampパターンを作成して保存する。

    ここで保存した TIFF ファイルに対して、Resolve で
    3DLUT を当てる。それは ST2084_TO_BT709_W_TM_FILE_NAME
    として保存しておく。
    """
    v = 1080
    h = 1920
    img_base = np.ones((v, h, 3), dtype=np.uint16)
    line_linear = get_log10_x_scale(
        sample_num=h, ref_val=1.0, min_exposure=-4.0, max_exposure=4.0)
    line_code_value = np.uint16(np.round(oetf_ST2084(line_linear) * 0xFFFF))
    line_rgb_code_value = np.dstack(
        (line_code_value, line_code_value, line_code_value))
    img = img_base * line_rgb_code_value
    img_file_write("./img/st2084_ramp.tif", img)


def plot_tone_mapping_characteristics():
    """
    今回使用したトーンマッピングの特性をプロットする。
    """
    src_code_value = img_file_read(ST2084_RAMP_FILE_NAME) / 0xFFFF
    dst_with_tone_mapping_code_value =\
        img_file_read(ST2084_TO_BT709_W_TM_FILE_NAME) / 0xFFFF
    src_luminance = eotf_ST2084(src_code_value)
    dst_with_tone_mapping_luminance =\
        eotf_BT1886(dst_with_tone_mapping_code_value) * NOMINAL_WHITE_LUMINANCE

    src_luminance_line = src_luminance[0, :, 0].flatten()
    dst_with_tone_mapping_luminance_line =\
        dst_with_tone_mapping_luminance[0, :, 0].flatten()
    dst_without_tone_mapping_luminance_line =\
        np.clip(src_luminance_line.copy(), 0.0, NOMINAL_WHITE_LUMINANCE)

    x = src_luminance_line.copy()

    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(12, 10),
        graph_title="Tone mapping characteristics",
        graph_title_size=None,
        xlabel="Input luminance [cd/m2]",
        ylabel="Output luminance [cd/m2]",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.set_yscale('log', basey=10.0)
    ax1.set_xscale('log', basex=10.0)
    x_val = [1.0 * (10 ** (x - 4)) for x in range(9)]
    x_caption = [r"$10^{{{}}}$".format(x - 4) for x in range(9)]
    plt.xticks(x_val, x_caption)
    y_val = [1.0 * (10 ** (x - 4)) for x in range(9)]
    y_caption = [r"$10^{{{}}}$".format(x - 4) for x in range(9)]
    plt.yticks(y_val, y_caption)
    ax1.plot(x, src_luminance_line, label="src_hdr", color='#606060', lw=4)
    ax1.plot(x, dst_with_tone_mapping_luminance_line, '--',
             color=RGB_COLOUR_LIST[0],
             label="dst_sdr_with_tone_mapping")
    ax1.plot(x, dst_without_tone_mapping_luminance_line, '--',
             label="dst_sdr_without_tone_mapping", color=RGB_COLOUR_LIST[1])
    plt.legend(loc='upper left')
    plt.savefig("./figures/tone_mapping_characteristics.png",
                bbox_inches='tight', pad_inches=0.1)
    plt.show()


def get_luminance_from_img(
        img, primaries=BT709_COLOURSPACE.primaries,
        whitepoint=BT709_COLOURSPACE.whitepoint):
    """
    R, G, B の画像データを輝度成分(Y)に変換する。
    作ってから気づいたけど、これ何もしてないね。
    作る意味なかった…。

    Parameters
    ----------
    img : ndarray
        An image data.
    primaries : ndarray
        rgb primaries.
    whitepoint : ndarray
        white point.

    Returns
    -------
    ndarray
        Luminance data.

    Examples
    --------
    >>> img = np.array(
    ...     [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
    ...      [0.0, 0.0, 1.0], [0.5, 0.5, 0.5]])
    >>> primaries = np.array([[ 0.64,  0.33], [ 0.3 ,  0.6 ], [ 0.15,  0.06]])
    >>> whitepoint = np.array([0.3127, 0.3290])
    >>> get_luminance(img, primaries, whitepoint)
    array([ 0.21263901,  0.71516868,  0.07219232,  0.5       ])
    """
    return RGB_luminance(img, primaries, whitepoint)


def get_line_luminance_from_file(
        file_name, v_pos=594,
        eotf=eotf_BT1886,
        primaries=BT709_COLOURSPACE.primaries,
        whitepoint=BT709_COLOURSPACE.whitepoint):
    """
    指定した画像ファイルの指定座標の 1 line の輝度成分を計算して返す。
    単位は[cd/m2]。
    """
    # 1 line 抽出の後で eotf をかけて Linear に戻す
    img = img_file_read(file_name) / 0xFFFF  # 16bit TIFF なので。。
    line_rgb_code_value = img[v_pos, :, :]
    line_rgb_linear = eotf(line_rgb_code_value)

    # 単位を cd/m2 に変換
    if eotf == eotf_BT1886:
        luminance_coef = NOMINAL_WHITE_LUMINANCE
    elif eotf == eotf_ST2084:
        # st2084 の EOTF は出力が既に cd/m2 単位になっている
        luminance_coef = 1.0
    else:
        print('warning: eotf parameter is invalid.')
        luminance_coef = 1.0

    line_rgb_luminance = line_rgb_linear * luminance_coef

    line_y_luminance = get_luminance_from_img(
        img=line_rgb_luminance, primaries=primaries, whitepoint=whitepoint)
    return line_y_luminance


def plot_h_luminance_waveform(v_pos=594):
    """
    1ライン分の輝度波形をプロットする。
    今回は3種類の画像の輝度波形を同一図にプロットする。
    """
    src_hdr = get_line_luminance_from_file(
        file_name=SRC_FILE_NAME, v_pos=v_pos,
        eotf=eotf_ST2084,
        primaries=BT2020_COLOURSPACE.primaries,
        whitepoint=BT2020_COLOURSPACE.whitepoint)
    dst_sdr_with_tm = get_line_luminance_from_file(
        file_name=DST_W_TM_FILE_NAME, v_pos=v_pos,
        eotf=eotf_BT1886,
        primaries=BT709_COLOURSPACE.primaries,
        whitepoint=BT709_COLOURSPACE.whitepoint)
    dst_sdr_without_tm = get_line_luminance_from_file(
        file_name=DST_WO_TM_FILE_NAME, v_pos=v_pos,
        eotf=eotf_BT1886,
        primaries=BT709_COLOURSPACE.primaries,
        whitepoint=BT709_COLOURSPACE.whitepoint)

    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(12, 10),
        graph_title="Comparison of luminance",
        graph_title_size=None,
        xlabel="Horizontal pixel index.",
        ylabel="Luminance [cd/m2]",
        axis_label_size=None,
        legend_size=17,
        xlim=(0, 1920),
        ylim=(1, 4000),
        linewidth=2.5,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.set_yscale('log', basey=10.0)
    ax1.plot(src_hdr.flatten(), label="src_hdr", color='k')
    ax1.plot(dst_sdr_with_tm.flatten(), color=RGB_COLOUR_LIST[0],
             label="dst_sdr_with_tone_mapping")
    ax1.plot(dst_sdr_without_tm.flatten(),
             label="dst_sdr_without_tone_mapping", color=RGB_COLOUR_LIST[1])
    plt.legend(loc='upper left')
    plt.savefig("./figures/comparison_of_luminance.png",
                bbox_inches='tight', pad_inches=0.1)
    plt.show()


def main():
    plot_h_luminance_waveform(v_pos=594)
    make_st2084_h_ramp()

    # 注意。本当は以下の関数の実行前に Resolve を使った 3DLUT変換が必要。
    # 今回は既に変換済みのデータをコミットした。
    plot_tone_mapping_characteristics()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
