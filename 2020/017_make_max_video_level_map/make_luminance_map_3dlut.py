# -*- coding: utf-8 -*-
"""
輝度マップ用の3DLUTを作る
========================

"""

# import standard libraries
import os
from itertools import product

# import third-party libraries
import numpy as np
from colour import write_LUT, read_LUT, LUT3D, write_image, read_image
from scipy import interpolate
from colour import RGB_luminance, RGB_COLOURSPACES
from colour.colorimetry import ILLUMINANTS
import matplotlib.pyplot as plt

# import my libraries
import turbo_colormap  # https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f
import transfer_functions as tf
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

CMFS_NAME = 'CIE 1931 2 Degree Standard Observer'
D65 = ILLUMINANTS[CMFS_NAME]['D65']

COLOR_SPACE_NAME_BT709 = 'ITU-R BT.709'
COLOR_SPACE_NAME_BT2020 = 'ITU-R BT.2020'
COLOR_SPACE_NAME_P3_D65 = 'P3-D65'
COLOR_SPACE_NAME_SRGB = 'sRGB'

LUMINANCE_METHOD = 'luminance'
CODE_VALUE_METHOD = 'code_value'

COLOR_SPACE_NAME = COLOR_SPACE_NAME_BT2020

OVER_RANGE_COLOR = np.array([1.0, 0.0, 1.0])

# 計算誤差とかも考慮した最大輝度
LUMINANCE_PYSICAL_MAX = 10000


def load_turbo_colormap():
    """
    Turbo の Colormap データを Numpy形式で取得する
    以下のソースコードを利用。
    https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f

    Examples
    --------
    >>> get_turbo_colormap()
    >>> [[ 0.18995  0.07176  0.23217]
    >>>  [ 0.19483  0.08339  0.26149]
    >>>  ...
    >>>  [ 0.49321  0.01963  0.00955]
    >>>  [ 0.4796   0.01583  0.01055]]
    """
    return np.array(turbo_colormap.turbo_colormap_data)


def apply_turbo_colormap(x):
    """
    1次元の入力データを Turbo で色付けする。
    入力データは Non-Linear かつ [0:1] の範囲とする。
    Turbo は 256エントリ数の LUT として定義されているが、
    入力データは浮動小数点であるため、補間計算が必要である。
    今回は scipy.interpolate.interp1d を使って、R, G, B の
    各種値を線形補間して使う。

    Parameters
    ----------
    x : array_like
        input data. the data range is 0.0 -- 1.0.
    """
    turbo = load_turbo_colormap()

    # scipy.interpolate.interp1d を使って線形補間する準備
    zz = np.linspace(0, 1, turbo.shape[0])
    func_rgb = [interpolate.interp1d(zz, turbo[:, idx]) for idx in range(3)]

    # 線形補間の実行
    out_rgb = [func(x) for func in func_rgb]

    return np.dstack(out_rgb)[0]


def calc_turbo_lut_luminance():
    turbo_lut = load_turbo_colormap()
    turbo_lut_linear = tf.eotf_to_luminance(turbo_lut, tf.SRGB)
    primaries = RGB_COLOURSPACES[COLOR_SPACE_NAME_SRGB].primaries
    turbo_lut_luminance = RGB_luminance(
        RGB=turbo_lut_linear, primaries=primaries, whitepoint=D65)

    return turbo_lut_luminance


def calc_turbo_code_value_from_luminance(luminance):
    turbo_lut_luminance = calc_turbo_lut_luminance()
    lut_len = turbo_lut_luminance.shape[0]
    luminance_max_idx = np.argmax(turbo_lut_luminance)
    luminance_max = turbo_lut_luminance[luminance_max_idx]

    if luminance > luminance_max:
        print("warning: luminance is too bright!")
        return luminance_max_idx / (lut_len - 1)

    # luminance_lut から該当の CodeValue を逆引きする
    func = interpolate.interp1d(
        turbo_lut_luminance[:luminance_max_idx + 1],
        np.linspace(0, luminance_max_idx, luminance_max_idx + 1))

    return func(luminance) / (lut_len - 1)


def check_turbo_luminance():
    """
    turbo を sRGB で見た場合の輝度値をプロットしてみる。
    """
    sample_num = 1024
    x = np.linspace(0, 1, sample_num)
    x10 = x * 1023
    turbo_rgb_srgb = apply_turbo_colormap(x)
    turbo_rgb_linear = tf.eotf_to_luminance(turbo_rgb_srgb, tf.SRGB)

    primaries = RGB_COLOURSPACES[COLOR_SPACE_NAME_SRGB].primaries
    turbo_y_linear = RGB_luminance(
        RGB=turbo_rgb_linear, primaries=primaries, whitepoint=D65)

    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="Luminance of the Turbo Colormap in sRGB Colorspace",
        graph_title_size=None,
        xlabel="Code Value (10bit)", ylabel="Luminance [nits]",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=[x * 128 for x in range(8)] + [1023],
        ytick=None,
        linewidth=3)
    ax1.scatter(
        x10, turbo_y_linear, color=turbo_rgb_srgb, label="Trubo Colormap")
    plt.legend(loc='upper left')
    plt.savefig("./figure/turbo_luminance.png", bbox_inches='tight')
    plt.show()


def normalize_and_fitting(x, x_min, x_max, target_min, target_max):
    normalized_val = (x - x_min) / (x_max - x_min)
    fitting_val = normalized_val * (target_max - target_min) + target_min

    return fitting_val


def calc_y_from_rgb_st2084(rgb_st2084, color_space_name, method):
    rgb_linear = tf.eotf_to_luminance(rgb_st2084, tf.ST2084)

    if method == LUMINANCE_METHOD:
        primaries = RGB_COLOURSPACES[color_space_name].primaries
        y_linear = RGB_luminance(
            RGB=rgb_linear, primaries=primaries, whitepoint=D65)
    elif method == CODE_VALUE_METHOD:
        y_linear = np.max(rgb_linear, axis=-1)
    else:
        print("warning: invalid method.")
        primaries = RGB_COLOURSPACES[color_space_name].primaries
        y_linear = RGB_luminance(
            RGB=rgb_linear, primaries=primaries, whitepoint=D65)

    y_linear = np.clip(y_linear, 0, LUMINANCE_PYSICAL_MAX)

    return y_linear


def make_3dlut_file_name(
        grid_num=65, sdr_pq_peak_luminance=100, turbo_peak_luminance=1000,
        color_space_name=COLOR_SPACE_NAME_BT2020, method=LUMINANCE_METHOD):
    dir_name = "./3dlut/"
    bt2020_luminance = "LuminanceMap_for_ST2084_BT2020_D65"
    dci_p3_luminance = "LuminanceMap_for_ST2084_DCI-P3_D65"
    bt2020_dci_p3_codevalue = "CodeValueMap_for_ST2084"
    map_range = f"MapRange_{sdr_pq_peak_luminance}-{turbo_peak_luminance}nits"
    suffix = f"{grid_num}x{grid_num}x{grid_num}.cube"

    if method == CODE_VALUE_METHOD:
        main_name = bt2020_dci_p3_codevalue
    elif method == LUMINANCE_METHOD:
        if color_space_name == COLOR_SPACE_NAME_BT2020:
            main_name = bt2020_luminance
        elif color_space_name == COLOR_SPACE_NAME_P3_D65:
            main_name = dci_p3_luminance
        else:
            print("warning: invalid color_space_name")
            main_name = bt2020_luminance

    file_name = dir_name + "_".join([main_name, map_range, suffix])

    return file_name


def make_3dlut_for_luminance_map(
        grid_num=65, sdr_pq_peak_luminance=100, turbo_peak_luminance=1000,
        sdr_turbo_st_luminance=18, sdr_srgb_peak_luminance=60,
        color_space_name=COLOR_SPACE_NAME_BT2020, method=LUMINANCE_METHOD):
    """
    輝度マップの3DLUTを作る。

    Parameters
    ----------
    grid_num : int
        3DLUT の格子点数。2^N + 1 が一般的(N=5～6)
    sdr_pq_peak_luminance : float
        SDR のピーク輝度を指定。100 nits or 203 nits が妥当かと。
    turbo_peak_luminance : float
        Turbo colormap を使って塗るHDR領域の最大輝度を指定。
        1000 nits or 4000 nits が妥当か？
    sdr_turbo_st_luminance : float
        Turbo colormap 空間の中での使用開始輝度を指定。
        sdr_pq_peak_luminance 付近が深い青だと違和感があったので、
        少し持ち上げて明るめの青が使われるように調整している。
    sdr_srgb_peak_luminance : float
        SDR領域はグレーで表示するが、そのグレーのピーク輝度を
        sRGB色空間(100nits想定)の中の何nitsの範囲にマッピングするか指定。
        100 nits だと明るすぎて違和感があったので、やや下げて運用中。
    color_space_name : str
        想定するカラースペースを選択。BT.2020 or DCI-P3-D65 を想定。
        このパラメータに依ってY成分の計算をする際の係数が変わる。
        後述の `method` が 'luminance' の場合にのみ有効
    method : str
        'luminance' or 'code_value' を指定。
        'code_value' の場合は 各ピクセルのRGBのうち最大値を使って
        3DLUTを生成する。
    """

    """ 3DLUT の元データ準備。ST2084 の データ """
    rgb_st2084 = LUT3D.linear_table(grid_num)

    """ Linear に戻して Y を計算 """
    y_linear = calc_y_from_rgb_st2084(rgb_st2084, color_space_name, method)

    """
    以後、3つのレンジで処理を行う。
    1. HDRレンジ(sdr_pq_peak_luminance -- turbo_peak_luminance)
    2. SDRレンジ(0 -- sdr_pq_peak_luminance)
    3. 超高輝度レンジ(turbo_peak_luminance -- 10000)
    """
    hdr_idx = (y_linear > sdr_pq_peak_luminance)\
        & (y_linear <= turbo_peak_luminance)
    sdr_idx = (y_linear <= sdr_pq_peak_luminance)
    over_idx = (y_linear > turbo_peak_luminance)

    """ 1. HDRレンジの処理 """
    # Turbo は Non-Linear のデータに適用するべきなので、OETF適用
    y_hdr_pq_code_value = tf.oetf_from_luminance(y_linear[hdr_idx], tf.ST2084)

    # HDRレンジのデータをTurboで変換する前に正規化を行う
    # sdr_pq_peak_luminance が青、turbo_peak_luminance が赤になるように。
    hdr_pq_min_code_value = tf.oetf_from_luminance(
        sdr_pq_peak_luminance, tf.ST2084)
    hdr_pq_max_code_value = tf.oetf_from_luminance(
        turbo_peak_luminance, tf.ST2084)
    turbo_min_code_value = calc_turbo_code_value_from_luminance(
        sdr_turbo_st_luminance)
    turbo_max_code_value = 1.0
    y_hdr_pq_normalized = normalize_and_fitting(
        y_hdr_pq_code_value, hdr_pq_min_code_value, hdr_pq_max_code_value,
        turbo_min_code_value, turbo_max_code_value)

    # 正規化した PQ の Code Value に対して Turbo を適用
    turbo_hdr = apply_turbo_colormap(y_hdr_pq_normalized)

    """ 2. SDRレンジの処理 """
    # SDRレンジは PQでエンコードしておく
    # こうすることで、sRGBモニターで見た時に低輝度領域の黒つぶれを防ぐ
    # 映像は歪むが、もともと輝度マップで歪んでるので気にしない
    y_sdr_pq_code_value = tf.oetf_from_luminance(y_linear[sdr_idx], tf.ST2084)

    # sRGBモニターで低輝度のSDR領域を表示した時に
    # 全体が暗すぎず・明るすぎずとなるように sdr_srgb_peak_luminance に正規化
    sdr_pq_min_code_value = 0
    sdr_pq_max_code_value = tf.oetf_from_luminance(
        sdr_pq_peak_luminance, tf.ST2084)
    sdr_srgb_min_code_value = 0
    sdr_srgb_max_code_value = tf.oetf_from_luminance(
        sdr_srgb_peak_luminance, tf.SRGB)
    y_sdr_pq_normalized = normalize_and_fitting(
        y_sdr_pq_code_value, sdr_pq_min_code_value, sdr_pq_max_code_value,
        sdr_srgb_min_code_value, sdr_srgb_max_code_value)

    # y_sdr_pq_normalized は単色データなので束ねてRGB値にする
    sdr_srgb_rgb = np.dstack(
        [y_sdr_pq_normalized, y_sdr_pq_normalized, y_sdr_pq_normalized])[0]

    """ 計算結果を3DLUTのバッファに代入 """
    lut_data = np.zeros_like(rgb_st2084)
    lut_data[hdr_idx] = turbo_hdr
    lut_data[sdr_idx] = sdr_srgb_rgb
    """ 3. 超高輝度レンジの処理 """
    lut_data[over_idx] = OVER_RANGE_COLOR

    """ 保存 """
    lut_name = f"tf: {tf.ST2084}, gamut: {color_space_name},"\
        + f"turbo_peak_luminance: {turbo_peak_luminance}"
    lut3d = LUT3D(table=lut_data, name=lut_name)

    file_name = make_3dlut_file_name(
        grid_num=grid_num, sdr_pq_peak_luminance=sdr_pq_peak_luminance,
        turbo_peak_luminance=turbo_peak_luminance,
        color_space_name=color_space_name, method=method)
    write_LUT(lut3d, file_name)


def apply_3dlut_for_blog_image(
        grid_num, sdr_pq_peak_luminance, turbo_peak_luminance,
        color_space_name, method, src_img_name="./img/step_ramp.tiff"):
    lut3d_file_name = make_3dlut_file_name(
        grid_num=grid_num, sdr_pq_peak_luminance=sdr_pq_peak_luminance,
        turbo_peak_luminance=turbo_peak_luminance,
        color_space_name=color_space_name, method=method)

    src_basename = os.path.basename(os.path.splitext(src_img_name)[0])
    src_dir = os.path.dirname(src_img_name)
    dst_basename = os.path.basename(os.path.splitext(lut3d_file_name)[0])
    dst_img_name = os.path.join(
        src_dir, dst_basename + "_" + src_basename + ".png")
    print(dst_img_name)

    hdr_img = read_image(src_img_name)
    lut3d = read_LUT(lut3d_file_name)
    luminance_map_img = lut3d.apply(hdr_img)
    write_image(luminance_map_img, dst_img_name, bit_depth='uint16')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    grid_num_list = [33, 65]
    turbo_peak_luminance_list = [1000, 4000, 10000]
    color_spece_name_list = [COLOR_SPACE_NAME_BT2020, COLOR_SPACE_NAME_P3_D65]
    method_list = [LUMINANCE_METHOD, CODE_VALUE_METHOD]
    # grid_num_list = [65]
    # turbo_peak_luminance_list = [1000]
    # color_spece_name_list = [COLOR_SPACE_NAME_BT2020]
    # method_list = [LUMINANCE_METHOD]
    sdr_pq_peak_luminance = 100
    sdr_turbo_st_luminance = 18
    sdr_srgb_peak_luminance = 60
    for grid_num, turbo_peak_luminance, color_space_name, method in product(
            grid_num_list, turbo_peak_luminance_list,
            color_spece_name_list, method_list):
        # 作成
        make_3dlut_for_luminance_map(
            grid_num=grid_num, sdr_pq_peak_luminance=sdr_pq_peak_luminance,
            turbo_peak_luminance=turbo_peak_luminance,
            sdr_turbo_st_luminance=sdr_turbo_st_luminance,
            sdr_srgb_peak_luminance=sdr_srgb_peak_luminance,
            color_space_name=color_space_name, method=method)
        # テストパターンに適用
        apply_3dlut_for_blog_image(
            grid_num=grid_num, sdr_pq_peak_luminance=sdr_pq_peak_luminance,
            turbo_peak_luminance=turbo_peak_luminance,
            color_space_name=color_space_name, method=method,
            src_img_name="./img/step_ramp.tiff")
        # apply_3dlut_for_blog_image(
        #     grid_num=grid_num, sdr_pq_peak_luminance=sdr_pq_peak_luminance,
        #     turbo_peak_luminance=turbo_peak_luminance,
        #     color_space_name=color_space_name, method=method,
        #     src_img_name="./img/src_riku.tif")
        # apply_3dlut_for_blog_image(
        #     grid_num=grid_num, sdr_pq_peak_luminance=sdr_pq_peak_luminance,
        #     turbo_peak_luminance=turbo_peak_luminance,
        #     color_space_name=color_space_name, method=method,
        #     src_img_name="./img/src_umi.tif")
        # apply_3dlut_for_blog_image(
        #     grid_num=grid_num, sdr_pq_peak_luminance=sdr_pq_peak_luminance,
        #     turbo_peak_luminance=turbo_peak_luminance,
        #     color_space_name=color_space_name, method=method,
        #     src_img_name="./img/umi_boost.tif")
        # apply_3dlut_for_blog_image(
        #     grid_num=grid_num, sdr_pq_peak_luminance=sdr_pq_peak_luminance,
        #     turbo_peak_luminance=turbo_peak_luminance,
        #     color_space_name=color_space_name, method=method,
        #     src_img_name="./img/riku_boost.tif")

    # apply_hdr10_to_turbo_3dlut(
    #     src_img_name="./figure/step_ramp.tiff",
    #     dst_img_name="./figure/test.png",
    #     lut_3d_name="./3dlut/test.cube")
    # apply_hdr10_to_turbo_3dlut(
    #     src_img_name="./figure/step_ramp.tiff",
    #     dst_img_name="./figure/test_before.png",
    #     lut_3d_name="./3dlut/PQ_BT2020_to_Turbo_sRGB.cube")
