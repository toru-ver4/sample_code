# -*- coding: utf-8 -*-
"""
輝度マップ用の3DLUTを作る
========================

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour import write_LUT, LUT3D
from scipy import interpolate
from colour import RGB_luminance, RGB_COLOURSPACES
from colour.colorimetry import ILLUMINANTS
import matplotlib.pyplot as plt

# import my libraries
import turbo_colormap  # https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f
import transfer_functions as tf
import plot_utility as pu
from apply_3dlut import apply_hdr10_to_turbo_3dlut

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

COLOR_SPACE_NAME = COLOR_SPACE_NAME_BT2020

SDR_LUMINANCE = 100
TURBO_PEAK_LUMINANCE = 1000
OVER_RANGE_COLOR = np.array([1.0, 0.0, 1.0])


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


def make_3dlut_for_luminance_map(
        grid_num=65, sdr_turbo_st_luminance=50, sdr_srgb_peak_luminance=50):
    """
    sdr_turbo_st_luminance: float
        turbo を使い始めるところの輝度のパラメータ
    sdr_srgb_peak_luminance: float
        輝度マップのモノクロの絵での SDR領域のピーク輝度の設定値
    """
    # 3DLUT の元データ準備
    rgb = LUT3D.linear_table(grid_num)

    # Linear に戻す
    rgb_linear = tf.eotf_to_luminance(rgb, tf.ST2084)

    # Yを計算
    primaries = RGB_COLOURSPACES[COLOR_SPACE_NAME].primaries
    y_linear = RGB_luminance(
        RGB=rgb_linear, primaries=primaries, whitepoint=D65)

    hdr_idx = (y_linear > SDR_LUMINANCE) & (y_linear <= TURBO_PEAK_LUMINANCE)
    over_idx = (y_linear > TURBO_PEAK_LUMINANCE)
    sdr_idx = (y_linear <= SDR_LUMINANCE)

    # HDRレンジの処理
    y_hdr_pq = tf.oetf_from_luminance(y_linear[hdr_idx], tf.ST2084)

    # Turbo に食わせる前に正規化をする
    turbo_min_code_value = calc_turbo_code_value_from_luminance(
        sdr_turbo_st_luminance)
    turbo_max_code_value = 1.0
    hdr_pq_min_code_value = tf.oetf_from_luminance(
        SDR_LUMINANCE, tf.ST2084)
    hdr_pq_max_code_value = tf.oetf_from_luminance(
        TURBO_PEAK_LUMINANCE, tf.ST2084)
    y_hdr_pq_normalized = normalize_and_fitting(
        y_hdr_pq, hdr_pq_min_code_value, hdr_pq_max_code_value,
        turbo_min_code_value, turbo_max_code_value)

    # 正規化した PQ の Code Value に対して Turbo を適用
    turbo_hdr = apply_turbo_colormap(y_hdr_pq_normalized)

    # SDRレンジの処理
    sdr_srgb_max_code_value = tf.oetf_from_luminance(
        sdr_turbo_st_luminance, tf.SRGB)
    sdr_normalized = y_linear[sdr_idx] / tf.PEAK_LUMINANCE[tf.SRGB]
    sdr_turbo_st_luminance_range = sdr_normalized * sdr_srgb_max_code_value
    sdr_code_value = tf.oetf(sdr_turbo_st_luminance_range, tf.SRGB)
    sdr_srgb_rgb = np.dstack(
        [sdr_code_value, sdr_code_value, sdr_code_value])[0]

    lut_data = np.zeros_like(rgb)
    lut_data[hdr_idx] = turbo_hdr
    lut_data[sdr_idx] = sdr_srgb_rgb
    lut_data[over_idx] = OVER_RANGE_COLOR

    lut_name = f"tf: {tf.ST2084}, gamut: {COLOR_SPACE_NAME},"\
        + f"sdr_turbo_st_luminance: {sdr_turbo_st_luminance}"
    lut3d = LUT3D(table=lut_data, name=lut_name)

    file_name = f"./3dlut/test.cube"
    write_LUT(lut3d, file_name)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # check_turbo_luminance()
    # calc_turbo_lut_luminance()
    # print(calc_turbo_code_value_from_luminance(78.1860129589))
    # print(calc_turbo_code_value_from_luminance(30)*1023)
    make_3dlut_for_luminance_map(grid_num=65, sdr_turbo_st_luminance=20)
    apply_hdr10_to_turbo_3dlut(
        src_img_name="./figure/step_ramp.tiff",
        dst_img_name="./figure/test.png",
        lut_3d_name="./3dlut/test.cube")
