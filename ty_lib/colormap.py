# -*- coding: utf-8 -*-
"""
make colormap image
===================

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from scipy import interpolate
from colour import RGB_luminance, RGB_COLOURSPACES, RGB_to_RGB
from colour.models import sRGB_COLOURSPACE
from colour.colorimetry import ILLUMINANTS

# import my libraries
import turbo_colormap  # https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f
import transfer_functions as tf

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


def apply_st2084_to_srgb_colormap(
        img, sdr_pq_peak_luminance=100, turbo_peak_luminance=1000,
        sdr_turbo_st_luminance=18, sdr_srgb_peak_luminance=60,
        color_space_name=COLOR_SPACE_NAME_BT2020, method=LUMINANCE_METHOD,
        out_on_hdr=False):
    """
    輝度マップの3DLUTを作る。

    Parameters
    ----------
    img : array_like
        A non-linear ST2084 image data. It should be normalized to 0.0-1.0.
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
    out_on_hdr : str
        出力値を ST2084 の箱に入れるか否か。
        True にすると ST2084 の 0～100nits にマッピングする

    Returns
    -------
    array_like
        A colormap image with sRGB OETF. The range is 0.0 -- 1.0.
    """

    """ Linear に戻して Y を計算 """
    y_linear = calc_y_from_rgb_st2084(img, color_space_name, method)

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
    lut_data = np.zeros_like(img)
    lut_data[hdr_idx] = turbo_hdr
    lut_data[sdr_idx] = sdr_srgb_rgb
    """ 3. 超高輝度レンジの処理 """
    lut_data[over_idx] = OVER_RANGE_COLOR

    """ Side by Side 用に sRGB on HDR の準備（オプション）"""
    if out_on_hdr:
        lut_data_srgb_linear = tf.eotf_to_luminance(lut_data, tf.SRGB)
        lut_data_wcg_linear = RGB_to_RGB(
            lut_data_srgb_linear, sRGB_COLOURSPACE,
            RGB_COLOURSPACES[color_space_name])
        lut_data_wcg_st2084 = tf.oetf_from_luminance(
            lut_data_wcg_linear, tf.ST2084)
        lut_data = lut_data_wcg_st2084

    return lut_data


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    pass
