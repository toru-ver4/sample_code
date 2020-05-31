# -*- coding: utf-8 -*-
"""
Chroma Colormap を作る
======================

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour import RGB_to_XYZ, XYZ_to_Lab, Lab_to_LCHab, LUT3D,\
    LCHab_to_Lab, Lab_to_XYZ, XYZ_to_RGB, write_LUT
from colour import RGB_COLOURSPACES
from scipy import interpolate

# import my libraries
import color_space as cs
import transfer_functions as tf
import turbo_colormap

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def make_linear_input_rgb_value(grid_num=33, tfc=tf.GAMMA24):
    rgb_non_linear = LUT3D.linear_table(grid_num)
    rgb_non_linear = rgb_non_linear.reshape((grid_num ** 3, 3))
    rgb_linear = tf.eotf(rgb_non_linear, tfc)
    return rgb_linear


def rgb_to_lch(rgb_linear, color_space_name=cs.BT709):
    large_xyz = RGB_to_XYZ(
        rgb_linear, cs.D65, cs.D65,
        RGB_COLOURSPACES[color_space_name].RGB_to_XYZ_matrix)
    lch = Lab_to_LCHab(XYZ_to_Lab(large_xyz))

    return lch


def lch_to_rgb(lch, color_space_name=cs.BT709):
    large_xyz = Lab_to_XYZ(LCHab_to_Lab(lch))
    rgb = XYZ_to_RGB(
        large_xyz, cs.D65, cs.D65,
        RGB_COLOURSPACES[color_space_name].XYZ_to_RGB_matrix)

    return rgb


def calc_gamut_boundary_specific_lightness_hue(
        lch, color_space_name=cs.BT709):
    chroma_init = 300
    trial_num = 30
    src_lightness = lch[..., 0]
    src_chroma = lch[..., 1]
    src_hue = lch[..., 2]

    dst_chroma = chroma_init * np.ones_like(src_chroma)

    for t_idx in range(trial_num):
        dst_lch = np.dstack((src_lightness, dst_chroma, src_hue))
        dst_rgb = lch_to_rgb(dst_lch)

        # dst_rgb is inside of the gamut boundary?
        r_judge = (dst_rgb[..., 0] >= 0) & (dst_rgb[..., 0] <= 1)
        g_judge = (dst_rgb[..., 1] >= 0) & (dst_rgb[..., 1] <= 1)
        b_judge = (dst_rgb[..., 2] >= 0) & (dst_rgb[..., 2] <= 1)
        judge = ((r_judge & g_judge) & b_judge)[0]

        # update dst_chroma
        add_diff = chroma_init / (2 ** (t_idx + 1))
        dst_chroma[judge] = dst_chroma[judge] + add_diff
        dst_chroma[~judge] = dst_chroma[~judge] - add_diff

    return dst_chroma


def calc_chroma_rate(lch, dst_chroma):
    src_chroma = lch[..., 1]
    rate = src_chroma / dst_chroma

    # 一応、Lightness=0, 100 は 0 に潰しておく
    ll = lch[..., 0]
    extreme_idx = (ll <= 0) | (ll >= 100)
    rate[extreme_idx] = 0.0

    # オーバーフロー防止
    rate = np.clip(rate, 0.0, 1.0)

    return rate


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


def save_3dlut(
        colormap, grid_num=33, color_space_name=cs.BT709, tfc=tf.GAMMA24):
    fname = f"./3DLUT/chroma_colormap_{tfc}_{color_space_name}_"\
        + f"{grid_num}x{grid_num}x{grid_num}.cube"
    fname = fname.replace(" ", "_")
    colormap = colormap.reshape((grid_num, grid_num, grid_num, 3))

    lut3d = LUT3D(table=colormap, name=fname)
    print(f"writing {fname}")
    write_LUT(lut3d, fname)


def make_chroma_colormap_3dlut(
        grid_num=33, color_space_name=cs.BT709, tfc=tf.GAMMA24):
    # 入力信号準備
    rgb_linear = make_linear_input_rgb_value(grid_num=grid_num, tfc=tfc)

    # 入力信号を LCH に変換
    lch = rgb_to_lch(rgb_linear, color_space_name=color_space_name)

    # LH値の Gamut Boundary の Chroma を計算
    dst_chroma = calc_gamut_boundary_specific_lightness_hue(
        lch=lch, color_space_name=color_space_name)

    # Chroma の使用率？を計算
    chroma_rate = calc_chroma_rate(lch=lch, dst_chroma=dst_chroma)

    # Colormap 適用
    colormap = apply_turbo_colormap(chroma_rate)

    # 結果を 3DLUT として保存
    save_3dlut(
        colormap=colormap, grid_num=grid_num,
        color_space_name=color_space_name, tfc=tfc)


def main_func():
    make_chroma_colormap_3dlut(
        grid_num=129, color_space_name=cs.BT709, tfc=tf.GAMMA24)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
