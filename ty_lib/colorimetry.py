#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Colorimetry
"""

import os
import numpy as np
from colour.colorimetry import STANDARD_OBSERVERS_CMFS
from colour.colorimetry import ILLUMINANTS
from colour.colorimetry.spectrum import SpectralShape
from colour.algebra import SpragueInterpolator, LinearInterpolator
from colour.colorimetry import MultiSpectralDistribution, SpectralDistribution
from colour.utilities import tstack
from colour.temperature import CCT_to_xy_CIE_D
from colour import sd_CIE_illuminant_D_series
from colour import XYZ_to_RGB, RGB_to_XYZ
import color_space as cs

CIE1931 = 'CIE 1931 2 Degree Standard Observer'
CIE2015_2 = 'CIE 2012 2 Degree Standard Observer'
D65_WHITE = ILLUMINANTS[CIE1931]['D65']
D50_WHITE = ILLUMINANTS[CIE1931]['D50']


def _make_multispectral_format_data(wavelengths, values):
    dic = dict(zip(np.uint16(wavelengths).tolist(), values.tolist()))

    return dic


def load_colorchecker_spectrum(interpolator=SpragueInterpolator):
    """
    Babel Average 2012 のデータをロード。
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv = base_dir + "/data/babel_spectrum_2012.csv"
    data = np.loadtxt(csv, delimiter=',', skiprows=0)

    wavelength = data[0, :]
    values = tstack([data[x, :] for x in range(1, 25)])
    m_data = _make_multispectral_format_data(wavelength, values)
    color_checker_spd = MultiSpectralDistribution(m_data)

    if interpolator is not None:
        color_checker_spd.interpolate(SpectralShape(interval=1),
                                      interpolator=interpolator)

    return color_checker_spd


def load_cie1931_1nm_data():
    """
    CIE S 014-1 に記載の2°視野の等色関数を得る
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cie_file = base_dir + "./data/CMFs_CIE_S_014-1.csv"
    cms_1931_2 = np.loadtxt(cie_file, delimiter=',')
    m_data = _make_multispectral_format_data(
        cms_1931_2[:, 0], cms_1931_2[:, 1:])
    cmfs_1931_spd = MultiSpectralDistribution(m_data)

    return cmfs_1931_spd


def load_cie2015_2_1nm_data():
    """
    CIE 170-2 に記載されている（らしい）2°視野の等色関数を得る
    """
    cmfs_spd = STANDARD_OBSERVERS_CMFS[CIE2015_2].copy()

    return cmfs_spd


def load_cmfs(cmfs_name=CIE1931):
    """
    Color Matching Functions を Load する。
    CIE1931の方は S 014-1 のデータを使用。
    CIE2015の方は colour のデータを使用。
    """
    if cmfs_name == CIE1931:
        cmfs_spd = load_cie1931_1nm_data()
    elif cmfs_name == CIE2015_2:
        cmfs_spd = load_cie2015_2_1nm_data()
    else:
        ValueError('parameter "cmfs_name" is invald.')

    return cmfs_spd


def load_d65_spd_1nmdata(interval):
    """
    CIE S 014-2 に記載の D65 の SPD をLoadする。
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cie_file = base_dir + "./data/d65_CIE_S_014-2.csv"
    cie = np.loadtxt(cie_file, delimiter=',')
    m_data = _make_multispectral_format_data(
        cie[::interval, 0], cie[::interval, 2])
    d65_spd = SpectralDistribution(m_data)

    return d65_spd


def make_xy_value_from_temperature(temperature=6500):
    return CCT_to_xy_CIE_D(temperature * 1.4388 / 1.4380)


def fit_significant_figures(x, significant_figures=3):
    """
    有効数字を指定桁に設定する。
    Numpy の関数で良い感じのが無かったので自作。
    D65とかの有効桁数を6桁にするのに使う。
    """
    exp_val = np.floor(np.log10(x))
    normalized_val = np.array(x) / (10 ** exp_val)
    round_val = np.round(normalized_val, significant_figures - 1)
    ret_val = round_val * (10 ** exp_val)

    return ret_val


def make_day_light_by_calculation(temperature=6500,
                                  interpolater=LinearInterpolator,
                                  interval=1):
    """
    計算でD光源を作る。

    interpolater: SpragueInterpolator or LinearInterpolator
    """
    xy = make_xy_value_from_temperature(temperature)
    spd = sd_CIE_illuminant_D_series(xy)
    spd = spd.interpolate(SpectralShape(interval=interval),
                          interpolator=interpolater)
    spd.values = fit_significant_figures(spd.values, 6)

    return spd


def get_day_light_spd(temperature=6500,
                      interpolater=LinearInterpolator,
                      interval=1):
    """
    D65の相対分光分布を計算する。
    計算によるD65値がどうしても有効数字6桁目でズレるため、
    6500Kに関しては規格値のcsvをLoadするようにした。
    """
    if temperature == 6500:
        spd = load_d65_spd_1nmdata(interval)
    else:
        spd = make_day_light_by_calculation(temperature=temperature,
                                            interpolater=interpolater,
                                            interval=interval)
    return spd


def calc_appropriate_shape(spd1, spd2):
    """
    等色関数と測定対象の波長範囲が一致しない場合に
    適切な Shape を設定する。

    例：
    ```
    a.shape = SpectralShape(390, 830, 1)
    b.shape = SpectralShape(360, 780, 5)

    calc_appropriate_shape(a, b)
    >>> SpectralShape(360, 780, 5)
    ```
    """
    start = np.max(np.array([spd1.shape.start, spd2.shape.start]))
    end = np.min(np.array([spd1.shape.end, spd2.shape.end]))
    interval = np.max(np.array([spd1.shape.interval, spd2.shape.interval]))

    return SpectralShape(start, end, interval)


def get_nomalize_large_y_coef(d_light_before_trim, cmfs_before_trim):
    """
    XYZ算出用の正規化係数を算出する。
    入力する d_light, cmfs は **.trim() する前の値とすること**。
    そうしないと、狭い波長範囲で正規化係数を算出することになってしまう。
    """
    shape = calc_appropriate_shape(d_light_before_trim, cmfs_before_trim)
    d_light_calc = d_light_before_trim.copy().trim(shape).values
    cmfs_y = cmfs_before_trim.copy().trim(shape).values[:, 1]
    large_y = np.sum(d_light_calc * cmfs_y)
    normalize_coef = 100 / large_y

    return normalize_coef


def colorchecker_spectrum_to_large_xyz(d_light, color_checker, cmfs):
    """
    ColorChecker のスペクトルと等色関数から
    24色の各パターンのXYZ値を算出する。
    """
    # calc large y normalize coef
    normalize_coef = get_nomalize_large_y_coef(
        d_light_before_trim=d_light, cmfs_before_trim=cmfs)
    normalize_coef /= 100

    # trim
    shape = calc_appropriate_shape(color_checker, cmfs)
    color_checker.trim(shape)
    cmfs.trim(shape)
    d_light.trim(shape)
    
    large_xyz_buf = []
    for idx in range(24):
        temp = d_light.values * color_checker.values[:, idx]
        temp = temp.reshape((d_light.values.shape[0], 1))
        large_xyz = np.sum(temp * cmfs.values * normalize_coef, axis=0)
        large_xyz_buf.append(large_xyz)

    return np.array(large_xyz_buf)


def color_checker_large_xyz_to_rgb(
        large_xyz, color_space=cs.RGB_COLOURSPACES[cs.SRTB]):
    illuminant_XYZ = D65_WHITE
    illuminant_RGB = D65_WHITE
    chromatic_adaptation_transform = None
    xyz_to_rgb_matrix = color_space.XYZ_to_RGB_matrix
    rgb = XYZ_to_RGB(large_xyz, illuminant_XYZ,
                     illuminant_RGB, xyz_to_rgb_matrix,
                     chromatic_adaptation_transform)

    # under flow check
    if np.min(rgb) < 0:
        print("under flow has occured, at color_checker_large_xyz_to_rgb.")
        rgb[rgb < 0] = 0

    # over flow check
    if np.max(rgb) > 1:
        print("over flow has occured, at color_checker_large_xyz_to_rgb.")
        rgb = rgb / np.max(rgb)

    return rgb


def temperature_convert(
       rgb_in, src_temperature=6500, dst_temperature=5000,
       chromatic_adaptation="CAT02", color_space=cs.RGB_COLOURSPACES[cs.SRTB]):
    """
    ColorCheckerの色温度を変更する。
    rgb_in は linear data とする。
    """
    src_xy = make_xy_value_from_temperature(src_temperature)
    dst_xy = make_xy_value_from_temperature(dst_temperature)
    rgb_to_xyz_matrix = color_space.RGB_to_XYZ_matrix
    xyz_to_rgb_matrix = color_space.XYZ_to_RGB_matrix
    large_xyz = RGB_to_XYZ(rgb_in, src_xy, src_xy, rgb_to_xyz_matrix,
                           chromatic_adaptation)
    rgb_out = XYZ_to_RGB(large_xyz, src_xy, dst_xy, xyz_to_rgb_matrix,
                         chromatic_adaptation)
    # under flow check
    if np.min(rgb_out) < 0:
        print("under flow has occured, at temperature_convert.")
        rgb_out[rgb_out < 0] = 0

    # over flow check
    if np.max(rgb_out) > 1:
        print("over flow has occured, at temperature_convert.")
        rgb_out = rgb_out / np.max(rgb_out)

    return rgb_out


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
