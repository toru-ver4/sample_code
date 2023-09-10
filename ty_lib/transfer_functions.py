#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# Transfer Functions モジュール

## 概要
OETF/EOTF の管理。Colour Science for Python っぽく
名称から OETF/EOTF にアクセスできるようにする。
また、ビデオレベルと輝度レベルの相互変換もお手軽にできるようにする。

## 設計思想
In-Out は原則 [0:1] のレンジで行う。
名前のOETFは[0:46.6]など1.0を余裕で超えてくるのだが、
ローカルで作っているテストパターンとの相性を考慮して [0:1] に正規化する。
別途、Global変数として最大輝度をパラメータとして持ち、
輝度⇔信号レベルの相互変換はその変数を使って行う。
"""

import os
import numpy as np
import colour

# NAME
BT709 = ""
GAMMA24 = 'Gamma 2.4'
GAMMA35 = 'Gamma 3.5'
ST2084 = 'SMPTE ST2084'
HLG = 'BT.2100 HLG'
LOGC = 'ARRI LOG_C'
VLOG_IRE = 'Panasonic VLog (IRE Base)'
VLOG = 'Panasonic VLog'
SLOG3 = "SONY S-Log3 (IRE Base)"
SLOG3_REF = "SONY S-Log3"
REDLOG = "RED REDLog"
LOG3G10 = "RED Log3G10"
LOG3G12 = "RED Log3G12"
NLOG = "Nikon N-Log"
DLOG = "DJI D-Log"
FLOG = "FUJIFILM F-Log"
SRGB = "sRGB"
LINEAR = 'Linear'
# ACES_CG = 'ACEScg'

slog_max = colour.models.log_decoding_SLog3((1023 / 1023),
                                            out_reflection=False)
slog_ref_max = colour.models.log_decoding_SLog3((1023 / 1023),
                                                out_reflection=True)
logc_max = colour.models.log_decoding_ARRILogC3(1.0)
vlog_ire_max = colour.models.log_decoding_VLog(1.0, out_reflection=False)
vlog_ref_max = colour.models.log_decoding_VLog(1.0, out_reflection=True)
red_max = colour.models.log_decoding_REDLog(1.0)
log3g10_max = colour.models.log_decoding_Log3G10(1.0)
log3g12_max = colour.models.log_decoding_Log3G12(1.0)
# nlog_max = n_log_decoding(1.0, out_reflection=True)
nlog_max = 14.78086344050015
# flog_max = f_log_decoding(1.0, out_reflection=True)
flog_max = 7.281324880488497
# dlog_max = d_log_decoding(1.0, out_reflection=True)
dlog_max = 41.99939267086707

REF_WHITE_LUMINANCE = 100

MAX_VALUE = {GAMMA24: 1.0, GAMMA35: 1.0, SRGB: 1.0, BT709: 1.0,
             ST2084: 10000, HLG: 1000,
             VLOG_IRE: vlog_ire_max, VLOG: vlog_ref_max,
             LOGC: logc_max,
             SLOG3: slog_max, SLOG3_REF: slog_ref_max,
             REDLOG: red_max,
             LOG3G10: log3g10_max, LOG3G12: log3g12_max,
             NLOG: nlog_max, FLOG: flog_max, DLOG: dlog_max}

PEAK_LUMINANCE = {GAMMA24: REF_WHITE_LUMINANCE,
                  GAMMA35: REF_WHITE_LUMINANCE,
                  SRGB: REF_WHITE_LUMINANCE,
                  BT709: REF_WHITE_LUMINANCE,
                  ST2084: 10000,
                  HLG: 1000,
                  VLOG_IRE: vlog_ire_max * REF_WHITE_LUMINANCE,
                  VLOG: vlog_ref_max * REF_WHITE_LUMINANCE,
                  LOGC: logc_max * REF_WHITE_LUMINANCE,
                  SLOG3: slog_max * REF_WHITE_LUMINANCE,
                  SLOG3_REF: slog_ref_max * REF_WHITE_LUMINANCE,
                  REDLOG: red_max * REF_WHITE_LUMINANCE,
                  LOG3G10: log3g10_max * REF_WHITE_LUMINANCE,
                  LOG3G12: log3g12_max * REF_WHITE_LUMINANCE,
                  NLOG: nlog_max * REF_WHITE_LUMINANCE,
                  FLOG: flog_max * REF_WHITE_LUMINANCE,
                  DLOG: dlog_max * REF_WHITE_LUMINANCE,
                  LINEAR: REF_WHITE_LUMINANCE}


def oetf(x, name=GAMMA24):
    """
    [0:1] で正規化された Scene Luminace 値から OETF値 を算出して返す。

    Parameters
    ----------
    x : numeric or array_like
        scene luminance. range is [0:1]
    name : unicode
        GAMMA24, ST2084, HLG, ... and so on.

    Returns
    -------
    numeric or ndarray
        encoded video level.

    Examples
    --------
    >>> oetf(0.18, GAMMA24)
    0.5 付近の値
    >>> oetf(1.0, GAMMA24)
    1.0
    >>> oetf(0.01, ST2084)
    0.5 付近の値
    """

    if name == GAMMA24:
        y = (x * MAX_VALUE[name]) ** (1/2.4)
    elif name == GAMMA35:
        y = (x * MAX_VALUE[name]) ** (1/3.5)
    elif name == SRGB:
        y = colour.models.eotf_inverse_sRGB(x * MAX_VALUE[name])
    elif name == BT709:
        y = colour.models.oetf_BT709(x * MAX_VALUE[name])
    elif name == HLG:
        y = colour.models.eotf_inverse_HLG_BT2100(x * MAX_VALUE[name])
    elif name == ST2084:
        # fix me!
        y = colour.models.eotf_inverse_ST2084(x * MAX_VALUE[name])
    elif name == SLOG3:
        y = colour.models.log_encoding_SLog3(x * MAX_VALUE[name],
                                             in_reflection=False)
    elif name == SLOG3_REF:
        y = colour.models.log_encoding_SLog3(x * MAX_VALUE[name],
                                             in_reflection=True)
    elif name == VLOG_IRE:
        y = colour.models.log_encoding_VLog(x * MAX_VALUE[name],
                                            in_reflection=False)
    elif name == VLOG:
        y = colour.models.log_encoding_VLog(x * MAX_VALUE[name],
                                            in_reflection=True)
    elif name == LOGC:
        y = colour.models.log_encoding_ARRILogC3(x * MAX_VALUE[name])
    elif name == REDLOG:
        y = colour.models.log_encoding_REDLog(x * MAX_VALUE[name])
    elif name == LOG3G10:
        y = colour.models.log_encoding_Log3G10(x * MAX_VALUE[name])
    elif name == LOG3G12:
        y = colour.models.log_encoding_Log3G12(x * MAX_VALUE[name])
    elif name == NLOG:
        y = n_log_encoding(x * MAX_VALUE[name], in_reflection=True)
    elif name == FLOG:
        y = f_log_encoding(x * MAX_VALUE[name], in_reflection=True)
    elif name == DLOG:
        y = d_log_encoding(x * MAX_VALUE[name], in_reflection=True)
    elif name == LINEAR:
        y = x
    else:
        raise ValueError("invalid transfer fucntion name")

    return y


def oetf_from_luminance(x, name=GAMMA24):
    """
    輝度値[cd/m2]からOETFを算出して返す。

    Parameters
    ----------
    x : numeric or array_like
        luminance. unit is [cd/m2].
    name : unicode
        GAMMA24, ST2084, HLG, ... and so on.

    Returns
    -------
    numeric or ndarray
        encoded video level.

    Examples
    --------
    >>> oetf_from_luminance(18, GAMMA24)
    0.5 付近の値
    >>> oetf_from_luminance(100, GAMMA24)
    1.0
    >>> oetf_from_luminance(100, ST2084)
    0.5 付近の値
    """
    return oetf(x / PEAK_LUMINANCE[name], name)


def eotf(x, name=GAMMA24):
    """
    Code Value から輝度値を算出する。
    返り値は[0:1]で正規化された値。

    Parameters
    ----------
    x : numeric or array_like
        code value. range is [0:1]
    name : unicode
        GAMMA24, ST2084, HLG, ... and so on.

    Returns
    -------
    numeric or ndarray
        decoded video level. range is [0:1]

    Examples
    --------
    >>> eotf(0.5, GAMMA24)
    0.18 付近の値
    >>> oetf(1.0, GAMMA24)
    1.0
    >>> oetf(0.5, ST2084)
    0.01 付近の値
    """
    if name == GAMMA24:
        y = x ** 2.4
    elif name == GAMMA35:
        y = x ** 3.5
    elif name == SRGB:
        y = colour.models.eotf_sRGB(x)
    elif name == BT709:
        y = colour.models.oetf_inverse_BT709(x) / MAX_VALUE[name]
    elif name == ST2084:
        # fix me!
        y = colour.models.eotf_ST2084(x) / MAX_VALUE[name]
    elif name == HLG:
        y = colour.models.eotf_HLG_BT2100(x) / MAX_VALUE[name]
    elif name == SLOG3:
        y = colour.models.log_decoding_SLog3(x, out_reflection=False)\
            / MAX_VALUE[name]
    elif name == SLOG3_REF:
        y = colour.models.log_decoding_SLog3(x, out_reflection=True)\
            / MAX_VALUE[name]
    elif name == VLOG_IRE:
        y = colour.models.log_decoding_VLog(x, out_reflection=False)\
            / MAX_VALUE[name]
    elif name == VLOG:
        y = colour.models.log_decoding_VLog(x, out_reflection=True)\
            / MAX_VALUE[name]
    elif name == LOGC:
        y = colour.models.log_decoding_ARRILogC3(x) / MAX_VALUE[name]
    elif name == REDLOG:
        y = colour.models.log_decoding_REDLog(x) / MAX_VALUE[name]
    elif name == LOG3G10:
        y = colour.models.log_decoding_Log3G10(x) / MAX_VALUE[name]
    elif name == LOG3G12:
        y = colour.models.log_decoding_Log3G12(x) / MAX_VALUE[name]
    elif name == NLOG:
        y = n_log_decoding(x, out_reflection=True) / MAX_VALUE[name]
    elif name == FLOG:
        y = f_log_decoding(x, out_reflection=True) / MAX_VALUE[name]
    elif name == DLOG:
        y = d_log_decoding(x, out_reflection=True) / MAX_VALUE[name]
    elif name == LINEAR:
        y = x
    else:
        raise ValueError("invalid transfer fucntion name")

    return y


def eotf_to_luminance(x, name=GAMMA24):
    """
    Code Value から輝度値を算出する。
    戻り値は [cd/m2] の単位。

    Parameters
    ----------
    x : numeric or array_like
        code value. range is [0:1]
    name : unicode
        GAMMA24, ST2084, HLG, ... and so on.

    Returns
    -------
    numeric or ndarray
        encoded video level.

    Examples
    --------
    >>> eotf(0.5, GAMMA24)
    18 付近の値
    >>> oetf(1.0, GAMMA24)
    100
    >>> oetf(0.5, ST2084)
    100 付近の値
    """
    return eotf(x, name) * PEAK_LUMINANCE[name]


def n_log_encoding(x, in_reflection=False):
    """
    Conversion from linear light to N-Log Value(not CodeValue).

    Parameters
    ----------
    x : numeric or array_like
        linear light value. The reference white is 1.0.
    out_reflection : boolean
        Whether the input light level is reflection.

    Returns
    -------
    numeric or ndarray
        encoded N-Log Value.

    Examples
    --------
    >>> n_log_encoding(0.0)
    0.12437263
    >>> n_log_encoding(0.2)  # 0.18 / 0.9 = 0.2
    0.36366777
    >>> n_log_encoding(1.0)
    0.58963433
    >>> n_log_encoding(16.4231816006)
    1.0
    """
    if not in_reflection:
        x = x * 0.9

    threshold = 0.328
    y = np.where(x < threshold,
                 650 * ((x + 0.0075) ** (1/3)),
                 150 * np.log(x) + 619)

    return y / 1023


def n_log_decoding(x, out_reflection=False):
    """
    Conversion from N-Log Value(not CodeValue) to linear light.

    Parameters
    ----------
    x : numeric or array_like
        N-log value. Valid domain range is [0.0:1.0].
    out_reflection : boolean
        Whether the output light level is reflection.

    Returns
    -------
    numeric or ndarray
        linear light value. ref white is 1.0

    Examples
    --------
    >>> n_log_decoding(0.12437263)
    0.0
    >>> n_log_decoding(0.36366777)
    0.2
    >>> n_log_decoding(0.58963433)
    1.0
    >>> n_log_decoding(1.0)
    16.4231816006
    """
    x = x * 1023
    threshold = 452
    y = np.where(x < threshold,
                 ((x / 650) ** 3.0) - 0.0075,
                 np.exp((x - 619) / 150))

    if not out_reflection:
        y = y / 0.9

    return y


def f_log_encoding(x, in_reflection=False):
    """
    Conversion from linear light to N-Log Value(not CodeValue).

    Parameters
    ----------
    x : numeric or array_like
        linear light value. The reference white is 1.0.
    out_reflection : boolean
        Whether the input light level is reflection.

    Returns
    -------
    numeric or ndarray
        encoded F-Log Value.

    Examples
    --------
    >>> n_log_encoding(0.0)
    0.092864
    >>> n_log_encoding(0.2)  # 0.18 / 0.9 = 0.2
    0.45931846
    >>> n_log_encoding(1.0)
    0.68950335
    >>> n_log_encoding(8.09036097832)
    1.0
    """
    a = 0.555556
    b = 0.009468
    c = 0.344676
    d = 0.790453
    e = 8.735631
    f = 0.092864
    cut1 = 0.00089

    if not in_reflection:
        x = x * 0.9

    y = np.where(x < cut1,
                 e * x + f,
                 c * np.log10(a * x + b) + d)

    return y


def f_log_decoding(x, out_reflection=False):
    """
    Conversion from F-Log Value(not CodeValue) to linear light.

    Parameters
    ----------
    x : numeric or array_like
        F-log value. Valid domain range is [0.0:1.0].
    out_reflection : boolean
        Whether the output light level is reflection.

    Returns
    -------
    numeric or ndarray
        linear light value. ref white is 1.0

    Examples
    --------
    >>> f_log_decoding(0.092864)
    0.0
    >>> f_log_decoding(0.45931846)
    0.2
    >>> f_log_decoding(0.68950335)
    1.0
    >>> f_log_decoding(1.0)
    8.09036097832
    """
    a = 0.555556
    b = 0.009468
    c = 0.344676
    d = 0.790453
    e = 8.735631
    f = 0.092864
    cut2 = 0.100537775223865

    y = np.where(x < cut2,
                 (x - f) / e,
                 (10 ** ((x - d) / c)) / a - (b / a))

    if not out_reflection:
        y = y / 0.9

    return y


def d_log_encoding(x, in_reflection=False):
    """
    Conversion from linear light to D-Log Value(not CodeValue).

    Parameters
    ----------
    x : numeric or array_like
        linear light value. The reference white is 1.0.
    out_reflection : boolean
        Whether the input light level is reflection.

    Returns
    -------
    numeric or ndarray
        encoded D-Log Value.

    Examples
    --------
    >>> n_log_encoding(0.0)
    0.0929
    >>> n_log_encoding(0.2)  # 0.18 / 0.9 = 0.2
    0.39876456
    >>> n_log_encoding(1.0)
    0.57294443
    >>> n_log_encoding(46.6659918565)
    1.0
    """
    if not in_reflection:
        x = x * 0.9

    y = np.where(x <= 0.0078,
                 6.025 * x + 0.0929,
                 (np.log10(x * 0.9892 + 0.0108)) * 0.256663 + 0.584555)

    return y


def d_log_decoding(x, out_reflection=False):
    """
    Conversion from D-Log Value(not CodeValue) to linear light.

    Parameters
    ----------
    x : numeric or array_like
        D-log value. Valid domain range is [0.0:1.0].
    out_reflection : boolean
        Whether the output light level is reflection.

    Returns
    -------
    numeric or ndarray
        linear light value. ref white is 1.0

    Examples
    --------
    >>> f_log_decoding(0.0929)
    0.0
    >>> f_log_decoding(0.39876456)
    0.2
    >>> f_log_decoding(0.57294443)
    1.0
    >>> f_log_decoding(1.0)
    46.6659918565
    """
    y = np.where(x <= 0.14,
                 (x - 0.0929) / 6.025,
                 ((10 ** (3.89616 * x - 2.27752)) - 0.0108) / 0.9892)

    if not out_reflection:
        y = y / 0.9

    return y


def bt1886_eotf(x, lw=100, lb=0.1):
    """
    Implementation of the Recommendation ITU-R BT.1886.
    """
    gm = 2.4
    lw_gm_inv = lw ** (1/gm)
    lb_gm_inv = lb ** (1/gm)
    a = (lw_gm_inv - lb_gm_inv) ** gm
    print(f"a={a}")
    b = lb_gm_inv / (lw_gm_inv - lb_gm_inv)
    print(f"b={b}")

    y = a * ((np.maximum(x + b, 0)) ** gm)

    return y


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(oetf(np.array([0.5, 0.5, 0.5]), SRGB))
