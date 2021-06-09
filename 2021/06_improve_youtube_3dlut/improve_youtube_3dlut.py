# -*- coding: utf-8 -*-
"""
improve the 3dlut.
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from sympy import symbols, solve
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt
from colour import XYZ_to_xyY, xyY_to_XYZ, XYZ_to_RGB, RGB_to_RGB
from colour.models import RGB_COLOURSPACES, BT2020_COLOURSPACE,\
    BT709_COLOURSPACE
from colour import write_LUT, read_LUT, LUT3D
from colour import YCbCr_to_RGB, RGB_to_YCbCr, YCBCR_WEIGHTS
from scipy import linalg

# import my libraries
import test_pattern_generator2 as tpg
import transfer_functions as tf
import plot_utility as pu
import color_space as cs
from bt2446_method_c import apply_cross_talk_matrix, rgb_to_xyz_in_hdr_space,\
    apply_chroma_correction, apply_inverse_cross_talk_matrix
from bt2047_gamut_mapping import bt2407_gamut_mapping_for_rgb_linear
from color_convert import rgb2yuv_rec2020mtx, rgb2yuv_rec709mtx


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def get_top_side_bezier(param=[[0.6, 0.7], [0.8, 1.0], [1.0, 1.0]]):
    """
    Note
    ----
    An example of ```top_param``` is bellow.
        {'x0': 0.5, 'y0': 0.5,
         'x1': 0.7, 'y1': 0.7,
         'x2': 1.0, 'y2': 0.7}
    """
    a_val = param[0][0]
    b_val = param[1][0]
    c_val = param[2][0]
    p_val = param[0][1]
    q_val = param[1][1]
    r_val = param[2][1]

    a, b, c, t, x = symbols('a, b, c, t, x')
    f = (1 - t)**2 * a + 2 * (1 - t) * t * b + t**2 * c - x

    # x について解く
    # ----------------------
    t = solve(f, t)[1]
    t = t.subs({a: a_val, b: b_val, c: c_val})

    # y と t(ここでは u と置いた) の関係式を記述
    # -------------------------------------------
    p, q, r, u, y = symbols('p, q, r, u, y')
    y = (1 - u)**2 * p + 2 * (1 - u) * u * q + u**2 * r

    # パラメータ u と事前に求めた t で置き換える
    # -------------------------------------------
    y = y.subs({p: p_val, q: q_val, r: r_val, u: t})

    func = lambdify(x, y, 'numpy')

    return func


def youtube_linear(x):
    return 0.74 * x + 0.01175


def youtube_tonemapping(x, ks_luminance=400, ke_luminance=1000):
    """
    YouTube の HDR to SDR のトーンマップを模倣してみる。
    中間階調までは直線で、高階調部だけ2次ベジェ曲線で丸める。

    直線の数式は $y = 0.74x + 0.01175$。
    y軸の最大値は 0.508078421517 (100nits)。
    この時の x は $x = (0.508078421517 - 0.01175) / 0.74$ より
    0.6707140831310812 。ちなみに、リニアなら 473.5 nits。

    ks は knee start の略。ke は knee end.

    Parameters
    ----------
    x : ndarray
        input data. range is 0.0-1.0.
    ks_luminance : float
        knee start luminance. the unit is cd/m2.
    ke_luminance : float
        knee end luminance. the unit is cd/m2.
    """
    ks_x = tf.oetf_from_luminance(ks_luminance, tf.ST2084)
    ks_y = youtube_linear(ks_x)
    ke_x = tf.oetf_from_luminance(ke_luminance, tf.ST2084)
    ke_y = tf.oetf_from_luminance(100, tf.ST2084)
    mid_x = 0.6707140831310812
    mid_y = ke_y
    bezie = get_top_side_bezier([[ks_x, ks_y], [mid_x, mid_y], [ke_x, ke_y]])
    y = np.select(
        (x < ks_x, x <= ke_x, x > ke_x),
        (youtube_linear(x), bezie(x), ke_y))

    return y


def main_func():
    x = np.linspace(0, 1, 1024)
    y = youtube_tonemapping(x)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="Title",
        graph_title_size=None,
        xlabel="X Axis Label", ylabel="Y Axis Label",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None,
        return_figure=True)
    ax1.plot(x, y, label="YouTube")
    plt.legend(loc='upper left')
    plt.show()
    plt.close(fig)


def bt2446_method_c_tonemapping_youtube_custom(
        img, src_color_space_name=cs.BT2020, tfc=tf.ST2084,
        alpha=0.15, sigma=0.5, hdr_ref_luminance=203, hdr_peak_luminance=1000,
        clip_output=True):
    """
    Apply tonemapping function described in BT.2446 Method C

    Parameters
    ----------
    img : ndarray
        hdr linear image data.
        img is should be linear. the range is 0.0 - 1.0.
        1.0 is corresponds to 10000 nits.
    src_color_space_name : str
        the name of the color space of the src image.
    tfc : str
        the name of the transfer characteristics of the src image.
        it is used for calculation of the eotf-peak-luminance.
    alpha : float
        parameter of chrosstalk matrix used to reduce the chroma
        before applying the tonecurve.
    sigma : float
        parameter of chroma reduction.
    hdr_ref_luminance : int
        luminance of the HDR reference white.
    hdr_peak_luminance : int
        peak luminance of the src image. This parameter is used for
        chroma correction
    clip_output : bool
        if the value is True, the data will be clipped to 0.0 - 1.0.

    Returns
    -------
    ndarray
        sdr linear data. data range is 0.0 -- 1.0.
    """
    img_desturated = apply_cross_talk_matrix(img=img, alpha=alpha)
    xyz_hdr = rgb_to_xyz_in_hdr_space(
        rgb=img_desturated, color_space_name=src_color_space_name)
    xyz_hdr_cor = apply_chroma_correction(
        xyz=xyz_hdr, sigma=sigma, src_color_space_name=src_color_space_name,
        tfc=tfc, hdr_ref_luminance=hdr_ref_luminance,
        hdr_peak_luminance=hdr_peak_luminance)
    xyY_hdr_cor = XYZ_to_xyY(xyz_hdr_cor)

    y_hdr = xyY_hdr_cor[..., 2]
    y_hdr_non_linear = tf.oetf(y_hdr, tf.ST2084)
    y_sdr_non_linear = youtube_tonemapping(y_hdr_non_linear)
    y_sdr = tf.eotf(y_sdr_non_linear, tf.ST2084)
    y_sdr = y_sdr / np.max(y_sdr)

    xyY_sdr_cor = xyY_hdr_cor.copy()
    xyY_sdr_cor[..., 2] = y_sdr
    xyz_sdr_cor = xyY_to_XYZ(xyY_sdr_cor)

    rgb_sdr_linear = XYZ_to_RGB(
        xyz_sdr_cor, cs.D65, cs.D65,
        RGB_COLOURSPACES[src_color_space_name].XYZ_to_RGB_matrix)
    rgb_sdr_linear = apply_inverse_cross_talk_matrix(
        img=rgb_sdr_linear, alpha=alpha)

    if clip_output:
        print("[Log][bt2446_method_c_tonemapping] clippint to 0.0 - 1.0")
        rgb_sdr_linear = np.clip(rgb_sdr_linear, 0.0, 1.0)
    else:
        None

    return rgb_sdr_linear


def debug_tone_mapping():
    x = np.linspace(0, 1, 1024)
    x = np.dstack((x, x, x))
    y = bt2446_method_c_tonemapping_youtube_custom(x)
    print(y.shape)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="Title",
        graph_title_size=None,
        xlabel="X Axis Label", ylabel="Y Axis Label",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None,
        return_figure=True)
    ax1.plot(x[..., 1].flatten(), y[..., 1].flatten(), label="YouTube")
    plt.legend(loc='upper left')
    plt.show()
    plt.close(fig)


def make_3dlut(
        src_color_space_name=cs.BT2020, tfc=tf.ST2084,
        alpha=0.15, sigma=0.5, gamma=2.4,
        hdr_ref_luminance=203, hdr_peak_luminance=1000,
        bt2407_gamut_mapping=True, grid_num=65, prefix="",
        on_hdr10=False, for_obs=False):

    x = LUT3D.linear_table(grid_num).reshape((1, grid_num ** 3, 3))
    print(x.shape)

    if for_obs:
        x = apply_obs_matrix(x)

    x_linear = tf.eotf(x, tf.ST2084)
    sdr_img_linear = bt2446_method_c_tonemapping_youtube_custom(
         img=x_linear,
         src_color_space_name=src_color_space_name,
         tfc=tfc, alpha=alpha, sigma=sigma,
         hdr_ref_luminance=hdr_ref_luminance,
         hdr_peak_luminance=hdr_peak_luminance)
    if bt2407_gamut_mapping:
        sdr_img_linear = bt2407_gamut_mapping_for_rgb_linear(
            rgb_linear=sdr_img_linear,
            outer_color_space_name=cs.BT2020,
            inner_color_space_name=cs.BT709)

    if on_hdr10:
        sdr_img_linear = RGB_to_RGB(
            sdr_img_linear, BT709_COLOURSPACE, BT2020_COLOURSPACE)
        sdr_img_nonlinear =\
            tf.oetf_from_luminance(sdr_img_linear * 100, tf.ST2084)
        suffix = "_on_HDR10"
    else:
        sdr_img_nonlinear = sdr_img_linear ** (1/gamma)
        suffix = ""
    lut_name = "ty tone mapping"
    if for_obs:
        suffix += "_for_obs_bt709"
    file_name = f"./3DLUT/{prefix}_a_{alpha:.2f}_s_{sigma:.2f}_"\
        + f"_grid_{grid_num}_gamma_{gamma:.1f}{suffix}.cube"

    sdr_img_nonlinear = sdr_img_nonlinear.reshape(
        ((grid_num, grid_num, grid_num, 3)))

    lut3d = LUT3D(table=sdr_img_nonlinear, name=lut_name)
    write_LUT(lut3d, file_name)


def create_youtube_org_on_hdr10():
    lut_3d_org = read_LUT("./3DLUT/HDR10_to_BT709_YouTube_Rev03.cube")
    grid_num = 65
    x = LUT3D.linear_table(grid_num).reshape((1, grid_num ** 3, 3))
    x_sdr = lut_3d_org.apply(x)
    sdr_img_linear = tf.eotf(x_sdr, tf.GAMMA24)
    sdr_img_linear = RGB_to_RGB(
        sdr_img_linear, BT709_COLOURSPACE, BT2020_COLOURSPACE)
    sdr_img_nonlinear =\
        tf.oetf_from_luminance(sdr_img_linear * 100, tf.ST2084)
    suffix = "_on_HDR10"

    sdr_img_nonlinear = sdr_img_nonlinear.reshape(
        ((grid_num, grid_num, grid_num, 3)))

    lut_name = "youtube tone mapping"
    file_name = "./3DLUT/HDR10_to_BT709_YouTube_Rev03_"
    file_name += f"{grid_num}grid_{suffix}.cube"
    lut3d = LUT3D(table=sdr_img_nonlinear, name=lut_name)
    write_LUT(lut3d, file_name)


def apply_wrong_ycbcr_matrix():
    img = tpg.img_read_as_float("./img/SMPTE_ST2084_ITU-R_BT.2020_D65_1920x1080_rev04_type1.png")
    img_10bit = np.uint16(np.round(img * 1023))
    ycbcr_10bit = RGB_to_YCbCr(
        img_10bit, K=YCBCR_WEIGHTS['ITU-R BT.2020'], in_bits=10,
        out_bits=10, out_int=True, in_int=True, in_legal=False)
    rgb_10bit = YCbCr_to_RGB(
        ycbcr_10bit, K=YCBCR_WEIGHTS['ITU-R BT.709'], in_bits=10,
        out_bits=10, out_int=True, in_legal=True, in_int=True)
    rgb = rgb_10bit / 1023
    tpg.img_wirte_float_as_16bit_int("./img/wrong_matrix.png", rgb)

    lut3d = read_LUT("./3DLUT/YouTube_Custom_BT2446_a_0.15_s_0.50__grid_65_gamma_2.4.cube")
    sdr_rgb = lut3d.apply(rgb)
    tpg.img_wirte_float_as_16bit_int("./img/sdr_wrong_matrix.png", sdr_rgb)

    sdr_rgb_right = lut3d.apply(img)
    tpg.img_wirte_float_as_16bit_int("./img/sdr_right_matrix.png", sdr_rgb_right)

    hdr_rgb_wrong_right = apply_obs_matrix(rgb)
    sdr_rgb_wrong_right = lut3d.apply(hdr_rgb_wrong_right)
    tpg.img_wirte_float_as_16bit_int("./img/sdr_wrong_right.png", sdr_rgb_wrong_right)


def apply_obs_matrix(rgb):
    r2y_bt709 = rgb2yuv_rec709mtx
    r2y_bt2020 = rgb2yuv_rec2020mtx
    y2r_bt2020 = linalg.inv(r2y_bt2020)
    mtx = y2r_bt2020.dot(r2y_bt709)

    org_shape = rgb.shape
    r, g, b = np.dsplit(rgb, 3)
    ro = mtx[0][0] * r + mtx[0][1] * g + mtx[0][2] * b
    go = mtx[1][0] * r + mtx[1][1] * g + mtx[1][2] * b
    bo = mtx[2][0] * r + mtx[2][1] * g + mtx[2][2] * b

    out = np.dstack((ro, go, bo)).reshape(org_shape)

    return out


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # main_func()
    # debug_tone_mapping()
    # make_3dlut(prefix="YouTube_Custom_BT2446", on_hdr10=False, grid_num=65,
    #            for_obs=True)
    # make_3dlut(prefix="YouTube_Custom_BT2446", on_hdr10=False, grid_num=65)
    # make_3dlut(prefix="YouTube_Custom_BT2446", on_hdr10=True, grid_num=65)
    # create_youtube_org_on_hdr10()
    # apply_wrong_ycbcr_matrix()
    # matrix_check()
    make_3dlut(
        src_color_space_name=cs.BT2020, tfc=tf.ST2084,
        alpha=0.15, sigma=0.0, gamma=2.2,
        hdr_ref_luminance=203, hdr_peak_luminance=1000,
        bt2407_gamut_mapping=True, grid_num=65,
        prefix="YouTube_Custom_BT2446",
        on_hdr10=False, for_obs=False)
