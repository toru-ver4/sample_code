# -*- coding: utf-8 -*-
"""
study Jzazbz color space
"""

# import standard libraries
from operator import eq
import os

# import third-party libraries
import numpy as np
from colour import xy_to_XYZ, RGB_to_XYZ, XYZ_to_RGB, Lab_to_XYZ, XYZ_to_Lab,\
    RGB_COLOURSPACES
from colour.models import RGB_COLOURSPACE_BT2020
from scipy.io import savemat, loadmat

# import my libraries
from jzazbz import large_xyz_to_jzazbz, jzazbz_to_large_xyz, st2084_oetf_like
from create_gamut_booundary_lut import is_out_of_gamut_rgb
import test_pattern_generator2 as tpg
import color_space as cs
import plot_utility as pu
import transfer_functions as tf
import font_control as fc

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def create_xyz_values_for_test():
    color_checker_linear = tpg.generate_color_checker_rgb_value() * 100
    rgbmycw_linear_100 = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1],
         [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1]]) * 100
    rgbmycw_linear_10000 = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1],
         [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1]]) * 10000
    test_rgb_linear = np.concatenate(
        [color_checker_linear, rgbmycw_linear_100, rgbmycw_linear_10000])

    large_xyz = RGB_to_XYZ(
        test_rgb_linear, cs.D65, cs.D65,
        RGB_COLOURSPACE_BT2020.matrix_RGB_to_XYZ)

    return large_xyz


def convert_ndarray_to_matlab_array(ndarray):
    pass


def sample_jzazbz_conv():
    d65_xy = np.array([0.3127, 0.3290])
    large_xyz = xy_to_XYZ(d65_xy) * 100
    large_xyz = np.array([95.047, 100, 108.883])
    print(large_xyz)

    jab = large_xyz_to_jzazbz(xyz=large_xyz)
    print(jab)


def compare_reference_code():
    ref_jzazbz = loadmat("./result.mat")['result']

    large_xyz = create_xyz_values_for_test()
    my_jzazbz = large_xyz_to_jzazbz(xyz=large_xyz)

    np.testing.assert_array_almost_equal(ref_jzazbz, my_jzazbz, decimal=7)
    # print(my_jzazbz)


def check_inverse_function():
    large_xyz = create_xyz_values_for_test()
    jzazbz = large_xyz_to_jzazbz(large_xyz)
    inv_xyz = jzazbz_to_large_xyz(jzazbz)
    np.testing.assert_array_almost_equal(large_xyz, inv_xyz, decimal=7)


def xyz_to_x2y2z2(xyz):
    """
    convert XYZ to X'Y'Z' using parameter `b` and `g`.
    """
    b = 1.15
    g = 0.66
    x2y2z2 = np.zeros_like(xyz)
    x2y2z2[..., 0] = (b * xyz[..., 0]) - (b - 1) * xyz[..., 2]
    x2y2z2[..., 1] = (g * xyz[..., 1]) - (g - 1) * xyz[..., 0]
    x2y2z2[..., 2] = xyz[..., 2]

    return x2y2z2


def iz_to_jz(iz, d=-0.56):
    d0 = -1.6295499532821566e-11
    return ((1 + d) * iz) / (1 + d * iz) - d0


def x2y2z2_to_ll(x2y2z2):
    return 0.41478972 * x2y2z2[..., 0]\
        + 0.579999 * x2y2z2[..., 1] + 0.014648 * x2y2z2[..., 2]


def create_valid_cie_ab_plane_image_linear(
        l_val=50, ab_max=200, ab_sample=1024, color_space_name=cs.BT2020,
        bg_rgb=np.array([0.8, 0.8, 0.8])):
    """
    Create an image that indicates the valid area of the CIELAB's ab plane.

    Parameters
    ----------
    l_val : float
        A Lightness value. range is 0.0 - 100.0
    ab_max : float
        A maximum value of the a, b range.
    ab_sapmle : int
        A number of samples in the image resolution.
    color_space_name : str
        color space name for colour.RGB_COLOURSPACES
    """
    dummy_rgb = bg_rgb
    aa_base = np.linspace(-ab_max, ab_max, ab_sample)
    bb_base = np.linspace(-ab_max, ab_max, ab_sample)
    aa = aa_base.reshape((1, ab_sample))\
        * np.ones_like(bb_base).reshape((ab_sample, 1))
    bb = bb_base.reshape((ab_sample, 1))\
        * np.ones_like(aa_base).reshape((1, ab_sample))
    ll = np.ones_like(aa) * l_val
    lab = np.dstack((ll, aa, bb[::-1])).reshape((ab_sample, ab_sample, 3))
    rgb_linear = cs.lab_to_rgb(lab=lab, color_space_name=color_space_name)
    ng_idx = is_out_of_gamut_rgb(rgb=rgb_linear)
    rgb_linear[ng_idx] = dummy_rgb

    return rgb_linear


def plot_diff_prime_function():
    lab_ll = 50
    ab_max = 120
    b = 1.15
    g = 0.66
    cs_name = cs.BT2020
    linear_rgb = create_valid_cie_ab_plane_image_linear(
        l_val=lab_ll, ab_max=ab_max, color_space_name=cs.BT2020)
    large_xyz = RGB_to_XYZ(
        linear_rgb, cs.D65, cs.D65,
        RGB_COLOURSPACES[cs_name].matrix_RGB_to_XYZ)

    eq5_xyz = np.zeros_like(large_xyz)
    eq5_xyz[..., 0] = b * large_xyz[..., 0] - (b - 1) * large_xyz[..., 2]
    eq5_xyz[..., 1] = large_xyz[..., 1]
    eq5_xyz[..., 2] = large_xyz[..., 2]

    eq8_xyz = np.zeros_like(large_xyz)
    eq8_xyz[..., 0] = b * large_xyz[..., 0] - (b - 1) * large_xyz[..., 2]
    eq8_xyz[..., 1] = g * large_xyz[..., 1] - (g - 1) * large_xyz[..., 0]
    eq8_xyz[..., 2] = large_xyz[..., 2]

    eq5_rgb = XYZ_to_RGB(
        eq5_xyz, cs.D65, cs.D65, RGB_COLOURSPACES[cs_name].matrix_XYZ_to_RGB)
    eq5_srgb = tf.oetf(np.clip(eq5_rgb, 0.0, 1.0), tf.SRGB)
    text_drawer = fc.TextDrawer(
        eq5_srgb, text="Eq. (5)", pos=(10, 5),
        font_color=(0.2, 0.2, 0.2), font_size=40)
    text_drawer.draw()
    tpg.img_wirte_float_as_16bit_int(
        f"./img/eq5_srgb_ll-{lab_ll}.png", eq5_srgb)
    tpg.img_wirte_float_as_16bit_int(
        f"./img/org_eq5_ll-{lab_ll}_01.png", eq5_srgb)

    eq8_rgb = XYZ_to_RGB(
        eq8_xyz, cs.D65, cs.D65, RGB_COLOURSPACES[cs_name].matrix_XYZ_to_RGB)
    eq8_srgb = tf.oetf(np.clip(eq8_rgb, 0.0, 1.0), tf.SRGB)
    text_drawer = fc.TextDrawer(
        eq8_srgb, text="Eq. (8)", pos=(10, 5),
        font_color=(0.2, 0.2, 0.2), font_size=40)
    text_drawer.draw()
    tpg.img_wirte_float_as_16bit_int(
        f"./img/eq8_srgb_ll-{lab_ll}.png", eq8_srgb)
    tpg.img_wirte_float_as_16bit_int(
        f"./img/org_eq8_ll-{lab_ll}_01.png", eq8_srgb)

    src_rgb_srgb = tf.oetf(np.clip(linear_rgb, 0.0, 1.0), tf.SRGB)
    text_drawer = fc.TextDrawer(
        src_rgb_srgb, text="Original", pos=(10, 5),
        font_color=(0.2, 0.2, 0.2), font_size=40)
    text_drawer.draw()
    tpg.img_wirte_float_as_16bit_int(
        f"./img/eq0_srgb_ll-{lab_ll}.png", src_rgb_srgb)
    tpg.img_wirte_float_as_16bit_int(
        f"./img/org_eq5_ll-{lab_ll}_00.png", src_rgb_srgb)
    tpg.img_wirte_float_as_16bit_int(
        f"./img/org_eq8_ll-{lab_ll}_00.png", src_rgb_srgb)


def plot_xyz_prime_function_with_mpl():
    lab_ll = 50
    ab_max = 140
    ab_sample = 512
    b = 1.15
    g = 0.66
    bg_xyz = Lab_to_XYZ([lab_ll, 0, 0])
    bg_rgb = XYZ_to_RGB(
        bg_xyz, cs.D65, cs.D65, RGB_COLOURSPACES[cs.BT2020].matrix_XYZ_to_RGB)
    cs_name = cs.BT2020
    linear_rgb = create_valid_cie_ab_plane_image_linear(
        l_val=lab_ll, ab_max=ab_max, ab_sample=ab_sample,
        color_space_name=cs.BT2020, bg_rgb=bg_rgb)
    src_srgb = tf.oetf(np.clip(linear_rgb, 0.0, 1.0), tf.SRGB)
    eq0_xyz = RGB_to_XYZ(
        linear_rgb, cs.D65, cs.D65,
        RGB_COLOURSPACES[cs_name].matrix_RGB_to_XYZ)
    eq0_lab = XYZ_to_Lab(eq0_xyz)

    eq5_xyz = np.zeros_like(eq0_xyz)
    eq5_xyz[..., 0] = b * eq0_xyz[..., 0] - (b - 1) * eq0_xyz[..., 2]
    eq5_xyz[..., 1] = eq0_xyz[..., 1]
    eq5_xyz[..., 2] = eq0_xyz[..., 2]
    eq5_lab = XYZ_to_Lab(eq5_xyz)

    eq8_xyz = np.zeros_like(eq0_xyz)
    eq8_xyz[..., 0] = b * eq0_xyz[..., 0] - (b - 1) * eq0_xyz[..., 2]
    eq8_xyz[..., 1] = g * eq0_xyz[..., 1] - (g - 1) * eq0_xyz[..., 0]
    eq8_xyz[..., 2] = eq0_xyz[..., 2]
    eq8_lab = XYZ_to_Lab(eq8_xyz)

    # plot!
    plot_xyz_prime_scatter(lab=eq0_lab, color=src_srgb, suffix="original")
    plot_xyz_prime_scatter(lab=eq5_lab, color=src_srgb, suffix="eq5")
    plot_xyz_prime_scatter(lab=eq8_lab, color=src_srgb, suffix="eq8")

    # src_srgb = tf.oetf(np.clip(linear_rgb, 0.0, 1.0), tf.SRGB)
    # tpg.img_wirte_float_as_16bit_int(
    #     f"./img/eq00_srgb_ll-{lab_ll}.png", src_srgb)


def plot_xyz_prime_scatter(lab, color, ab_max=140, suffix=""):
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 10),
        bg_color=(0.96, 0.96, 0.96),
        graph_title=f"{suffix}",
        graph_title_size=None,
        xlabel="a", ylabel="b",
        axis_label_size=None,
        legend_size=17,
        xlim=[-ab_max, ab_max],
        ylim=[-ab_max, ab_max],
        xtick=None,
        ytick=None)
    ax1.scatter(
        lab[..., 1], lab[..., 2], c=color.reshape(-1, 3))
    fname = f"./img/prime_on_lab_{suffix}.png"
    pu.show_and_save(
        fig=fig, legend_loc=None, show=False, save_fname=fname)


def plot_oetf_like_and_pq_oetf():
    x = np.linspace(0, 1, 1024) * 10000
    pq = tf.oetf_from_luminance(x, tf.ST2084)
    eq10 = st2084_oetf_like(x)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 7),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Comparison between BT.2100 PQ OETF and Eq. (10)",
        graph_title_size=None,
        xlabel="Luminance [nits]",
        ylabel="Code Value",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        linewidth=3)
    ax1.plot(x, pq, color=pu.RED, label="BT.2100 PQ OETF")
    ax1.plot(x, eq10, color=pu.BLUE, label="Eq. (10)")
    fname = "./img/bt210_pq_oetf_eq10.png"
    pu.show_and_save(
        fig=fig, legend_loc="lower right", show=False, save_fname=fname)


def check_iz_jz():
    iz = np.linspace(0, 1, 16)
    jz_org = iz_to_jz(iz)
    print(jz_org)
    jz_d0 = iz_to_jz(iz, d=0)
    jz_d_m064 = iz_to_jz(iz, d=-0.64)
    jz_d_11 = iz_to_jz(iz, d=1.1)
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 7),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Relationship between Iz and Jz",
        graph_title_size=None,
        xlabel="Iz", ylabel="Jz",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(iz, jz_org, pu.RED, label='d=-0.56')
    # ax1.plot(iz, jz_d0, pu.GREEN, label='d=0')
    # ax1.plot(iz, jz_d_m064, pu.BLUE, label='d=-0.64')
    # ax1.plot(iz, jz_d_11, pu.SKY, label='d=1.1')
    fname = "./img/iz_vs_jz.png"
    pu.show_and_save(
        fig=fig, legend_loc='lower right', show=False, save_fname=fname)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # sample_jzazbz_conv()
    # large_xyz = create_xyz_values_for_test()
    # save_data = dict(large_xyz=large_xyz)
    # savemat("./test_data.mat", save_data)

    # plot_diff_prime_function()
    # plot_xyz_prime_function_with_mpl()

    # plot eq.10
    # plot_oetf_like_and_pq_oetf()

    # plot eq.12
    check_iz_jz()

    # # test code
    # compare_reference_code()
    # check_inverse_function()
