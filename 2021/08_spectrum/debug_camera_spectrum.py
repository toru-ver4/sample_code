# -*- coding: utf-8 -*-
"""
spectrum
"""

# import standard libraries
import os
from colour.colorimetry.spectrum import MultiSpectralDistributions
from colour.models.rgb.datasets import srgb

# import third party libraries
import numpy as np
from colour import SpectralShape, XYZ_to_RGB, XYZ_to_xyY
from colour.models import RGB_COLOURSPACE_BT709
from sympy import Symbol, diff
from colour.utilities import tstack

# import my libraries
import plot_utility as pu
import spectrum_calculation as scl
from spectrum_calculation import VALID_WAVELENGTH_ST, VALID_WAVELENGTH_ED,\
    REFRECT_100P_SD
import color_space as cs
import test_pattern_generator2 as tpg
import transfer_functions as tf

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def load_camera_spectral_sensitivity_database():
    sony_ss = scl.get_sony_nex5_ss()

    fig, ax1 = pu.plot_1_graph(
        fontsize=18,
        figsize=(10, 6),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="SONY NEX-5N",
        graph_title_size=None,
        xlabel="Wavelength [nm]", ylabel="???",
        axis_label_size=None,
        legend_size=14,
        xlim=[380, 730],
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=2,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(
        sony_ss.wavelengths, sony_ss.values[..., 0], label="R",
        color=pu.RED, alpha=1.0)
    ax1.plot(
        sony_ss.wavelengths, sony_ss.values[..., 1], label="G",
        color=pu.GREEN, alpha=1.0)
    ax1.plot(
        sony_ss.wavelengths, sony_ss.values[..., 2], label="B",
        color=pu.BLUE, alpha=1.0)

    pu.show_and_save(
        fig=fig, legend_loc='upper right', save_fname="./img/sony_ssd.png")
    # pu.show_and_save(
    #     fig=fig, legend_loc='upper right', save_fname=None)


def plot_camera_gamut():
    sony_ss = scl.get_sony_nex5_ss()
    sony_csd = scl.CameraSpectralDistribution(sony_ss)
    primaries, white = sony_csd.calc_primary_xyY_and_white_xyY()
    print(primaries)
    print(white)

    fig, ax1 = pu.plot_1_graph(
        fontsize=18,
        figsize=(10, 10),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="SONY NEX-5N",
        graph_title_size=None,
        xlabel="x", ylabel="y",
        axis_label_size=None,
        legend_size=14,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=2,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(primaries[..., 0], primaries[..., 1], label="Gamut")
    ax1.plot(white[0], white[1], 'x', label="Gamut", ms=10, mew=3)

    pu.show_and_save(
        fig=fig, legend_loc='upper right', save_fname="./img/sony_gamut.png")


def debug_least_square_method():
    var_str_list = [
        ['m11', 'm12', 'm13'],
        ['m21', 'm22', 'm23'],
        ['m31', 'm32', 'm33']]
    mtx = [[Symbol(var_str_list[i][j]) for j in range(3)] for i in range(3)]

    xx = Symbol('xx')
    yy = Symbol('yy')
    zz = Symbol('zz')

    rr = Symbol('rr')
    gg = Symbol('gg')
    bb = Symbol('bb')

    jr = (xx - (mtx[0][0] * rr + mtx[0][1] * gg + mtx[0][2] * bb)) ** 2
    jg = (yy - (mtx[1][0] * rr + mtx[1][1] * gg + mtx[1][2] * bb)) ** 2
    jb = (zz - (mtx[2][0] * rr + mtx[2][1] * gg + mtx[2][2] * bb)) ** 2

    jj = jr + jg + jb

    m11_diff = diff(jr, mtx[0][0])
    m12_diff = diff(jr, mtx[0][1])
    m13_diff = diff(jr, mtx[0][2])

    print(m11_diff)
    print(m12_diff)
    print(m13_diff)


def debug_cct_matrix():
    color_temp = 6504
    light_sd = scl.calc_illuminant_d_spectrum(color_temp)
    color_checker_sd = scl.load_color_checker_spectrum()
    camera_ss = scl.get_sony_nex5_ss()
    cmfs = scl.get_cie_2_1931_cmf()
    cct_matrix = scl.calc_cct_matrix_from_color_checker(camera_ss=camera_ss)

    camera_rgb = scl.calc_tristimulus_values_from_multi_spectrum(
        src_sd=light_sd, ref_sd=color_checker_sd, ss=camera_ss)

    measure_xyz = scl.calc_xyz_from_multi_spectrum(
        src_sd=light_sd, ref_sd=color_checker_sd, cmfs=cmfs)

    print(cct_matrix)

    camera_xyz_using_mtx = scl.apply_matrix(src=camera_rgb, mtx=cct_matrix)

    true_rgb = XYZ_to_RGB(
        measure_xyz, cs.D65, cs.D65, RGB_COLOURSPACE_BT709.matrix_XYZ_to_RGB)
    estimated_rgb = XYZ_to_RGB(
        camera_xyz_using_mtx, cs.D65, cs.D65,
        RGB_COLOURSPACE_BT709.matrix_XYZ_to_RGB)

    true_rgb_srgb = tf.oetf(np.clip(true_rgb, 0.0, 1.0), tf.SRGB)
    est_rgb_srgb = tf.oetf(np.clip(estimated_rgb, 0.0, 1.0), tf.SRGB)
    img = tpg.plot_color_checker_image(
        rgb=true_rgb_srgb, rgb2=est_rgb_srgb)
    tpg.img_wirte_float_as_16bit_int("./img/cct_mtx.png", img)

    # primaries
    xmin = 0.0
    xmax = 0.8
    ymin = -0.4
    ymax = 1.2
    primary_rgb = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 1]])
    primary_xyz = scl.apply_matrix(primary_rgb, cct_matrix)
    primary_xyY = XYZ_to_xyY(primary_xyz)
    bt709_gamut, _ = tpg.get_primaries(name=cs.BT709)
    bt2020_gamut, _ = tpg.get_primaries(name=cs.BT2020)
    dci_p3_gamut, _ = tpg.get_primaries(name=cs.P3_D65)
    xy_image = tpg.get_chromaticity_image(
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(8, 14),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Chromaticity Diagram?",
        graph_title_size=None,
        xlabel="x", ylabel="y",
        axis_label_size=None,
        legend_size=17,
        xlim=[xmin, xmax],
        ylim=[ymin, ymax],
        xtick=[0.1 * x for x in range(9)],
        ytick=[0.1 * x - 0.4 for x in range(17)],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    cmf_xy = tpg._get_cmfs_xy()
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', label=None)
    ax1.plot(bt709_gamut[:, 0], bt709_gamut[:, 1],
             c=pu.RED, label="BT.709", lw=2, alpha=0.8)
    ax1.plot(bt2020_gamut[:, 0], bt2020_gamut[:, 1],
             c=pu.YELLOW, label="BT.2020", lw=2, alpha=0.8)
    ax1.plot(dci_p3_gamut[:, 0], dci_p3_gamut[:, 1],
             c=pu.BLUE, label="DCI-P3", lw=2, alpha=0.8)
    ax1.plot(
        (cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
        '-k', label=None)
    ax1.plot(
        primary_xyY[:4, 0], primary_xyY[:4, 1], color='k', label="SONY NEX-5N")
    ax1.imshow(xy_image, extent=(xmin, xmax, ymin, ymax))
    pu.show_and_save(
        fig=fig, legend_loc='upper right',
        save_fname="img/camera_chroma_test.png")


def calc_camera_gamut_from_ss():
    color_temp = 6504
    light_sd = scl.REFRECT_100P_SD
    camera_ss = scl.get_sony_nex5_ss()
    cmfs = scl.get_cie_2_1931_cmf()
    cr = camera_ss.values[..., 0]
    cg = camera_ss.values[..., 1]
    cb = camera_ss.values[..., 2]

    rr = cmfs.values[..., 0]
    gg = cmfs.values[..., 1]
    bb = cmfs.values[..., 2]

    r_base = cr - cr*cg - cr*cb
    g_base = cg - cg*cr - cg*cb
    b_base = cb - cb*cr - cb*cg

    rx = np.sum(r_base * rr)
    ry = np.sum(r_base * gg)
    rz = np.sum(r_base * bb)

    gx = np.sum(g_base * rr)
    gy = np.sum(g_base * gg)
    gz = np.sum(g_base * bb)

    bx = np.sum(b_base * rr)
    by = np.sum(b_base * gg)
    bz = np.sum(b_base * bb)

    r_xyY = XYZ_to_xyY(tstack([rx, ry, rz]))
    g_xyY = XYZ_to_xyY(tstack([gx, gy, gz]))
    b_xyY = XYZ_to_xyY(tstack([bx, by, bz]))
    print(r_xyY)
    print(g_xyY)
    print(b_xyY)


def plot_camera_capture_xy_value():
    wavelengths = REFRECT_100P_SD.wavelengths
    cmfs = scl.get_cie_2_1931_cmf()
    length = len(wavelengths)
    spectrum_array = np.zeros((length, length))
    for idx in range(length):
        spectrum_array[idx, idx] = 1

    data = dict(zip(wavelengths, spectrum_array))
    src_sd = MultiSpectralDistributions(data=data)
    camera_ss = scl.get_sony_nex5_ss()
    camera_rgb = scl.calc_tristimulus_values_from_multi_spectrum(
        src_sd=REFRECT_100P_SD, ref_sd=src_sd, ss=camera_ss)
    cct_matrix = scl.calc_cct_matrix_from_color_checker(camera_ss=camera_ss)
    camera_xyz_using_mtx = scl.apply_matrix(src=camera_rgb, mtx=cct_matrix)
    camera_xyY = XYZ_to_xyY(camera_xyz_using_mtx)
    # ok_idx = camera_xyY[..., 2] != 0
    ok_idx = (wavelengths >= 400) & (wavelengths <= 720)
    ok_wavelength = wavelengths[ok_idx]
    ok_xyY = camera_xyY[ok_idx]

    linear_rgb_from_line_spectrum = scl.calc_linear_rgb_from_spectrum(
        src_sd=REFRECT_100P_SD, ref_sd=src_sd, cmfs=cmfs,
        color_space=RGB_COLOURSPACE_BT709)
    linear_rgb_from_line_spectrum = linear_rgb_from_line_spectrum[ok_idx]
    linear_rgb_from_line_spectrum =\
        linear_rgb_from_line_spectrum / np.max(linear_rgb_from_line_spectrum,
                                               -1)[0]
    # primaries
    xmin = 0.0
    xmax = 0.8
    ymin = -0.4
    ymax = 1.2
    primary_rgb = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 1]])
    primary_xyz = scl.apply_matrix(primary_rgb, cct_matrix)
    primary_xyY = XYZ_to_xyY(primary_xyz)
    bt709_gamut, _ = tpg.get_primaries(name=cs.BT709)
    bt2020_gamut, _ = tpg.get_primaries(name=cs.BT2020)
    dci_p3_gamut, _ = tpg.get_primaries(name=cs.P3_D65)
    xy_image = tpg.get_chromaticity_image(
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(8, 14),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Chromaticity Diagram?",
        graph_title_size=None,
        xlabel="x", ylabel="y",
        axis_label_size=None,
        legend_size=17,
        xlim=[xmin, xmax],
        ylim=[ymin, ymax],
        xtick=[0.1 * x for x in range(9)],
        ytick=[0.1 * x - 0.4 for x in range(17)],
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    cmf_xy = tpg._get_cmfs_xy()
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', label=None)
    ax1.plot(bt709_gamut[:, 0], bt709_gamut[:, 1],
             c=pu.RED, label="BT.709", lw=2, alpha=0.8)
    ax1.plot(bt2020_gamut[:, 0], bt2020_gamut[:, 1],
             c=pu.YELLOW, label="BT.2020", lw=2, alpha=0.8)
    ax1.plot(dci_p3_gamut[:, 0], dci_p3_gamut[:, 1],
             c=pu.BLUE, label="DCI-P3", lw=2, alpha=0.8)
    ax1.plot(
        (cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
        '-k', label=None)
    ax1.plot(
        primary_xyY[:4, 0], primary_xyY[:4, 1], color='k', label="SONY NEX-5N")
    ax1.scatter(
        ok_xyY[..., 0], ok_xyY[..., 1], label="monochromatic light",
        edgecolors=None, c=(0.4, 0.4, 0.4)
    )
    ax1.imshow(xy_image, extent=(xmin, xmax, ymin, ymax))
    pu.show_and_save(
        fig=fig, legend_loc='upper right',
        save_fname="img/camera_chroma_with_line_spectrum.png")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # load_camera_spectral_sensitivity_database()
    # plot_camera_gamut()
    # debug_least_square_method()
    # debug_cct_matrix()
    # calc_camera_gamut_from_ss()
    plot_camera_capture_xy_value()
