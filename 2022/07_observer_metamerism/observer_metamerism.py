# -*- coding: utf-8 -*-
"""
"""

# import standard libraries
import os
from unittest.mock import patch
import requests

# import third-party libraries
import cv2
import numpy as np
from colour.continuous import MultiSignals, Signal
from colour import sd_to_XYZ, MultiSpectralDistributions, MSDS_CMFS,\
    SDS_ILLUMINANTS, SpectralShape, SpectralDistribution, XYZ_to_xyY,\
    XYZ_to_xy, xy_to_XYZ, XYZ_to_Lab
from colour.algebra import vector_dot
from colour.utilities import tstack
from colour.io import write_image, read_image

# import my libraries
import test_pattern_generator2 as tpg
from test_pattern_coordinate import GridCoordinate
import transfer_functions as tf
import color_space as cs
import plot_utility as pu
from spectrum import DisplaySpectrum, create_display_sd,\
    CIE1931_CMFS, CIE2012_CMFS, ILLUMINANT_E, START_WAVELENGTH,\
    STOP_WAVELENGTH, WAVELENGTH_STEP,\
    calc_xyz_to_rgb_matrix_from_spectral_distribution, trim_and_iterpolate,\
    wavelength_to_color
import color_space as cc
import font_control2 as fc2

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


DEFAULT_SPECTRAL_SHAPE = SpectralShape(
    START_WAVELENGTH, STOP_WAVELENGTH, WAVELENGTH_STEP)

SPECTRAL_SHAPE_FOR_10_CATEGORY_CMFS = SpectralShape(390, 730, 1)


bt709_msd = create_display_sd(
    r_mu=649, r_sigma=35, g_mu=539, g_sigma=33, b_mu=460, b_sigma=13,
    normalize_y=True)
p3_msd = create_display_sd(
    r_mu=620, r_sigma=12, g_mu=535, g_sigma=18, b_mu=458, b_sigma=8,
    normalize_y=True)
bt2020_msd = create_display_sd(
    r_mu=639, r_sigma=3, g_mu=530, g_sigma=4, b_mu=465, b_sigma=4,
    normalize_y=True)


def download_file(url, color_checker_sr_fname):
    response = requests.get(url)
    open(color_checker_sr_fname, "wb").write(response.content)


def prepaere_color_checker_sr_data():
    """
    Returns
    -------
    MultiSpectralDistributions
        multi-spectral distributions data.
    """
    color_checker_sr_fname = "./ref_data/color_checker_sr.txt"
    data = np.loadtxt(
        fname=color_checker_sr_fname, delimiter='\t', skiprows=1).T
    domain = np.arange(380, 740, 10)
    color_checker_signals = MultiSignals(data=data, domain=domain)
    color_checker_sds = MultiSpectralDistributions(data=color_checker_signals)

    return color_checker_sds


def color_checker_calc_sd_to_XYZ_D65_illuminant(
        spectral_shape=DEFAULT_SPECTRAL_SHAPE,
        cmfs=MSDS_CMFS['CIE 1931 2 Degree Standard Observer'],
        illuminant=SDS_ILLUMINANTS['D65']):
    color_checker_sds = prepaere_color_checker_sr_data()
    color_checker_sds = trim_and_iterpolate(color_checker_sds, spectral_shape)
    cmfs_trimed = trim_and_iterpolate(cmfs, spectral_shape)
    illuminant_intp = trim_and_iterpolate(illuminant, spectral_shape)
    # print(illuminant)
    XYZ = sd_to_XYZ(
        sd=color_checker_sds, cmfs=cmfs_trimed, illuminant=illuminant_intp)

    return XYZ


def create_color_checker_plus_d65_sd():
    color_checker_sr_fname = "./ref_data/color_checker_sr.txt"
    data = np.loadtxt(
        fname=color_checker_sr_fname, delimiter='\t', skiprows=1).T
    domain = np.arange(380, 740, 10)
    data_white = np.ones((len(domain), 1)) * 1.0
    data = np.append(data, data_white, axis=1)
    color_checker_signals = MultiSignals(data=data, domain=domain)
    color_checker_plut_d65_sds = MultiSpectralDistributions(
        data=color_checker_signals)

    return color_checker_plut_d65_sds


def calc_cc_plus_d65_xyz_for_each_cmfs_D65_illuminant(cmfs_list):
    """
    Parameters
    ----------
    cmfs_list : List
        A list of MultiSpectralDistributions

    Returns
    -------
    ndarray
        XYZ data.
        Shape is (num_of_cmfs, num_of_patch, 3).
    """
    spectral_shape = SPECTRAL_SHAPE_FOR_10_CATEGORY_CMFS
    color_checker_plut_d65_sds = create_color_checker_plus_d65_sd()
    color_checker_plut_d65_sds = trim_and_iterpolate(
        color_checker_plut_d65_sds, spectral_shape)
    illuminant = SDS_ILLUMINANTS['D65']
    illuminant = trim_and_iterpolate(illuminant, spectral_shape)

    # dummy_large_xyz = calc_cc_plus_d65_xyz(cmfs=cmfs_list[0])
    num_of_cmfs = len(cmfs_list)
    # num_of_patch = dummy_large_xyz.shape[0]
    num_of_patch = color_checker_plut_d65_sds.values.shape[1]

    large_xyz_out_buf = np.zeros((num_of_cmfs, num_of_patch, 3))
    for idx, cmfs in enumerate(cmfs_list):
        cmfs = trim_and_iterpolate(cmfs, spectral_shape)
        large_xyz = sd_to_XYZ(
            sd=color_checker_plut_d65_sds, cmfs=cmfs, illuminant=illuminant)
        large_xyz_out_buf[idx] = large_xyz

    return large_xyz_out_buf


def calc_xyz_to_rgb_matrix_each_display_sd_each_cmfs(
        display_sd_list, cmfs_list):
    """
    Returns
    -------
    ndarray
        Matrix list.
        Shape is (num_of_display, num_of_cmfs, 3, 3)
    """
    spectral_shape = SPECTRAL_SHAPE_FOR_10_CATEGORY_CMFS
    xyz_to_rgb_mtx_list = np.zeros(
        (len(display_sd_list), len(cmfs_list), 3, 3))
    for d_idx, display_sd in enumerate(display_sd_list):
        for c_idx, cmfs in enumerate(cmfs_list):
            mtx = calc_xyz_to_rgb_matrix_from_spectral_distribution(
                spd=display_sd, cmfs=cmfs, spectral_shape=spectral_shape)
            xyz_to_rgb_mtx_list[d_idx, c_idx] = mtx

    return xyz_to_rgb_mtx_list


def calc_tristimulus_value_for_each_sd_patch_cmfs(
        large_xyz, xyz_to_rgb_mtx, rgb_nomalize_val):
    """
    Parameters
    ----------
    large_xyz : ndarray
        XYZ. shape is (num_of_cmfs, 25, 3)
    xyz_to_rgb_mtx : ndarray
        XYZ to RGB matrix.
        shape is (num_of_display, num_of_cmfs, 3, 3)
    rgb_normalize_val : ndarray
        Normalize val.
        Normalize val is typically 100 because display'w Y_n = 100.
        But `Y_n = 100` is only valid when using CIE1931 CMFs.
        Therefore I prepare this normalize val.
        shape is (num_of_display, num_of_cmfs)

    Returns
    -------
    ndarray
        RGB value.
        Shape is (num_of_display, num_of_cmfs, num_of_patch, 3)
    """
    num_of_display = xyz_to_rgb_mtx.shape[0]
    num_of_cmfs = xyz_to_rgb_mtx.shape[1]
    num_of_patch = large_xyz.shape[1]
    out_rgb = np.zeros((num_of_display, num_of_cmfs, num_of_patch, 3))
    for d_idx in range(num_of_display):
        for c_idx in range(num_of_cmfs):
            mtx = xyz_to_rgb_mtx[d_idx, c_idx]
            large_xyz_temp = large_xyz[c_idx]
            rgb = vector_dot(mtx, large_xyz_temp)
            out_rgb[d_idx, c_idx] = rgb / rgb_nomalize_val[d_idx, c_idx]

    return out_rgb


def create_modified_display_sd_based_on_rgb_gain_core(
        sd: MultiSpectralDistributions, rgb):
    """
    Parameters
    ----------
    sd : DisplaySpectrum
        display spectrum
    """
    r_sd = sd.values[..., 0]
    g_sd = sd.values[..., 1]
    b_sd = sd.values[..., 2]
    r_sd_new = r_sd * rgb[0]
    g_sd_new = g_sd * rgb[1]
    b_sd_new = b_sd * rgb[2]
    w_sd_new = r_sd_new + g_sd_new + b_sd_new
    domain = sd.domain
    gained_signal = Signal(data=w_sd_new, domain=domain)
    modified_sd = SpectralDistribution(data=gained_signal)

    return modified_sd


def calc_display_Yn_for_each_cmfs(display_sd_list, cmfs_list):
    """
    Returns
    -------
    ndarray
        Normalize val.
        Normalize val is typically 100 because display's `Y_n` is 100.
        But `Y_n = 100` is only valid when using CIE1931 CMFs.
        Therefore I prepare this normalize val.
        shape is (num_of_display, num_of_cmfs)
    """
    num_of_display = len(display_sd_list)
    num_of_cmfs = len(cmfs_list)
    out_buf = np.zeros((num_of_display, num_of_cmfs))
    spectral_shape = SPECTRAL_SHAPE_FOR_10_CATEGORY_CMFS
    illuminant = trim_and_iterpolate(ILLUMINANT_E, spectral_shape)
    for d_idx in range(num_of_display):
        sd = display_sd_list[d_idx]
        sd = trim_and_iterpolate(sd, spectral_shape)
        for c_idx in range(num_of_cmfs):
            cmfs = cmfs_list[c_idx]
            cmfs = trim_and_iterpolate(cmfs, spectral_shape)
            xyz = sd_to_XYZ(sd=sd, cmfs=cmfs, illuminant=illuminant)
            out_buf[d_idx, c_idx] = xyz[3, 1]

    return out_buf


def create_modified_display_sd_based_on_rgb_gain(
        display_sd_list, rgb_list):
    """
    Parameters
    ----------
    display_sd_list : list
        A list of the display spectrum.
    rgb_list : ndarray
        RGB value for reproducting the specific color.
        Shape is (num_of_display, num_of_cmfs, num_of_patch)

    Returns
    -------
    A list of SpectralDistribution
        Shape is (num_of_display, num_of_cmfs, num_of_patch)
    """
    num_of_display = rgb_list.shape[0]
    num_of_cmfs = rgb_list.shape[1]
    num_of_patch = rgb_list.shape[2]
    out_buf = [
        [[0] * num_of_patch for ii in range(num_of_cmfs)]
        for jj in range(num_of_display)]
    spectral_shape = SPECTRAL_SHAPE_FOR_10_CATEGORY_CMFS

    for d_idx in range(num_of_display):
        sd = display_sd_list[d_idx]
        for c_idx in range(num_of_cmfs):
            for p_idx in range(num_of_patch):
                out_sd = create_modified_display_sd_based_on_rgb_gain_core(
                    sd=sd, rgb=rgb_list[d_idx, c_idx, p_idx])
                out_sd = trim_and_iterpolate(out_sd, spectral_shape)
                out_buf[d_idx][c_idx][p_idx] = out_sd

    return out_buf


def calc_XYZ_from_adjusted_display_sd_using_cie1931(sd_list, rgb_list):
    """
    Parameters
    ----------
    sd_list : list
        A list of the display spectrum.
    rgb_list : ndarray
        RGB value for reproducting the specific color.
        Shape is (num_of_display, num_of_cmfs, num_of_patch)

    Returns
    -------
    ndarray
        A list of XYZ.
        Shape is (num_of_display, num_of_cmfs, num_of_patch, 3)
    """
    num_of_display = rgb_list.shape[0]
    num_of_cmfs = rgb_list.shape[1]
    num_of_patch = rgb_list.shape[2]
    out_buf = np.zeros((num_of_display, num_of_cmfs, num_of_patch, 3))

    spectral_shape = SPECTRAL_SHAPE_FOR_10_CATEGORY_CMFS
    cmfs = trim_and_iterpolate(CIE1931_CMFS, spectral_shape)
    illuminant = trim_and_iterpolate(ILLUMINANT_E, spectral_shape)

    for d_idx in range(num_of_display):
        for c_idx in range(num_of_cmfs):
            for p_idx in range(num_of_patch):
                sd_data = sd_list[d_idx][c_idx][p_idx]
                xyz = sd_to_XYZ(
                    sd=sd_data, cmfs=cmfs, illuminant=illuminant)
                out_buf[d_idx, c_idx, p_idx] = xyz / 100

    return out_buf


def calc_display_sd_using_metamerism(
        large_xyz, base_msd):
    """
    Parameters
    ----------
    large_xyz : ndarray
        Target XYZ value. Its shape must be (N, 3).
    base_msd : MultiSpectralDistributions
        Default display spectral distribution.

    Returns
    -------
    MultiSpectralDistributions
        Display spectral distribution. Its shape is (W, N).
        "W" means the number of wavelengths.
    """
    ds = DisplaySpectrum(msd=base_msd)
    xyz_to_rgb_mtx = ds.get_rgb_to_xyz_mtx()

    rgb_gain = vector_dot(xyz_to_rgb_mtx, large_xyz) / 100
    rgb_gain = np.clip(rgb_gain, 0.0, 1.0)

    metamerism_spectrum = ds.calc_msd_from_rgb_gain(rgb=rgb_gain)

    return metamerism_spectrum


def debug_numpy_mult_check():
    r_coef = np.arange(6) + 1
    r_coef = r_coef.reshape(1, 6)
    sd = np.linspace(0, 1, 8).reshape(8, 1)
    print(r_coef)
    print(sd)
    yy = r_coef * sd
    print(yy)


def calc_cc_xyz_ref_val_and_actual_cmfs2(
        msd=None, cmfs2=CIE2012_CMFS, spectral_shape=DEFAULT_SPECTRAL_SHAPE):
    """
    Returns
    -------
    spectral_shape : ndarray
        A XYZ value calculated from spectral reflectance using cmfs2.
    mismatch_large_xyz_cmfs2 : ndarray
        A XYZ value calculated from display spectral distribution
        using cmfs2.
        display spectral distribution is adjusted based on cmfs1.
    """
    cmfs1 = CIE1931_CMFS
    cmfs1 = trim_and_iterpolate(cmfs1, spectral_shape)
    cmfs2 = trim_and_iterpolate(cmfs2, spectral_shape)

    # calc reference XYZ by seeing ColorChecker under D65 illuminant
    cc_large_xyz_1931 = color_checker_calc_sd_to_XYZ_D65_illuminant(
        spectral_shape=spectral_shape, cmfs=cmfs1)
    correct_ref_xyz_cmfs2 = color_checker_calc_sd_to_XYZ_D65_illuminant(
        spectral_shape=spectral_shape, cmfs=cmfs2)

    # create display spectrum using metamerism (cmfs is CIE1931)
    cc_spectrum = calc_display_sd_using_metamerism(
        large_xyz=cc_large_xyz_1931, base_msd=msd)
    cc_spectrum = trim_and_iterpolate(cc_spectrum, spectral_shape)

    # calc each observer's XYZ by seeing the display
    # adjusted for CIE1931 observer
    illuminant = ILLUMINANT_E
    illuminant = trim_and_iterpolate(illuminant, spectral_shape)
    mismatch_large_xyz_cmfs2 = sd_to_XYZ(
        sd=cc_spectrum, cmfs=cmfs2, illuminant=illuminant)

    return correct_ref_xyz_cmfs2, mismatch_large_xyz_cmfs2


def draw_mismatch_cmfs2_color_checker_image(
        large_xyz, mismatch_large_xyz, fname="./figure/hoge.png"):
    rgb = cs.large_xyz_to_rgb(
        xyz=large_xyz, color_space_name=cs.BT709)
    rgb = np.clip(rgb / 100, 0.0, 1.0)

    rgb_mismatch = cs.large_xyz_to_rgb(
        xyz=mismatch_large_xyz, color_space_name=cs.BT709)
    rgb_mismatch = np.clip(rgb_mismatch / 100, 0.0, 1.0)

    cc_image = tpg.plot_color_checker_image(
        rgb=rgb, rgb2=rgb_mismatch, size=(1280, 720))

    cc_image_srgb = tf.oetf(cc_image, tf.SRGB)

    tpg.img_wirte_float_as_16bit_int(fname, cc_image_srgb)


def debug_calc_msd_from_rgb_gain(
        r_mu=620, r_sigma=12, g_mu=535, g_sigma=18, b_mu=458, b_sigma=8,
        fname="./figure/sample222.png"):
    msd = create_display_sd(
        r_mu=r_mu, r_sigma=r_sigma,
        g_mu=g_mu, g_sigma=g_sigma,
        b_mu=b_mu, b_sigma=b_sigma,
        normalize_y=True)
    spectral_shape = DEFAULT_SPECTRAL_SHAPE
    cmfs_1931 = CIE1931_CMFS
    cmfs_2012 = CIE2012_CMFS
    illuminant = ILLUMINANT_E
    cmfs_1931 = trim_and_iterpolate(cmfs_1931, spectral_shape)
    cmfs_2012 = trim_and_iterpolate(cmfs_2012, spectral_shape)
    illuminant = trim_and_iterpolate(illuminant, spectral_shape)

    cc_large_xyz = color_checker_calc_sd_to_XYZ_D65_illuminant(
        spectral_shape=spectral_shape, cmfs=cmfs_1931)
    cc_large_xyz_2012 = color_checker_calc_sd_to_XYZ_D65_illuminant(
        spectral_shape=spectral_shape, cmfs=cmfs_2012)

    cc_spectrum = calc_display_sd_using_metamerism(
        large_xyz=cc_large_xyz, base_msd=msd)
    cc_spectrum = trim_and_iterpolate(cc_spectrum, spectral_shape)

    xyz_1931 = sd_to_XYZ(
        sd=cc_spectrum, cmfs=cmfs_1931, illuminant=illuminant)
    xyz_2012 = sd_to_XYZ(
        sd=cc_spectrum, cmfs=cmfs_2012, illuminant=illuminant)

    rgb_1931 = cs.large_xyz_to_rgb(
        xyz=xyz_1931, color_space_name=cs.BT709)
    rgb_2012 = cs.large_xyz_to_rgb(
        xyz=xyz_2012, color_space_name=cs.BT709)

    rgb_1931 = np.clip(rgb_1931 / 100, 0.0, 1.0)
    rgb_2012 = np.clip(rgb_2012 / 100, 0.0, 1.0)

    img_cat = tpg.plot_color_checker_image(
        rgb=rgb_1931, rgb2=rgb_2012, size=(1280, 720))
    img_1931 = tpg.plot_color_checker_image(
        rgb=rgb_1931, rgb2=None, size=(1280, 720))
    img_2012 = tpg.plot_color_checker_image(
        rgb=rgb_2012, rgb2=None, size=(1280, 720))

    tpg.img_wirte_float_as_16bit_int(fname, img_cat**(1/2.4))
    tpg.img_wirte_float_as_16bit_int(fname+"_1931.png", img_1931**(1/2.4))
    tpg.img_wirte_float_as_16bit_int(fname+"_2012.png", img_2012**(1/2.4))

    # print(rgb_gain)


def calc_delta_xyz(xyz1, xyz2):
    diff_x = xyz1[..., 0] - xyz2[..., 0]
    diff_y = xyz1[..., 1] - xyz2[..., 1]
    diff_z = xyz1[..., 2] - xyz2[..., 2]

    delta_xyz = ((diff_x ** 2) + (diff_y ** 2) + (diff_z ** 2)) ** 0.5

    return delta_xyz


def debug_plot_color_checker_delta_xyz(
        ok_xyz, ng_xyz_709, ng_xyz_p3, ng_xyz_2020, fname_suffix="0"):
    delta_709 = calc_delta_xyz(ok_xyz, ng_xyz_709)
    delta_p3 = calc_delta_xyz(ok_xyz, ng_xyz_p3)
    delta_2020 = calc_delta_xyz(ok_xyz, ng_xyz_2020)

    # label = [
    #     "dark skin", "light skin", "blue sky", "foliage", "blue flower",
    #     "bluish green", "orange", "purplish blue", "moderate red", "purple",
    #     "yellow green", "orange yellow", "blue", "green", "red", "yellow",
    #     "magenta", "cyan", "white 9.5", "neutral 8", "neutral 6.5",
    #     "neutral 5", "neutral 3.5", "black 2"]
    label = [
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
        "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24"]

    x = np.arange(24) + 1
    x_709 = x - 0.25
    x_p3 = x + 0.0
    x_2020 = x + 0.25

    fig, ax1 = pu.plot_1_graph()
    ax1.bar(
        x_709, delta_709, width=0.25, color=pu.RED,
        align="center", label="BT.709")
    ax1.bar(
        x_p3, delta_p3, width=0.25, color=pu.GREEN,
        align="center", label="DCI-P3")
    ax1.bar(
        x_2020, delta_2020, width=0.25, color=pu.BLUE,
        align="center", label="BT.2020")

    ax1.set_xticks(x, label)
    pu.show_and_save(
        fig=fig, legend_loc='upper left',
        save_fname=f"./figure/delta_xyz_{fname_suffix}.png")


def load_2deg_151_cmfs():
    """
    source: https://www.rit.edu/science/sites/rit.edu.science/files/2019-01/MCSL-Observer_Function_Database.pdf
    """
    fname_x = "./ref_data/RIT_MCSL_CMFs_151_02deg_x.csv"
    fname_y = "./ref_data/RIT_MCSL_CMFs_151_02deg_y.csv"
    fname_z = "./ref_data/RIT_MCSL_CMFs_151_02deg_z.csv"

    xx_base = np.loadtxt(fname=fname_x, delimiter=',')
    domain = xx_base[..., 0]
    xx = xx_base[..., 1:]
    yy = np.loadtxt(fname=fname_y, delimiter=',')[..., 1:]
    zz = np.loadtxt(fname=fname_z, delimiter=',')[..., 1:]

    num_of_cmfs = xx.shape[1]

    cmfs_array_151 = []

    for cmfs_idx in range(num_of_cmfs):
        sd = tstack([xx[..., cmfs_idx], yy[..., cmfs_idx], zz[..., cmfs_idx]])
        signals = MultiSignals(data=sd, domain=domain)
        sds = MultiSpectralDistributions(data=signals)
        cmfs_array_151.append(sds)

    return cmfs_array_151


def load_2deg_10_cmfs():
    """
    source: https://www.rit.edu/science/sites/rit.edu.science/files/2019-01/MCSL-Observer_Function_Database.pdf
    """
    fname_x = "./ref_data/RIT_MSCL_CMFs_10_02deg_x.csv"
    fname_y = "./ref_data/RIT_MSCL_CMFs_10_02deg_y.csv"
    fname_z = "./ref_data/RIT_MSCL_CMFs_10_02deg_z.csv"

    xx_base = np.loadtxt(fname=fname_x, delimiter=',')
    domain = xx_base[..., 0]
    xx = xx_base[..., 1:]
    yy = np.loadtxt(fname=fname_y, delimiter=',')[..., 1:]
    zz = np.loadtxt(fname=fname_z, delimiter=',')[..., 1:]

    num_of_cmfs = xx.shape[1]

    cmfs_array_10 = []

    for cmfs_idx in range(num_of_cmfs):
        sd = tstack([xx[..., cmfs_idx], yy[..., cmfs_idx], zz[..., cmfs_idx]])
        signals = MultiSignals(data=sd, domain=domain)
        sds = MultiSpectralDistributions(data=signals)
        cmfs_array_10.append(sds)

    return cmfs_array_10


def debug_plot_151_cmfs():
    cmfs_array = load_2deg_151_cmfs()

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="151 color-normal human cmfs",
        graph_title_size=None,
        xlabel="Wavelength [nm]",
        ylabel="Tristimulus Values",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=1,
        minor_xtick_num=None,
        minor_ytick_num=None)
    for cmfs in cmfs_array:
        ax1.plot(
            cmfs.wavelengths, cmfs.values[..., 0], '-',
            color=pu.RED, alpha=1/5)
        ax1.plot(
            cmfs.wavelengths, cmfs.values[..., 1], '-',
            color=pu.GREEN, alpha=1/5)
        ax1.plot(
            cmfs.wavelengths, cmfs.values[..., 2], '-',
            color=pu.BLUE, alpha=1/5)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', save_fname="./figure/cmfs_151.png")


def debug_plot_10_cmfs():
    cmfs_array = load_2deg_10_cmfs()
    spectral_shape = SpectralShape(300, 830, 1)

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="10 categorical observers (xyz 2 degree)",
        graph_title_size=None,
        xlabel="Wavelength [nm]",
        ylabel="Tristimulus Values",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=1,
        minor_xtick_num=None,
        minor_ytick_num=None)
    for cmfs in cmfs_array:
        cmfs = cmfs.extrapolate(spectral_shape)
        cmfs = cmfs.interpolate(spectral_shape)
        ax1.plot(
            cmfs.wavelengths, cmfs.values[..., 0], '-',
            color=pu.RED, alpha=1/2)
        ax1.plot(
            cmfs.wavelengths, cmfs.values[..., 1], '-',
            color=pu.GREEN, alpha=1/2)
        ax1.plot(
            cmfs.wavelengths, cmfs.values[..., 2], '-',
            color=pu.BLUE, alpha=1/2)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', save_fname="./figure/cmfs_10.png")


def create_709_p3_2020_display_sd():
    # bt709_msd = create_display_sd(
    #     r_mu=649, r_sigma=35, g_mu=539, g_sigma=33, b_mu=460, b_sigma=13,
    #     normalize_y=True)
    # p3_msd = create_display_sd(
    #     r_mu=620, r_sigma=12, g_mu=535, g_sigma=18, b_mu=458, b_sigma=8,
    #     normalize_y=True)
    # bt2020_msd = create_display_sd(
    #     r_mu=639, r_sigma=3, g_mu=530, g_sigma=4, b_mu=465, b_sigma=4,
    #     normalize_y=True)

    return bt709_msd, p3_msd, bt2020_msd


def debug_calc_and_plot_metamerism_delta():
    bt709_msd, p3_msd, bt2020_msd = create_709_p3_2020_display_sd()
    cmfs_list = load_2deg_10_cmfs()
    result_709_list = []
    result_p3_list = []
    result_2020_list = []
    ref_xyz_list = []

    for idx, cmfs in enumerate(cmfs_list):
        print(f"calc cmfs idx: {idx}")
        ok_xyz, ng_xyz_709 =\
            calc_cc_xyz_ref_val_and_actual_cmfs2(
                msd=bt709_msd, cmfs2=cmfs)
        ok_xyz, ng_xyz_p3 =\
            calc_cc_xyz_ref_val_and_actual_cmfs2(
                msd=p3_msd, cmfs2=cmfs)
        ok_xyz, ng_xyz_2020 =\
            calc_cc_xyz_ref_val_and_actual_cmfs2(
                msd=bt2020_msd, cmfs2=cmfs)
        ref_xyz_list.append(ok_xyz)
        result_709_list.append(ng_xyz_709)
        result_p3_list.append(ng_xyz_p3)
        result_2020_list.append(ng_xyz_2020)

    for cmfs_idx in range(len(cmfs_list)):
        debug_plot_color_checker_delta_xyz(
            ok_xyz=ref_xyz_list[cmfs_idx],
            ng_xyz_709=result_709_list[cmfs_idx],
            ng_xyz_p3=result_p3_list[cmfs_idx],
            ng_xyz_2020=result_2020_list[cmfs_idx],
            fname_suffix=f"cmfs_idx-{cmfs_idx:02d}")


def debug_save_white_patch(large_xyz):
    # org_shape = large_xyz.shape
    # large_xyz = large_xyz.reshape(1, -1, 3)
    xy = XYZ_to_xy(large_xyz)
    large_xyz_nomalized = xy_to_XYZ(xy)
    rgb = cc.large_xyz_to_rgb(
        xyz=large_xyz_nomalized, color_space_name=cs.BT709,
        xyz_white=cs.D65, rgb_white=cs.D65)
    # rgb = rgb.reshape(org_shape)
    print(f"max={np.max(rgb[:, :, 24])}, min={np.min(rgb[:, :, 24])}")
    rgb = rgb / np.max(rgb[:, :, 24])
    img_size = 200
    base_img = np.ones((img_size, img_size, 3))
    v_img_buf = []
    for d_idx in range(3):
        h_img_buf = []
        for c_idx in range(11):
            img = base_img * rgb[d_idx, c_idx, 24]  # 24 is white
            tpg.draw_outline(img, fg_color=[0.2, 0.2, 0.2], outline_width=1)
            h_img_buf.append(img)
        v_img_buf.append(np.hstack(h_img_buf))
    out_img = np.vstack(v_img_buf)
    out_img = tf.oetf(np.clip(out_img, 0.0, 1.0), tf.SRGB)

    write_image(out_img, "./debug/white_with_multi_cmfs.png")


def debug_save_color_checker(large_xyz):
    # org_shape = large_xyz.shape
    # large_xyz = large_xyz.reshape(1, -1, 3)
    rgb = cc.large_xyz_to_rgb(
        xyz=large_xyz, color_space_name=cs.BT709,
        xyz_white=cs.D65, rgb_white=cs.D65)
    # rgb = rgb.reshape(org_shape)
    print(f"max={np.max(rgb[:, :, :24])}, min={np.min(rgb[:, :, :24])}")
    rgb = rgb / np.max(rgb[:, :, :24])
    ref_rgb = rgb[0, 10, :24]
    num_of_display = rgb.shape[0]
    num_of_cmfs = rgb.shape[1]
    for d_idx in range(num_of_display):
        for c_idx in range(num_of_cmfs):
            cc_data = rgb[d_idx, c_idx, :24]
            img = tpg.plot_color_checker_image(
                rgb=cc_data, rgb2=ref_rgb, side_trim=True)
            img = tf.oetf(np.clip(img, 0.0, 1.0), tf.SRGB)
            fname = "./figure/color_checker_cmfs-"
            fname += f"{c_idx:02d}_display-{d_idx:02d}.png"
            print(fname)
            write_image(img, fname)


def plot_color_checker_image_11_patch(
        rgb, size=(1920, 1080), block_size=1/4.5, side_trim=True):
    """
    ColorCheckerをプロットする

    Parameters
    ----------
    rgb : array_like
        RGB value of the ColorChecker using 11 cmfs.
        RGB's shape must be (11, 24, 3).
    size : tuple
        canvas size.
    block_size : float
        A each block's size.
        This value is ratio to height of the canvas.

    Returns
    -------
    array_like
        A ColorChecker image.

    """
    # 基本パラメータ算出
    # --------------------------------------
    cc_h_num = 6
    cc_v_num = 4
    img_height = size[1]
    img_width = size[0]
    patch_width = int(img_height * block_size)
    patch_height = patch_width
    patch_space = int(
        (img_height - patch_height * cc_v_num) / (cc_v_num + 1))

    patch_st_h = int(
        img_width / 2.0
        - patch_width * cc_h_num / 2.0
        - patch_space * (cc_h_num / 2.0 - 0.5)
    )
    patch_st_v = int(
        img_height / 2.0
        - patch_height * cc_v_num / 2.0
        - patch_space * (cc_v_num / 2.0 - 0.5)
    )

    # 24ループで1枚の画像に24パッチを描画
    # -------------------------------------------------
    img_all_patch = np.zeros((img_height, img_width, 3))
    for idx in range(cc_h_num * cc_v_num):
        v_idx = idx // cc_h_num
        h_idx = (idx % cc_h_num)
        patch = np.ones((patch_height, patch_width, 3))
        patch[:, :] = plot_11_patch_rectangle(
            data=rgb[:, idx],
            patch_width=patch_width, patch_height=patch_height)
        st_h = patch_st_h + (patch_width + patch_space) * h_idx
        st_v = patch_st_v + (patch_height + patch_space) * v_idx
        img_all_patch[st_v:st_v+patch_height, st_h:st_h+patch_width] = patch

    if side_trim:
        img_trim_h_st = patch_st_h - patch_space
        img_trim_h_ed = patch_st_h + (patch_width + patch_space) * 6
        img_all_patch = img_all_patch[:, img_trim_h_st:img_trim_h_ed]

    return img_all_patch


def debug_save_color_checker_11_patch(large_xyz):
    rgb = cc.large_xyz_to_rgb(
        xyz=large_xyz, color_space_name=cs.BT709,
        xyz_white=cs.D65, rgb_white=cs.D65)
    rgb = rgb / np.max(rgb)
    # rgb = rgb.reshape(org_shape)
    print(f"max={np.max(rgb)}, min={np.min(rgb)}")
    num_of_display = rgb.shape[0]
    for d_idx in range(num_of_display):
        cc_data = rgb[d_idx]
        # cc_data = cc_data / np.max(cc_data)
        img = plot_color_checker_image_11_patch(rgb=cc_data)
        img = tf.oetf(np.clip(img, 0.0, 1.0), tf.SRGB)
        fname = "./figure/11_patch_color_checker_cmfs-"
        fname += f"display-{d_idx:02d}.png"
        print(fname)
        write_image(img, fname)


def debug_verify_calibrated_sd(modified_sd_list, cmfs_list, large_xyz):
    spectral_shape = SPECTRAL_SHAPE_FOR_10_CATEGORY_CMFS
    illuminant = trim_and_iterpolate(ILLUMINANT_E, spectral_shape)
    for d_idx in range(3):
        for c_idx in range(10):
            cmfs = cmfs_list[c_idx]
            cmfs = trim_and_iterpolate(cmfs, spectral_shape)
            for p_idx in range(25):
                # if p_idx < 24:
                #     continue
                sd = modified_sd_list[d_idx][c_idx][p_idx]
                sd = trim_and_iterpolate(sd, spectral_shape)
                xyz = sd_to_XYZ(
                    sd=sd, cmfs=cmfs, illuminant=illuminant)
                ref_xyz = large_xyz[c_idx, p_idx]
                xyY = XYZ_to_xyY(xyz)
                ref_xyY = XYZ_to_xyY(ref_xyz)
                diff = ref_xyY - xyY
                msg = f"(d, c, p)=({d_idx}, {c_idx}, {p_idx}), "
                msg += f"{ref_xyY}-{xyY}={diff}"
                print(msg)


def debug_xyz_to_rgb_matrix(display_sd, cmfs):
    calc_xyz_to_rgb_matrix_from_spectral_distribution(
        spd=display_sd, cmfs=cmfs)


def plot_11_patch_rectangle(
        data, patch_width=640, patch_height=640):
    width_list = tpg.equal_devision(patch_width, 4)
    height_list = tpg.equal_devision(patch_height, 4)
    big_height = height_list[1] + height_list[2]
    big_width = width_list[1] + width_list[2]

    # upper side
    img_buf = []
    for idx in range(4):
        img = np.ones((height_list[0], width_list[idx], 3))
        img = img * data[idx]
        # print(f"data_idx={idx}, {data[idx]}")
        img_buf.append(img)
    top_img = np.hstack(img_buf)

    # bottom side
    img_buf = []
    idx_cnt = 0
    for idx in range(5, 9)[::-1]:
        img = np.ones((height_list[3], width_list[idx_cnt], 3))
        idx_cnt += 1
        img = img * data[idx]
        # print(f"data_idx={idx}, {data[idx]}")
        img_buf.append(img)
    bottom_img = np.hstack(img_buf)

    # center side
    img_buf = []
    img = np.ones((big_height, width_list[0], 3))
    img = img * data[9]
    # print(f"data_idx={9}, {data[9]}")
    img_buf.append(img)
    img = np.ones((big_height, big_width, 3))
    img = img * data[10]
    # print(f"data_idx={10}, {data[9]}")
    img_buf.append(img)
    img = np.ones((big_height, width_list[3], 3))
    img = img * data[4]
    # print(f"data_idx={4}, {data[9]}")
    img_buf.append(img)
    center_img = np.hstack(img_buf)
    img = np.vstack([top_img, center_img, bottom_img])

    return img


def large_xyz_to_xy(large_xyz):
    """
    Parameters
    ----------
    large_xyz : ndarray
        XYZ. shape is (..., 3)

    Returns
    -------
    ndarray
        xy. shape is (..., 2)
    """
    org_shape = large_xyz.shape
    large_xyz_temp = large_xyz.reshape(-1, 3)
    large_x = large_xyz_temp[..., 0]
    large_y = large_xyz_temp[..., 1]
    large_z = large_xyz_temp[..., 2]
    sum_val = (large_x + large_y + large_z).reshape(-1, 1)
    small_xyz = (large_xyz_temp / sum_val).reshape(org_shape)

    return small_xyz[..., :2]


def calc_delta_xy(large_xyz_1, large_xyz_2):
    xy1 = large_xyz_to_xy(large_xyz_1)
    xy2 = large_xyz_to_xy(large_xyz_2)
    diff_xy =\
        ((xy2[..., 0] - xy1[..., 0]) ** 2) + ((xy2[..., 1] - xy1[..., 1]) ** 2)
    diff_xy = diff_xy ** 0.5

    return diff_xy


def calc_intra_observer_error():
    """
    Shape is (num_of_display, num_of_cmfs, num_of_patch, 3)

    Returns
    -------
    ndarray
        delta xy value between CC under D65 and CC displayed.
        dislay spd is adjusted based on CIE1931 CMFS.
        Shape is (num_of_display, num_of_cmfs, num_of_patch, 3)
    """
    display_list = create_709_p3_2020_display_sd()
    cmfs_list = load_2deg_10_cmfs()
    cmfs_list.append(CIE1931_CMFS)
    num_of_display = len(display_list)
    num_of_cmfs = len(cmfs_list)
    num_of_patch = 24
    debug_fname = "./debug/intra_error_diff_xy.npy"

    # delta_xy_all = np.zeros(
    #     (num_of_display, num_of_cmfs, num_of_patch))

    # for d_idx in range(num_of_display):
    #     for c_idx in range(num_of_cmfs):
    #         print(f"calc_delta_xy_disp-{d_idx}_cmfs-{c_idx}")
    #         display_sd = display_list[d_idx]
    #         cmfs = cmfs_list[c_idx]
    #         ok_xyz, ng_xyz =\
    #             calc_cc_xyz_ref_val_and_actual_cmfs2(
    #                 msd=display_sd, cmfs2=cmfs,
    #                 spectral_shape=SPECTRAL_SHAPE_FOR_10_CATEGORY_CMFS)
    #         diff_xy = calc_delta_xy(ok_xyz, ng_xyz)
    #         delta_xy_all[d_idx, c_idx] = diff_xy
    # np.save(debug_fname, delta_xy_all)

    delta_xy_all = np.load(debug_fname)

    return delta_xy_all


def create_intra_error_single_patch_name(c_idx, p_idx):
    fname = f"./debug/intra_error_single_{c_idx:02d}_{p_idx:02d}.png"
    return fname


def plot_intra_error_single_patch(
        delta_xy, y_max, patch_color, c_idx, p_idx):
    """
    """
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(1.5, 1.5),
        bg_color=patch_color,
        graph_title=None,
        graph_title_size=None,
        xlabel=None,
        ylabel=None,
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=[0, y_max*1.03],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    x = np.arange(3) + 1
    color_list = [
        pu.RED, pu.YELLOW, pu.GREEN, pu.BLUE, pu.SKY,
        pu.PINK, pu.ORANGE, pu.MAJENTA, pu.BROWN, pu.GRAY50,
        pu.GRAY05]
    for idx in range(len(x)):
        ax1.bar(
            x[idx], delta_xy[..., idx], color=color_list[idx],
            edgecolor=pu.GRAY05)
    fname = create_intra_error_single_patch_name(
        c_idx=c_idx, p_idx=p_idx)
    print(fname)
    pu.show_and_save(
        fig=fig, save_fname=fname, only_graph_area=True)


def get_color_checker_srgb_val():
    cc_rgb = tpg.generate_color_checker_rgb_value()

    return tf.oetf(np.clip(cc_rgb, 0.0, 1.0), tf.SRGB)


def get_intra_observer_error_single_patch_size():
    fname = create_intra_error_single_patch_name(0, 0)
    img = read_image(fname)
    width = img.shape[1]
    height = img.shape[0]

    return width, height


def get_inter_observer_error_single_patch_size():
    fname = create_inter_error_single_patch_name(0, 0)
    img = read_image(fname)
    width = img.shape[1]
    height = img.shape[0]

    return width, height


def draw_24_cc_patch_intra_observer_error(c_idx):
    h_num = 6
    v_num = 4
    margin_rate = 0.15
    patch_width, patch_height = get_intra_observer_error_single_patch_size()
    margin_h = patch_width * margin_rate
    margin_v = patch_height * margin_rate
    bg_width = int(
        (patch_width * h_num) + margin_h * (h_num + 1))
    bg_height = int(
        (patch_height * v_num) + margin_v * (v_num + 1))

    gc = GridCoordinate(
        bg_width=bg_width, bg_height=bg_height,
        fg_width=patch_width, fg_height=patch_height,
        h_num=6, v_num=4, remove_tblr_margin=False)
    pos_list = gc.get_st_pos_list()
    img = np.zeros((bg_height, bg_width, 3))
    for v_idx in range(v_num):
        for h_idx in range(h_num):
            p_idx = v_idx * h_num + h_idx
            patch_fname = create_intra_error_single_patch_name(
                c_idx=c_idx, p_idx=p_idx)
            patch_img = read_image(patch_fname)[..., :3]
            tpg.merge(img, patch_img, pos_list[h_idx][v_idx])

    # composite legend
    legend_img_name = create_intra_observer_bar_legend_img_fname()
    legend_img = read_image(legend_img_name)[..., :3]
    rate = 0.6
    dst_size = (int(legend_img.shape[1] * rate),
                int(legend_img.shape[0] * rate))
    legend_img = cv2.resize(legend_img, dst_size)
    legend_bg = np.zeros((bg_height, int(legend_img.shape[1]+margin_h), 3))
    tpg.merge(legend_bg, legend_img, (0, int(margin_v)))
    img = np.hstack([img, legend_bg])

    # draw categorical observer info
    info_str_list = [
        f"Cat.Obs. {ii+1:02d}" for ii in range(10)]
    info_str_list.append("CIE1931 Observer")
    print(f"c_idx={c_idx}")
    font_color = tf.eotf(np.array([0.96, 0.96, 0.96]), tf.SRGB)
    text_draw_ctrl = fc2.TextDrawControl(
        text=info_str_list[c_idx], font_color=font_color,
        font_size=20, font_path=fc2.NOTO_SANS_CJKJP_BOLD,
        stroke_width=0, stroke_fill=None)
    # calc position
    pos_h = bg_width
    pos_v = int(margin_v) * 2 + legend_img.shape[0]
    pos = (pos_h, pos_v)
    # draw text
    img_linear = tf.eotf(img, tf.SRGB)
    text_draw_ctrl.draw(img=img_linear, pos=pos)
    img = tf.oetf(img_linear, tf.SRGB)

    fname = f"./figure/intra_observer_error_cmfs-{c_idx:02d}.png"
    write_image(img, fname, 'uint8')


def plot_intra_observer_error(delta_xy_all):
    """
    Parameters
    ----------
    delta_xy_all : ndarray
        delta xy value between CC under D65 and CC displayed.
        dislay spd is adjusted based on CIE1931 CMFS.
        Shape is (num_of_display, num_of_cmfs, num_of_patch, 3)
    """
    dummy_plot_for_display_gamut_legend_intra_error()
    max_err = np.max(delta_xy_all)
    cc_rgb_srgb = get_color_checker_srgb_val()
    num_of_cmfs = delta_xy_all.shape[1]
    num_of_patch = delta_xy_all.shape[2]

    # for c_idx in range(num_of_cmfs):
    #     for p_idx in range(num_of_patch):
    #         plot_intra_error_single_patch(
    #             delta_xy=delta_xy_all[:, c_idx, p_idx],
    #             patch_color=cc_rgb_srgb[p_idx], y_max=max_err,
    #             c_idx=c_idx, p_idx=p_idx)

    for c_idx in range(num_of_cmfs):
        draw_24_cc_patch_intra_observer_error(c_idx=c_idx)


def create_intra_observer_bar_legend_img_fname():
    return "./figure/debug_display_legend_trim.png"


def dummy_plot_for_display_gamut_legend_intra_error():
    fig, ax1 = pu.plot_1_graph(
        figsize=(10, 8))
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.bar([1], [1], label="BT.709 Display", color=pu.RED)
    ax1.bar([2], [2], label="DCI-P3 Display", color=pu.YELLOW)
    ax1.bar([3], [3], label="BT.2020 Display", color=pu.GREEN)
    print("hoge")
    graph_name = "./debug/debug_display_legend.png"
    pu.show_and_save(
        fig=fig, legend_loc='upper left', save_fname=graph_name)

    img = read_image(graph_name)
    st_h = 47
    st_v = 47
    ed_h = st_h + 318
    ed_v = st_v + 129
    trim_img = img[st_v:ed_v, st_h:ed_h]

    out_fname = create_intra_observer_bar_legend_img_fname()
    write_image(trim_img, out_fname)


def create_inter_observer_bar_legend_img_fname_inter_error():
    return "./figure/debug_display_legend_trim_inter_error.png"


def dummy_plot_for_display_gamut_legend_inter_error():
    fig, ax1 = pu.plot_1_graph(
        figsize=(10, 8))
    color_list = [
        pu.RED, pu.YELLOW, pu.GREEN, pu.BLUE, pu.SKY,
        pu.PINK, pu.ORANGE, pu.MAJENTA, pu.BROWN, pu.GRAY50,
        pu.GRAY05]
    base_str = "Cat.Obs."
    label_list = [f"{base_str} {idx+1:02d}" for idx in range(10)]
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    for idx in range(9):
        ax1.bar([idx], [1], label=label_list[idx], color=color_list[idx])
    ax1.bar([idx], [100], label=label_list[9], color=color_list[9])
    graph_name = "./debug/debug_display_legend_inter_error.png"
    pu.show_and_save(
        fig=fig, legend_loc='upper left', save_fname=graph_name)

    img = read_image(graph_name)
    st_h = 46
    st_v = 46
    ed_h = st_h + 259
    ed_v = st_v + 423
    trim_img = img[st_v:ed_v, st_h:ed_h]

    out_fname = create_inter_observer_bar_legend_img_fname_inter_error()
    write_image(trim_img, out_fname)


def calc_inter_observer_error():
    """
    calculation delta Eab between reference CIE1931 viewing and
    each observer-viewing?

    Returns
    -------
    ndarray
        A list of XYZ.
        Shape is (num_of_display, num_of_cmfs, num_of_patch, 3)
    """
    # cmfs_list = load_2deg_10_cmfs()
    # cmfs_list.append(CIE1931_CMFS)

    # # calc reference XYZ value using D65 illuminant
    # large_xyz_il_d65 = calc_cc_plus_d65_xyz_for_each_cmfs_D65_illuminant(
    #     cmfs_list=cmfs_list)

    # # calc RGB tristimulus value for each display to color match on each CMFS
    # display_sd_list = create_709_p3_2020_display_sd()
    # xyz_to_rgb_mtx = calc_xyz_to_rgb_matrix_each_display_sd_each_cmfs(
    #     display_sd_list=display_sd_list, cmfs_list=cmfs_list)
    # large_y_n = calc_display_Yn_for_each_cmfs(
    #     display_sd_list=display_sd_list, cmfs_list=cmfs_list)
    # rgb = calc_tristimulus_value_for_each_sd_patch_cmfs(
    #     large_xyz=large_xyz_il_d65, xyz_to_rgb_mtx=xyz_to_rgb_mtx,
    #     rgb_nomalize_val=large_y_n)

    # # apply RGB tristimulus value for each display
    # modified_sd_list = create_modified_display_sd_based_on_rgb_gain(
    #     display_sd_list=display_sd_list, rgb_list=rgb)

    # # calc each dislay spd using cie1931
    # large_xyz_1931 = calc_XYZ_from_adjusted_display_sd_using_cie1931(
    #     sd_list=modified_sd_list, rgb_list=rgb)

    # np.save("./debug/calibrated_xyz.npy", large_xyz_1931)

    large_xyz_1931 = np.load("./debug/calibrated_xyz.npy")

    # debug_verify_calibrated_sd(
    #     modified_sd_list=modified_sd_list, cmfs_list=cmfs_list,
    #     large_xyz=large_xyz_il_d65)

    return large_xyz_1931


def calc_delta_eab_inter_observer_error(large_xyz_1931):
    """
    Parameters
    ----------
    large_xyz_1931 : ndarray
        XYZ value. CIE1931 observer see the display adjusted for each CMFS.
        Shape is (num_of_display, num_of_cmfs, num_of_patch, 3)

    Returns
    -------
    ndarray
        delta Eab (CIE1931 XYZ - Cat 01~10 CMFS XYZ)
    """
    lab_1976 = XYZ_to_Lab(large_xyz_1931)
    ref_lab = (lab_1976[0, 10]).reshape(1, 1, 25, 3)
    delta_a = ref_lab[..., 1] - lab_1976[..., 1]
    delta_b = ref_lab[..., 2] - lab_1976[..., 2]
    delta_eab = ((delta_a ** 2) + (delta_b ** 2)) ** 0.5

    return delta_eab

    # # debug code
    # num_of_display = large_xyz_1931.shape[0]
    # num_of_cmfs = large_xyz_1931.shape[1]
    # for d_idx in range(num_of_display):
    #     for c_idx in range(num_of_cmfs):
    #         print(f"{d_idx:02d} - {c_idx:02d}")
    #         print(delta_eab[d_idx, c_idx, :24])


def create_inter_error_single_patch_name(d_idx, p_idx):
    fname = f"./debug/inter_error_single_{d_idx:02d}_{p_idx:02d}.png"
    return fname


def plot_inter_error_single_patch(
        delta_xy, y_max, patch_color, d_idx, p_idx):
    """
    """
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(1.5, 1.5),
        bg_color=patch_color,
        graph_title=None,
        graph_title_size=None,
        xlabel=None,
        ylabel=None,
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=[0, y_max*1.03],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    x = np.arange(10) + 1
    color_list = [
        pu.RED, pu.YELLOW, pu.GREEN, pu.BLUE, pu.SKY,
        pu.PINK, pu.ORANGE, pu.MAJENTA, pu.BROWN, pu.GRAY50,
        pu.GRAY05]
    for idx in range(len(x)):
        ax1.bar(
            x[idx], delta_xy[..., idx], color=color_list[idx],
            edgecolor=pu.GRAY05)
    fname = create_inter_error_single_patch_name(
        d_idx=d_idx, p_idx=p_idx)
    print(fname)
    pu.show_and_save(
        fig=fig, save_fname=fname, only_graph_area=True)


def draw_24_cc_patch_inter_observer_error(d_idx):
    h_num = 6
    v_num = 4
    margin_rate = 0.15
    patch_width, patch_height = get_inter_observer_error_single_patch_size()
    margin_h = patch_width * margin_rate
    margin_v = patch_height * margin_rate
    bg_width = int(
        (patch_width * h_num) + margin_h * (h_num + 1))
    bg_height = int(
        (patch_height * v_num) + margin_v * (v_num + 1))

    gc = GridCoordinate(
        bg_width=bg_width, bg_height=bg_height,
        fg_width=patch_width, fg_height=patch_height,
        h_num=6, v_num=4, remove_tblr_margin=False)
    pos_list = gc.get_st_pos_list()
    img = np.zeros((bg_height, bg_width, 3))
    for v_idx in range(v_num):
        for h_idx in range(h_num):
            p_idx = v_idx * h_num + h_idx
            patch_fname = create_inter_error_single_patch_name(
                d_idx=d_idx, p_idx=p_idx)
            patch_img = read_image(patch_fname)[..., :3]
            tpg.merge(img, patch_img, pos_list[h_idx][v_idx])

    # composite legend
    legend_img_name = create_inter_observer_bar_legend_img_fname_inter_error()
    legend_img = read_image(legend_img_name)[..., :3]
    rate = 0.6
    dst_size = (int(legend_img.shape[1] * rate),
                int(legend_img.shape[0] * rate))
    legend_img = cv2.resize(legend_img, dst_size)
    legend_bg = np.zeros((bg_height, int(legend_img.shape[1]+margin_h), 3))
    tpg.merge(legend_bg, legend_img, (0, int(margin_v)))
    img = np.hstack([img, legend_bg])

    # draw display gamut info
    info_str_list = [
        "Display Gamut:\n    BT.709",
        "Display Gamut:\n    DCI-P3",
        "Display Gamut:\n    BT.2020"
    ]
    font_color = tf.eotf(np.array([0.96, 0.96, 0.96]), tf.SRGB)
    text_draw_ctrl = fc2.TextDrawControl(
        text=info_str_list[d_idx], font_color=font_color,
        font_size=20, font_path=fc2.NOTO_SANS_CJKJP_BOLD,
        stroke_width=0, stroke_fill=None)
    # calc position
    pos_h = bg_width
    pos_v = int(margin_v) * 2 + legend_img.shape[0]
    pos = (pos_h, pos_v)
    # draw text
    img_linear = tf.eotf(img, tf.SRGB)
    text_draw_ctrl.draw(img=img_linear, pos=pos)
    img = tf.oetf(img_linear, tf.SRGB)

    fname = f"./figure/inter_observer_error_display-{d_idx:02d}.png"
    print(fname)
    write_image(img, fname, 'uint8')


def plot_inter_observer_error(large_xyz_1931):
    """
    Parameters
    ----------
    large_xyz_1931 : ndarray
        XYZ value. CIE1931 observer see the display adjusted for each CMFS.
        Shape is (num_of_display, num_of_cmfs, num_of_patch, 3)
    """

    # debug_save_white_patch(large_xyz=large_xyz_1931)
    # debug_save_color_checker(large_xyz=large_xyz_1931)
    # debug_save_color_checker_11_patch(large_xyz=large_xyz_1931[:, :, :24])

    delta_eab = calc_delta_eab_inter_observer_error(
        large_xyz_1931=large_xyz_1931)
    dummy_plot_for_display_gamut_legend_inter_error()
    max_err = np.max(delta_eab)
    print(max_err)
    cc_rgb_srgb = get_color_checker_srgb_val()
    num_of_display = large_xyz_1931.shape[0]
    # num_of_cmfs = large_xyz_1931.shape[1]
    num_of_patch = large_xyz_1931.shape[2] - 1

    # for d_idx in range(num_of_display):
    #     for p_idx in range(num_of_patch):
    #         plot_inter_error_single_patch(
    #             delta_xy=delta_eab[d_idx, :, p_idx],
    #             patch_color=cc_rgb_srgb[p_idx], y_max=max_err,
    #             d_idx=d_idx, p_idx=p_idx)

    for d_idx in range(num_of_display):
        draw_24_cc_patch_inter_observer_error(d_idx=d_idx)


def plot_inter_observer_error_ab_plane(large_xyz_1931):
    """
    Parameters
    ----------
    large_xyz_1931 : ndarray
        XYZ value. CIE1931 observer see the display adjusted for each CMFS.
        Shape is (num_of_display, num_of_cmfs, num_of_patch, 3)

    Returns
    -------
    ndarray
        delta Eab (CIE1931 XYZ - Cat 01~10 CMFS XYZ)
    """
    color_list = np.array([
        pu.RED, pu.YELLOW, pu.GREEN, pu.BLUE, pu.SKY,
        pu.PINK, pu.ORANGE, pu.MAJENTA, pu.BROWN, pu.GRAY50])
    lab_1976 = XYZ_to_Lab(large_xyz_1931)
    print(lab_1976[:, :10, 18])
    aa = lab_1976[:, :10, 18, 1]
    bb = lab_1976[:, :10, 18, 2]
    num_of_display = large_xyz_1931.shape[0]
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 10),
        bg_color=(0.96, 0.96, 0.96),
        graph_title="Title",
        graph_title_size=None,
        xlabel="a*",
        ylabel="b*",
        axis_label_size=None,
        legend_size=17,
        xlim=[-17, 17],
        ylim=[-17, 17],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    marker_list = ["s", "D", "o"]
    label_list = ["BT.709", "DCI-P3", "BT.2020"]
    marker_size_list = [230, 200, 250]
    for d_idx in range(num_of_display):
        marker = marker_list[d_idx]
        marker_size = marker_size_list[d_idx]
        label = label_list[d_idx]
        ax1.scatter(
            aa[d_idx].flatten(), bb[d_idx].flatten(),
            marker=marker, s=marker_size, c=color_list,
            edgecolors='k', linewidths=1, label=label)
    fname = "./debug/ab_plane_cie1931.png"
    pu.show_and_save(
        fig=fig, legend_loc='upper right', save_fname=fname)


def create_cc_sds_under_d65_illuminant():
    spectral_shape = SPECTRAL_SHAPE_FOR_10_CATEGORY_CMFS
    illuminant = SDS_ILLUMINANTS['D65']
    color_checker_sds_under_d65 = prepaere_color_checker_sr_data()
    color_checker_sds_under_d65 = trim_and_iterpolate(
        color_checker_sds_under_d65, spectral_shape)
    illuminant_intp = trim_and_iterpolate(illuminant, spectral_shape)

    domain = color_checker_sds_under_d65.domain
    cc_sds = color_checker_sds_under_d65.values
    d65_sds = illuminant_intp.values.reshape(-1, 1)

    cc_d65_sds = cc_sds * d65_sds

    color_checker_signals = MultiSignals(
        data=cc_d65_sds, domain=domain)
    color_checker_sds_under_d65 = MultiSpectralDistributions(
        data=color_checker_signals)

    return color_checker_sds_under_d65


def create_cc_display_metamerism():
    """
    Returns
    -------
    A list of MultiSpectralDistributions
        Shape is (num_of_display)
    """
    cmfs_list = [CIE1931_CMFS]

    # calc reference XYZ value using D65 illuminant
    large_xyz_il_d65 = calc_cc_plus_d65_xyz_for_each_cmfs_D65_illuminant(
        cmfs_list=cmfs_list)

    # calc RGB tristimulus value for each display to color match on each CMFS
    display_sd_list = create_709_p3_2020_display_sd()
    xyz_to_rgb_mtx = calc_xyz_to_rgb_matrix_each_display_sd_each_cmfs(
        display_sd_list=display_sd_list, cmfs_list=cmfs_list)
    large_y_n = calc_display_Yn_for_each_cmfs(
        display_sd_list=display_sd_list, cmfs_list=cmfs_list)
    rgb = calc_tristimulus_value_for_each_sd_patch_cmfs(
        large_xyz=large_xyz_il_d65, xyz_to_rgb_mtx=xyz_to_rgb_mtx,
        rgb_nomalize_val=large_y_n)

    # apply RGB tristimulus value for each display
    modified_sd_list = create_modified_display_sd_based_on_rgb_gain(
        display_sd_list=display_sd_list, rgb_list=rgb)

    domain = modified_sd_list[0][0][0].domain
    num_of_display = len(display_sd_list)
    num_of_patch = 24

    output_sds = []
    for d_idx in range(num_of_display):
        sd_list = []
        for p_idx in range(num_of_patch):
            sd_value = modified_sd_list[d_idx][0][p_idx].values
            sd_list.append(sd_value)
        color_checker_signals = MultiSignals(
            data=tstack(sd_list), domain=domain)
        color_checker_sds = MultiSpectralDistributions(
            data=color_checker_signals)
        output_sds.append(color_checker_sds)

    return output_sds


def create_cc_spectrum_with_metamerism_each_patch_name(
        d_idx, p_idx):
    out_fname = f"./debug/metamerism_sds_patch_{d_idx:02d}-{p_idx:02}.png"
    return out_fname


def plot_color_checker_spectrum_with_metamerism_each_patch(
        domain, value, y_max, patch_color, d_idx, p_idx,
        wl_intp, line_color_intp):
    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(2, 2),
        # figsize=(10, 10),
        bg_color=patch_color,
        graph_title=None,
        graph_title_size=None,
        xlabel=None,
        ylabel=None,
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=[0, y_max*1.03],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    x_intp = wl_intp
    y_intp = np.interp(x_intp, domain, value)
    ax1.plot(domain, value, lw=7, color=pu.GRAY90)
    ax1.scatter(x_intp, y_intp, s=10, c=line_color_intp, zorder=50)
    fname = create_cc_spectrum_with_metamerism_each_patch_name(
        d_idx=d_idx, p_idx=p_idx)
    print(fname)
    pu.show_and_save(
        fig=fig, save_fname=fname, only_graph_area=True)


def get_cc_sd_each_display_single_patch_size():
    fname = create_cc_spectrum_with_metamerism_each_patch_name(0, 0)
    img = read_image(fname)
    width = img.shape[1]
    height = img.shape[0]

    return width, height


def draw_color_checker_sd_each_display(d_idx):
    h_num = 6
    v_num = 4
    margin_rate = 0.12
    patch_width, patch_height = get_cc_sd_each_display_single_patch_size()
    margin_h = patch_width * margin_rate
    margin_v = patch_height * margin_rate
    bg_width = int(
        (patch_width * h_num) + margin_h * (h_num + 1))
    bg_height = int(
        (patch_height * v_num) + margin_v * (v_num + 1))

    gc = GridCoordinate(
        bg_width=bg_width, bg_height=bg_height,
        fg_width=patch_width, fg_height=patch_height,
        h_num=h_num, v_num=v_num, remove_tblr_margin=False)
    pos_list = gc.get_st_pos_list()
    img = np.zeros((bg_height, bg_width, 3))
    for v_idx in range(v_num):
        for h_idx in range(h_num):
            p_idx = v_idx * h_num + h_idx
            patch_fname = create_cc_spectrum_with_metamerism_each_patch_name(
                d_idx=d_idx, p_idx=p_idx)
            patch_img = read_image(patch_fname)[..., :3]
            tpg.merge(img, patch_img, pos_list[h_idx][v_idx])

    # draw categorical observer info
    info_str_list = [
        "Illuminant D65",
        "BT.709 Display", "DCI-P3 Display", "BT.2020 Display"]
    font_color = tf.eotf(np.array([0.96, 0.96, 0.96]), tf.SRGB)
    text_draw_ctrl_dummy = fc2.TextDrawControl(
        text=info_str_list[1], font_color=font_color,
        font_size=20, font_path=fc2.NOTO_SANS_CJKJP_BOLD,
        stroke_width=0, stroke_fill=None)
    _, text_height = text_draw_ctrl_dummy.get_text_width_height()

    text_draw_ctrl = fc2.TextDrawControl(
        text=info_str_list[d_idx], font_color=font_color,
        font_size=20, font_path=fc2.NOTO_SANS_CJKJP_BOLD,
        stroke_width=0, stroke_fill=None)
    text_draw_img_height = int(text_height * 1.2)
    text_draw_img = np.zeros((text_draw_img_height, bg_width, 3))

    # calc position
    pos_h = int(margin_h)
    pos_v = 0
    pos = (pos_h, pos_v)
    # draw text
    text_draw_ctrl.draw(img=text_draw_img, pos=pos)
    text_draw_img = tf.oetf(text_draw_img, tf.SRGB)

    fname = f"./figure/metamerism_spectrum-{d_idx:02d}.png"
    print(fname)
    img = np.vstack([img, text_draw_img])
    write_image(img, fname, 'uint8')


def plot_color_checker_spectrum_with_metamerism():
    cc_sds_under_d65 = create_cc_sds_under_d65_illuminant()
    cc_sds_each_display = create_cc_display_metamerism()
    cc_sds_list = [
        cc_sds_under_d65, cc_sds_each_display[0],
        cc_sds_each_display[1], cc_sds_each_display[2]]

    # cc_rgb_srgb = get_color_checker_srgb_val()
    # wl = cc_sds_under_d65.domain
    # intp_step = 0.1
    # wl_intp = np.arange(wl[0], wl[-1] + intp_step, intp_step)
    # rgb = wavelength_to_color(
    #     wl=wl, chroma_rate=0.8) ** (1/2.4)
    # line_color_intp = np.zeros((len(wl_intp), 3))
    # for idx in range(3):
    #     line_color_intp[..., idx] = np.interp(wl_intp, wl, rgb[..., idx])

    # domain = cc_sds_under_d65.domain
    # for d_idx, cc_sds in enumerate(cc_sds_list):
    #     y_max = np.max(cc_sds.values)
    #     for p_idx in range(24):
    #         value = cc_sds.values[..., p_idx]
    #         plot_color_checker_spectrum_with_metamerism_each_patch(
    #             domain=domain, value=value, y_max=y_max,
    #             patch_color=cc_rgb_srgb[p_idx], d_idx=d_idx, p_idx=p_idx,
    #             wl_intp=wl_intp, line_color_intp=line_color_intp)
    #     #     break
    #     # break

    for d_idx, cc_sds in enumerate(cc_sds_list):
        draw_color_checker_sd_each_display(d_idx=d_idx)
        # break


def debug_func():
    # debug_numpy_mult_check()
    # debug_plot_151_cmfs()
    # debug_plot_10_cmfs()

    # delta_xy_all = calc_intra_observer_error()
    # plot_intra_observer_error(delta_xy_all=delta_xy_all)

    # large_xyz_1931 = calc_inter_observer_error()
    # plot_inter_observer_error(large_xyz_1931=large_xyz_1931)

    # plot_inter_observer_error_ab_plane(
    #     large_xyz_1931=large_xyz_1931)

    plot_color_checker_spectrum_with_metamerism()
    pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug_func()
