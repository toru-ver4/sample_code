# -*- coding: utf-8 -*-
"""
spectrum
"""

# import standard libraries
import os

# import third party libraries
import numpy as np
import colour_datasets
from colour.plotting import plot_multi_cmfs
from colour import LinearInterpolator, SpectralShape, XYZ_to_RGB
from colour.models import RGB_COLOURSPACE_BT709
from sympy import Symbol, expand, diff

# import my libraries
import plot_utility as pu
import spectrum_calculation as scl
import transfer_functions as tf
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


VALID_WAVELENGTH_ST = 360
VALID_WAVELENGTH_ED = 830
# VALID_WAVELENGTH_ST = 380
# VALID_WAVELENGTH_ED = 730
VALID_SHAPE = SpectralShape(
    VALID_WAVELENGTH_ST, VALID_WAVELENGTH_ED, 1)


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


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # load_camera_spectral_sensitivity_database()
    # plot_camera_gamut()
    debug_least_square_method()
