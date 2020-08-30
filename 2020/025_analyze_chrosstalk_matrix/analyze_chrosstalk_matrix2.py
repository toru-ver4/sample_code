# -*- coding: utf-8 -*-
"""
Visualize the effect of the Crosstalk Matrix
============================================

1. design the tone mapping parameters to cover 10,000 cd/m2
2. do the simple tone mapping and comare the two color volumes,
   one is the sdr color volume and the other is sdr color volume
   that is converted by tone mapping.
3. do the tone mapping with crosstalk matrix
   and compare the two color volumes.

"""

# import standard libraries
import os
import sys
from multiprocessing import Pool, cpu_count

# import third-party libraries
from colour import xyY_to_XYZ, XYZ_to_xyY, chromatic_adaptation
from colour.models import BT2020_COLOURSPACE
from colour import COLOURCHECKERS
import matplotlib.pyplot as plt
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
from sympy import Matrix, symbols, latex, N

# import my libraries
from color_volume_boundary_data import calc_xyY_boundary_data,\
    GamutBoundaryData
import color_space as cs
from bt2446_method_c import bt2446_method_c_tonemapping,\
    apply_cross_talk_matrix
import transfer_functions as tf
import plot_utility as pu
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def plot_bt2446_method_c_without_crosstalk_matrix(
        gmb_hdr_to_sdr, gmb_sdr, ctm_alpha):
    mesh_xyY_hdr_to_sdr = gmb_hdr_to_sdr.get_outline_mesh_data_as_abL(
        ab_plane_div_num=40, rad_rate=8.0, l_step=1)
    mesh_xyY_sdr = gmb_sdr.get_outline_mesh_data_as_abL(
        ab_plane_div_num=40, rad_rate=8.0, l_step=1)
    if ctm_alpha:
        title = f""
    fig, ax = pu.plot_3d_init(
        figsize=(9, 9),
        title="Title",
        title_font_size=18,
        color_preset='light',
        x_label="X",
        y_label="Y",
        z_label="Z",
        xlim=[0.0, 0.8],
        ylim=[0.0, 0.9],
        zlim=[0.0, 1.1])

    # mesh
    pu.plot_xyY_with_scatter3D(
        ax, mesh_xyY_hdr_to_sdr, ms=1, color="#000000", alpha=0.3)
    pu.plot_xyY_with_scatter3D(ax, mesh_xyY_sdr, ms=3)

    ax.view_init(elev=20, azim=-120)
    fname = "./blog_img/bt2446_wo_CM.png"
    print(fname)
    plt.savefig(
        fname, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close(fig)


def plot_bt2446_method_c_matrix_pyqtgraph(
        gmb_hdr_to_sdr, gmb_sdr):
    mesh_xyY_hdr_to_sdr = gmb_hdr_to_sdr.get_outline_mesh_data_as_abL(
        ab_plane_div_num=40, rad_rate=8.0, l_step=1)
    mesh_xyY_sdr = gmb_sdr.get_outline_mesh_data_as_abL(
        ab_plane_div_num=40, rad_rate=8.0, l_step=1)

    app, w = pu.pyqtgraph_plot_3d_init(
        title="Title",
        distance=1.7,
        center=(cs.D65[0], cs.D65[1], 0.5),
        elevation=30,
        angle=-120)

    pu.plot_xyY_with_gl_GLScatterPlotItem(
        w=w, xyY=mesh_xyY_hdr_to_sdr, size=0.01, color=(0.1, 0.1, 0.1, 0.5))
    pu.plot_xyY_with_gl_GLScatterPlotItem(
        w=w, xyY=mesh_xyY_sdr, size=0.01)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


def plot_bt2446_method_c_with_crosstalk_matrix_seq(
        gmb_hdr_to_sdr, gmb_sdr, idx=0, angle=120, alpha_ctm=0.0):
    mesh_xyY_hdr_to_sdr = gmb_hdr_to_sdr.get_outline_mesh_data_as_abL(
        ab_plane_div_num=40, rad_rate=8.0, l_step=1)
    mesh_xyY_sdr = gmb_sdr.get_outline_mesh_data_as_abL(
        ab_plane_div_num=40, rad_rate=8.0, l_step=1)
    fig, ax = pu.plot_3d_init(
        figsize=(9, 9),
        title=f"xyY Color Volume {angle:.1f}°",
        title_font_size=18,
        face_color=(0.9, 0.9, 0.9),
        plane_color=(0.7, 0.7, 0.7, 1.0),
        text_color=(0.1, 0.1, 0.1),
        grid_color=(0.8, 0.8, 0.8),
        color_preset='light',
        x_label="x",
        y_label="y",
        z_label="Y",
        xlim=[0.0, 0.8],
        ylim=[0.0, 0.9],
        zlim=[0.0, 1.1])

    # mesh
    pu.plot_xyY_with_scatter3D(
        ax, mesh_xyY_hdr_to_sdr, ms=1, color="#111111", alpha=0.3)
    pu.plot_xyY_with_scatter3D(ax, mesh_xyY_sdr, ms=3)

    ax.view_init(elev=20, azim=angle)
    dir_name = "/work/overuse/2020/025_analyze_crosstalk_matrix/img_seq"
    fname = f"{dir_name}/hdt_to_sdr_ctm_{alpha_ctm:.2f}_{idx:04d}.png"
    print(fname)
    plt.savefig(
        fname, bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    plt.close(fig)


def apply_bt2446_method_c_without_crosstalk_matrix(
        y_num=1025, h_num=1024, ctm_alpha=0.0):
    tf_for_hdr_src = tf.ST2084
    tf_for_sdr_src = tf.GAMMA24

    gmb_hdr = calc_xyY_boundary_data(
       color_space_name=cs.BT2020, white=cs.D65, y_num=y_num, h_num=h_num,
       eotf_name=tf_for_hdr_src, overwirte_lut=False)
    rgb_hdr = cs.calc_rgb_from_XYZ(
        XYZ=xyY_to_XYZ(gmb_hdr.get_as_abL()), color_space_name=cs.BT2020)
    rgb_hdr_to_sdr = bt2446_method_c_tonemapping(
        img=rgb_hdr, src_color_space_name=cs.BT2020, tfc=tf.ST2084,
        alpha=ctm_alpha, sigma=0.0, hdr_ref_luminance=203,
        hdr_peak_luminance=1000, k1=0.69, k3=0.74, y_sdr_ip=41.0,
        clip_output=False)
    XYZ_hdr_to_sdr = cs.calc_XYZ_from_rgb(
        rgb=rgb_hdr_to_sdr, color_space_name=cs.BT2020, white=cs.D65)
    gmb_hdr_to_sdr = GamutBoundaryData(
        XYZ_to_xyY(XYZ_hdr_to_sdr), Lab_to_abL_swap=True)

    gmb_sdr = calc_xyY_boundary_data(
        color_space_name=cs.BT2020, white=cs.D65, y_num=y_num, h_num=h_num,
        eotf_name=tf_for_sdr_src, overwirte_lut=False)

    # plot_bt2446_method_c_without_crosstalk_matrix(
    #     gmb_hdr_to_sdr=gmb_hdr_to_sdr, gmb_sdr=gmb_sdr, ctm_alpha=ctm_alpha)
    # plot_bt2446_method_c_matrix_pyqtgraph(gmb_hdr_to_sdr, gmb_sdr)

    args = []
    angle_list = np.arange(360)
    for idx, angle in enumerate(angle_list):
        d = dict(
            gmb_hdr_to_sdr=gmb_hdr_to_sdr, gmb_sdr=gmb_sdr,
            idx=idx, angle=angle, alpha_ctm=ctm_alpha)
        # plot_bt2446_method_c_with_crosstalk_matrix_seq()
        args.append(d)
    with Pool(cpu_count()) as pool:
        pool.map(th_plot_bt2446_method_c_with_crosstalk_matrix_seq, args)


def th_plot_bt2446_method_c_with_crosstalk_matrix_seq(args):
    plot_bt2446_method_c_with_crosstalk_matrix_seq(**args)


def calc_crosstalk_matrix_in_xyz(alpha=0.5):
    alpha = symbols('alpha')
    ctm_rgb = Matrix([
        [1 - 2 * alpha, alpha, alpha],
        [alpha, 1 - 2 * alpha, alpha],
        [alpha, alpha, 1 - 2 * alpha]])
    rgb_to_xyz_mtx = Matrix(BT2020_COLOURSPACE.RGB_to_XYZ_matrix)
    xyz_to_rgb_mtx = Matrix(BT2020_COLOURSPACE.XYZ_to_RGB_matrix)
    print(rgb_to_xyz_mtx)
    print(xyz_to_rgb_mtx)

    ctm_xyz = rgb_to_xyz_mtx * (ctm_rgb * xyz_to_rgb_mtx)

    print(latex(N(ctm_xyz, 4)))


def get_color_checker_xyY_value():
    colour_checker_param = COLOURCHECKERS.get('ColorChecker 2005')
    _name, data, whitepoint = colour_checker_param
    xyY = []
    for key in data.keys():
        xyY.append(data[key])
    xyY_D50 = np.array(xyY)
    XYZ_D50 = xyY_to_XYZ(xyY_D50)
    L_A = 200
    XYZ_D65 = chromatic_adaptation(
        XYZ_D50, XYZ_w=cs.D50_XYZ, XYZ_wr=cs.D65_XYZ, method="CMCCAT2000",
        L_A1=L_A, L_A2=L_A)
    xyY_D65 = XYZ_to_xyY(XYZ_D65)

    return xyY_D65


def plot_xy_plane_displacement_seq(alpha=0.05):
    rate = 480/755.0*2
    xmin = 0.0
    xmax = 0.8
    ymin = 0.0
    ymax = 0.9
    xyY = get_color_checker_xyY_value()
    rgb = cs.calc_rgb_from_XYZ(
        XYZ=xyY_to_XYZ(xyY), color_space_name=cs.BT2020)
    rgb_ctm = apply_cross_talk_matrix(rgb, alpha=alpha)
    xyY_ctm = XYZ_to_xyY(
        cs.calc_XYZ_from_rgb(rgb=rgb_ctm, color_space_name=cs.BT2020))

    # プロット用データ準備
    # ---------------------------------
    cmf_xy = tpg._get_cmfs_xy()

    bt2020_gamut, _ = tpg.get_primaries(name=cs.BT2020)
    bt709_gamut, _ = tpg.get_primaries(name=cs.BT709)
    xlim = (min(0, xmin), max(0.8, xmax))
    ylim = (min(0, ymin), max(0.9, ymax))

    ax1 = pu.plot_1_graph(fontsize=20 * rate,
                          figsize=((xmax - xmin) * 10 * rate,
                                   (ymax - ymin) * 10 * rate),
                          graph_title=f"Chromaticity Diagram, α={alpha:.02f}",
                          graph_title_size=None,
                          xlabel=None, ylabel=None,
                          axis_label_size=None,
                          legend_size=18 * rate,
                          xlim=xlim, ylim=ylim,
                          xtick=[x * 0.1 + xmin for x in
                                 range(int((xlim[1] - xlim[0])/0.1) + 1)],
                          ytick=[x * 0.1 + ymin for x in
                                 range(int((ylim[1] - ylim[0])/0.1) + 1)],
                          xtick_size=17 * rate,
                          ytick_size=17 * rate,
                          linewidth=4 * rate,
                          minor_xtick_num=2,
                          minor_ytick_num=2)
    ax1.set_facecolor("#E0E0E0")
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=3.5*rate, label=None)
    ax1.plot((cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
             '-k', lw=3.5*rate, label=None)
    ax1.plot(bt2020_gamut[:, 0], bt2020_gamut[:, 1],
             c='#404040', label="BT.2020 Gamut", lw=2.75*rate)
    ax1.plot(bt709_gamut[:, 0], bt709_gamut[:, 1], '--',
             c='#404040', label="BT.709 Gamut", lw=1.5*rate)

    # plot color checker
    rgb_mplc = pu.calc_rgb_from_XYZ_for_mpl(
        XYZ=xyY_to_XYZ(xyY), color_space_name=cs.BT709, oetf_str=tf.GAMMA24)
    ax1.scatter(xyY[..., 0], xyY[..., 1], color=rgb_mplc, s=200)
    ax1.scatter(xyY_ctm[..., 0], xyY_ctm[..., 1], color="#404040", s=40)

    # annotation
    diff = ((xyY[..., 0] - xyY_ctm[..., 0]) ** 2
            + (xyY[..., 1] - xyY_ctm[..., 1]) ** 2) ** 0.5
    arrowprops = dict(
        facecolor='#D0D0D0', shrink=0.0, headwidth=8, headlength=8,
        width=1)
    for idx in range(len(xyY[..., 0])):
        if diff[idx] > 0.005:
            st_pos = (xyY[idx, 0], xyY[idx, 1])
            ed_pos = (xyY_ctm[idx, 0], xyY_ctm[idx, 1])
            ax1.annotate(
                "", xy=ed_pos, xytext=st_pos, xycoords='data',
                textcoords='data', ha='left', va='bottom',
                arrowprops=arrowprops)

    plt.legend(loc='upper right')
    plt.savefig(f'./blog_img/xy_a_{alpha:.02f}.png', bbox_inches='tight')
    plt.show()


def plot_xyY_color_volume_displacement_seq(alpha=0.20):
    gmb_sdr = calc_xyY_boundary_data(
       color_space_name=cs.BT2020, white=cs.D65, y_num=1025, h_num=1024,
       eotf_name=None, overwirte_lut=False)
    mesh_xyY_sdr = gmb_sdr.get_outline_mesh_data_as_abL(
        ab_plane_div_num=40, rad_rate=8.0, l_step=2)

    xyY = get_color_checker_xyY_value()
    rgb = cs.calc_rgb_from_XYZ(
        XYZ=xyY_to_XYZ(xyY), color_space_name=cs.BT2020)
    rgb_ctm = apply_cross_talk_matrix(rgb, alpha=alpha)
    xyY_ctm = XYZ_to_xyY(
        cs.calc_XYZ_from_rgb(rgb=rgb_ctm, color_space_name=cs.BT2020))

    angle_list = np.arange(360)
    args = []
    for idx, angle in enumerate(angle_list):
        d = dict(
            a_idx=idx, angle=angle,
            mesh_xyY_sdr=mesh_xyY_sdr, xyY=xyY, xyY_ctm=xyY_ctm, alpha=alpha)
        # plot_xyY_color_volume_displacement(**d)
        args.append(d)
    with Pool(cpu_count()) as pool:
        pool.map(thread_wrapper_plot_xyY_color_volume_displacement, args)


def thread_wrapper_plot_xyY_color_volume_displacement(args):
    plot_xyY_color_volume_displacement(**args)


def plot_xyY_color_volume_displacement(
        a_idx, angle, mesh_xyY_sdr, xyY, xyY_ctm, alpha=0.20):
    fig, ax = pu.plot_3d_init(
        figsize=(9, 9),
        title=f"xyY, alpha={alpha:.02f}, {angle}°",
        title_font_size=18,
        color_preset='light',
        x_label="x",
        y_label="y",
        z_label="Y",
        xlim=[0.0, 0.8],
        ylim=[0.0, 0.9],
        zlim=[0.0, 1.1])
    pu.plot_xyY_with_scatter3D(
        ax, mesh_xyY_sdr, ms=1, color='#404040', alpha=0.1)
    pu.plot_xyY_with_scatter3D(
        ax, xyY, ms=50, color='rgb', alpha=1.0)
    pu.plot_xyY_with_scatter3D(
        ax, xyY_ctm, ms=20, color='#404040', alpha=1.0)

    for idx in range(len(xyY)):
        ax.arrow3D(
            xyY[idx, 0], xyY[idx, 1], xyY[idx, 2],
            xyY_ctm[idx, 0] - xyY[idx, 0],
            xyY_ctm[idx, 1] - xyY[idx, 1],
            xyY_ctm[idx, 2] - xyY[idx, 2],
            mutation_scale=16, facecolor='#D0D0D0',
            arrowstyle="-|>")

    ax.view_init(elev=20, azim=angle)
    out_dir = "/work/overuse/2020/025_analyze_crosstalk_matrix/img_seq"
    fname = f"{out_dir}/xyY_seq_a_{alpha:.02f}_{a_idx:04d}.png"
    print(fname)
    plt.savefig(
        fname, bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    plt.close(fig)


def plot_hdr10_gamut_boundary():
    gmb_hdr = calc_xyY_boundary_data(
        color_space_name=cs.BT2020, white=cs.D65, y_num=1025, h_num=1024,
        eotf_name=None, overwirte_lut=False)
    mesh_xyY = gmb_hdr.get_outline_mesh_data_as_abL(
        ab_plane_div_num=30, rad_rate=8, l_step=2)
    mesh_xyY[..., 2] = mesh_xyY[..., 2] * 100  # change to the HDR range.

    fig, ax = pu.plot_3d_init(
        figsize=(9, 9),
        title="HDR10 Gamut Boundary",
        title_font_size=18,
        color_preset='dark',
        x_label="x",
        y_label="y",
        z_label="Y",
        xlim=[0.0, 0.8],
        ylim=[0.0, 0.9],
        zlim=[0.0, 110])

    # mesh
    pu.plot_xyY_with_scatter3D(
        ax, mesh_xyY, ms=3, color="rgb", alpha=0.3)

    ax.view_init(elev=20, azim=-120)
    fname = "./blog_img/HDR10_GamutBoundary.png"
    print(fname)
    plt.savefig(
        fname, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close(fig)


def plot_hdr10_to_sdr_gamut_boundary():
    tf_for_hdr_src = tf.ST2084
    y_num = 1025
    h_num = 1024
    ctm_alpha = 0.0
    gmb_hdr = calc_xyY_boundary_data(
       color_space_name=cs.BT2020, white=cs.D65, y_num=y_num, h_num=h_num,
       eotf_name=tf_for_hdr_src, overwirte_lut=False)
    rgb_hdr = cs.calc_rgb_from_XYZ(
        XYZ=xyY_to_XYZ(gmb_hdr.get_as_abL()), color_space_name=cs.BT2020)
    rgb_hdr_to_sdr = bt2446_method_c_tonemapping(
       img=rgb_hdr, src_color_space_name=cs.BT2020, tfc=tf.ST2084,
       alpha=ctm_alpha, sigma=0.0, hdr_ref_luminance=203,
       hdr_peak_luminance=1000, k1=0.69, k3=0.74, y_sdr_ip=41.0,
       clip_output=False)
    XYZ_hdr_to_sdr = cs.calc_XYZ_from_rgb(
        rgb=rgb_hdr_to_sdr, color_space_name=cs.BT2020, white=cs.D65)
    gmb_hdr_to_sdr = GamutBoundaryData(
        XYZ_to_xyY(XYZ_hdr_to_sdr), Lab_to_abL_swap=True)
    mesh_xyY = gmb_hdr_to_sdr.get_outline_mesh_data_as_abL(
        ab_plane_div_num=40, rad_rate=16, l_step=2)

    fig, ax = pu.plot_3d_init(
        figsize=(9, 9),
        title="Gamut Boundary after HDR10 to SDR convertion",
        title_font_size=18,
        color_preset='dark',
        x_label="x",
        y_label="y",
        z_label="Y",
        xlim=[0.0, 0.8],
        ylim=[0.0, 0.9],
        zlim=[0.0, 1.1])

    # mesh
    pu.plot_xyY_with_scatter3D(
        ax, mesh_xyY, ms=3, color="rgb", alpha=0.3)

    ax.view_init(elev=20, azim=-120)
    fname = "./blog_img/HDR10_to_SDR_GamutBoundary.png"
    print(fname)
    plt.savefig(
        fname, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close(fig)


def apply_ctm_with_tp_image(luminance=100):
    alpha_list = [0.0, 0.05, 0.10, 0.20]

    for alpha in alpha_list:
        img_sdr = tpg.img_read_as_float(
            "./blog_img/bt2020_tp_src_1920x1080.png")
        img_sdr = tf.eotf(img_sdr, tf.GAMMA24) * luminance / 10000

        rgb_hdr_to_sdr = bt2446_method_c_tonemapping(
            img=img_sdr, src_color_space_name=cs.BT2020, tfc=tf.ST2084,
            alpha=alpha, sigma=0.0, hdr_ref_luminance=203,
            hdr_peak_luminance=1000, k1=0.69, k3=0.74, y_sdr_ip=41.0,
            clip_output=False)

        sdr_rgb_non_linear = tf.oetf(
            np.clip(rgb_hdr_to_sdr, 0.0, 1.0), tf.GAMMA24)
        tpg.img_wirte_float_as_16bit_int(
            f"./blog_img/tp_ctm_{alpha:.02f}.png", sdr_rgb_non_linear)


def make_blog_image():
    # color checker on xy plane
    # plot_xy_plane_displacement_seq(alpha=0.10)
    # plot_xy_plane_displacement_seq(alpha=0.20)

    # color checker in xyY space
    # plot_xyY_color_volume_displacement_seq(alpha=0.10)
    # plot_xyY_color_volume_displacement_seq(alpha=0.20)

    # color volume in xyY space
    # alpha_list = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    # for alpha in alpha_list:
    #     apply_bt2446_method_c_without_crosstalk_matrix(ctm_alpha=alpha)

    # gamut boundary sample
    plot_hdr10_gamut_boundary()
    plot_hdr10_to_sdr_gamut_boundary()

    # apply bt2446 with ctm for colorchecker.
    # apply_ctm_with_tp_image(luminance=400)


def main_func():
    # apply_bt2446_method_c_without_crosstalk_matrix(ctm_alpha=0.05)
    # calc_crosstalk_matrix_in_xyz()
    make_blog_image()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
