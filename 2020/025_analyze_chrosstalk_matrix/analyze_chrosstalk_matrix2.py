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
from colour import xyY_to_XYZ, XYZ_to_xyY
import matplotlib.pyplot as plt
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui

# import my libraries
from color_volume_boundary_data import calc_xyY_boundary_data,\
    GamutBoundaryData
import color_space as cs
from bt2446_method_c import bt2446_method_c_tonemapping
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
        gmb_hdr_to_sdr, gmb_sdr):
    mesh_xyY_hdr_to_sdr = gmb_hdr_to_sdr.get_outline_mesh_data_as_abL(
        ab_plane_div_num=40, rad_rate=8.0, l_step=1)
    mesh_xyY_sdr = gmb_sdr.get_outline_mesh_data_as_abL(
        ab_plane_div_num=40, rad_rate=8.0, l_step=1)
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
        ax, mesh_xyY_hdr_to_sdr, ms=1, color="#000000", alpha=1.0)
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
        w=w, xyY=mesh_xyY_hdr_to_sdr, size=0.001)
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
        # face_color=(0.1, 0.1, 0.1),
        # plane_color=(0.2, 0.2, 0.2, 1.0),
        # text_color=(0.8, 0.8, 0.8),
        # grid_color=(0.5, 0.5, 0.5),
        x_label="X",
        y_label="Y",
        z_label="Z",
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

    # ref_rgb = cs.calc_rgb_from_XYZ(
    #     XYZ=xyY_to_XYZ(gmb_sdr.get_as_abL()), color_space_name=cs.BT709)

    # before_img = tf.oetf(np.clip(rgb_hdr, 0.0, 1.0), tf_for_hdr_src)
    # print(np.max(rgb_hdr_to_sdr), np.min(rgb_hdr_to_sdr))
    # after_img = tf.oetf(rgb_hdr_to_sdr, tf_for_sdr_src)
    # ref_img = tf.oetf(ref_rgb, tf_for_sdr_src)

    # img = np.hstack([before_img, after_img, ref_img])
    # tpg.img_wirte_float_as_16bit_int("./debug.png", img[::-1])

    plot_bt2446_method_c_without_crosstalk_matrix(
        gmb_hdr_to_sdr=gmb_hdr_to_sdr, gmb_sdr=gmb_sdr)

    # args = []
    # angle_list = np.arange(360)
    # for idx, angle in enumerate(angle_list):
    #     d = dict(
    #         gmb_hdr_to_sdr=gmb_hdr_to_sdr, gmb_sdr=gmb_sdr,
    #         idx=idx, angle=angle, alpha_ctm=ctm_alpha)
    #     # plot_bt2446_method_c_with_crosstalk_matrix_seq()
    #     args.append(d)
    # with Pool(cpu_count()) as pool:
    #     pool.map(th_plot_bt2446_method_c_with_crosstalk_matrix_seq, args)


def th_plot_bt2446_method_c_with_crosstalk_matrix_seq(args):
    plot_bt2446_method_c_with_crosstalk_matrix_seq(**args)


def main_func():
    apply_bt2446_method_c_without_crosstalk_matrix(ctm_alpha=0.0)
    apply_bt2446_method_c_without_crosstalk_matrix(ctm_alpha=0.05)
    apply_bt2446_method_c_without_crosstalk_matrix(ctm_alpha=0.10)
    apply_bt2446_method_c_without_crosstalk_matrix(ctm_alpha=0.20)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
