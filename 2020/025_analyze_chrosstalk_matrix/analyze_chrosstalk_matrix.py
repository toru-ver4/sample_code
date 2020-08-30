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

# import third-party libraries
from colour import xyY_to_XYZ, XYZ_to_xyY
import matplotlib.pyplot as plt
import numpy as np

# import my libraries
from color_volume_boundary_data import calc_xyY_boundary_data,\
    GamutBoundaryData
import color_space as cs
from bt2446_method_c import bt2446_method_c_tonemapping
import transfer_functions as tf
import plot_utility as pu

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def plot_bt2446_method_c_without_crosstalk_matrix(
        gmb_hdr_to_sdr):
    reduced_xyY = gmb_hdr_to_sdr.get_reduced_data_as_abL()
    mesh_xyY = gmb_hdr_to_sdr.get_outline_mesh_data_as_abL(
        ab_plane_div_num=40, rad_rate=8.0, l_step=1)

    fig, ax = pu.plot_3d_init(
        figsize=(9, 9),
        title="Title",
        title_font_size=18,
        face_color=(0.1, 0.1, 0.1),
        plane_color=(0.2, 0.2, 0.2, 1.0),
        text_color=(0.5, 0.5, 0.5),
        grid_color=(0.3, 0.3, 0.3),
        x_label="X",
        y_label="Y",
        z_label="Z",
        xlim=[0.0, 0.8],
        ylim=[0.0, 0.9],
        zlim=[0.0, 1.1])

    # mesh
    pu.plot_xyY_with_scatter3D(ax, mesh_xyY, ms=2)

    # outline
    # pu.plot_xyY_with_scatter3D(ax, reduced_xyY, ms=5)

    ax.view_init(elev=20, azim=-120)
    fname = "./blog_img/hdr_to_sdr.png"
    print(fname)
    plt.savefig(
        fname, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close(fig)


def apply_bt2446_method_c_without_crosstalk_matrix():
    gmb_hdr = calc_xyY_boundary_data(
       color_space_name=cs.BT2020, white=cs.D65, y_num=1025, h_num=1024,
       eotf_name=tf.ST2084, overwirte_lut=False)
    rgb_hdr = cs.calc_rgb_from_XYZ(xyY_to_XYZ(gmb_hdr.get_as_abL()))
    rgb_hdr_to_sdr = bt2446_method_c_tonemapping(
       img=rgb_hdr, src_color_space_name=cs.BT2020, tfc=tf.ST2084,
       alpha=0.0, sigma=0.0, hdr_ref_luminance=203, hdr_peak_luminance=10000,
       k1=0.69, k3=0.78, y_sdr_ip=40.7)
    print(np.max(rgb_hdr_to_sdr))

     gmb_hdr = calc_xyY_boundary_data(
       color_space_name=cs.BT2020, white=cs.D65, y_num=1025, h_num=1024,
       eotf_name=tf.ST2084, overwirte_lut=False)


    XYZ_hdr_to_sdr = cs.calc_XYZ_from_rgb(
        rgb=rgb_hdr_to_sdr, color_space_name=cs.BT2020, white=cs.D65)
    gmb_hdr_to_sdr = GamutBoundaryData(
        XYZ_to_xyY(XYZ_hdr_to_sdr), Lab_to_abL_swap=True)
    plot_bt2446_method_c_without_crosstalk_matrix(
        gmb_hdr_to_sdr=gmb_hdr_to_sdr)


def main_func():
    apply_bt2446_method_c_without_crosstalk_matrix()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
