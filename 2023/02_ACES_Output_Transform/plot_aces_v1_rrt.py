# -*- coding: utf-8 -*-
"""
debug code
==========

"""

# import standard libraries
import os
import subprocess

# import third-party libraries
import numpy as np
from colour.utilities import tsplit
from colour import XYZ_to_xyY
from colour.io import write_image, read_image

# import my libraries
import plot_utility as pu
import test_pattern_generator2 as tpg
import transfer_functions as tf
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2023 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def debug_plot_rgb_envelope():
    rgb = tpg.create_rgb_envelop_data(grid_num=11)
    print(rgb.shape)
    fig, ax = pu.plot_3d_init(
        figsize=(9, 9),
        title="Title",
        title_font_size=18,
        face_color=(0.1, 0.1, 0.1),
        plane_color=(0.2, 0.2, 0.2, 1.0),
        text_color=(0.5, 0.5, 0.5),
        grid_color=(0.3, 0.3, 0.3),
        x_label="R",
        y_label="G",
        z_label="B")

    # mesh
    r, g, b = tsplit(rgb)
    ax.scatter3D(r, g, b, color=rgb, s=100)

    ax.view_init(elev=20, azim=-120)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', save_fname=None, show=True)


def create_aces_ap0_envelope_fname(grid_num):
    return f"./img/aces_ap0_grid_{grid_num}.exr"


def create_aces_ap0_to_bt709_envelope_fname(grid_num):
    return f"./img/aces_ap0_grid_{grid_num}_to_bt709.exr"


def create_aces_ap0_envelope(grid_num):
    rgb = tpg.create_rgb_envelop_data(grid_num=grid_num)
    rgb_linear = tf.eotf(rgb, tf.ST2084) * 100
    rgb_linear = rgb_linear.reshape(1, -1, 3)

    fname = create_aces_ap0_envelope_fname(grid_num=grid_num)
    write_image(rgb_linear, path=fname, bit_depth='float32')


def convert_aces_ap0_to_bt709_using_aces_ot(grid_num):
    in_fname = create_aces_ap0_envelope_fname(grid_num)
    out_fname = create_aces_ap0_to_bt709_envelope_fname(grid_num)
    ocio_path = "/work/src/2023/02_ACES_Output_Transform/config/"
    ocio_path += "reference-config-v1.0.0_aces-v1.3_ocio-v2.1.ocio"
    cmd = [
        'ocioconvert', '--view', in_fname, "ACES2065-1",
        out_fname, "Rec.1886 Rec.709 - Display",
        "ACES 1.0 - SDR Video"]
    print(" ".join(cmd))
    my_env = os.environ.copy()
    my_env["OCIO"] = ocio_path
    subprocess.run(cmd, env=my_env)


def plot_aces_v1_rrt_core(grid_num):
    # rgb value
    rgb = tpg.create_rgb_envelop_data(grid_num=grid_num)

    # ACES AP0
    img_fname_ap0 = create_aces_ap0_envelope_fname(grid_num)
    rgb_ap0_linear = read_image(img_fname_ap0)
    large_xyz_ap0 = cs.rgb_to_large_xyz(
        rgb=rgb_ap0_linear, color_space_name=cs.ACES_AP0,
        rgb_white=cs.D60_ACES, xyz_white=cs.D60_ACES)
    xyY_ap0 = XYZ_to_xyY(large_xyz_ap0)
    x_ap0, y_ap0, Y_ap0 = tsplit(xyY_ap0)

    # converted BT.709
    img_fname_bt709 = create_aces_ap0_to_bt709_envelope_fname(grid_num)
    rgb_bt709 = read_image(img_fname_bt709)
    rgb_bt709_linear = tf.eotf(rgb_bt709, tf.GAMMA24)
    large_xyz_bt709 = cs.rgb_to_large_xyz(
        rgb=rgb_bt709_linear, color_space_name=cs.BT709)
    xyY_bt709 = XYZ_to_xyY(large_xyz_bt709)

    fig, ax = pu.plot_3d_init(
        figsize=(9, 9),
        title="ACES AP0 to BT709 using ACES_OT",
        title_font_size=18,
        face_color=(0.1, 0.1, 0.1),
        plane_color=(0.2, 0.2, 0.2, 1.0),
        text_color=(0.5, 0.5, 0.5),
        grid_color=(0.3, 0.3, 0.3),
        x_label="x",
        y_label="y",
        z_label="Y")

    # mesh
    x_bt709, y_bt709, Y_bt709 = tsplit(xyY_bt709)
    # color_rgb = np.clip(rgb_bt709.reshape(-1, 3), 0.0, 1.0)
    ax.scatter3D(x_bt709, y_bt709, Y_bt709, c=pu.GRAY50, s=10)
    ax.scatter3D(x_ap0, y_ap0, Y_ap0, color=rgb, s=20)

    ax.view_init(elev=20, azim=-120)
    pu.show_and_save(
        fig=fig, legend_loc='upper left', save_fname=None, show=True)


def plot_aces_v1_rrt():
    grid_num = 65
    create_aces_ap0_envelope(grid_num=grid_num)
    convert_aces_ap0_to_bt709_using_aces_ot(grid_num=grid_num)
    plot_aces_v1_rrt_core(grid_num=grid_num)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # debug_plot_rgb_envelope()
    plot_aces_v1_rrt()
