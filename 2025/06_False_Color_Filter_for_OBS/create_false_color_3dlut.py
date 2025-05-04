# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour import write_LUT, LUT3D

# my libraries
import color_space as cs
import transfer_functions as tf
from apply_false_color import (
    convert_to_bt2020_linear,
    apply_false_color_filter_for_Y,
    apply_false_color_filter_for_maxRGB,
    palette_0,
    palette_1,
    palette_2,
    palette_3,
    palette_4,
    palette_5,
    palette_6,
)


def convert_to_source_cs_linear(linear_bt2020, source_cs_name, source_tf_name):
    """
    Returns
    -------
    ndarray
        The output image in souce color space with oetf.
        Nominal white is represented as (1, 1, 1) in the Rec.2020-Linear color space.
        Peak white is represented as (100, 100, 100) in the Rec.2020-Linear color space.
    """

    if source_cs_name == cs.BT2020:
        linear_source_cs = linear_bt2020
    else:
        large_xyz = cs.rgb_to_large_xyz(linear_bt2020, cs.BT2020)
        linear_source_cs = cs.large_xyz_to_rgb(large_xyz, source_cs_name)

    rgb = tf.oetf_from_luminance(linear_source_cs * 100.0, source_tf_name)

    return rgb


def convert_to_sRGB(linear_bt2020):
    """
    Returns
    -------
    ndarray
        The output image in sRGB.
    """
    large_xyz = cs.rgb_to_large_xyz(linear_bt2020, cs.BT2020)
    linear_bt709 = cs.large_xyz_to_rgb(large_xyz, cs.BT709)

    srgb_img = tf.oetf(np.clip(linear_bt709, 0.0, 1.0), tf.SRGB)

    return srgb_img


def create_false_color_3dlut(num_of_grid=65, color_space_name=cs.BT2020):
    """
    Create a 3D LUT for false color filter.

    Parameters
    ----------
    num_of_grid : int, optional
        The number of grid points in the LUT. Default is 65.
    color_space_name : str, optional
        The name of the target color space. Default is 'BT.2020'.
    """
    palette_list = np.array(
        [palette_0, palette_1, palette_2, palette_3, palette_4, palette_5, palette_6]
    )
    if color_space_name == cs.BT2020:
        cs_suffix = "Rec.2100-PQ"
    elif color_space_name == cs.P3_D65:
        cs_suffix = "P3D65-PQ"
    else:
        raise ValueError("Invalid color space name.")

    rgb_st2084 = LUT3D.linear_table(num_of_grid)

    img_rec2100_linear = convert_to_bt2020_linear(
        img=rgb_st2084, source_cs_name=color_space_name, source_tf_name=tf.ST2084
    )

    false_color_img_y_base_bt2020_linear = apply_false_color_filter_for_Y(
        img=img_rec2100_linear, palette_list=palette_list
    )
    false_color_img_maxRGB_base_bt2020_linear = apply_false_color_filter_for_maxRGB(
        img=img_rec2100_linear, palette_list=palette_list
    )

    false_color_img_y_base = convert_to_source_cs_linear(
        linear_bt2020=false_color_img_y_base_bt2020_linear,
        source_cs_name=color_space_name,
        source_tf_name=tf.ST2084
    )
    false_color_img_maxRGB_base = convert_to_source_cs_linear(
        linear_bt2020=false_color_img_maxRGB_base_bt2020_linear,
        source_cs_name=color_space_name,
        source_tf_name=tf.ST2084
    )

    false_color_img_y_base_sRGB = convert_to_sRGB(
        linear_bt2020=false_color_img_y_base_bt2020_linear,
    )
    false_color_img_maxRGB_base_sRGB = convert_to_sRGB(
        linear_bt2020=false_color_img_maxRGB_base_bt2020_linear,
    )

    lut3d_y = LUT3D(
        table=false_color_img_y_base,
        name="7JzCzhz False Color (Y-based)"
    )
    lut3d_maxRGB = LUT3D(
        table=false_color_img_maxRGB_base,
        name="7JzCzhz False Color (maxRGB-based)"
    )

    lut3d_y_sRGB = LUT3D(
        table=false_color_img_y_base_sRGB,
        name="7JzCzhz False Color (Y-based) conv to sRGB"
    )
    lut3d_maxRGB_sRGB = LUT3D(
        table=false_color_img_maxRGB_base_sRGB,
        name="7JzCzhz False Color (maxRGB-based) conv to sRGB"
    )

    grid_str = f"{num_of_grid}x{num_of_grid}x{num_of_grid}"
    out_fname_y = f"./lut/7JzCzhz_false_color_{cs_suffix}_{grid_str}_Y.cube"
    out_fname_maxRGB = f"./lut/7JzCzhz_false_color_{cs_suffix}_{grid_str}_maxRGB.cube"
    out_fname_y_sRGB = f"./lut/7JzCzhz_false_color_{cs_suffix}_{grid_str}_Y_sRGB.cube"
    out_fname_maxRGB_sRGB = f"./lut/7JzCzhz_false_color_{cs_suffix}_{grid_str}_maxRGB_sRGB.cube"
    print(out_fname_y)
    print(out_fname_maxRGB)

    write_LUT(lut3d_y, out_fname_y)
    write_LUT(lut3d_y_sRGB, out_fname_y_sRGB)
    write_LUT(lut3d_maxRGB, out_fname_maxRGB)
    write_LUT(lut3d_maxRGB_sRGB, out_fname_maxRGB_sRGB)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    create_false_color_3dlut(num_of_grid=65, color_space_name=cs.BT2020)
    create_false_color_3dlut(num_of_grid=65, color_space_name=cs.P3_D65)
