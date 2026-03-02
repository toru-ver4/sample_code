# import standard libraries
import os

# import third-party libraries
import numpy as np

# my libraries
import color_space as cs
import test_pattern_generator2 as tpg
import transfer_functions as tf
from ty_utility import add_suffix_to_filename
from flase_color_palette import(
    palette_0, palette_1, palette_2, palette_3, palette_4, palette_5, palette_6
)


def convert_to_bt2020_linear(img, source_cs_name, source_tf_name):
    """
    Returns
    -------
    ndarray
        The output image in BT.2020-Linear color space.
        Nominal white is represented as (1, 1, 1) in the Rec.2020-Linear color space.
        Peak white is represented as (100, 100, 100) in the Rec.2020-Linear color space.
    """
    linear = tf.eotf_to_luminance(img, source_tf_name) / 100.0

    if source_cs_name == cs.BT2020:
        linear_bt2020 = linear
    else:
        large_xyz = cs.rgb_to_large_xyz(linear, source_cs_name)
        linear_bt2020 = cs.large_xyz_to_rgb(large_xyz, cs.BT2020)

    return linear_bt2020


def convert_bt2020_linear_to_sRGB(img_linear_bt2020):
    """
    Returns
    -------
    ndarray
        The output image is sRGB.
    """
    large_xyz = cs.rgb_to_large_xyz(img_linear_bt2020, cs.BT2020)
    linear_bt709 = cs.large_xyz_to_rgb(large_xyz, cs.BT709)

    srgb_img = tf.oetf(np.clip(linear_bt709, 0.0, 1.0), tf.SRGB)

    return srgb_img


def apply_hdr_false_color_filter(y_like, palette_list, out_img):
    threshold_list = [1.0, 2.0, 4.0, 6.0, 10.0, 20.0, 40.0, 101.0]
    for p_idx in range(len(threshold_list) - 1):
        lower_bound = threshold_list[p_idx]
        upper_bound = threshold_list[p_idx + 1]
        palette = palette_list[p_idx]
        x_org = np.linspace(0, 1, len(palette))

        # Find indices where the luminance value falls in the current threshold range
        idx = (y_like > lower_bound) & (y_like <= upper_bound)
        x = (y_like[idx] - lower_bound) / (upper_bound - lower_bound)

        # Interpolate each channel from the palette
        rr = np.interp(x, x_org, palette[:, 0])
        gg = np.interp(x, x_org, palette[:, 1])
        bb = np.interp(x, x_org, palette[:, 2])
        # Combine the interpolated channels into an RGB value
        rgb = np.column_stack((rr, gg, bb))
        out_img[idx] = rgb

    return out_img


def apply_sdr_false_color_filter(y_like, out_img):
    mono_idx = y_like < 1.0
    mono_img = np.dstack((y_like[mono_idx], y_like[mono_idx], y_like[mono_idx])).reshape(-1, 3)
    out_img[mono_idx] = mono_img ** (1 / 1.1)


def apply_false_color_filter_for_Y(img, palette_list):
    """
    Parameters
    ----------
    img : ndarray
        The input image in BT.2020-Linear color space.
        Nominal white is represented as (1, 1, 1) in the Rec.2020-Linear color space.
        Peak white is represented as (100, 100, 100) in the Rec.2020-Linear color space.
    palette_list : ndarray
        The false color palette to be applied to the image.
        The palette should be in the same color space as the input image.

    Returens
    --------
    ndarray
    """
    out_img = np.zeros_like(img)

    y_like = 0.262700212 * img[..., 0] + 0.677998072 * img[..., 1] + 0.0593017165 * img[..., 2]
    apply_sdr_false_color_filter(y_like=y_like, out_img=out_img)
    apply_hdr_false_color_filter(y_like=y_like, out_img=out_img, palette_list=palette_list)

    return out_img


def apply_false_color_filter_for_maxRGB(img, palette_list):
    """
    Parameters
    ----------
    img : ndarray
        The input image in BT.2020-Linear color space.
        Nominal white is represented as (1, 1, 1) in the Rec.2020-Linear color space.
        Peak white is represented as (100, 100, 100) in the Rec.2020-Linear color space.
    palette_list : ndarray
        The false color palette to be applied to the image.
        The palette should be in the same color space as the input image.

    Returens
    --------
    ndarray
    """
    out_img = np.zeros_like(img)

    y_like = np.max(img, axis=-1)
    apply_sdr_false_color_filter(y_like=y_like, out_img=out_img)
    apply_hdr_false_color_filter(y_like=y_like, out_img=out_img, palette_list=palette_list)

    return out_img


def debug_func():
    palette_list = np.array(
        [palette_0, palette_1, palette_2, palette_3, palette_4, palette_5, palette_6]
    )

    from colour.io import read_image
    img_rec709_linear = read_image("./debug/RE4.exr")
    large_xyz = cs.rgb_to_large_xyz(img_rec709_linear, cs.BT709)
    img_rec2100_linear = cs.large_xyz_to_rgb(large_xyz, cs.BT2020)

    false_color_img_y_base_bt2020_linear = apply_false_color_filter_for_Y(
        img=img_rec2100_linear, palette_list=palette_list
    )
    false_color_img_maxRGB_base_bt2020_linear = apply_false_color_filter_for_maxRGB(
        img=img_rec2100_linear, palette_list=palette_list
    )

    srgb_img_y = convert_bt2020_linear_to_sRGB(
        img_linear_bt2020=false_color_img_y_base_bt2020_linear
    )
    srgb_img_maxRGB = convert_bt2020_linear_to_sRGB(
        img_linear_bt2020=false_color_img_maxRGB_base_bt2020_linear
    )

    tpg.img_wirte_float_as_16bit_int("./debug/RE4_y.png", srgb_img_y)
    tpg.img_wirte_float_as_16bit_int("./debug/RE4_maxRGB.png", srgb_img_maxRGB)


def debug_func_2_calc_rgb_to_rgb_mtx():
    from colour import matrix_RGB_to_RGB
    import color_space as cs

    p3d65_to_bt2020 = matrix_RGB_to_RGB(cs.P3_D65, cs.BT2020)
    bt2020_to_p3d65 = matrix_RGB_to_RGB(cs.BT2020, cs.P3_D65)
    bt2020_to_rec709 = matrix_RGB_to_RGB(cs.BT2020, cs.BT709)

    print(p3d65_to_bt2020)
    print(bt2020_to_p3d65)
    print(bt2020_to_rec709)


def main(src_file="./img/src_tp_rec2100-pq.png"):
    palette_list = np.array(
        [palette_0, palette_1, palette_2, palette_3, palette_4, palette_5, palette_6]
    )
    is_alpha = False
    alpha = None
    dst_file = src_file.replace("src", "dst")
    dst_file_y = add_suffix_to_filename(fname=dst_file, suffix="_y")
    dst_file_maxRGBx = add_suffix_to_filename(fname=dst_file, suffix="_maxRGB")


    img_rec2100_pq = tpg.img_read_as_float(src_file)
    if img_rec2100_pq.shape[2] == 4:
        is_alpha = True
        alpha = img_rec2100_pq[..., 3].copy()
        img_rec2100_pq = img_rec2100_pq[..., :3]

    img_rec2100_linear = convert_to_bt2020_linear(
        img=img_rec2100_pq, source_cs_name=cs.BT2020, source_tf_name=tf.ST2084
    )

    false_color_img_y_base_bt2020_linear = apply_false_color_filter_for_Y(
        img=img_rec2100_linear, palette_list=palette_list
    )
    false_color_img_maxRGB_base_bt2020_linear = apply_false_color_filter_for_maxRGB(
        img=img_rec2100_linear, palette_list=palette_list
    )

    srgb_img_y = convert_bt2020_linear_to_sRGB(
        img_linear_bt2020=false_color_img_y_base_bt2020_linear
    )
    srgb_img_maxRGB = convert_bt2020_linear_to_sRGB(
        img_linear_bt2020=false_color_img_maxRGB_base_bt2020_linear
    )

    if is_alpha:
        srgb_img_y = np.dstack((srgb_img_y, alpha))
        srgb_img_maxRGB = np.dstack((srgb_img_maxRGB, alpha))

    tpg.img_wirte_float_as_16bit_int(dst_file_y, srgb_img_y)
    tpg.img_wirte_float_as_16bit_int(dst_file_maxRGBx, srgb_img_maxRGB)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # main("./img/step_ramp_step_65.png")
    main("./img/scale_img.png")
    # main("./img/src_tp_rec2100-pq.png")
    # debug_func()
    # debug_func_2_calc_rgb_to_rgb_mtx()
