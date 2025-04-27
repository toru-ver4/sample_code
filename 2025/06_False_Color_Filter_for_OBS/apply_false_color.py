# import standard libraries
import os

# import third-party libraries
import numpy as np

# my libraries
import color_space as cs
import test_pattern_generator2 as tpg
import transfer_functions as tf
# from flase_color_palette import(
#     palette_0, palette_1, palette_2, palette_3, palette_4, palette_5, palette_6
# )

# This color palette is for BT.2100-PQ color space.
palette_0 = [
    [0.006870, 0.010806, 0.066433],
    [0.011951, 0.016830, 0.136649],
    [0.018471, 0.021998, 0.243564],
    [0.026109, 0.024525, 0.397274],
    [0.034571, 0.022095, 0.609755],
    [0.043458, 0.011752, 0.895132],
]

palette_1 = [
    [0.043260, 0.077137, 0.079256],
    [0.081920, 0.156688, 0.161881],
    [0.133998, 0.274117, 0.284756],
    [0.199854, 0.436833, 0.456306],
    [0.279404, 0.652628, 0.685586],
    [0.372089, 0.929636, 0.982266],
]

palette_2 = [
    [0.037669, 0.076590, 0.018682],
    [0.071234, 0.155267, 0.031800],
    [0.116685, 0.271308, 0.046693],
    [0.174674, 0.432080, 0.062113],
    [0.245594, 0.645368, 0.076575],
    [0.329557, 0.919339, 0.088376],
]

palette_3 = [
    [0.080057, 0.084641, 0.024562],
    [0.162542, 0.171343, 0.041712],
    [0.283944, 0.298159, 0.060600],
    [0.451536, 0.471938, 0.079072],
    [0.672855, 0.699505, 0.094600],
    [0.955659, 0.987578, 0.104328],
]

palette_4 = [
    [0.062827, 0.040983, 0.012828],
    [0.126368, 0.078263, 0.020906],
    [0.219572, 0.129472, 0.029339],
    [0.348266, 0.195778, 0.037102],
    [0.518677, 0.278128, 0.043051],
    [0.737423, 0.377211, 0.045940],
]

palette_5 = [
    [0.054922, 0.014994, 0.006472],
    [0.109362, 0.025329, 0.009976],
    [0.188642, 0.037021, 0.013254],
    [0.297598, 0.049073, 0.015790],
    [0.441452, 0.060248, 0.017033],
    [0.625815, 0.069046, 0.016407],
]

palette_6 = [
    [0.058208, 0.019449, 0.071704],
    [0.116440, 0.032952, 0.146631],
    [0.201438, 0.047866, 0.258881],
    [0.318343, 0.062392, 0.417113],
    [0.472651, 0.074221, 0.630991],
    [0.670206, 0.080494, 0.911242],
]


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
    threshold_list = [1.0, 2.0, 4.0, 6.0, 10.0, 20.0, 40.0, 101.0]

    out_img = np.zeros_like(img)

    yy = 0.262700212 * img[..., 0] + 0.677998072 * img[..., 1] + 0.0593017165 * img[..., 2]

    # mono
    mono_idx = yy < 1.0
    mono_img = np.dstack((yy[mono_idx], yy[mono_idx], yy[mono_idx])).reshape(-1, 3)
    out_img[mono_idx] = mono_img ** (1 / 1.1)

    for p_idx in range(len(threshold_list) - 1):
        lower_bound = threshold_list[p_idx]
        upper_bound = threshold_list[p_idx + 1]
        palette = palette_list[p_idx]
        x_org = np.linspace(0, 1, len(palette))

        # Y < lower_bound
        idx = (yy > lower_bound) & (yy <= upper_bound)
        x = (yy[idx] - lower_bound) / (upper_bound - lower_bound)
        print(np.min(x), np.max(x))

        rr = np.interp(x, x_org, palette[:, 0])
        gg = np.interp(x, x_org, palette[:, 1])
        bb = np.interp(x, x_org, palette[:, 2])
        rgb = np.dstack([rr, gg, bb]).reshape(-1, 3)
        out_img[idx] = rgb

    return out_img


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    palette_list = np.array(
        [palette_0, palette_1, palette_2, palette_3, palette_4, palette_5, palette_6]
    )

    img_rec2100_pq = tpg.img_read_as_float("./img/src_tp_rec2100-pq.png")
    img_rec2100_linear = convert_to_bt2020_linear(
        img=img_rec2100_pq, source_cs_name=cs.BT2020, source_tf_name=tf.ST2084
    )

    false_color_img_bt2020_linear = apply_false_color_filter_for_Y(
        img=img_rec2100_linear, palette_list=palette_list
    )

    srgb_img = convert_bt2020_linear_to_sRGB(img_linear_bt2020=false_color_img_bt2020_linear)

    tpg.img_wirte_float_as_16bit_int("./img/dst_tp_rec2100-pq_y.png", srgb_img)
