# -*- coding: utf-8 -*-
"""

```
"""

# import standard libraries
import os
from colour.utilities import tstack
from itertools import product

# import third-party libraries
import numpy as np

# import my libraries
import color_space as cs
import transfer_functions as tf
from create_gamut_booundary_lut import TyLchLut,\
    make_jzazbz_gb_lut_fname_method_c, is_outer_gamut_jzazbz
from jzazbz import jzczhz_to_jzazbz, delta_Ez_jzazbz, delta_Ez_jzczhz
import test_pattern_generator2 as tpg
import font_control as fc

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


if os.name == 'nt':
    FONT_PATH = "C:/Users/toruv/OneDrive/work/sample_code/font/NotoSansJP-Regular.otf"
else:
    FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf"


def calc_rgb_each_hue_dEz_base(
        hue, cusp, focal_point_l, chroma_num, delta_Ez,
        color_space_name, luminance):
    """
    Parameters
    ----------
    hue : float
        hue angle (0.0-360)
    focal_point_l : float
        lightness value of the focal_point.
    chroma_num : int
        the number of chroma.
    delta_Ez : float
        delta_Ez
    color_space_name : str
        name
    luminance : float
        luminance (ex. 1000)
    """
    hue_array_this_h_idx = np.ones(chroma_num) * hue
    cusp_l = cusp[0]
    cusp_c = cusp[1]
    aa = (cusp_l - focal_point_l) / cusp_c
    each_chroma = ((delta_Ez ** 2) / (1 + aa ** 2)) ** 0.5
    chroma_array = np.arange(chroma_num) * each_chroma
    # print(f"hue={hue}, chroma_array={chroma_array}")
    bb = focal_point_l
    # bb = cusp_l
    lightness_array = aa * chroma_array + bb

    ng_idx = (chroma_array > cusp_c)
    chroma_array[ng_idx] = 0.0
    lightness_array[ng_idx] = 0.0

    lch_array = tstack(
        [lightness_array, chroma_array, hue_array_this_h_idx])

    rgb_array = cs.jzazbz_to_rgb(
        jzazbz=jzczhz_to_jzazbz(lch_array),
        color_space_name=color_space_name, luminance=luminance)
    # rgb_array_lumiannce = rgb_array * luminance
    # print(f"hue={hue}")
    # for c_idx in range(len(chroma_array)):
    #     print(f"{lch_array[c_idx]}, {rgb_cv[c_idx]}")

    return rgb_array


def calc_jzazbz_each_hue_Cz_base(
        hue, cusp, focal_point_l, chroma_max, chroma_num):
    """
    Parameters
    ----------
    hue : float
        hue angle (0.0-360)
    focal_point_l : float
        lightness value of the focal_point.
    chroma_max : float
        maximum chroma
    chroma_num : int
        the number of chroma.
    """
    hue_array_this_h_idx = np.ones(chroma_num) * hue
    cusp_l = cusp[0]
    cusp_c = cusp[1]
    aa = (cusp_l - focal_point_l) / cusp_c
    bb = focal_point_l
    chroma_array = np.linspace(0, chroma_max, chroma_num)
    lightness_array = aa * chroma_array + bb

    ng_idx = (chroma_array > cusp_c)

    lch_array = tstack(
        [lightness_array, chroma_array, hue_array_this_h_idx])

    jzazbz_array = jzczhz_to_jzazbz(lch_array)

    return jzazbz_array, ng_idx


def calc_maximum_delta_Ez(
        focal_point_l, hue_array, hue_num, lut: TyLchLut):
    # find maximum chroma from the cups of all hue angles
    cusp = np.zeros((hue_num, 3))
    delta_Ez = np.zeros(hue_num)
    for h_idx in range(hue_num):
        hue = hue_array[h_idx]
        # cusp[h_idx] = lut.get_cusp(hue=hue)
        cusp[h_idx] = lut.get_cusp_without_intp(hue=hue)
        focal_point = np.array([focal_point_l, 0.0, hue])
        delta_Ez[h_idx] = delta_Ez_jzczhz(focal_point, cusp[h_idx])

    return np.max(delta_Ez)


def create_cl_pattern_contant_dEz():
    color_space_name = cs.BT709
    width = 1920
    height = 1080
    hue_num = 42
    hue_array = np.linspace(0, 360, hue_num, endpoint=False)
    chroma_num = int(round(height / width * hue_num))
    luminance = 100
    normalize_luminance = 10000 if luminance > 100 else 100
    block_width_list = tpg.equal_devision(width, hue_num)
    block_height_list = tpg.equal_devision(height, chroma_num)

    # load lut
    lut_name = make_jzazbz_gb_lut_fname_method_c(
        color_space_name=color_space_name, luminance=luminance)
    lut = TyLchLut(np.load(lut_name))
    focal_point_l = lut.ll_max / 2

    # find maximum chroma from the cups of all hue angles
    cusp = np.zeros((hue_num, 3))
    for h_idx in range(hue_num):
        hue = hue_array[h_idx]
        # cusp[h_idx] = lut.get_cusp(hue=hue)
        cusp[h_idx] = lut.get_cusp_without_intp(hue=hue)

    # calc each delta_Ez
    delta_Ez_total = calc_maximum_delta_Ez(
        focal_point_l=focal_point_l,
        hue_array=hue_array, hue_num=hue_num, lut=lut)
    delta_Ez = delta_Ez_total / (chroma_num - 1)

    h_buf = []
    for h_idx in range(hue_num):
        hue = hue_array[h_idx]
        rgb = calc_rgb_each_hue_dEz_base(
            hue=hue, cusp=cusp[h_idx], focal_point_l=focal_point_l,
            chroma_num=chroma_num, delta_Ez=delta_Ez,
            color_space_name=color_space_name, luminance=normalize_luminance)
        block_width = block_width_list[h_idx]

        v_buf = []
        for c_idx in range(chroma_num):
            block_height = block_height_list[c_idx]
            block_img = np.ones((block_height, block_width, 3)) * rgb[c_idx]
            v_buf.append(block_img)
        h_buf.append(np.vstack(v_buf))
    img_linear = np.hstack(h_buf)

    tf_str = tf.ST2084 if luminance > 100 else tf.GAMMA24
    img = tf.oetf(np.clip(img_linear, 0.0, 1.0), tf_str)

    tpg.img_wirte_float_as_16bit_int(f"./img/abb_dEz-{luminance}.png", img)


def create_cl_pattern_contant_Cz(
        color_space_name=cs.BT709, width=1920, height=1080,
        hue_num=46, luminance=100):
    hue_array = np.linspace(0, 360, hue_num, endpoint=False)

    tf_str = tf.ST2084 if luminance > 100 else tf.GAMMA24
    bg_luminance = 1
    bg_linear = tf.eotf(
        tf.oetf_from_luminance(bg_luminance, tf_str), tf_str)
    mark_luminance = 0
    mark_linear = tf.eotf(
        tf.oetf_from_luminance(mark_luminance, tf_str), tf_str)
    text_bg_luminance = 0.0
    text_bg_value = tf.oetf_from_luminance(text_bg_luminance, tf_str)
    text_luminance = 20
    text_value = tf.oetf_from_luminance(text_luminance, tf_str)
    normalize_luminance = 10000 if luminance > 100 else 100
    revision = 1  # new addition

    # information text
    font_path = FONT_PATH
    font_size = int(20 * height / 1080)
    text = f'Constant Hue-Chroma Pattern (Jzazbz color space),   {tf_str},   '
    text += f"{color_space_name},   D65,   {luminance} nits,   "
    text += f"Revision {revision}"
    text_width, text_height = fc.get_text_width_height(
        text=text, font_path=font_path, font_size=font_size)
    text_h_margin = int(6 * height / 1080)
    text_v_margin = int(text_height * 0.3)
    text_img = np.ones((text_height+text_v_margin*2, width, 3)) * text_bg_value
    text_drawer = fc.TextDrawer(
        text_img, text=text,
        pos=(text_h_margin, text_v_margin),
        font_color=(text_value, text_value, text_value),
        font_size=font_size, font_path=font_path)
    text_drawer.draw()

    # recalc coordinate
    height_rest = height - text_img.shape[0]
    chroma_num = int(round(height_rest / width * hue_num))
    block_width_list = tpg.equal_devision(width, hue_num)
    block_height_list = tpg.equal_devision(height_rest, chroma_num)

    # prepare marker
    mark_size = max(block_width_list[0] // 10, 5)
    mark_img_2020 = np.ones((mark_size, mark_size, 3)) * mark_linear

    # load lut
    lut_name = make_jzazbz_gb_lut_fname_method_c(
        color_space_name=color_space_name, luminance=luminance)
    lut = TyLchLut(np.load(lut_name))
    focal_point_l = lut.ll_max / 2

    # find maximum chroma from the cups of all hue angles
    cusp = np.zeros((hue_num, 3))
    for h_idx in range(hue_num):
        hue = hue_array[h_idx]
        cusp[h_idx] = lut.get_cusp(hue=hue)
        # cusp[h_idx] = lut.get_cusp_without_intp(hue=hue)
    chroma_max = np.max(cusp[..., 1])

    h_buf = []
    for h_idx in range(hue_num):
        hue = hue_array[h_idx]
        jzazbz, ng_idx = calc_jzazbz_each_hue_Cz_base(
            hue=hue, cusp=cusp[h_idx], focal_point_l=focal_point_l,
            chroma_max=chroma_max, chroma_num=chroma_num)
        is_oog_709 = is_outer_gamut_jzazbz(
            jzazbz=jzazbz, color_space_name=cs.BT709, luminance=luminance,
            delta=10**-6)
        is_oog_p3 = is_outer_gamut_jzazbz(
            jzazbz=jzazbz, color_space_name=cs.P3_D65, luminance=luminance,
            delta=10**-6)
        rgb = cs.jzazbz_to_rgb(
            jzazbz=jzazbz,
            color_space_name=color_space_name, luminance=normalize_luminance)
        rgb[ng_idx] = np.array([bg_linear, bg_linear, bg_linear])
        # print(f"hue={hue:.1f}, rgb={rgb}")

        v_buf = []
        block_width = block_width_list[h_idx]
        for c_idx in range(chroma_num):
            block_height = block_height_list[c_idx]
            block_img = np.ones((block_height, block_width, 3)) * rgb[c_idx]
            if is_oog_709[c_idx] and not ng_idx[c_idx]:
                img_p3 = mark_img_2020.copy()
                img_p3[1:-1, 1:-1]\
                    = np.ones_like(img_p3[1:-1, 1:-1]) * rgb[c_idx]
                tpg.merge(block_img, img_p3, (0, 0))
            if is_oog_p3[c_idx] and not ng_idx[c_idx]:
                tpg.merge(block_img, mark_img_2020, (0, 0))
            v_buf.append(block_img)
        h_buf.append(np.vstack(v_buf))
    img_linear = np.hstack(h_buf)

    img = tf.oetf(np.clip(img_linear, 0.0, 1.0), tf_str)
    img = np.vstack([img, text_img])

    fname = "./test_pattern_output/constant_chroma_pattern_"
    fname += f"{width}x{height}_{color_space_name}_{luminance}-nits"
    fname += f"_rev{revision}.png"
    print(fname)

    tpg.img_wirte_float_as_16bit_int(fname, img)


def create_cl_pattern_normal(
        color_space_name=cs.BT709, width=1920, height=1080,
        hue_num=46, luminance=100):
    hue_array = np.linspace(0, 360, hue_num, endpoint=False)

    tf_str = tf.ST2084 if luminance > 100 else tf.GAMMA24
    bg_luminance = 1
    bg_linear = tf.eotf(
        tf.oetf_from_luminance(bg_luminance, tf_str), tf_str)
    mark_luminance = 0
    mark_linear = tf.eotf(
        tf.oetf_from_luminance(mark_luminance, tf_str), tf_str)
    text_bg_luminance = 0.0
    text_bg_value = tf.oetf_from_luminance(text_bg_luminance, tf_str)
    text_luminance = 20
    text_value = tf.oetf_from_luminance(text_luminance, tf_str)
    normalize_luminance = 10000 if luminance > 100 else 100
    revision = 4  # new lut, focal point = ll.max/2

    # information text
    font_path = FONT_PATH
    font_size = int(20 * height / 1080)
    text = f'Hue-Chroma Pattern (Jzazbz color space),   {tf_str},   '
    text += f"{color_space_name},   D65,   {luminance} nits,   "
    text += f"Revision {revision}"
    text_width, text_height = fc.get_text_width_height(
        text=text, font_path=font_path, font_size=font_size)
    text_h_margin = int(6 * height / 1080)
    text_v_margin = int(text_height * 0.3)
    text_img = np.ones((text_height+text_v_margin*2, width, 3)) * text_bg_value
    text_drawer = fc.TextDrawer(
        text_img, text=text,
        pos=(text_h_margin, text_v_margin),
        font_color=(text_value, text_value, text_value),
        font_size=font_size, font_path=font_path)
    text_drawer.draw()

    # recalc coordinate
    height_rest = height - text_img.shape[0]
    chroma_num = int(round(height_rest / width * hue_num))
    block_width_list = tpg.equal_devision(width, hue_num)
    block_height_list = tpg.equal_devision(height_rest, chroma_num)

    # prepare marker
    mark_size = max(block_width_list[0] // 10, 5)
    mark_img_2020 = np.ones((mark_size, mark_size, 3)) * mark_linear

    # load lut
    lut_name = make_jzazbz_gb_lut_fname_method_c(
        color_space_name=color_space_name, luminance=luminance)
    lut = TyLchLut(np.load(lut_name))
    focal_point_l = lut.ll_max / 2

    # find maximum chroma from the cups of all hue angles
    cusp = np.zeros((hue_num, 3))
    for h_idx in range(hue_num):
        hue = hue_array[h_idx]
        # cusp[h_idx] = lut.get_cusp(hue=hue)
        cusp[h_idx] = lut.get_cusp_without_intp(hue=hue)

    h_buf = []
    for h_idx in range(hue_num):
        hue = hue_array[h_idx]
        cusp = lut.get_cusp(hue=hue)
        chroma_max = cusp[1]
        jzazbz, ng_idx = calc_jzazbz_each_hue_Cz_base(
            hue=hue, cusp=cusp, focal_point_l=focal_point_l,
            chroma_max=chroma_max, chroma_num=chroma_num)
        is_oog_709 = is_outer_gamut_jzazbz(
            jzazbz=jzazbz, color_space_name=cs.BT709, luminance=luminance,
            delta=10**-5)
        is_oog_p3 = is_outer_gamut_jzazbz(
            jzazbz=jzazbz, color_space_name=cs.P3_D65, luminance=luminance,
            delta=10**-5)
        rgb = cs.jzazbz_to_rgb(
            jzazbz=jzazbz,
            color_space_name=color_space_name, luminance=normalize_luminance)
        rgb[ng_idx] = np.array([bg_linear, bg_linear, bg_linear])
        # print(f"hue={hue:.1f}, rgb={rgb}")

        v_buf = []
        block_width = block_width_list[h_idx]
        for c_idx in range(chroma_num):
            block_height = block_height_list[c_idx]
            block_img = np.ones((block_height, block_width, 3)) * rgb[c_idx]
            if is_oog_709[c_idx] and not ng_idx[c_idx]:
                img_p3 = mark_img_2020.copy()
                img_p3[1:-1, 1:-1]\
                    = np.ones_like(img_p3[1:-1, 1:-1]) * rgb[c_idx]
                tpg.merge(block_img, img_p3, (0, 0))
            if is_oog_p3[c_idx] and not ng_idx[c_idx]:
                tpg.merge(block_img, mark_img_2020, (0, 0))
            v_buf.append(block_img)
        h_buf.append(np.vstack(v_buf))
    img_linear = np.hstack(h_buf)

    img = tf.oetf(np.clip(img_linear, 0.0, 1.0), tf_str)
    img = np.vstack([img, text_img])

    fname = "./test_pattern_output/hue-chroma_pattern_"
    fname += f"{width}x{height}_{color_space_name}_{luminance}-nits"
    fname += f"_rev{revision}.png"
    print(fname)

    tpg.img_wirte_float_as_16bit_int(fname, img)


def debug_cl_pattern():
    y1 = (np.array([816, 763, 410]) / 1023) ** 2.4
    y2 = (np.array([842, 782, 392]) / 1023) ** 2.4
    r1 = (np.array([851, 383, 495]) / 1023) ** 2.4
    r2 = (np.array([877, 361, 493]) / 1023) ** 2.4
    b1 = (np.array([413, 511, 684]) / 1023) ** 2.4
    b2 = (np.array([396, 507, 707]) / 1023) ** 2.4

    y1_jab = cs.rgb_to_jzazbz(y1, color_space_name=cs.BT709, luminance=100)
    y2_jab = cs.rgb_to_jzazbz(y2, color_space_name=cs.BT709, luminance=100)
    r1_jab = cs.rgb_to_jzazbz(r1, color_space_name=cs.BT709, luminance=100)
    r2_jab = cs.rgb_to_jzazbz(r2, color_space_name=cs.BT709, luminance=100)
    b1_jab = cs.rgb_to_jzazbz(b1, color_space_name=cs.BT709, luminance=100)
    b2_jab = cs.rgb_to_jzazbz(b2, color_space_name=cs.BT709, luminance=100)

    print(y1_jab, y2_jab, r1_jab, r2_jab)

    delta_y = delta_Ez_jzazbz(y1_jab, y2_jab)
    delta_r = delta_Ez_jzazbz(r1_jab, r2_jab)
    delta_b = delta_Ez_jzazbz(b1_jab, b2_jab)

    print(f"dEz_y={delta_y}")
    print(f"dEz_r={delta_r}")
    print(f"dEz_b={delta_b}")

    debug_data_rgb_non_linear = np.load("debug_data.npy")
    debug_data_rgb = debug_data_rgb_non_linear ** 2.4
    dd = cs.rgb_to_jzazbz(
        debug_data_rgb, color_space_name=cs.BT709, luminance=100)
    # print(debug_data.shape)
    # chroma_num, hue_num, _ = debug_data.shape
    # for h_idx in range(hue_num):
    #     print(debug_data[:, h_idx, :])

    delta_1 = delta_Ez_jzazbz(dd[5, 3], dd[6, 3])
    delta_2 = delta_Ez_jzazbz(dd[6, 3], dd[7, 3])
    delta_3 = delta_Ez_jzazbz(dd[7, 3], dd[8, 3])
    print(delta_1)
    print(delta_2)
    print(delta_3)

    delta_1 = delta_Ez_jzazbz(dd[5, 20], dd[6, 20])
    delta_2 = delta_Ez_jzazbz(dd[6, 20], dd[7, 20])
    delta_3 = delta_Ez_jzazbz(dd[7, 20], dd[8, 20])
    print(delta_1)
    print(delta_2)
    print(delta_3)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_cl_pattern_contant_dEz()
    color_space_name_list = [cs.BT709, cs.BT2020, cs.P3_D65]
    luminance_list = [100, 300, 600, 1000, 2000, 4000, 10000]
    resolution_list = [[1920, 1080], [3840, 2160]]

    for color_space_name, luminance, resolution in product(
            color_space_name_list, luminance_list, resolution_list):
        width = resolution[0]
        height = resolution[1]
        create_cl_pattern_contant_Cz(
            color_space_name=color_space_name,
            width=width, height=height, hue_num=46, luminance=luminance)
        create_cl_pattern_normal(
            color_space_name=color_space_name,
            width=width, height=height, hue_num=46, luminance=luminance)

    # debug_cl_pattern()
