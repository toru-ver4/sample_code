# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
import subprocess
from itertools import product
from math import ceil

# import third-party libraries
import numpy as np
import cv2
from colour import RGB_to_RGB, RGB_COLOURSPACES, RGB_to_XYZ, XYZ_to_xyY

# import my libraries
import test_pattern_generator2 as tpg
import ty_utility as util
import font_control as fc
import color_space as cs
import transfer_functions as tf
import plot_utility as pu
from create_gamut_booundary_lut import make_jzazbz_gb_lut_fname_method_c,\
    TyLchLut
from jzazbz import jzczhz_to_jzazbz

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def modify_to_n_times(x, n):
    if x % n != 0:
        y = int(ceil(x / n)) * n
    else:
        y = x

    return y


def dot_mesh_pattern(
        width_org=1920, height_org=1080, dot_size=2, color=[1, 1, 0]):

    width = modify_to_n_times(width_org, dot_size * 2)
    height = modify_to_n_times(height_org, dot_size * 2)
    img = np.ones((height, width, 3)) * np.array(color)

    fname = f"./img/{width_org}x{height_org}_dotsize-{dot_size}_rgbmask-"
    fname += f"{color[0]}{color[1]}{color[2]}.png"

    zero_idx_h = ((np.arange(width) // (2**(dot_size-1))) % 2) == 0
    idx_even = np.hstack([zero_idx_h for x in range(dot_size)])
    idx_odd = np.hstack([~zero_idx_h for x in range(dot_size)])
    idx_even_odd = np.hstack([idx_even, idx_odd])
    idx_all_line = np.tile(
        idx_even_odd, height//(2 * dot_size)).reshape(height, width)

    img[idx_all_line] = 0

    img = img[:height_org, :width_org]
    print(fname)
    tpg.img_wirte_float_as_16bit_int(fname, img)

    fname_icc = util.add_suffix_to_filename(fname=fname, suffix="_with_icc")
    icc_profile = './icc_profile/Gamma2.4_DCI-P3_D65.icc'
    cmd = ['convert', fname, '-profile', icc_profile, fname_icc]
    subprocess.run(cmd)


def create_dot_pattern():
    resolution_list = [[2778, 1284], [2532, 1170]]
    dot_size_list = [1, 2]
    color_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
    for resolution, dot_size, color in product(
            resolution_list, dot_size_list, color_list):
        width = resolution[0]
        height = resolution[1]
        dot_mesh_pattern(
            width_org=width, height_org=height, dot_size=dot_size, color=color)


def create_abl_check_pattern(width_panel=2778, height_panel=1284):
    fps = 60
    sec = 8
    frame = fps * sec
    width_total = 1920
    height_total = 1080
    width = width_total
    height = int(round(height_panel/width_panel * width))

    for idx in range(frame):
        rate = (np.sin(np.pi/(frame - 1)*idx - np.pi/2) + 1) / 2
        img = np.zeros((height_total, width_total, 3))

        local_width = int(round(width * rate))
        local_height = int(round(height * rate))
        st_pos = (
            (width_total//2) - (local_width//2),
            (height_total//2) - (local_height//2))
        ed_pos = (st_pos[0]+local_width, st_pos[1]+local_height)
        cv2.rectangle(img, st_pos, ed_pos, (1.0, 1.0, 1.0), -1)

        percent = (local_width * local_height)\
            / (width * height) * 100
        text_drawer = fc.TextDrawer(
            img, text=f"{percent:.02f}%",
            pos=(int(width_total*0.04), int(width_total*0.08)),
            font_color=(0.25, 0.25, 0.25), font_size=30)
        text_drawer.draw()

        fname = "/work/overuse/2021/00_iphone_movie/img_seq/"
        fname += f"iPhone_abl_{width_panel}x{height_panel}_{idx:04d}.png"
        print(fname)
        tpg.img_wirte_float_as_16bit_int(fname, img)


def create_patch_specific_area(
        panel_width=2778, panel_height=1284, area_rate=20.0,
        color_st2084=[0.8, 0.5, 0.2], luminance=1000, src_cs=cs.BT709):
    width = 1920
    height = 1080
    tf_str = tf.ST2084
    img = np.zeros((height, width, 3))
    width_vertual = panel_width/panel_height*height
    height_vertual = height
    block_size = int(
        round((area_rate/100 * width_vertual * height_vertual) ** 0.5))
    st_pos = ((width//2) - (block_size//2), (height//2) - (block_size//2))
    ed_pos = (st_pos[0]+block_size, st_pos[1]+block_size)

    color_with_lumiannce = calc_linear_color_from_primary(
        color=color_st2084, luminance=luminance)
    cv2.rectangle(img, st_pos, ed_pos, color_with_lumiannce, -1)

    large_xyz = RGB_to_XYZ(
        color_with_lumiannce, cs.D65, cs.D65,
        RGB_COLOURSPACES[src_cs].matrix_RGB_to_XYZ)
    xyY = XYZ_to_xyY(large_xyz)

    img = tf.oetf(np.clip(img, 0.0, 1.0), tf_str)

    text = f"for_{panel_width}x{panel_height}, {src_cs}, "
    text += f"xyY=({xyY[0]:.03f}, "
    text += f"{xyY[1]:.03f}, {xyY[2]*10000:.1f})"
    text_drawer = fc.TextDrawer(
        img, text=text, pos=(10, 10),
        font_color=(0.25, 0.25, 0.25), font_size=20)
    text_drawer.draw()

    fname = f"./img/iPhone13_color_patch_for_{panel_width}x{panel_height}_"
    fname += f"{src_cs}_"
    fname += f"rgb_{color_with_lumiannce[0]:.2f}-"
    fname += f"{color_with_lumiannce[1]:.2f}-"
    fname += f"{color_with_lumiannce[2]:.2f}_{tf_str}_{luminance}-nits.png"
    print(fname)
    tpg.img_wirte_float_as_16bit_int(fname, img)


def calc_linear_color_from_primary(color=[1, 0, 0], luminance=1000):
    """
    supported transfer characteristics is ST2084 only.
    """
    color_luminance\
        = tf.eotf(np.array(color), tf.ST2084)\
        / tf.PEAK_LUMINANCE[tf.ST2084] * luminance

    return color_luminance


def create_iphone_13_primary_patch(area_rate=0.4*100):
    color_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
    resolution_list = [[2778, 1284], [2532, 1170]]
    luminance_list = [100, 1000, 4000, 10000]
    src_cs_list = [cs.BT709, cs.P3_D65, cs.BT2020]

    for resolution, luminance, color, src_cs in product(
            resolution_list, luminance_list, color_list, src_cs_list):
        width = resolution[0]
        height = resolution[1]
        create_patch_specific_area(
            panel_width=width, panel_height=height, area_rate=area_rate,
            color_st2084=color, luminance=luminance, src_cs=src_cs)


def conv_img_from_bt2020_to_bt709_using_3x3_matrix():
    # in_fname = "./img/bt2020_bt709_hue_chroma_1920x1080_h_num-32.png"
    in_fname = "./img/iPhone13_color_patch_for_2778x1284_P3-D65-on-ITU-R "
    in_fname += "BT.2020_rgb_0.10-0.00-0.00_SMPTE ST2084_1000-nits.png"
    out_fname = util.add_suffix_to_filename(
        fname=in_fname, suffix="_bt709_with_matrix")
    tf_str = tf.ST2084

    img_non_linear = tpg.img_read_as_float(in_fname)
    img_linear_2020 = tf.eotf(img_non_linear, tf_str)
    img_linear_709 = RGB_to_RGB(
        RGB=img_linear_2020,
        input_colourspace=RGB_COLOURSPACES[cs.BT2020],
        output_colourspace=RGB_COLOURSPACES[cs.P3_D65])
    img_non_linear_709 = tf.oetf(np.clip(img_linear_709, 0.0, 1.0), tf_str)

    tpg.img_wirte_float_as_16bit_int(out_fname, img_non_linear_709)


def plot_bt2020_vs_dci_p3():
    bt2020 = tpg.get_primaries(cs.BT2020)[0]
    p3_d65 = tpg.get_primaries(cs.P3_D65)[0]
    cmf_xy = tpg._get_cmfs_xy()

    fig, ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(8, 9),
        bg_color=(0.90, 0.90, 0.90),
        graph_title="Chromaticity Diagram",
        graph_title_size=None,
        xlabel="x", ylabel="y",
        axis_label_size=None,
        legend_size=17,
        xlim=[0.65, 0.72],
        ylim=[0.28, 0.34],
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=4)
    ax1.plot(
        (cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
        '-k', lw=4)
    ax1.plot(
        bt2020[..., 0], bt2020[..., 1], '-', color=pu.RED, label="BT.2020")
    ax1.plot(
        p3_d65[..., 0], p3_d65[..., 1], '-', color=pu.SKY, label="DCI-P3")
    pu.show_and_save(
        fig=fig, legend_loc='upper right', save_fname="./img/p3_gamut.png")


def create_gray_patch_core(
        width, height, panel_width, panel_height,
        st_pos, ed_pos, cv_float):
    img = np.zeros((height, width, 3))
    color = [cv_float, cv_float, cv_float]
    cv2.rectangle(img, st_pos, ed_pos, color, -1)

    text = f"for_{panel_width}x{panel_height}, {cv_float*1023:.0f} CV"
    text_drawer = fc.TextDrawer(
        img, text=text, pos=(10, 10),
        font_color=(0.25, 0.25, 0.25), font_size=20)
    text_drawer.draw()

    fname = f"./img/iPhone13_color_patch_for_{panel_width}x{panel_height}_"
    fname += f"{int(cv_float*1023):04d}-CV.png"
    print(fname)
    tpg.img_wirte_float_as_16bit_int(fname, img)


def create_gray_patch(panel_width=2778, panel_height=1284, area_rate=0.4*100):
    width = 1920
    height = 1080
    step_num = 33  # 33 or 65
    each_step = 1024 // (step_num - 1)
    cv_list = np.arange(step_num) * each_step
    cv_list[-1] = 1023

    width_vertual = panel_width/panel_height*height
    height_vertual = height
    block_size = int(
        round((area_rate/100 * width_vertual * height_vertual) ** 0.5))
    st_pos = ((width//2) - (block_size//2), (height//2) - (block_size//2))
    ed_pos = (st_pos[0]+block_size, st_pos[1]+block_size)

    for cv in cv_list:
        create_gray_patch_core(
            width=width, height=height,
            panel_width=panel_width, panel_height=panel_height,
            st_pos=st_pos, ed_pos=ed_pos, cv_float=cv/1023)


def get_rgb_st2084_cv_from_luminance(luminance):
    val = tf.oetf_from_luminance(luminance, tf.ST2084)
    return np.array([val, val, val])


def careate_tone_mapping_check_pattern(bg_luminance_start=600):
    g_width = 3840
    g_height = 2160
    b_h_num = 4
    b_v_num = 4
    tile_num = 6
    font_size = 28
    fg_luminance = 10000

    height = int(round(g_height / 1.5))
    width = height
    b_width = width // b_h_num
    b_height = height // b_v_num

    comp_st_pos = [
        (g_width // 2) - int(b_width * (b_h_num/2)),
        (g_height // 2) - int(b_height * (b_v_num/2))
    ]

    img = np.zeros((g_height, g_width, 3))

    v_img_buf = []
    for v_idx in range(b_v_num):
        h_img_buf = []
        for h_idx in range(b_h_num):
            idx = v_idx * b_h_num + h_idx
            bg_luminance = bg_luminance_start + 100 * idx
            print(f"bg_luminance = {bg_luminance}")
            b_img_nonlinear = tpg.make_tile_pattern(
                width=b_width, height=b_height,
                h_tile_num=tile_num, v_tile_num=tile_num,
                low_level=get_rgb_st2084_cv_from_luminance(bg_luminance),
                high_level=get_rgb_st2084_cv_from_luminance(fg_luminance),
                dtype=np.float32)
            text = f"bg_luminance = {bg_luminance} nit"
            _, text_height = fc.get_text_width_height(
                text, fc.NOTO_SANS_MONO_BOLD, font_size)
            text_drawer = fc.TextDrawer(
                b_img_nonlinear, text=text,
                pos=(int(text_height*0.2), int(text_height*0.2)),
                font_color=(0, 0, 0),
                font_size=font_size,
                bg_transfer_functions=tf.ST2084,
                fg_transfer_functions=tf.ST2084,
                font_path=fc.NOTO_SANS_MONO_BOLD)
            text_drawer.draw()

            tpg.draw_outline(b_img_nonlinear, np.array([0, 0, 0]), 1)
            h_img_buf.append(b_img_nonlinear)
        v_img_buf.append(np.hstack(h_img_buf))
    tp_img = np.vstack(v_img_buf)

    tpg.merge(img, tp_img, comp_st_pos)

    text = f"fg_luminance = {fg_luminance} nit"
    _, text_height = fc.get_text_width_height(
        text, fc.NOTO_SANS_MONO_BOLD, font_size)
    text_drawer = fc.TextDrawer(
        img, text=text,
        pos=(comp_st_pos[0], comp_st_pos[1] - int(text_height * 1.2)),
        font_color=(0.5, 0.5, 0.5),
        font_size=font_size,
        bg_transfer_functions=tf.ST2084,
        fg_transfer_functions=tf.ST2084,
        font_path=fc.NOTO_SANS_MONO_BOLD)
    text_drawer.draw()

    fname = f"./img/tone_map_tp_{bg_luminance_start}.png"
    tpg.img_wirte_float_as_16bit_int(fname, img)


def create_tone_mapping_check_pattern_all():
    careate_tone_mapping_check_pattern(bg_luminance_start=600)
    careate_tone_mapping_check_pattern(bg_luminance_start=600+1600*1)
    careate_tone_mapping_check_pattern(bg_luminance_start=600+1600*2)

    careate_tone_mapping_check_pattern(bg_luminance_start=100)
    careate_tone_mapping_check_pattern(bg_luminance_start=100+1600*1)
    careate_tone_mapping_check_pattern(bg_luminance_start=100+1600*2)


def calc_cusp_rgb_value(hue_num, color_space_name, luminance):
    lut_name = make_jzazbz_gb_lut_fname_method_c(
        color_space_name=color_space_name, luminance=luminance)
    lut = TyLchLut(np.load(lut_name))
    hue_list = np.linspace(0, 360, hue_num, endpoint=False)
    cusp_list = np.zeros((hue_num, 3))
    for h_idx, hue in enumerate(hue_list):
        cusp_list[h_idx] = lut.get_cusp_without_intp(hue)

    jzazbz = jzczhz_to_jzazbz(cusp_list)
    rgb = cs.jzazbz_to_rgb(
        jzazbz=jzazbz, color_space_name=color_space_name)

    return rgb


def create_per_nit_patch_img(
        b_width_array, b_height_array, rgb_array):
    # base_img = np.ones((b_height, b_width, 3))
    luminance_num = rgb_array.shape[0]
    hue_num = rgb_array.shape[1]
    v_buf = []
    for l_idx in range(luminance_num):
        b_height = b_height_array[l_idx]
        h_buf = []
        for h_idx in range(hue_num):
            b_width = b_width_array[h_idx]
            base_img = np.ones((b_height, b_width, 3))
            temp_img = base_img * rgb_array[l_idx][h_idx]
            tpg.draw_outline(temp_img, [0, 0, 0], 1)
            h_buf.append(temp_img)
        v_buf.append(np.hstack(h_buf))
    img = np.vstack(v_buf)

    return img


def create_cusp_per_nit_image_without_text(
        width=1600, height=1080, color_space_name=cs.P3_D65,
        luminance_list=[108, 643, 1143, 2341, 6470]):
    aspect_reate = width / height
    v_block_num = len(luminance_list)
    h_block_num = int(round(aspect_reate * v_block_num))

    b_height_array = tpg.equal_devision(height, v_block_num)
    b_width_array = tpg.equal_devision(width, h_block_num)

    rgb_array = np.zeros((v_block_num, h_block_num, 3))
    for idx, luminance in enumerate(luminance_list):
        rgb_array[idx] = calc_cusp_rgb_value(
            h_block_num, color_space_name, luminance)

    patch_img = create_per_nit_patch_img(
        b_width_array=b_width_array, b_height_array=b_height_array,
        rgb_array=rgb_array)

    return patch_img, b_height_array, b_width_array


def font_v_size_determinater(font, text="sample", limit_height=48):
    st_font_size = 100
    font_size_list = np.arange(st_font_size + 1)[::-1]

    for font_size in font_size_list:
        _, height = fc.get_text_width_height(
            text=text, font_path=font, font_size=font_size)
        if height < limit_height:
            break

    return font_size


def font_h_size_determinater(font, text="sample", limit_width=48):
    st_font_size = 100
    font_size_list = np.arange(st_font_size + 1)[::-1]

    for font_size in font_size_list:
        width, _ = fc.get_text_width_height(
            text=text, font_path=font, font_size=font_size)
        if width < limit_width:
            break

    return font_size


def font_size_determinater(font, text, limit_width, limit_heigt):
    font_v = font_v_size_determinater(
        font=font, text=text, limit_height=limit_heigt)
    font_h = font_h_size_determinater(
        font=font, text=text, limit_width=limit_width)

    font_size = min(font_v, font_h)
    width, height = fc.get_text_width_height(
            text=text, font_path=font, font_size=font_size)

    return font_size, width, height


def calc_luminance_list(min_lumi=100, max_lumi=2000):
    cv_list = [x * 16 for x in range(65)]
    cv_list[-1] = cv_list[-1] - 1
    luminance_list = [
        int(round(tf.eotf_to_luminance(x/1023, tf.ST2084)))
        for x in cv_list]
    luminance_list = [
        x for x in luminance_list if (x >= min_lumi) and (x <= max_lumi)]

    return luminance_list


def add_luminance_text_information(
        img, luminance_list, height_list, v_idx, font_size, font, font_height,
        font_color):
    v_offset = height_list[v_idx] // 2 - font_height // 2
    st_pos = [0, int(np.sum(height_list[:v_idx])) + v_offset]

    text = f" Peak Luminance {luminance_list[v_idx]:5d} nits "
    font_drawer = fc.TextDrawer(
        img=img, text=text, pos=st_pos, font_color=font_color,
        font_size=font_size, bg_transfer_functions=tf.ST2084,
        fg_transfer_functions=tf.ST2084, font_path=font)
    font_drawer.draw()


def cusp_per_nit_pattern(min_lumi=100, max_lumi=2000, width=1920, height=1080):
    width_rate = 0.8
    bg_luminance = 0.5
    bg_linear = tf.eotf(
        tf.oetf_from_luminance(bg_luminance, tf.ST2084), tf.ST2084)
    text_luminance = 20
    text_cv = tf.oetf_from_luminance(text_luminance, tf.ST2084)
    font_color = (text_cv, text_cv, text_cv)
    total_block_width = int(width * width_rate)
    rate = height // 1080
    font = fc.NOTO_SANS_MONO_REGULAR
    info_font_size = 24 * rate
    _, info_text_height = fc.get_text_width_height(
        text="sample", font_path=font, font_size=info_font_size)
    info_text_margin = int(info_text_height * 0.4)
    info_height = info_text_height + info_text_margin * 2
    total_block_height = height - info_height
    print(f"total_block_height={total_block_height}")
    color_space_name = cs.P3_D65
    font_size = None
    luminance_list = calc_luminance_list(min_lumi=min_lumi, max_lumi=max_lumi)
    print(luminance_list)

    patch_img, height_array, _ = create_cusp_per_nit_image_without_text(
        width=total_block_width, height=total_block_height,
        color_space_name=color_space_name, luminance_list=luminance_list)
    max_text = " Peak Luminance 10000 nits "
    font_size, _, font_height = font_size_determinater(
        font=font, text=max_text,
        limit_width=int((width-total_block_width)*0.9),
        limit_heigt=int(np.min(height_array)*0.9))
    print(f"font_size={font_size}")
    img = np.ones((height, width, 3)) * bg_linear

    tpg.merge(img, patch_img, (width - patch_img.shape[1], 0))
    img_non_linear = tf.oetf(np.clip(img, 0.0, 1.0), tf.ST2084)

    # add luminance information
    for v_idx in range(len(height_array)):
        add_luminance_text_information(
            img=img_non_linear, luminance_list=luminance_list,
            height_list=height_array, v_idx=v_idx, font_color=font_color,
            font_size=font_size, font=font, font_height=font_height)

    # add basic text information
    info_img = np.zeros((info_height, width, 3))
    text = f" Jzazbz Cusp Pattern,  ST2084,  {color_space_name},   "
    text += f"{width}x{height},  Revision 2"
    st_pos = [0, info_text_margin]
    font_drawer = fc.TextDrawer(
        img=info_img, text=text, pos=st_pos, font_color=font_color,
        font_size=font_size, bg_transfer_functions=tf.ST2084,
        fg_transfer_functions=tf.ST2084, font_path=font)
    font_drawer.draw()

    tpg.merge(img_non_linear, info_img, [0, height - info_height])

    fname = f"./img/jzazbz_cusp_{width}x{height}_"
    fname += f"{color_space_name}_{min_lumi}-{max_lumi}_nits.png"
    tpg.img_wirte_float_as_16bit_int(fname, img_non_linear)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_dot_pattern()
    # create_abl_check_pattern(width_panel=2778, height_panel=1284)
    # create_abl_check_pattern(width_panel=2532, height_panel=1170)
    # create_patch_specific_area(
    #     panel_width=2778, panel_height=1284, area_rate=0.4*100,
    #     color_linear=[1, 0, 0],
    #     luminance=1000, src_cs=cs.BT709, dst_cs=cs.BT2020, tf_str=tf.ST2084)
    # create_iphone_13_primary_patch(area_rate=0.4*100)
    # conv_img_from_bt2020_to_bt709_using_3x3_matrix()
    # plot_bt2020_vs_dci_p3()
    # create_gray_patch(panel_width=2778, panel_height=1284, area_rate=0.4*100)

    cusp_per_nit_pattern(width=3840, height=2160, min_lumi=100, max_lumi=1000)
    # cusp_per_nit_pattern(width=3840, height=2160, min_lumi=100, max_lumi=2000)
    # cusp_per_nit_pattern(width=3840, height=2160, min_lumi=100, max_lumi=10000)
