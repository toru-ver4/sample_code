# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
import cv2

# import my libraries
import test_pattern_generator2 as tpg
import transfer_functions as tf
import font_control as fc

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


FONT_PATH = "../../font/NotoSansMono-Bold.ttf"


def get_video_level_text_img(
        step_num, width, bit_depth, fontsize, stroke_width):
    """
    ステップカラーに付与する VideoLevel & Luminance 情報。
    最初は縦向きで作って、最後に横向きにする
    """
    max_level = (2 ** bit_depth) - 1
    font_color = [
        int(tf.oetf_from_luminance(100, tf.ST2084) * 0xFF + 0.5)
        for x in range(3)]
    dummy_text = "{:>6.1f} ".format(1023, 10000.0)
    text_width, text_height = fc.get_text_size(
        text=dummy_text, font_size=fontsize, font_path=FONT_PATH,
        stroke_width=stroke_width, stroke_fill='black')
    # print(f"text_size={text_width, text_height}")
    text_height_list = tpg.equal_devision(width, step_num)
    video_level = np.linspace(0, 2 ** bit_depth, step_num)
    video_level[-1] -= 1
    video_level_float = video_level / max_level
    bright_list = tf.eotf_to_luminance(video_level_float, tf.ST2084)
    # txt_img = np.zeros((width, text_width, 3))
    txt_img = np.ones((width, text_width, 3)) * 0.1
    txt_img[..., 0] = 0.0
    st_pos_h = 0
    st_pos_v = 0
    for idx in range(step_num):
        pos = (st_pos_h, st_pos_v + (text_height_list[idx] - text_height) // 2)
        if bright_list[idx] < 999.99999:
            text_data = "{:>6.1f} ".format(bright_list[idx])
        else:
            text_data = "{:>5.0f} ".format(bright_list[idx])
        text_drawer = fc.TextDrawer(
            txt_img, text=text_data, pos=pos,
            font_color=font_color, font_size=fontsize,
            bg_transfer_functions=tf.ST2084,
            fg_transfer_functions=tf.ST2084,
            font_path=FONT_PATH,
            stroke_width=stroke_width, stroke_fill='black')
        text_drawer.draw()

        st_pos_v += text_height_list[idx]
    txt_img = np.rot90(txt_img)
    return txt_img


def get_video_level_text_img_v2(
        num_of_step, width, cv_list, fontsize, stroke_width):
    """
    ステップカラーに付与する VideoLevel & Luminance 情報。
    最初は縦向きで作って、最後に横向きにする
    """
    font_color = [
        int(tf.oetf_from_luminance(400, tf.ST2084) * 0xFF + 0.5)
        for x in range(3)]
    dummy_text = "{:>6.1f} ".format(10000.0)
    text_width, text_height = fc.get_text_size(
        text=dummy_text, font_size=fontsize, font_path=FONT_PATH,
        stroke_width=stroke_width, stroke_fill='black')

    text_height_list = tpg.equal_devision(width, num_of_step)
    luminance_list = tf.eotf_to_luminance(cv_list, tf.ST2084)

    txt_img = np.ones((width, text_width, 3)) * 0.1
    txt_img[..., 0] = 0.0
    st_pos_h = 0
    st_pos_v = 0
    for idx in range(num_of_step):
        pos = (st_pos_h, st_pos_v + (text_height_list[idx] - text_height) // 2)
        if luminance_list[idx] < 999.99999:
            text_data = "{:>6.1f} ".format(luminance_list[idx])
        else:
            text_data = "{:>5.0f} ".format(luminance_list[idx])
        text_drawer = fc.TextDrawer(
            txt_img, text=text_data, pos=pos,
            font_color=font_color, font_size=fontsize,
            bg_transfer_functions=tf.ST2084,
            fg_transfer_functions=tf.ST2084,
            font_path=FONT_PATH,
            stroke_width=stroke_width, stroke_fill='black'
        )
        text_drawer.draw()

        st_pos_v += text_height_list[idx]
    txt_img = np.rot90(txt_img)
    return txt_img


def add_frame(img, frame_width):
    """
    img に frame_rate 分の余白を付けて出力する。
    余白は横幅基準で作る
    """
    bg_img = np.zeros((img.shape[0] + frame_width * 2,
                       img.shape[1] + frame_width * 2, 3))
    tpg.merge(bg_img, img, pos=(frame_width, frame_width))

    return bg_img


def generate_wrgb_step_ramp(
        step_num=65, width=1920, height=540, fontsize=20,
        stroke_width=4, frame_rate=0.007):
    """
    WRGB の 階調飛び飛び Ramp を作る。
    """
    color_list = [[1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]]
    bit_depth = 10
    max_value = (2 ** bit_depth) - 1
    frame_width = int(width * frame_rate + 0.5)
    internal_width = width - 2 * frame_width
    internal_height = height - 2 * frame_width

    bar_height_list = tpg.equal_devision(internal_height, len(color_list))
    bar_img_list = []
    for color, bar_height in zip(color_list, bar_height_list):
        color_bar = tpg.gen_step_gradation(
            width=internal_width, height=bar_height, step_num=step_num,
            bit_depth=10, color=color, direction='h')
        bar_img_list.append(color_bar)
    color_bar = np.vstack(bar_img_list)

    text_img = get_video_level_text_img(
        step_num, internal_width, bit_depth, fontsize,
        stroke_width) * max_value

    color_bar = np.vstack([text_img, color_bar])
    color_bar = add_frame(color_bar, frame_width=frame_width)

    alpha_idx = (color_bar[..., 0] < 10)\
        & (color_bar[..., 1] > 10)\
        & (color_bar[..., 2] > 10)
    alpha = (np.logical_not(alpha_idx) * max_value).reshape(
        color_bar.shape[0], color_bar.shape[1], 1)

    color_bar = np.dstack(
        (color_bar[..., 0], color_bar[..., 1], color_bar[..., 2], alpha))

    return color_bar / max_value


def generate_wrgb_step_ramp_v2():
    width = 3840
    height = 64
    num_of_step = 49
    font_size = 48
    stroke_width = 5
    max_value_int = 769
    max_value_float = (max_value_int - 1) / 1023

    img_buf = []
    color_list = [[1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]]
    for color in color_list:
        temp_img, cv_list = tpg.gen_step_ramp_v2(
            width=width, height=height, num_of_step=num_of_step,
            max_val=max_value_float, color=color)
        img_buf.append(temp_img)
    img = np.vstack(img_buf)
    print(cv_list * 1023)

    text_img = get_video_level_text_img_v2(
        num_of_step=num_of_step, width=width, cv_list=cv_list,
        fontsize=font_size, stroke_width=stroke_width)

    img = np.vstack([text_img, img])

    alpha_idx = (img[..., 0] < 0.1)\
        & (img[..., 1] >= 0.1)\
        & (img[..., 2] >= 0.1)
    alpha = (np.logical_not(alpha_idx) * max_value_int).reshape(
        img.shape[0], img.shape[1], 1)

    img = np.dstack(
        (img[..., 0], img[..., 1], img[..., 2], alpha))

    tpg.img_wirte_float_as_16bit_int("./tp_1000_mode/step_ramp_1000.png", img)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    step_num = 65
    height = 90
    font_size = 24
    stroke_width = 8
    width = 1920
    # img = generate_wrgb_step_ramp(
    #     step_num=step_num, width=width,
    #     height=height, fontsize=font_size, stroke_width=stroke_width,
    #     frame_rate=0.0)
    # fname = f"./tp_1000_mode/step_ramp_{width}px.png"
    # tpg.img_wirte_float_as_16bit_int(fname, img)
    generate_wrgb_step_ramp_v2()
