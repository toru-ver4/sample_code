# -*- coding: utf-8 -*-
"""
Luminance Map 確認用のテストパターンを作る
=========================================

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
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansMonoCJKjp-Bold.otf"


def get_video_level_text_img(step_num, width, bit_depth, fontsize):
    """
    ステップカラーに付与する VideoLevel & Luminance 情報。
    最初は縦向きで作って、最後に横向きにする
    """
    max_level = (2 ** bit_depth) - 1
    font_color = [
        int(tf.oetf_from_luminance(100, tf.ST2084) * 0xFF + 0.5)
        for x in range(3)]
    dummy_text = " {:>4.0f},{:>5.0f}  nit".format(1023, 10000.0)
    text_width, text_height = fc.get_text_size(
        text=dummy_text, font_size=fontsize, font_path=FONT_PATH)
    text_height_list = tpg.equal_devision(width, step_num)
    video_level = np.linspace(0, 2 ** bit_depth, step_num)
    video_level[-1] -= 1
    video_level_float = video_level / max_level
    bright_list = tf.eotf_to_luminance(video_level_float,
                                       tf.ST2084)
    txt_img = np.zeros((width, text_width, 3))
    st_pos_h = 0
    st_pos_v = 0
    for idx in range(step_num):
        pos = (st_pos_h, st_pos_v + (text_height_list[idx] - text_height) // 2)
        if bright_list[idx] < 999.99999:
            text_data = " {:>4.0f},{:>6.1f} nit".format(video_level[idx],
                                                        bright_list[idx])
        else:
            text_data = " {:>4.0f},{:>5.0f}  nit".format(video_level[idx],
                                                         bright_list[idx])
        text_drawer = fc.TextDrawer(
            txt_img, text=text_data, pos=pos,
            font_color=font_color, font_size=fontsize,
            bg_transfer_functions=tf.ST2084,
            fg_transfer_functions=tf.ST2084,
            font_path=FONT_PATH)
        text_drawer.draw()

        st_pos_v += text_height_list[idx]
    txt_img = np.rot90(txt_img)
    return txt_img


def add_frame(img, frame_rate=0.03):
    """
    img に frame_rate 分の余白を付けて出力する。
    余白は横幅基準で作る
    """
    img_width = img.shape[1]
    frame_width = int(img_width * frame_rate + 0.5)
    bg_img = np.zeros((img.shape[0] + frame_width * 2,
                       img.shape[1] + frame_width * 2, 3))
    tpg.merge(bg_img, img, pos=(frame_width, frame_width))

    return bg_img


def generate_wrgb_step_ramp(step_num=65, width=1920, height=540, fontsize=20):
    """
    WRGB の 階調飛び飛び Ramp を作る。
    """
    color_list = [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    bit_depth = 10
    max_value = (2 ** bit_depth) - 1

    bar_height_list = tpg.equal_devision(height, len(color_list))
    bar_img_list = []
    for color, bar_height in zip(color_list, bar_height_list):
        color_bar = tpg.gen_step_gradation(width=width, height=bar_height,
                                           step_num=step_num,
                                           bit_depth=10,
                                           color=color, direction='h')
        bar_img_list.append(color_bar)
    color_bar = np.vstack(bar_img_list)

    text_img = get_video_level_text_img(
        step_num, width, bit_depth, fontsize) * max_value

    color_bar = np.vstack([text_img, color_bar])
    color_bar = add_frame(color_bar, frame_rate=0.007)

    return color_bar / max_value


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    img = generate_wrgb_step_ramp(
        step_num=65, width=1920, height=200, fontsize=20)
    img = np.uint16(np.round(img * 0xFFFF))
    fname = f"./figure/step_ramp.tiff"
    cv2.imwrite(fname, img[..., ::-1])
