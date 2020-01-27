# -*- coding: utf-8 -*-
"""
WRGBMYCチェッカー＋可変のMAX輝度領域のテストパターン作成
=======================================================

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from colour import write_image

# import my libraries
import test_pattern_generator2 as tpg
import transfer_functions as tf

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

IMG_WIDTH = 1920
IMG_HEIGHT = 1080
WRGBMYC_RATE = 0.6
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansMonoCJKjp-Regular.otf"
WHITE_PAT_V_LIMIT = 0.77


def get_color_bar_text_font_size(text_height):
    """
    カラーバー横に表示する階調＋輝度のフォントサイズを取得する
    """
    font_size = int(text_height / 96 * 72)
    return font_size


def get_video_level_text_img(scale_step, width):
    """
    ステップカラーに付与する VideoLevel & Luminance 情報。
    最初は縦向きで作って、最後に横向きにする
    """
    fg_color = tuple([384, 384, 384])
    text_height_list = tpg.equal_devision(width, scale_step)
    font_size = get_color_bar_text_font_size(width / scale_step)
    video_level = np.linspace(0, 1024, scale_step)
    video_level[-1] -= 1
    video_level_float = video_level / 1023
    bright_list = tf.eotf_to_luminance(video_level_float, tf.ST2084)
    text_width = IMG_HEIGHT // 2

    txt_img = Image.new("RGB", (text_width, width), (0x00, 0x00, 0x00))
    draw = ImageDraw.Draw(txt_img)
    font = ImageFont.truetype(FONT_PATH, font_size)
    st_pos_h = 0
    st_pos_v = 0
    for idx in range(scale_step):
        pos = (st_pos_h, st_pos_v)
        if bright_list[idx] < 999.99999:
            text_data = " {:>4.0f},{:>6.1f} nit".format(video_level[idx],
                                                        bright_list[idx])
        else:
            text_data = " {:>4.0f},{:>5.0f}  nit".format(video_level[idx],
                                                         bright_list[idx])
        draw.text(pos, text_data, font=font, fill=fg_color)
        st_pos_v += text_height_list[idx]
    txt_img = np.uint16(np.asarray(txt_img)) * 4  # 8bit to 10bit
    txt_img = np.rot90(txt_img)
    return txt_img


def draw_wrgbmyc_color_bar(tp_width, tp_height):
    """
    画面下部のチェッカー柄＋テキストをプロットする
    """
    img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint16)
    scale_step = 65
    high_level = 1023
    color_list = [(1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                  (1, 0, 1), (1, 1, 0), (0, 1, 1)]
    color_bar_st_pos_h = IMG_WIDTH // 2 - tp_width // 2
    color_bar_st_pos_v = IMG_HEIGHT - tp_height
    st_pos = (color_bar_st_pos_h, color_bar_st_pos_v)
    bar_height_list = tpg.equal_devision(tp_height, len(color_list))
    bar_width_list = tpg.equal_devision(tp_width, scale_step)
    low_level_list = np.arange(scale_step) * (1024 // (scale_step - 1))
    low_level_list = np.uint16(np.round(low_level_list))
    low_level_list[-1] = low_level_list[-1] - 1

    v_img_list = []
    for color, bar_height in zip(color_list, bar_height_list):
        h_img_list = []
        for low_level, bar_width in zip(low_level_list, bar_width_list):
            # print(low_level, bar_width, bar_height)
            low_level_color = [low_level * color[idx] for idx in range(3)]
            high_level_color = [high_level * color[idx] for idx in range(3)]
            temp_img = tpg.make_tile_pattern(
                width=bar_width, height=bar_height, h_tile_num=2, v_tile_num=2,
                low_level=low_level_color, high_level=high_level_color)
            h_img_list.append(temp_img)
        v_img_list.append(np.hstack(h_img_list))
    scale_img = np.vstack(v_img_list)

    tpg.merge(img, scale_img, st_pos)

    # # ここからテキスト。あらかじめV方向で作っておき、最後に回転させる
    txt_img = get_video_level_text_img(scale_step, tp_width)

    text_pos = (st_pos[0], st_pos[1] - txt_img.shape[0])
    tpg.merge(img, txt_img, text_pos)

    return img


def make_base_tp():
    """
    下の方に WRGBMYC のチェッカー柄のあるパターンを作る。
    """
    tp_width = int(IMG_WIDTH * WRGBMYC_RATE)
    tp_height = int(tp_width / 65 + 0.5) * 7
    img = draw_wrgbmyc_color_bar(tp_width, tp_height)

    return img


def make_tp_sequence():
    """
    テストパターンの静止画連番ファイルを作る。
    """
    g_idx = 0  # 連番ファイルのインデックス
    fps = 30
    one_way_seq_sec = 8
    peak_white_seq_sec = 6

    one_way_seq_num = one_way_seq_sec * fps
    peak_white_seq_num = peak_white_seq_sec * fps

    rate_list = np.linspace(0, 1, one_way_seq_num)

    # 下地のテストパターン作成
    base_img = make_base_tp()

    # 増加
    for white_rate in rate_list:
        height = int((white_rate ** 2.4) * (IMG_HEIGHT * WHITE_PAT_V_LIMIT))
        if height > 0:
            white_img = np.ones((height, IMG_WIDTH, 3), dtype=np.uint16) * 1023
            img = base_img.copy()
            tpg.merge(img, white_img, (0, 0))
        else:
            img = base_img
        fname = "./img/tp_{:05d}.tiff".format(g_idx)
        write_image(img/0x3FF, fname, bit_depth='uint16')
        g_idx = g_idx + 1

    # 停止
    white_img = np.ones((int(IMG_HEIGHT * WHITE_PAT_V_LIMIT), IMG_WIDTH, 3),
                        dtype=np.uint16) * 1023
    for nominal_count in range(peak_white_seq_num):
        fname = "./img/tp_{:05d}.tiff".format(g_idx)
        write_image(img/0x3FF, fname, bit_depth='uint16')
        g_idx = g_idx + 1

    # 減少
    for white_rate in rate_list[::-1]:
        height = int((white_rate ** 2.4) * (IMG_HEIGHT * WHITE_PAT_V_LIMIT))
        if height > 0:
            white_img = np.ones((height, IMG_WIDTH, 3), dtype=np.uint16) * 1023
            img = base_img.copy()
            tpg.merge(img, white_img, (0, 0))
        else:
            img = base_img
        fname = "./img/tp_{:05d}.tiff".format(g_idx)
        write_image(img/0x3FF, fname, bit_depth='uint16')
        g_idx = g_idx + 1


def main_func():
    make_tp_sequence()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
