# -*- coding: utf-8 -*-
"""
KJ-65X9500G の輝度制限と階調特性の関係調査
=========================================

Description.

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour import write_image
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

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

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansMonoCJKjp-Regular.otf"


def convert_from_pillow_to_numpy(img):
    img = np.uint16(np.asarray(img)) * 2 ** (10 - 8)

    return img


def merge_text(img, txt_img, pos):
    """
    テキストを合成する作業の最後の部分。
    pos は テキストの (st_pos_h, st_pos_v) 。
    ## 個人的実装メモ
    今回はちゃんとアルファチャンネルを使った合成をしたかったが、
    PILは8bit, それ以外は 10～16bit により BG_COLOR に差が出るので断念。
    """
    st_pos_v = pos[1]
    ed_pos_v = pos[1] + txt_img.shape[0]
    st_pos_h = pos[0]
    ed_pos_h = pos[0] + txt_img.shape[1]
    # かなり汚い実装。0x00 で無いピクセルのインデックスを抽出し、
    # そのピクセルのみを元の画像に上書きするという処理をしている。
    text_index = txt_img > 0
    temp_img = img[st_pos_v:ed_pos_v, st_pos_h:ed_pos_h]
    temp_img[text_index] = txt_img[text_index]
    img[st_pos_v:ed_pos_v, st_pos_h:ed_pos_h] = temp_img


def merge_each_spec_text(img, pos, font_size, text_img_size, text):
    """
    各パーツの説明テキストを合成。
    pos は テキストの (st_pos_h, st_pos_v) 。
    text_img_size = (size_h, size_v)
    ## 個人的実装メモ
    今回はちゃんとアルファチャンネルを使った合成をしたかったが、
    PILは8bit, それ以外は 10～16bit により BG_COLOR に差が出るので断念。
    """
    # テキストイメージ作成
    text_width = text_img_size[0]
    text_height = text_img_size[1]
    fg_color = (0x00, 0x60, 0x60)
    bg_coor = (0x00, 0x00, 0x00)
    txt_img = Image.new("RGB", (text_width, text_height), bg_coor)
    draw = ImageDraw.Draw(txt_img)
    font = ImageFont.truetype(FONT_PATH, font_size)
    draw.text((0, 0), text, font=font, fill=fg_color)
    txt_img = convert_from_pillow_to_numpy(txt_img)
    merge_text(img, txt_img, pos)


def research_recognizable_peak_luminance(target_luminance=1000):
    """
    識別可能な最大輝度を調査するためのパッチを作成する。
    """
    bg_width = 1920
    bg_height = 1080

    fg_width = int(bg_width * (0.1 ** 0.5) + 0.5)
    fg_height = int(bg_height * (0.1 ** 0.5) + 0.5)

    img = np.zeros((bg_height, bg_width, 3), dtype=np.uint16)

    low_level = tf.oetf_from_luminance(target_luminance, tf.ST2084)
    low_level = np.uint16(np.round(low_level * 1023))
    print(low_level)
    low_level = (low_level, low_level, low_level)

    fg_img = tpg.make_tile_pattern(
        width=fg_width, height=fg_height, h_tile_num=16, v_tile_num=9,
        low_level=low_level, high_level=(1023, 1023, 1023))

    tpg.merge(img, fg_img, pos=(0, 0))
    merge_each_spec_text(
        img, pos=(630, 5), font_size=30, text_img_size=(960, 100),
        text="target luminance = {:d} cd/m2".format(target_luminance))

    fname = "./img/{:05d}_peak_lumiance.tiff".format(target_luminance)
    write_image(img / 0x3FF, fname, bit_depth='uint16')


def research_st2084_with_bg_luminance_change(
        target_luminance=1600, bg_luminance=1000):
    """
    識別可能な最大輝度を調査するためのパッチを作成する。
    """
    bg_width = 1920
    bg_height = 1080

    fg_width = int(bg_width * (0.1 ** 0.5) + 0.5)
    fg_height = int(bg_height * (0.1 ** 0.5) + 0.5)

    bg_level = tf.oetf_from_luminance(bg_luminance, tf.ST2084)
    bg_level = np.uint16(np.round(bg_level * 1023))

    img = np.ones((bg_height, bg_width, 3), dtype=np.uint16) * bg_level

    low_level = tf.oetf_from_luminance(target_luminance, tf.ST2084)
    low_level = np.uint16(np.round(low_level * 1023))
    low_level = (low_level, low_level, low_level)

    fg_img = tpg.make_tile_pattern(
        width=fg_width, height=fg_height, h_tile_num=16, v_tile_num=9,
        low_level=low_level, high_level=(1023, 1023, 1023))

    tpg.merge(img, fg_img, pos=(0, 0))
    text_base = "target_luminance = {:d} cd/m2, bg_luminance = {:d} cd/m2"
    merge_each_spec_text(
        img, pos=(630, 5), font_size=30, text_img_size=(960, 100),
        text=text_base.format(target_luminance, bg_luminance))

    fname_base = "./img/target_{:05d}_bg_{:05d}_lumiance.tiff"
    fname = fname_base.format(target_luminance, bg_luminance)
    write_image(img / 0x3FF, fname, bit_depth='uint16')


def main_func():
    # KJ-65X9500G の表示限界を調査
    for idx in range(10):
        luminance = 1000 + 100 * idx
        research_recognizable_peak_luminance(luminance)

    # BG Luminance を変化させた場合の挙動確認
    bg_luminance_list = [0, 100, 300, 500, 800, 1000, 2000, 4000, 10000]
    target_list = [500, 750, 1000, 1500, 2000]
    for target in target_list:
        for bg in bg_luminance_list:
            research_st2084_with_bg_luminance_change(
                target_luminance=target, bg_luminance=bg)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
