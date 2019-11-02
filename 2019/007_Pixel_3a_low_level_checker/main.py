#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Turbo を使ってHDR10信号の輝度マップを作成する
"""

# 外部ライブラリのインポート
import os
import numpy as np
import cv2
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from colour.models import eotf_ST2084, oetf_ST2084, RGB_to_RGB,\
    RGB_COLOURSPACES

# 自作ライブラリのインポート
from TyImageIO import TyWriter
import test_pattern_generator2 as tpg
import color_space as cs


PQ_10BIT_LOW_CV_PAIR_LIST = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64],
                             [0, 96], [0, 128], [0, 160], [0, 192], [0, 224]]
PQ_10BIT_HIGH_CV_PAIR_LIST = [[688, 1023], [704, 1023], [720, 1023],
                              [736, 1023], [752, 1023], [768, 1023],
                              [848, 1023], [928, 1023]]

PQ_REF_WHITE_NITS = 203  # Report ITU-R BT.2408-2


def convert_from_pillow_to_numpy(img):
    img = np.uint16(np.asarray(img)) * 2 ** (10 - 8)

    return img


def convert_from_float_to_int_10bit(x):
    return np.uint16(np.round(x * 1023))


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
    fg_color = (0x60, 0x60, 0x60)
    bg_coor = (0x00, 0x00, 0x00)
    txt_img = Image.new("RGB", (text_width, text_height), bg_coor)
    draw = ImageDraw.Draw(txt_img)
    font = ImageFont.truetype("./fonts/NotoSansMonoCJKjp-Regular.otf",
                              font_size)
    draw.text((0, 0), text, font=font, fill=fg_color)
    txt_img = convert_from_pillow_to_numpy(txt_img)
    merge_text(img, txt_img, pos)


def _make_tile_pattern_vdirection(
        width=3840, height=2160, h_tile_num=32, v_tile_num=18,
        low_level=[940, 940, 940], high_level=[1019, 1019, 1019],
        color_list=[[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]):
    """
    make_tile_pattern_wwrgbmyc のサブルーチン。
    縦方向に(W, R, G, B) or (W, M, Y, C) の 4色分のパターンを作成する。
    """
    img_buf = []
    for v_idx, color in enumerate(color_list):
        low_temp = [low_level[idx] * color[idx] for idx in range(3)]
        high_temp = [high_level[idx] * color[idx] for idx in range(3)]
        temp = tpg.make_tile_pattern(
            width=width//2, height=height//4, h_tile_num=32, v_tile_num=9,
            low_level=low_temp, high_level=high_temp)
        temp = temp[:, ::-1, :] if v_idx % 2 != 0 else temp
        img_buf.append(temp)

    return np.vstack(img_buf)


def make_tile_pattern_wwrgbmyc(
        width=3840, height=2160, h_tile_num=32, v_tile_num=18,
        low_level=[940, 940, 940], high_level=[1019, 1019, 1019]):
    """
    画面を8分割し、WWRGBMYC タイル状パターンを作る。
    White が 2つあるのは隙間を埋めるため
    """
    # wrgb
    color_list = [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    wrgb_img = _make_tile_pattern_vdirection(
        width=width, height=height,
        h_tile_num=h_tile_num, v_tile_num=v_tile_num,
        low_level=low_level, high_level=high_level,
        color_list=color_list)

    # wcmy
    color_list = [[1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1]]
    wmyc_img = _make_tile_pattern_vdirection(
        width=width, height=height,
        h_tile_num=h_tile_num, v_tile_num=v_tile_num,
        low_level=low_level, high_level=high_level,
        color_list=color_list)

    # tpg.preview_image(np.hstack([wrgb_img, wmyc_img])/1023)
    return np.hstack([wrgb_img, wmyc_img])


def make_tile_pattern_sequence(cv_pair_list=PQ_10BIT_LOW_CV_PAIR_LIST):
    width = 3840
    height = 2160
    code_value_text_fmt = "low : ( {} / 1023 )\nhigh: ( {} / 1023 )"
    luminance_text_fmt = "low : {:.4g} nits\nhigh: {:.5g} nits"
    out_name_fmt = "./out_img/low_{}_high_{}.dpx"
    for cv_pair in cv_pair_list:
        # 画像作成
        low_level = (cv_pair[0], cv_pair[0], cv_pair[0])
        high_level = (cv_pair[1], cv_pair[1], cv_pair[1])
        img = make_tile_pattern_wwrgbmyc(
            width=width, height=height, h_tile_num=32, v_tile_num=18,
            low_level=low_level, high_level=high_level)

        # Code Value値をテキストとして付与
        text = code_value_text_fmt.format(cv_pair[0], cv_pair[1])
        merge_each_spec_text(img, pos=(50, 50), font_size=100,
                             text_img_size=(2000, 1920), text=text)

        # 絶対輝度値をテキストとして付与
        low_luminance = eotf_ST2084(cv_pair[0] / 1023)
        high_luminance = eotf_ST2084(cv_pair[1] / 1023)
        text = luminance_text_fmt.format(low_luminance, high_luminance)
        merge_each_spec_text(img, pos=(1920 + 50, 50), font_size=100,
                             text_img_size=(1870, 2000), text=text)

        # ファイル書き出し
        out_name = out_name_fmt.format(cv_pair[0], cv_pair[1])
        attr = {"oiio:BitsPerSample": 10}
        writer = TyWriter(img=img/0x3FF, fname=out_name, attr=attr)
        writer.write()


def make_primary_color_code_value_on_bt2020(color_space=cs.BT709):
    """
    任意の色域の RGB の Primary Colors が BT.2020 の
    Code Value値(10bit) を算出する。

    Returns
    -------
    array_like
        10bit code value array of the primary colors.

    Examples
    --------
    >>> make_primary_color_code_value_on_bt2020('ITU-R BT.709')
    array([  1.58232402e-13   1.00000000e+00])
    """
    primaries = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    in_cs = RGB_COLOURSPACES[color_space]
    out_cs = RGB_COLOURSPACES[cs.BT2020]
    primaries_on_bt2020 = RGB_to_RGB(
        RGB=primaries, input_colourspace=in_cs, output_colourspace=out_cs)

    # 範囲外はハードクリップ
    primaries_on_bt2020 = np.clip(primaries_on_bt2020, 0.0, 1.0)

    # PQ の Ref White(203nits) に正規化しつつ、PQ の OETF を適用
    primaries_on_bt2020 = primaries_on_bt2020 * PQ_REF_WHITE_NITS
    primaries_on_bt2020 = oetf_ST2084(primaries_on_bt2020)

    return convert_from_float_to_int_10bit(primaries_on_bt2020)


def make_primaries_tile_pattern_on_bt2020_half(
        inner_cs=cs.BT709, outer_cs=cs.P3_D65,
        width=1920, height=2160):
    """
    make_primaries_tile_pattern_on_bt2020 のサブルーチン。
    make_primaries_tile_pattern_on_bt2020 は画面の左右で
    色域を変えるため、この関数では画面の半分を作る。
    """
    inner_primaries = make_primary_color_code_value_on_bt2020(inner_cs)
    outer_primaries = make_primary_color_code_value_on_bt2020(outer_cs)
    img_buf = []
    for in_val, out_val in zip(inner_primaries, outer_primaries):
        temp = tpg.make_tile_pattern(
            width=width, height=height//3, h_tile_num=32, v_tile_num=12,
            low_level=in_val, high_level=out_val)
        img_buf.append(temp)

    img = np.vstack(img_buf)
    text = "chroma_low : {}\nchroma_high: {}".format(inner_cs, outer_cs)
    merge_each_spec_text(img, pos=(50, 50), font_size=100,
                         text_img_size=(1870, 1870), text=text)

    # tpg.preview_image(img / 1023)

    return img


def make_primaries_tile_pattern_on_bt2020(width=3840, height=2160):
    """
    2つの異なる Primary Colors に対するタイルパターンを作る。
    Primary Colors は **BT.2020** の Gamut にマッピングする。
    """
    left_img = make_primaries_tile_pattern_on_bt2020_half(
        inner_cs=cs.BT709, outer_cs=cs.P3_D65, width=width//2, height=height)
    right_img = make_primaries_tile_pattern_on_bt2020_half(
        inner_cs=cs.P3_D65, outer_cs=cs.BT2020, width=width//2, height=height)
    img = np.hstack([left_img, right_img])

    # tpg.preview_image(img / 1023)

    # ファイル書き出し
    out_name = "./out_img/chroma_tile_checker.dpx"
    attr = {"oiio:BitsPerSample": 10}
    writer = TyWriter(img=img/0x3FF, fname=out_name, attr=attr)
    writer.write()


def main_func():
    """
    Pixel 3a 用のパターンを作成する。
    """
    # low level, hight level checker
    # make_tile_pattern_sequence(PQ_10BIT_LOW_CV_PAIR_LIST)
    # make_tile_pattern_sequence(PQ_10BIT_HIGH_CV_PAIR_LIST)
    make_primaries_tile_pattern_on_bt2020(width=3840, height=2160)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
