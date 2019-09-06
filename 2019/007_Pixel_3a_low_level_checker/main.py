#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Turbo を使ってHDR10信号の輝度マップを作成する
"""

# 外部ライブラリのインポート
import os
import numpy as np
import math
import cv2
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

# 自作ライブラリのインポート
from TyImageIO import TyWriter


PQ_10BIT_CV_PAIR_LIST = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64],
                         [0, 96], [0, 128], [0, 160], [0, 192], [0, 224]]


def preview_image(img, order='rgb', over_disp=False):
    """
    画像をプレビューする。何かキーを押すとウィンドウは消える。
    """
    if order == 'rgb':
        cv2.imshow('preview', img[:, :, ::-1])
    elif order == 'bgr':
        cv2.imshow('preview', img)
    else:
        raise ValueError("order parameter is invalid")

    if over_disp:
        cv2.resizeWindow('preview', )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def equal_devision(length, div_num):
    """
    # 概要
    length を div_num で分割する。
    端数が出た場合は誤差拡散法を使って上手い具合に分散させる。
    """
    base = length / div_num
    ret_array = [base for x in range(div_num)]

    # 誤差拡散法を使った辻褄合わせを適用
    # -------------------------------------------
    diff = 0
    for idx in range(div_num):
        diff += math.modf(ret_array[idx])[0]
        if diff >= 1.0:
            diff -= 1.0
            ret_array[idx] = int(math.floor(ret_array[idx]) + 1)
        else:
            ret_array[idx] = int(math.floor(ret_array[idx]))

    # 計算誤差により最終点が +1 されない場合への対処
    # -------------------------------------------
    diff = length - sum(ret_array)
    if diff != 0:
        ret_array[-1] += diff

    # 最終確認
    # -------------------------------------------
    if length != sum(ret_array):
        raise ValueError("the output of equal_division() is abnormal.")

    return ret_array


def make_tile_pattern(width=480, height=960, h_tile_num=4,
                      v_tile_num=4, low_level=(940, 940, 940),
                      high_level=(1023, 1023, 1023)):
    """
    タイル状の縞々パターンを作る
    """
    width_array = equal_devision(width, h_tile_num)
    height_array = equal_devision(height, v_tile_num)
    high_level = np.array(high_level, dtype=np.uint16)
    low_level = np.array(low_level, dtype=np.uint16)

    v_buf = []

    for v_idx, height in enumerate(height_array):
        h_buf = []
        for h_idx, width in enumerate(width_array):
            tile_judge = (h_idx + v_idx) % 2 == 0
            h_temp = np.zeros((height, width, 3), dtype=np.uint16)
            h_temp[:, :] = high_level if tile_judge else low_level
            h_buf.append(h_temp)

        v_buf.append(np.hstack(h_buf))
    img = np.vstack(v_buf)
    return img


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
    縦方向に4色分のパターンを作成する。
    """
    img_buf = []
    for v_idx, color in enumerate(color_list):
        low_temp = [low_level[idx] * color[idx] for idx in range(3)]
        high_temp = [high_level[idx] * color[idx] for idx in range(3)]
        temp = make_tile_pattern(
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

    # preview_image(np.hstack([wrgb_img, wmyc_img])/1023)
    return np.hstack([wrgb_img, wmyc_img])


def main_func():
    """
    Pixel 3a 用のパターンを作成する。
    """
    cv_pair_list = PQ_10BIT_CV_PAIR_LIST
    width = 3840
    height = 2160
    text_fmt = "low = ( {} / 1023 )\nhigh = ( {} / 1023 )"
    out_name_fmt = "./out_img/low_{}_high_{}.dpx"
    for cv_pair in cv_pair_list:
        text = text_fmt.format(cv_pair[0], cv_pair[1])
        low_level = (cv_pair[0], cv_pair[0], cv_pair[0])
        high_level = (cv_pair[1], cv_pair[1], cv_pair[1])
        img = make_tile_pattern_wwrgbmyc(
            width=width, height=height, h_tile_num=32, v_tile_num=18,
            low_level=low_level, high_level=high_level)
        merge_each_spec_text(img, pos=(50, 50), font_size=100,
                             text_img_size=(2000, 2000), text=text)
        out_name = out_name_fmt.format(cv_pair[0], cv_pair[1])
        attr = {"oiio:BitsPerSample": 10}
        writer = TyWriter(img=img/0x3FF, fname=out_name, attr=attr)
        writer.write()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
