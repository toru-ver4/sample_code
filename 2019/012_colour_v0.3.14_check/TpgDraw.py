#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# TpgDraw モジュール

## 概要
TestPattern作成時の各種描画関数をまとめたもの。
"""

import os
import numpy as np
import transfer_functions as tf
import color_space as cs
import test_pattern_generator2 as tpg
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import colour
import imp
imp.reload(tpg)


class TpgDraw:
    """
    テストパターン用の各種描画を行う。
    """
    def __init__(self, draw_param, preview):
        # TpgControl から受け通るパラメータ
        self.preview = preview
        self.bg_color = draw_param['bg_color']
        self.fg_color = draw_param['fg_color']
        self.img_width = draw_param['img_width']
        self.img_height = draw_param['img_height']
        self.bit_depth = draw_param['bit_depth']
        self.img_max = (2 ** self.bit_depth) - 1
        self.transfer_function = draw_param['transfer_function']
        self.color_space = draw_param['color_space']
        self.white_point = draw_param['white_point']
        self.revision = draw_param['revision']

        self.convert_to_8bit_coef = 2 ** (self.bit_depth - 8)
        self.convert_from_10bit_coef = 2 ** (16 - self.bit_depth)

        # TpgDraw 内部パラメータ(細かい座標など)
        self.ramp_height_coef = 0.079  # range is [0.0:1.0]
        # self.ramp_st_pos_h_coef = 0.442  # range is [0.0:1.0]
        self.ramp_st_pos_h_coef = 0.03  # range is [0.0:1.0]
        self.ramp_st_pos_v_coef = 0.235  # range is [0.0:1.0]
        self.checker_8bit_st_pos_v_coef = self.ramp_st_pos_v_coef + 0.115
        self.checker_10bit_st_pos_v_coef\
            = self.checker_8bit_st_pos_v_coef + 0.115
        self.each_spec_text_size_coef = 0.02  # range is [0.0:1.0]
        self.outline_text_size_coef = 0.02  # range is [0.0:1.0]
        self.step_bar_width_coef = 0.95
        self.step_bar_height_coef = 0.2
        self.step_bar_st_pos_v_coef = 0.74
        self.step_bar_text_width = 0.3

        self.color_checker_size_coef = 0.060
        self.color_checker_padding_coef = 0.0035
        self.color_checker_st_pos_v_coef = 0.1
        # self.color_checker_st_pos_h_coef = 0.03
        self.color_checker_st_pos_h_coef = 0.59

        self.csf_img_size_coef = 0.017
        self.csf_img_st_pos_v_coef = 0.03
        self.csf_img_low_st_pos_h_coef = 0.8
        self.csf_img_high_st_pos_h_coef = 0.842

        self.ycbcr_img_st_pos_v_coef = 0.03
        self.ycbcr_img_st_pos_h_coef = 0.862
        self.ycbcr_img_size_coef = 0.017

        self.dot_img_st_pos_v_coef = 0.03
        self.dot_img_st_pos_h_coef = 0.90

        self.info_text_st_pos_h_coef = 0.3
        self.info_text_st_pos_v_coef = 0.043
        self.info_text_size_coef = 0.3

        self.info2_text_st_pos_h_coef = 0.83
        self.info2_text_st_pos_v_coef = 0.03
        self.info2_text_size_coef = 0.02

        self.step_bar_width_coef_type2 = 0.8
        self.step_bar_height_coef_type2 = 0.09 * self.step_bar_width_coef_type2
        self.step_bar_st_pos_h_coef_type2 = 0.02
        self.step_bar_st_pos_v_coef_type2 = 0.17
        self.step_bar_v_space_coef_type2 = 0.24
        self.step_bar_text_width_type2 = 0.15

        self.set_fg_code_value()
        self.set_bg_code_value()

    def set_bg_code_value(self):
        code_value = tf.oetf_from_luminance(self.bg_color,
                                            self.transfer_function)
        code_value = int(round(code_value * self.img_max))
        self.bg_code_value = code_value

    def get_bg_code_value(self):
        return self.bg_code_value

    def set_fg_code_value(self):
        code_value = tf.oetf_from_luminance(self.fg_color,
                                            self.transfer_function)
        code_value = int(round(code_value * self.img_max))
        self.fg_code_value = code_value

    def get_fg_code_value(self):
        return self.fg_code_value

    def preview_iamge(self, img, order='rgb'):
        tpg.preview_image(img, order)

    def draw_bg_color(self):
        """
        背景色を描く。
        nits で指定したビデオレベルを使用
        """
        code_value = self.get_bg_code_value()
        self.img *= code_value

    def draw_outline(self):
        """
        外枠として1pxの直線を描く。
        nits で指定したビデオレベルを使用
        """
        code_value = self.get_fg_code_value()

        st_h, st_v = (0, 0)
        ed_h, ed_v = (self.img_width - 1, self.img_height - 1)

        self.img[st_v, st_h:ed_h, :] = code_value
        self.img[ed_v, st_h:ed_h, :] = code_value
        self.img[st_v:ed_v, st_h, :] = code_value
        self.img[st_v:ed_v, ed_h, :] = code_value

    def get_each_spec_text_height_and_size(self):
        """
        各パーツ説明のテキストの高さ(px)とフォントサイズを吐く
        """
        font_size = self.img_height * self.each_spec_text_size_coef
        text_height = font_size / 72 * 96 * 1.1

        return int(text_height), int(font_size)

    def get_info_text_height_and_size(self):
        """
        冒頭の解説用テキストの情報を取得
        """
        font_size = self.img_height * self.info_text_size_coef
        text_height = font_size / 72 * 96 * 1.1

        return int(text_height), int(font_size)

    def get_info2_text_height_and_size(self):
        """
        冒頭の解説用テキストの情報を取得
        """
        font_size = self.img_height * self.info2_text_size_coef
        text_height = font_size / 72 * 96 * 1.1

        return int(text_height), int(font_size)

    def get_color_bar_text_font_size(self, text_height):
        """
        カラーバー横に表示する階調＋輝度のフォントサイズを取得する
        """
        font_size = int(text_height / 96 * 72)
        return font_size

    def get_text_st_pos_for_over_info(self, tp_pos, text_height):
        return (tp_pos[0], tp_pos[1] - text_height)

    def get_fg_color_for_pillow(self):
        """
        Pillow 用 に 8bit精度のFG COLORを算出する
        """
        text_video_level_8bit\
            = int(self.fg_code_value / self.convert_to_8bit_coef)
        fg_color = tuple([text_video_level_8bit for x in range(3)])
        return fg_color

    def get_bg_color_for_pillow(self):
        """
        Pillow 用 に 8bit精度のFG COLORを算出する
        """
        text_video_level_8bit\
            = int(self.bg_code_value / self.convert_to_8bit_coef)
        bg_color = tuple([text_video_level_8bit for x in range(3)])
        return bg_color

    def convert_from_pillow_to_numpy(self, img):
        img = np.uint16(np.asarray(img)) * self.convert_to_8bit_coef

        return img

    def merge_text(self, txt_img, pos):
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
        bg_code_value\
            = self.get_bg_color_for_pillow() * self.convert_to_8bit_coef
        text_index = txt_img > bg_code_value[0]
        temp_img = self.img[st_pos_v:ed_pos_v, st_pos_h:ed_pos_h]
        temp_img[text_index] = txt_img[text_index]
        self.img[st_pos_v:ed_pos_v, st_pos_h:ed_pos_h] = temp_img

    def merge_each_spec_text(self, pos, font_size, text_img_size, text):
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
        fg_color = self.get_fg_color_for_pillow()
        bg_color = self.get_bg_color_for_pillow()
        txt_img = Image.new("RGB", (text_width, text_height), bg_color)
        draw = ImageDraw.Draw(txt_img)
        font = ImageFont.truetype("./fonts/NotoSansMonoCJKjp-Regular.otf",
                                  font_size)
        draw.text((0, 0), text, font=font, fill=fg_color)
        txt_img = self.convert_from_pillow_to_numpy(txt_img)

        self.merge_text(txt_img, pos)

    def draw_10bit_ramp(self):
        """
        10bitのRampパターンを描く
        """
        global g_cuurent_pos_v

        # パラメータ計算
        width = (self.img_height // 1080) * 1024
        height = int(self.img_height * self.ramp_height_coef)
        ramp_st_pos_h = self.get_ramp_st_pos_h(width)
        ramp_st_pos_v = int(self.img_height * self.ramp_st_pos_v_coef)
        ramp_pos = (ramp_st_pos_h, ramp_st_pos_v)
        text_height, font_size = self.get_each_spec_text_height_and_size()
        text_sub = " 10bit gray ramp (0, 1, 2, ..., 1021, 1022, 1023 level)."
        text = "▼" + text_sub
        text_pos = self.get_text_st_pos_for_over_info(ramp_pos, text_height)

        # ramp パターン作成
        ramp_10bit = tpg.gen_step_gradation(width=width, height=height,
                                            step_num=1025,
                                            bit_depth=self.bit_depth,
                                            color=(1.0, 1.0, 1.0),
                                            direction='h')
        tpg.merge(self.img, ramp_10bit, pos=ramp_pos)
        self.merge_each_spec_text(text_pos, font_size,
                                  (width, text_height), text)

    def get_bit_depth_checker_grad_width(self):
        """
        画面中央のグラデーション(256～768)の幅を求める
        """
        if self.img_height == 1080:
            width = 2048
        elif self.img_height == 2160:
            width = 4096
        else:
            raise ValueError('invalid img_height')

        return width

    def get_bit_depth_checker_grad_st_ed(self):
        """
        8bit/10bit チェック用のグラデーションは長めに作ってあるので
        最後にトリミングが必要となる。
        トリミングポイントの st, ed を返す
        """
        grad_width = self.get_bit_depth_checker_grad_width()
        grad_st_h = grad_width // 4
        grad_ed_h = grad_st_h + (grad_width // 2)

        return grad_st_h, grad_ed_h

    def get_ramp_st_pos_h(self, width):
        return int(self.img_width * self.ramp_st_pos_h_coef)

    def draw_8bit_10bit_checker(self, bit_depth='8bit', pos_v_coef=0.5):
        """
        256～768 の 8bit/10bitのRampパターンを描く
        """
        # パラメータ計算
        width = self.get_bit_depth_checker_grad_width()
        height = int(self.img_height * self.ramp_height_coef)
        grad_st_pos_h, grad_ed_pos_h = self.get_bit_depth_checker_grad_st_ed()
        width_after_trim = grad_ed_pos_h - grad_st_pos_h
        ramp_st_pos_h = self.get_ramp_st_pos_h(width_after_trim)
        ramp_st_pos_v = int(self.img_height * pos_v_coef)
        ramp_pos = (ramp_st_pos_h, ramp_st_pos_v)
        text_height, font_size = self.get_each_spec_text_height_and_size()
        text_8 = "8bit gray ramp (256, 260, 264, ..., 756, 760, 764 level)."
        text_10 = "10bit gray ramp (256, 257, 258, ..., 765, 766, 767 level)."
        text_sub = text_8 if bit_depth == '8bit' else text_10
        text = "▼ " + text_sub
        text_pos = self.get_text_st_pos_for_over_info(ramp_pos, text_height)

        # ramp パターン作成
        step_num = 257 if bit_depth == '8bit' else 1025
        ramp = tpg.gen_step_gradation(width=width, height=height,
                                      step_num=step_num,
                                      bit_depth=self.bit_depth,
                                      color=(1.0, 1.0, 1.0),
                                      direction='h')
        tpg.merge(self.img, ramp[:, grad_st_pos_h:grad_ed_pos_h],
                  pos=ramp_pos)
        self.merge_each_spec_text(text_pos, font_size,
                                  (width_after_trim, text_height), text)

    def get_color_bar_st_pos_h(self, width):
        return (self.img_width - width) // 2

    def draw_wrgbmyc_color_bar(self):
        """
        階段状のカラーバーをプロットする
        """
        scale_step = 65
        color_list = [(1, 1, 1), (1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                      (1, 0, 1), (1, 1, 0), (0, 1, 1)]
        width = int(self.img_width * self.step_bar_width_coef)
        height = int(self.img_height * self.step_bar_height_coef)
        color_bar_st_pos_h = self.get_color_bar_st_pos_h(width)
        color_bar_st_pos_v = int(self.img_height * self.step_bar_st_pos_v_coef)
        st_pos = (color_bar_st_pos_h, color_bar_st_pos_v)

        bar_height_list = tpg.equal_devision(height, len(color_list))
        bar_img_list = []
        for color, bar_height in zip(color_list, bar_height_list):
            color_bar = tpg.gen_step_gradation(width=width, height=bar_height,
                                               step_num=scale_step,
                                               bit_depth=self.bit_depth,
                                               color=color, direction='h')
            bar_img_list.append(color_bar)
        color_bar = np.vstack(bar_img_list)
        tpg.merge(self.img, color_bar, st_pos)

        # ここからテキスト。あらかじめV方向で作っておき、最後に回転させる
        txt_img = self.get_video_level_text_img(scale_step, width)
        text_pos = self.get_text_st_pos_for_over_info(st_pos, txt_img.shape[0])
        self.merge_text(txt_img, text_pos)

        # 説明文を下に追加する
        text_pos_v = st_pos[1] + color_bar.shape[0]
        text_pos = (st_pos[0], text_pos_v)
        text_height, font_size = self.get_each_spec_text_height_and_size()
        level_text = " (0, 16, 32, ..., 992, 1008, 1023 Level)"
        text = "▲ WRGBMYC Color Gradation" + level_text
        self.merge_each_spec_text(text_pos, font_size,
                                  (width, text_height), text)

    def get_video_level_text_img(self, scale_step, width, type=1):
        """
        ステップカラーに付与する VideoLevel & Luminance 情報。
        最初は縦向きで作って、最後に横向きにする
        """
        fg_color = self.get_fg_color_for_pillow()
        text_height_list = tpg.equal_devision(width, scale_step)
        font_size = self.get_color_bar_text_font_size(width / scale_step)
        video_level = np.linspace(0, 2 ** self.bit_depth, scale_step)
        video_level[-1] -= 1
        video_level_float = video_level / self.img_max
        bright_list = tf.eotf_to_luminance(video_level_float,
                                           self.transfer_function)
        if type == 1:
            text_width = int(self.step_bar_text_width * self.img_height)
        elif type == 2:
            text_width = int(self.step_bar_text_width_type2 * self.img_height)
        else:
            raise ValueError('invalid tpg-type')
        txt_img = Image.new("RGB", (text_width, width), (0x00, 0x00, 0x00))
        draw = ImageDraw.Draw(txt_img)
        font = ImageFont.truetype("./fonts/NotoSansMonoCJKjp-Regular.otf",
                                  font_size)
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

        txt_img = self.convert_from_pillow_to_numpy(txt_img)
        txt_img = np.rot90(txt_img)

        return txt_img

    def get_color_checker_rgb_value(self):
        """
        24パターンの Color Checker の RGB値を得る
        """
        colour_checker_param\
            = colour.CCS_COLOURCHECKERS.get('ColorChecker 2005')

        # 今回の処理では必要ないデータもあるので xyY と whitepoint だけ抽出
        # -------------------------------------------------------------
        # _name, data, whitepoint = colour_checker_param
        data = colour_checker_param.data
        whitepoint = colour_checker_param.illuminant
        temp_xyY = []
        for key in data.keys():
            temp_xyY.append(data[key])
        temp_xyY = np.array(temp_xyY)
        large_xyz = colour.models.xyY_to_XYZ(temp_xyY)

        # rgb_white_point\
        #     = colour.colorimetry.CCS_ILLUMINANTS['cie_2_1931'][self.white_point]

        # illuminant_XYZ = whitepoint   # ColorCheckerのオリジナルデータの白色点
        # illuminant_RGB = rgb_white_point  # RGBの白色点を設定
        # chromatic_adaptation_transform = 'CAT02'
        # large_xyz_to_rgb_matrix\
        #     = cs.get_xyz_to_rgb_matrix(self.color_space.name)
        # large_xyz_to_rgb_matrix = self.color_space.XYZ_to_RGB_matrix
        # rgb = colour.models.XYZ_to_RGB(large_xyz, illuminant_XYZ,
        #                                illuminant_RGB,
        #                                large_xyz_to_rgb_matrix,
        #                                chromatic_adaptation_transform)
        rgb = cs.large_xyz_to_rgb(
            xyz=large_xyz, color_space_name=self.color_space.name,
            xyz_white=whitepoint)

        # overflow, underflow check
        # -----------------------------
        rgb[rgb < 0.0] = 0.0
        rgb[rgb > 1.0] = 1.0

        point_100nits = 100 / tf.PEAK_LUMINANCE[self.transfer_function]
        # point_100nits = 203 / tf.PEAK_LUMINANCE[self.transfer_function]
        rgb = tf.oetf(rgb * point_100nits, self.transfer_function)
        rgb = np.uint16(np.round(rgb * self.img_max))

        return rgb

    def draw_color_checker(self):
        # 基本パラメータ算出
        # --------------------------------------
        h_num = 6
        v_num = 4
        patch_width = int(self.color_checker_size_coef * self.img_width)
        patch_height = patch_width
        patch_space = int(self.color_checker_padding_coef * self.img_width)
        img_width = patch_width * h_num + patch_space * (h_num - 1)
        img_height = patch_height * v_num + patch_space * (v_num - 1)
        rgb = self.get_color_checker_rgb_value()
        st_pos_h = int(self.color_checker_st_pos_h_coef * self.img_width)
        st_pos_v = int(self.color_checker_st_pos_v_coef * self.img_height)
        pos = (st_pos_h, st_pos_v)
        text = "▼ Color Checker"
        text_height, font_size = self.get_each_spec_text_height_and_size()
        text_pos = self.get_text_st_pos_for_over_info(pos, text_height)

        # 24ループで1枚の画像に24パッチを描画
        # -------------------------------------------------
        img_all_patch = np.zeros((img_height, img_width, 3))
        for idx in range(h_num * v_num):
            v_idx = idx // h_num
            h_idx = (idx % h_num)
            patch = np.ones((patch_height, patch_width, 3))
            patch[:, :] = rgb[idx]
            st_h = (patch_width + patch_space) * h_idx
            st_v = (patch_height + patch_space) * v_idx
            img_all_patch[st_v:st_v+patch_height, st_h:st_h+patch_width]\
                = patch

        tpg.merge(self.img, img_all_patch, pos)
        self.merge_each_spec_text(text_pos, font_size,
                                  (img_width, text_height), text)

    def draw_info_text(self):
        text_list = ["■ Information",
                     "    OETF/EOTF: {}".format(self.transfer_function),
                     "    White Point: {}".format(self.white_point),
                     "    Gamut: {}".format(self.color_space.name),
                     "    Revision: {:02d}".format(self.revision)]
        text_height, font_size = self.get_each_spec_text_height_and_size()
        st_pos_h = int(self.info_text_st_pos_h_coef * self.img_width)
        st_pos_v = int(self.info_text_st_pos_v_coef * self.img_height)
        for text in text_list:
            pos = (st_pos_h, st_pos_v)
            # text_img_size = (self.img_width - st_pos_h, text_height)
            text_img_size = (self.img_width // 4, text_height)
            self.merge_each_spec_text(pos, font_size, text_img_size, text)
            st_pos_v += text_height

    def draw_info_text_type2(self):
        text_list = ["■ Information",
                     "    OETF/EOTF: {}".format(self.transfer_function),
                     "    Revison: {:02d}".format(self.revision)]
        text_height, font_size = self.get_info2_text_height_and_size()
        st_pos_h = int(self.info2_text_st_pos_h_coef * self.img_width)
        st_pos_v = int(self.info2_text_st_pos_v_coef * self.img_height)
        for text in text_list:
            pos = (st_pos_h, st_pos_v)
            text_img_size = (self.img_width - st_pos_h, text_height)
            self.merge_each_spec_text(pos, font_size, text_img_size, text)
            st_pos_v += text_height

    def draw_dot_mesh(self):
        """
        dot mesh パターンを描画。
        """
        # UHD画像の場合はトリミングを行う
        triming = True if self.img_height != 1080 else False

        kind_num = 2 if self.img_height == 1080 else 3
        fg_color_list = [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        fg_color_list = np.uint16(np.array(fg_color_list) * self.img_max)
        bg_color = np.uint16(np.array([self.get_bg_code_value()
                                       for x in range(3)]))

        args = {'kind_num': kind_num, 'whole_repeat': 1,
                'bg_color': bg_color}
        dot_img_list = []
        for idx in range(4):
            args['fg_color'] = fg_color_list[idx]
            temp_img = tpg.complex_dot_pattern(**args)
            if triming:
                length = (temp_img.shape[0] // 3) * 2
                temp_img = temp_img[:length, :length, :]
            dot_img_list.append(temp_img)
        dot_img = np.hstack(dot_img_list)
        st_pos_h = int(self.dot_img_st_pos_h_coef * self.img_width)
        st_pos_v = int(self.dot_img_st_pos_v_coef * self.img_height)
        dot_img_pos = (st_pos_h, st_pos_v)
        tpg.merge(self.img, dot_img, pos=dot_img_pos)

    def draw_csf_pattern(self):
        """
        Limited Range 確認用のパターンを追加。
        """
        lv_0 = np.array([0, 0, 0], dtype=np.uint16)
        lv_64 = np.array([64, 64, 64], dtype=np.uint16)
        lv_940 = np.array([940, 940, 940], dtype=np.uint16)
        lv_1023 = np.array([1023, 1023, 1023], dtype=np.uint16)
        width = int(self.csf_img_size_coef * self.img_width)
        num = 2
        low_img = tpg.make_csf_color_image(width, width, lv_0, lv_64, num)
        high_img = tpg.make_csf_color_image(width, width, lv_940, lv_1023, num)

        high_img = tpg.make_tile_pattern(width=width, height=width,
                                         h_tile_num=4, v_tile_num=4,
                                         low_level=lv_940, high_level=lv_1023)
        low_img = tpg.make_tile_pattern(width=width, height=width,
                                        h_tile_num=4, v_tile_num=4,
                                        low_level=lv_0, high_level=lv_64)

        st_pos_h = int(self.csf_img_low_st_pos_h_coef * self.img_width)
        st_pos_v = int(self.csf_img_st_pos_v_coef * self.img_height)
        csf_img_pos = (st_pos_h, st_pos_v)
        tpg.merge(self.img, low_img, csf_img_pos)

        st_pos_h = int(self.csf_img_high_st_pos_h_coef * self.img_width)
        csf_img_pos = (st_pos_h, st_pos_v)
        tpg.merge(self.img, high_img, csf_img_pos)
        # tpg.preview_image(csf_img_pos)

    def draw_ycbcr_err_checker(self):
        width = int(self.ycbcr_img_size_coef * self.img_width)
        img = tpg.make_ycbcr_checker(height=width, v_tile_num=4)

        st_pos_h = int(self.ycbcr_img_st_pos_h_coef * self.img_width)
        st_pos_v = int(self.ycbcr_img_st_pos_v_coef * self.img_height)
        ycbcr_img_pos = (st_pos_h, st_pos_v)
        tpg.merge(self.img, img, ycbcr_img_pos)

    def draw_tpg_type1(self):
        """
        Color Checker付きの基本的なパターンを描画。
        """
        self.img = np.ones((self.img_height, self.img_width, 3),
                           dtype=np.uint16)
        self.draw_bg_color()
        self.draw_outline()
        self.draw_10bit_ramp()
        self.draw_8bit_10bit_checker('8bit', self.checker_8bit_st_pos_v_coef)
        self.draw_8bit_10bit_checker('10bit', self.checker_10bit_st_pos_v_coef)
        self.draw_wrgbmyc_color_bar()
        self.draw_color_checker()
        self.draw_dot_mesh()
        self.draw_csf_pattern()
        self.draw_ycbcr_err_checker()
        self.draw_info_text()

        if self.preview:
            self.preview_iamge(self.img / self.img_max)

        return self.img

    def draw_wrgbmyc_color_bar_type2(self):
        """
        階段状のカラーバーをプロットする
        """
        scale_step = 257
        color_list = [(1, 1, 1)]
        width = int(self.img_width * self.step_bar_width_coef_type2) * 4
        height = int(self.img_height * self.step_bar_height_coef_type2)
        vspace = int(self.img_height * self.step_bar_v_space_coef_type2)
        color_bar_st_pos_h = int(self.img_width
                                 * self.step_bar_st_pos_h_coef_type2)
        color_bar_st_pos_v = int(self.img_height
                                 * self.step_bar_st_pos_v_coef_type2)
        st_pos = (color_bar_st_pos_h, color_bar_st_pos_v)

        bar_height_list = tpg.equal_devision(height, len(color_list))
        bar_img_list = []
        for color, bar_height in zip(color_list, bar_height_list):
            color_bar = tpg.gen_step_gradation(width=width, height=bar_height,
                                               step_num=scale_step,
                                               bit_depth=self.bit_depth,
                                               color=color, direction='h')
            bar_img_list.append(color_bar)
        color_bar = np.vstack(bar_img_list)

        # テキストデータ作成
        txt_img = self.get_video_level_text_img(scale_step, width, type=2)

        # 最終データだけ削除（後で4分割できるように）
        width = int(round(width/257*256))
        color_bar = color_bar[:, :width, :]
        txt_img = txt_img[:, :width, :]

        for idx in range(4):
            img_pos_h = (width // 4) * idx
            img_merge_width = (width // 4)
            color_bar_temp = color_bar[:, img_pos_h:img_pos_h+img_merge_width]
            txt_img_temp = txt_img[:, img_pos_h:img_pos_h+img_merge_width]
            tpg.merge(self.img, color_bar_temp, st_pos)
            text_pos = self.get_text_st_pos_for_over_info(st_pos,
                                                          txt_img.shape[0])
            self.merge_text(txt_img_temp, text_pos)
            st_pos = (st_pos[0], st_pos[1] + vspace)

    def draw_tpg_type2(self):
        """
        グレースケールを4stepで描画した画像
        """
        self.img = np.ones((self.img_height, self.img_width, 3),
                           dtype=np.uint16)
        self.draw_bg_color()
        self.draw_outline()

        self.draw_info_text_type2()
        self.draw_wrgbmyc_color_bar_type2()
        if self.preview:
            self.preview_iamge(self.img / self.img_max)
        return self.img


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
