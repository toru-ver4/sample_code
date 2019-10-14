#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GamutPattern の描画を担当
"""

# 外部ライブラリのインポート
import os
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from colour.models import eotf_ST2084, oetf_ST2084, RGB_to_RGB,\
    BT2020_COLOURSPACE, xy_to_xyY, xyY_to_XYZ, XYZ_to_RGB, RGB_to_RGB,\
    BT709_COLOURSPACE

# 自作ライブラリのインポート
import test_pattern_generator2 as tpg
import color_space as cs

# 位置パラメータ。原則、width に乗算してピクセル値に変換する
global_tb_margin = 0.05
global_left_margin = 0.02
global_right_margin = 0.02
each_size = 0.045
D65 = tpg.D65_WHITE
tile_num = 5


class DrawGamutPattern:
    """
    ## cordinate_base_param の詳細

    ```
    typedef struct{
        double global_top_bottom_margin = 0.03;  // 上下マージン
        double global_left_margin = 0.1;  // 少し情報を書くので多めに確保
        double global_right_margin = 0.02;
        double each_size = 0.05; // 1つ1つの正方形のサイズ
    }cordinate_base_param;
    ```
    """
    def __init__(self, base_param, draw_param, img):

        """
        <--------------- global_width ----------->
                   <-- rectangle_area_width -->
        +----------------------------------------+----------------+
        |                                        |                |
        |          +----+                +----+  |                |  ^
        |          |    |                |    |  |                |  |
        |          +----+                +----+  |                |  |
        |                                        |        rectangle_area_height
        |          +----+                        |                |  |
        |          |    |                        |                |  |
        |          +----+                        |                |  |
        |                                        |                |  |
        |          +----+                        |                |  |
        |          |    |                        |                |  |
        |          +----+                        |                |  v
        |                                        |                |
        +----------------------------------------+----------------+
        """
        self.base_param = base_param
        self.draw_param = draw_param
        self.img = img
        self.img_width = self.get_img_width()
        self.img_height = self.get_img_height()
        self.xy_array = np.append(self.draw_param['inner_xy'],
                                  self.draw_param['outer_xy'],
                                  axis=1)
        self.calc_plot_parameters()

        # debug area
        idx = 8
        large_y = self.draw_param['min_large_y'][idx]
        large_xyz = xyY_to_XYZ(xy_to_xyY(self.xy_array[idx], large_y))
        xyz_to_rgb_matrix = BT2020_COLOURSPACE.XYZ_to_RGB_matrix
        src_rgb = XYZ_to_RGB(large_xyz, D65, D65, xyz_to_rgb_matrix)
        dst_rgb = RGB_to_RGB(src_rgb, BT2020_COLOURSPACE, BT709_COLOURSPACE)
        # print(src_rgb * 100)
        print(dst_rgb)
        for idx in range(8):
            print(dst_rgb[idx] / np.max(dst_rgb, axis=-1)[idx])
        # end of debug area

    def int(self, x):
        return int(x + 0.5)

    def get_img_width(self):
        return self.img.shape[1]

    def get_img_height(self):
        return self.img.shape[0]

    def draw_gamut_tile_pattern(self):
        print("sample implement.")
        luminance = self.base_param['reference_white']
        for hue_idx in range(self.get_hue_idx_num()):
            ref_xy = self.draw_param['ref_xy']
            ref_rgb = self.calc_rgb(ref_xy[hue_idx], hue_idx)
            rgb_array = self.calc_rgb(self.xy_array[hue_idx], hue_idx)

            for sat_idx in reversed(range(self.get_sat_idx_num())):
                temp_img = tpg.make_tile_pattern2(
                    width=self.rect_width, height=self.rect_height,
                    h_tile_num=tile_num, v_tile_num=tile_num,
                    low_level=ref_rgb * luminance,
                    high_level=rgb_array[sat_idx] * luminance)
                tpg.merge(self.img, temp_img,
                          self.rect_pos_list[hue_idx, sat_idx])

    def calc_rgb(self, xy, hue_idx):
        xyz_to_rgb_matrix = BT2020_COLOURSPACE.XYZ_to_RGB_matrix
        min_large_y = self.draw_param['min_large_y'][hue_idx]
        large_xyz = xyY_to_XYZ(xy_to_xyY(xy, min_large_y))
        ref_rgb = XYZ_to_RGB(large_xyz, D65, D65, xyz_to_rgb_matrix)
        return ref_rgb

    def calc_plot_parameters(self):
        """
        プロット位置やプロット幅などの情報を計算
        """
        self.rectangle_area_st_pos = self.calc_rectangle_area_st_pos()
        self.rect_width = self.int(each_size * self.img_width)
        self.rect_height = self.rect_width
        self.local_lr_margin = self.calc_local_lr_margin(
            width=self.calc_rectangle_area_width(),
            rect_width=self.rect_width,
            rect_h_num=self.get_hue_idx_num()
        )
        self.local_tb_margin = self.calc_local_tb_margin(
            height=self.calc_rectangle_area_height(),
            rect_height=self.rect_height,
            rect_v_num=self.get_sat_idx_num()
        )
        self.rect_pos_list = self.calc_each_rect_st_pos()

    def calc_each_rect_st_pos(self):
        pos_list = []
        for hue_idx in range(self.get_hue_idx_num()):
            temp_buf = []
            h_st = self.rectangle_area_st_pos[0]\
                + (self.rect_width + self.local_lr_margin) * hue_idx
            v_st = self.rectangle_area_st_pos[1]
            for sat_idx in range(self.get_sat_idx_num()):
                h_st_temp = h_st
                v_st_temp = v_st\
                    + (self.rect_height + self.local_tb_margin) * sat_idx
                temp_buf.append((h_st_temp, v_st_temp))
            pos_list.append(temp_buf)
        return np.array(pos_list, dtype=np.uint16)

    def get_hue_idx_num(self):
        """
        色相方向のサンプル数を取得する。
        """
        return 3 * self.base_param['hue_devide_num']

    def get_sat_idx_num(self):
        """
        彩度方向のサンプル数を取得する。
        """
        return self.base_param['inner_sample_num']\
            + self.base_param['outer_sample_num']

    def calc_local_lr_margin(self, width, rect_width, rect_h_num):
        """
        正方形を並べるエリアの H方向の正方形と正方形の間の
        マージンを求める。

        ```
        width = \
            rect_width * rect_h_num + margin * (rect_h_num - 1)
        margin = \
            (width - rect_width * rect_h_num) / (rect_h_num - 1)
        ```
        """
        return self.int((width - rect_width * rect_h_num) / (rect_h_num - 1))

    def calc_local_tb_margin(self, height, rect_height, rect_v_num):
        """
        正方形を並べるエリアの V方向の正方形と正方形の間の
        マージンを求める。

        ```
        height = \
            rect_height * rect_v_num + margin * (rect_vnum - 1)
        margin = \
            (height - rect_height * rect_v_num) / (rect_vnum - 1)
        ```
        """
        return self.int((height - rect_height * rect_v_num) / (rect_v_num - 1))

    def calc_rectangle_area_width(self):
        pattern_space_rate = self.base_param['pattern_space_rate']
        global_width = self.int(pattern_space_rate * self.img_width)
        right_margin = self.int(global_right_margin * self.img_width)
        left_margin = self.int(global_left_margin * self.img_width)
        return global_width - right_margin - left_margin

    def calc_rectangle_area_height(self):
        return self.img_height\
            - self.int(global_tb_margin * self.img_width) * 2

    def calc_rectangle_area_st_pos(self):
        """
        タイルパターンのプロット開始位置(左上)の
        座標を計算する。
        pos = (x, y) の順序ね。
        """
        x = self.int(global_left_margin * self.img_width)
        y = self.int(global_tb_margin * self.img_width)
        return (x, y)


if __name__ == '__main__':
    pass
