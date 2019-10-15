#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SRI International社の Inter Gamut パターンを改良して、
Gamut の境界を検出するパターンを作成する。
"""

# 外部ライブラリのインポート
import os
import numpy as np
import cv2

# 自作ライブラリのインポート
import test_pattern_generator2 as tpg
import color_space as cs
import transfer_functions as tf
from DrawGamutPattern import DrawGamutPattern
from DrawChromaticityDiagram import DrawChromaticityDiagram
from DrawInformation import DrawInformation
from CalcParameters import CalcParameters

mini_primaries = [[0.45, 0.25], [0.30, 0.45], [0.25, 0.20], [0.45, 0.25]]

BASE_PARAM = {
    'revision': 1,
    'inner_sample_num': 4,
    'outer_sample_num': 4,
    'hue_devide_num': 4,
    'img_width': 1920,
    'img_height': 1080,
    'pattern_space_rate': 0.71,
    'inner_gamut_name': 'ITU-R BT.709',
    'outer_gamut_name': 'ITU-R BT.2020',
    'inner_primaries': np.array(tpg.get_primaries(cs.BT709)[0]),
    'outer_primaries': np.array(tpg.get_primaries(cs.BT2020)[0]),
    'transfer_function': tf.SRGB,
    'background_luminance': 2,
    'reference_white': 100
}


class GamutEdgeChecker:
    """
    ## 概要

    Gamutのエッジをチェックする。

    ## 全体の処理の概要

    ```
    gamut_edge_checker = GamutEdgeChecker(base_param)
    gamut_edge_checker.make()
    gamut_edge_checker.preview()
    gamut_edge_checker.save()
    ```

    ## Gec.make() の概要

    ```
    def make(self):
        self.make_base_layer()  // 大元の背景画像を準備

        calc_param = CalcParameters(self.base_param)
        draw_param = calc_param.calc_parameters()

        draw_pattern = DrawGamutPattern(draw_param, self.img)
        draw_pattern.draw_gamut_tile_pattern()

        draw_diagram = DrawChromaticityDiagram(draw_param, self.img)
        draw_diagram.draw_chromaticity_diagram()

        text_info = self.make_text_information()
        draw_information = DrawInformation(text_imfo, self.img)
        draw_information.draw_information()

        self.apply_oetf()
    ```

    ## self.make_base_param() の吐き出す値

    ```
    typedef struct{
        int revision;
        int inner_sample_num;  // 内側の描画点の数
        int outer_sample_num;  // 外側の描画点の数
        int hue_devide_num;  // 色相方向の分割数。原則4固定。
        int img_width;
        int img_height;
        char *inner_gamut_name;  // 内側の Gamut名
        char *outer_gamut_name;  // 外側の Gamut名
        double inner_primaries[3][3];  // 内側のxy色度座標
        double outer_primaries[3][3];  // 外側のxy色度座標
        char *transfer_function;  // OETF の指定
        int reference_white;  // ref white の設定。単位は [cd/m2]。
    }base_param;
    ```

    ## self.make_text_information() の吐き出す値

    ```
    typedef struct{
        int diagram_width;
        int diagram_height;
    }text_info;
    ```

    ## calc_param.calc_parameters() の吐き出す draw_param 値

    typedef struct{
        double inner_xyY[12][inner_sample_num][3];
        double outer_xyY[12][outer_sample_num][3];
        double innnr_ref_xyY[12][inner_sample_num][3];
        double outer_ref_xyY[12][inner_sample_num][3];
        double min_large_y[12];  // inner_xy, outer_xy の largeY最小値。
                                 // これに合わせて xyY to RGB 変換を行う
    }draw_param // 12 は 3(RGB) * 4(hue_devide_num) から算出
    """
    def __init__(self, base_param=BASE_PARAM):
        self.base_param = base_param

    def make(self):
        """
        画像生成
        """
        self.make_base_layer()
        calc_param = CalcParameters(self.base_param)
        draw_param = calc_param.calc_parameters()
        draw_pattern = DrawGamutPattern(self.base_param, draw_param, self.img)
        draw_pattern.draw_gamut_tile_pattern()
        draw_diagram = DrawChromaticityDiagram(
            self.base_param, draw_param, self.img)
        draw_diagram.draw_chromaticity_diagram()
        diagram_width, diagram_height =\
            draw_diagram.get_diagram_widgh_height()
        draw_information = DrawInformation(
            self.base_param, draw_param, self.img,
            diagram_width, diagram_height)
        draw_information.draw_information()
        self.apply_oetf()

    def int(self, x):
        return int(x + 0.5)

    def make_base_layer(self):
        """
        ベースとなる背景画像を生成。
        """
        # 大枠準備
        width = self.base_param['img_width']
        height = self.base_param['img_height']
        self.img = np.zeros((height, width, 3))

        # Gamut パターン配置場所のBG Colorを設定
        pattern_space_rate = self.base_param['pattern_space_rate']
        background_luminance = self.base_param['background_luminance']
        gamut_area_width = int(width * pattern_space_rate)
        bg_img = np.ones((height, gamut_area_width, 3)) * background_luminance
        self.img[:, :gamut_area_width, :] = bg_img

    def apply_oetf(self):
        oetf_name = self.base_param['transfer_function']
        self.img = tf.oetf_from_luminance(self.img, oetf_name)
        if np.sum(self.img > 1.0) > 0:
            print("warning. over flow")
        elif np.sum(self.img < 0.0) > 0:
            print("warning. under flow")
        self.img = np.clip(self.img, 0, 1)
        self.img = np.uint16(np.round(self.img * 0xFFFF))

    def preview(self):
        tpg.preview_image(self.img)

    def save(self):
        cv2.imwrite("test.tiff", self.img[:, :, ::-1])


def main_func():
    gamut_edge_checker = GamutEdgeChecker(base_param=BASE_PARAM)
    gamut_edge_checker.make()
    gamut_edge_checker.preview()
    gamut_edge_checker.save()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
