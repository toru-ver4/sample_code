#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SRI International社の Inter Gamut パターンを改良して、
Gamut の境界を検出するパターンを作成する。
"""

# 外部ライブラリのインポート
import os
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from colour.models import eotf_ST2084, oetf_ST2084, RGB_to_RGB,\
    RGB_COLOURSPACES

# 自作ライブラリのインポート
from TyImageIO import TyWriter
import test_pattern_generator2 as tpg
import color_space as cs


class GamutEdgeChecker:
    """
    ## 概要

    Gamutのエッジをチェックする。

    ## 全体の処理の概要

    ```
    gamut_edge_checker = GamutEdgeChecker()
    gamut_edge_checker.make()
    gamut_edge_checker.preview()
    gamut_edge_checker.save()
    ```

    ## Gec.make() の概要

    ```
    def make(self):
        base_param = self.make_base_param()
        calc_param = CalcParameters()
        draw_param = calc_param.calc_parameters()

        draw_pattern = DrawPattern(draw_param, self.img)
        draw_pattern.draw_gamut_tile_pattern()

        draw_diagram = DrawChromaticityDiagram(draw_param, self.img)
        draw_diagram.draw_chromaticity_diagram()

        text_info = self.make_text_information()
        draw_information = DrawInformation(text_imfo, self.img)
        draw_information.draw_information()
    ```

    ## self.make_base_param() の吐き出す値

    ```
    typedef struct{
        double inner_primaries[3][2];  // 内側のxy色度座標
        double outer_primaries[3][2];  // 外側のxy色度座標
        int inner_sample_num;  // 内側の描画点の数
        int outer_sample_num;  // 外側の描画点の数
    }base_param;
    ```

    ## self.make_text_information() の吐き出す値

    ```
    typedef struct{
        int revision;
        char *outer_gamut_name;  // 外側の Gamut名
        char *inner_gamut_name;  // 内側の Gamut名
        double inner_primaries[3][2];  // 内側のxy色度座標
        double outer_primaries[3][2];  // 外側のxy色度座標
    }text_info;
    ```

    ## calc_param.calc_parameters() の吐き出す値

    typedef struct{
        inter
    }draw_param[12]  // 12 は RGBMYC とその中間点の意味。

    """




if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
