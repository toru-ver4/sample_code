#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gamutパターン作成に必要なxy座標の計算を行う。
"""

# 外部ライブラリのインポート
import numpy as np

# 自作ライブラリのインポート
import test_pattern_generator2 as tpg
import transfer_functions as tf
from sympy import Point, Segment, Line, intersection

# define
D65 = tpg.D65_WHITE

class CalcParameters:
    """
    typedef struct{
        self.outer_edge[12];
        self.inner_edge[12];
    }
    self.calc_outer_edge()
    self.calc_intersection()
    self.calc_inner_edge()
    """
    def __init__(self, base_param):
        self.base_param = base_param

    def calc_parameters(self):
        self.calc_outer_edge()
        self.calc_inner_edge()
        print(self.inner_edge)

    def calc_inner_edge(self):
        """
        内側の Gamut の端点を求める。
        R, Y, G, C, B, M の順序で計算して配列に入れる。
        """
        primary_points = [Point(x) for x in self.base_param['inner_primaries']]
        primary_segments = [Segment(primary_points[0], primary_points[1]),
                            Segment(primary_points[1], primary_points[2]),
                            Segment(primary_points[2], primary_points[0])]
        d65_point = Point(D65)
        buf = []
        for outer_edge in self.outer_edge:
            outer_edge_point = Point(outer_edge)
            segment_from_d65 = Segment(d65_point, outer_edge_point)

            # Primaryの三角形の3辺に対して総当りで交点を調べる
            temp = None
            for p_idx, primary_segment in enumerate(primary_segments):
                cross_point = intersection(segment_from_d65, primary_segment)
                if cross_point:
                    cross_point = cross_point[0].evalf()
                    temp = np.array(
                        [cross_point[0], cross_point[1]], dtype=np.float)
                    break
            if temp is None:
                print("error, intersection wa not found.")
            buf.append(temp)
        self.inner_edge = buf

    def calc_outer_edge(self):
        """
        外側の Gamut の端点を求める。
        R, Y, G, C, B, M の順序で計算して配列に入れる。
        """
        hue_divide_num = self.base_param['hue_devide_num']
        primaries = self.base_param['outer_primaries']
        buf = []
        for color_idx in range(3):
            # R, G, B のループ
            diff = (primaries[color_idx + 1] - primaries[color_idx])\
                / hue_divide_num
            base_xy = primaries[color_idx]
            for div_idx in range(hue_divide_num):
                # 色度図の三角形の一辺の中のループ
                temp = base_xy + diff * div_idx
                buf.append(temp)

        self.outer_edge = np.array(buf)


if __name__ == '__main__':
    pass
