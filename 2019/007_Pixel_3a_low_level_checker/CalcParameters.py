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
        self.calc_inner_xy()
        self.calc_outer_xy()

        ret_dict = {
            'inner_xy': self.inner_xy,
            'outer_xy': self.outer_xy,
            'ref_xy': None,
            'min_large_y': None
        }

        return ret_dict

    def calc_inner_xy(self):
        """
        D65(中心)から inner_edge への xy値を求める
        """
        sample_num = self.base_param['inner_sample_num']
        src = D65
        buf = []
        for dst in self.inner_edge:
            temp = self.linear_interpolation(src, dst, sample_num + 1)[1:]
            buf.append(temp)

        self.inner_xy = np.array(buf)

    def calc_outer_xy(self):
        """
        inner_edge から outer_edge への xy値を求める
        """
        sample_num = self.base_param['outer_sample_num']
        buf = []
        for hue_idx, dst in enumerate(self.outer_edge):
            src = self.inner_edge[hue_idx]
            temp = self.linear_interpolation(src, dst, sample_num + 1)[1:]
            buf.append(temp)

        self.outer_xy = np.array(buf)

    def linear_interpolation(self, st_xy, ed_xy, sample_num):
        """
        xy座標の単純な線形補間を実施。
        """
        x = np.linspace(st_xy[0], ed_xy[0], sample_num)
        y = np.linspace(st_xy[1], ed_xy[1], sample_num)
        ret_value = np.dstack((x, y)).reshape((sample_num, 2))
        return ret_value

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
                print("error, intersection was not found.")
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
