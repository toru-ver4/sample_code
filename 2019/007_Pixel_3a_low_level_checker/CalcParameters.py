#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gamutパターン作成に必要なxy座標の計算を行う。
"""

# 外部ライブラリのインポート
import numpy as np
from scipy import linalg

# 自作ライブラリのインポート
import test_pattern_generator2 as tpg
from sympy import Point, Segment, intersection
from colour import xy_to_xyY, xyY_to_XYZ, XYZ_to_RGB, RGB_to_XYZ, XYZ_to_xyY
from colour.models import BT2020_COLOURSPACE
import color_space as cs

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
        self.calc_large_y()
        self.calc_ref_xy()
        self.calc_inner_outer_xyY()
        self.calc_inner_ref_xyY()
        self.calc_outer_ref_xyY()

        ret_dict = {
            'inner_xyY': self.inner_xyY,
            'outer_xyY': self.outer_xyY,
            'inner_ref_xyY': self.inner_ref_xyY,
            'outer_ref_xyY': self.outer_ref_xyY,
            # 'ref_xy': self.ref_xy,
            # 'min_large_y': self.min_large_y
        }

        return ret_dict

    def calc_outer_ref_xyY(self):
        """
        外側領域のrefとなるxyY値を求める。
        """
        # outer_gamut --> inter_gamut の色域変換を実施して
        # 無理やり inner_gamut に貼り付けさせる
        src_primaries = self.base_param['outer_primaries'][:3]
        dst_primaries = self.base_param['inner_primaries'][:3]

        # 各種 Matrix を計算
        white = xyY_to_XYZ(xy_to_xyY(D65))
        xyz_to_rgb_matrix_2020 = BT2020_COLOURSPACE.XYZ_to_RGB_matrix
        rgb_to_xyz_matrix_2020 = BT2020_COLOURSPACE.RGB_to_XYZ_matrix
        outer_rgb_to_inner_rgb_mtx = self.calc_rgb_to_rgb_matrix(
            src_primaries, dst_primaries, white)
        inner_rgb_to_outer_rgb_mtx = self.calc_rgb_to_rgb_matrix(
            dst_primaries, src_primaries, white)

        # BT.2020 空間でRGBに戻す
        rgb_2020 = XYZ_to_RGB(
                xyY_to_XYZ(self.outer_xyY), D65, D65, xyz_to_rgb_matrix_2020)

        # BT.709 の RGB に変換してクリップする
        rgb_inner = cs.color_cvt(rgb_2020, outer_rgb_to_inner_rgb_mtx)
        clip_max_value = self.base_param['reference_white'] / 100.0
        rgb_inner_cliped = np.clip(rgb_inner, 0.0, clip_max_value)

        # クリップ後の値を BT.2020 空間での xyY 値に変換する
        rgb_2020_2 = cs.color_cvt(rgb_inner_cliped, inner_rgb_to_outer_rgb_mtx)
        large_xyz_inner = RGB_to_XYZ(
            rgb_2020_2, D65, D65, rgb_to_xyz_matrix_2020)
        self.outer_ref_xyY = XYZ_to_xyY(large_xyz_inner)

    def calc_rgb_to_rgb_matrix(self, src_primaries, dst_primaries, white):
        """
        色域変換のMatrixを計算。
        なお、白色点はD65固定とする。
        """
        rgb_to_xyz_matrix_src = cs.calc_rgb_to_xyz_matrix(src_primaries, white)
        rgb_to_xyz_matrix_dst = cs.calc_rgb_to_xyz_matrix(dst_primaries, white)
        xyz_to_rgb_matrix_dst = linalg.inv(rgb_to_xyz_matrix_dst)
        return xyz_to_rgb_matrix_dst.dot(rgb_to_xyz_matrix_src)

    def calc_inner_ref_xyY(self):
        self.calc_inner_ref_xyY = None
        inner_ref_xy = np.ones_like(self.inner_xy)\
            * self.ref_xy[:, np.newaxis, :]
        buf = []
        for hue_idx, xy in enumerate(inner_ref_xy):
            temp = xy_to_xyY(xy, self.min_large_y[hue_idx])
            buf.append(temp)
        self.inner_ref_xyY = np.array(buf)

    def calc_inner_outer_xyY(self):
        self.inner_xyY = self.calc_xyY(self.inner_xy)
        self.outer_xyY = self.calc_xyY(self.outer_xy)

    def calc_xyY(self, xy_array):
        """
        外側領域の xyY を求める。
        """
        buf = []
        for hue_idx, xy in enumerate(xy_array):
            temp = xy_to_xyY(xy, self.min_large_y[hue_idx])
            buf.append(temp)
        return np.array(buf)

    def calc_ref_xy(self):
        """
        正方形のタイルの模様の基準のxy値。
        とりあえず現状は inner_edge と同一で良いでしょ(適当)。
        """
        self.ref_xy = self.inner_edge.copy()

    def calc_large_y(self):
        """
        各 hue ごとの Y値を求める。
        一度、RGBに戻して、正規化して、またXYZに変換して、
        そのY値を抽出。
        """
        xyz_to_rgb_matrix = BT2020_COLOURSPACE.XYZ_to_RGB_matrix
        rgb_to_xyz_matrix = BT2020_COLOURSPACE.RGB_to_XYZ_matrix
        buf = []
        for edge_xy in self.outer_edge:
            large_xyz = xyY_to_XYZ(xy_to_xyY(edge_xy))
            rgb = XYZ_to_RGB(large_xyz, D65, D65, xyz_to_rgb_matrix)
            normalized_rgb = rgb / np.max(rgb)
            normalized_rgb[normalized_rgb < (10 ** -14)] = 0.0
            large_xyz = RGB_to_XYZ(normalized_rgb, D65, D65, rgb_to_xyz_matrix)
            xyY = XYZ_to_xyY(large_xyz)
            buf.append(xyY[2])
        self.min_large_y = np.array(buf)

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
        self.inner_edge = np.array(buf)

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
