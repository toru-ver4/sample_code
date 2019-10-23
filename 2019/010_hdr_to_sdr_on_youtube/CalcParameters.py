#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
パラメータ計算する。
"""
import os
import numpy as np
import lut


class CalcParameters:
    def __init__(self, base_param):
        self.img_width = base_param['img_width']
        self.img_height = base_param['img_height']
        self.patch_size = base_param['patch_size']
        self.grid_num = base_param['grid_num']

    def calc(self):
        self.calc_h_v_num()
        self.calc_coordinate()
        self.calc_frame_num()
        self.calc_patch_rgb()

        ctrl_param = {
            "frame_num": self.frame,
            "coord": self.coord,
            "rgb": self.rgb,
            'v_num': self.v_num,
            'h_num': self.h_num
        }   

        return ctrl_param

    def calc_h_v_num(self):
        """
        1枚の画像に対する、H方向、V方向のパッチ数を求める。
        """
        self.h_num = self.img_width // self.patch_size
        self.v_num = self.img_height // self.patch_size

    def calc_coordinate(self):
        """
        各パッチの開始座標(左上)を算出する
        """
        x = np.arange(self.h_num) * self.patch_size
        y = np.arange(self.v_num) * self.patch_size
        xx, yy = np.meshgrid(x, y)
        xy = np.dstack([yy, xx])
        self.coord = xy

    def calc_frame_num(self):
        """
        3DLUTの全格子点数のパッチ生成に必要なフレーム数の算出
        """
        total_patch = self.grid_num ** 3
        self.frame = int(round(total_patch / (self.h_num * self.v_num) + 0.5))

    def calc_patch_rgb(self):
        rgb = lut.get_3d_grid_cube_format(grid_num=self.grid_num)
        self.rgb = np.zeros((self.frame * self.h_num * self.v_num, 3))
        self.rgb[:rgb.shape[0], :] = rgb
        self.rgb = self.rgb.reshape((self.frame, self.v_num, self.h_num, 3))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
