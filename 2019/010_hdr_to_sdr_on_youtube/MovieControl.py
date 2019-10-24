#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
動画の管理。
"""
import os
import shutil


class MovieControl:
    def __init__(self, base_param, ctrl_param):
        self.img_width = base_param['img_width']
        self.img_height = base_param['img_height']
        self.patch_size = base_param['patch_size']
        self.grid_num = base_param['grid_num']
        self.name_base = base_param['patch_file_name']
        self.patch_frame_num = ctrl_param['frame_num']
        self.frame_rate = 24
        self.each_patch_sec = 2  # unit is [sec]

    def make_sequence(self):
        counter = 0
        dst_base = "./sequence/grid_{:03d}_sequence_{:06d}.tiff"
        for p_idx in range(self.patch_frame_num):
            src = self.name_base.format(p_idx, self.grid_num)
            for f_idx in range(self.frame_rate * self.each_patch_sec):
                dst = dst_base.format(self.grid_num, counter)
                print(src, dst)
                shutil.copyfile(src=src, dst=dst)
                counter += 1

    def parse_sequence(self):
        print("parse_sequence is not implemented.")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
