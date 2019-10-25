#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
動画の管理。
"""
import os
import shutil
import subprocess


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
        start_offset = self.each_patch_sec // 2
        each_patch_offset = self.each_patch_sec
        cmd_base = "./bin/ffmpeg -ss {:d} -i ./movie/sdr_movie.webm "
        cmd_base += "-vframes 1 -f image2 "
        cmd_base += "./after_frame/frame_{:03d}_grid_{:02d}.tiff -y"
        for p_idx in range(self.patch_frame_num):
            start_second = each_patch_offset * p_idx + start_offset
            cmd = cmd_base.format(start_second, p_idx, self.grid_num)
            print(cmd)
            subprocess.run(cmd.split(" "))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
