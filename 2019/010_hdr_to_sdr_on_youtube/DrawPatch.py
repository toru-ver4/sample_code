#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
パッチをモリモリ描画する
"""
import os
import numpy as np
import cv2
from colour.models import oetf_ST2084


class DrawPatch:
    def __init__(self, base_param, draw_param):
        self.img_width = base_param['img_width']
        self.img_height = base_param['img_height']
        self.patch_size = base_param['patch_size']
        self.grid_num = base_param['grid_num']
        self.frame_num = draw_param['frame_num']
        self.coord = draw_param['coord']
        self.rgb = draw_param['rgb']
        self.v_num = draw_param['v_num']
        self.h_num = draw_param['h_num']

        self.name_base = "./base_frame/frame_{:03d}_grid_{:02d}.tiff"

    def preview_image(self, img, order='rgb', over_disp=False):
        if order == 'rgb':
            cv2.imshow('preview', img[:, :, ::-1])
        elif order == 'bgr':
            cv2.imshow('preview', img)
        else:
            raise ValueError("order parameter is invalid")
        if over_disp:
            cv2.resizeWindow('preview', )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def merge(self, img_a, img_b, pos=(0, 0)):
        """
        img_a に img_b をマージする。
        img_a にデータを上書きする。

        pos = (horizontal_st, vertical_st)
        """
        b_width = img_b.shape[1]
        b_height = img_b.shape[0]

        img_a[pos[1]:b_height+pos[1], pos[0]:b_width+pos[0]] = img_b

    def draw(self):
        base_img = np.zeros((self.img_height, self.img_width, 3))
        for f_idx in range(self.frame_num):
            v_buf = []
            for v_idx in range(self.v_num):
                h_buf = []
                for h_idx in range(self.h_num):
                    temp_img = np.ones((self.patch_size, self.patch_size, 3))
                    temp_img = temp_img * self.rgb[f_idx][v_idx][h_idx]
                    h_buf.append(temp_img)
                v_buf.append(np.hstack(h_buf))
            patch_img = np.vstack(v_buf)
            frame_img = base_img.copy()
            self.merge(frame_img, patch_img, (0, 0))
            print(f_idx, np.max(frame_img), np.min(frame_img))
            out_name = self.name_base.format(f_idx, self.grid_num)
            cv2.imwrite(out_name,
                        np.uint16(np.round(frame_img * 0xFFFF))[:, :, ::-1])


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
