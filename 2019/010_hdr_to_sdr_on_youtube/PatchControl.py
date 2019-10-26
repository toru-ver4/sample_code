#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
パッチをモリモリ描画する
"""
import os
import numpy as np
import cv2


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

        self.name_base = base_param['patch_file_name']
        self.after_name_base = base_param['patch_after_name']

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

    def restore(self):
        """
        フレームデータから RGB値を復元する。
        初版のアルゴリズムは、中心座標の値をすっぱ抜くだけ。
        将来的には平均値取るとか、もう少しオシャレなことをしたい。
        """
        frame_buf = []
        for frame_idx in range(self.frame_num):
            rgb_each_frame = self.get_each_frame_rgb_value(frame_idx)
            frame_buf.append(rgb_each_frame)
            # self.check_restored_rgb_visual(rgb_each_frame)

        # 扱いやすい形でに変更。あと、最後にゴミデーをを除外する
        sdr_rgb = np.array(frame_buf)
        s = sdr_rgb.shape
        sdr_rgb = sdr_rgb.reshape((s[0] * s[1] * s[2], 3))

        return sdr_rgb[:self.grid_num**3]

    def get_each_frame_rgb_value(self, frame_idx):
        v_offset = self.patch_size // 2
        h_offset = self.patch_size // 2
        f_name = self.after_name_base.format(frame_idx, self.grid_num)
        img = cv2.imread(
            f_name, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[:, :, ::-1]
        img = img / 0xFF
        v_buf = []
        for v_idx in range(self.v_num):
            h_buf = []
            v_pos = self.patch_size * v_idx + v_offset
            for h_idx in range(self.h_num):
                h_pos = self.patch_size * h_idx + h_offset
                h_buf.append(img[v_pos, h_pos])
            v_buf.append(h_buf)
        return np.array(v_buf)

    def check_restored_rgb_visual(self, rgb):
        """
        実装のデバッグ用。リストアした RGB値をもう一度パッチとして描画。
        キャプチャした パッチと目で見比べてみよう！
        """
        base_img = np.zeros((self.img_height, self.img_width, 3))
        v_buf = []
        for v_idx in range(self.v_num):
            h_buf = []
            for h_idx in range(self.h_num):
                temp_img = np.ones((self.patch_size, self.patch_size, 3))
                temp_img = temp_img * rgb[v_idx][h_idx]
                h_buf.append(temp_img)
            v_buf.append(np.hstack(h_buf))
        patch_img = np.vstack(v_buf)
        frame_img = base_img.copy()
        self.merge(frame_img, patch_img, (0, 0))
        out_name = "debug_resoterd_rgb.tiff"
        cv2.imwrite(out_name,
                    np.uint16(np.round(frame_img * 0xFFFF))[:, :, ::-1])


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
