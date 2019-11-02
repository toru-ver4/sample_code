#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2

# 自作ライブラリ
from CalcParameters import CalcParameters
from PatchControl import PatchControl
from MovieControl import MovieControl
import lut


# definition
base_param = {
    "img_width": 3840,
    "img_height": 2160,
    "patch_size": 64,
    'grid_num': 65,
    'patch_file_name': "./base_frame/frame_{:03d}_grid_{:02d}.tiff",
    'patch_after_name': "./after_frame/frame_{:03d}_grid_{:02d}.png",
    'patch_after_name2': "./after_frame/frame_%3d_grid_{:02d}.png"
}

"""
各 Revision の意味は以下の通り。
* Rev 01: 失敗。なんか全体的に Gain 低い
* Rev 02: 成功。初版。
* Rev 03: ソースコードリファクタリング後。中身は Rev 02 と同じ。
"""
lut_file_name = "./luts/HDR10_to_BT709_YouTube_Rev03.cube"

"""
# 設計

## 動作イメージ

```
calc_parameters = CalcParameters(base_param)
ctrl_param = calc_parameters.calc()

patch_control = PatchControl(base_param, ctrl_param)
patch_control.draw()  # もりもりパッチ画像作成。10bit DPX で保存

movie_ctrl = MovieControl(base_param, ctrl_param)
movie_ctrl.make_sequence()  # 指定の fps に合わせて基準フレームをコピー

# Davinci Resolve でエンコード
# YouTube にアップロード
# スクショを保存

movie_ctrl.parse_sequence()  # データから基準フレームを復元
sdr_rgb = patch_control.restore()

lut.save_3dlut(sdr_rgb)  # 3DLUT として保存
```

## base_param

```
typedef struct{
    int img_width;  // 3840
    int img_height; // 2160
    int patch_size;  // 32 [pixel]
    int grid_num;  // 65 [grid]
    str *patch_file_name;  // for DrawPatch
    str *patch_file_name2;  // for ffmpeg
}base_param;
```

## ctrl_param

```
typedef struct{
    int frame_num; // total の Frame数
    int coord[V_PATCH_NUM][H_PATCH_NUM];  // (Hpos, Vpos) の情報が入った配列
    float rgb[FRAME_NUM][V_PATCH_NUM][H_PATCH_NUM];  // np.zeros() で初期化。端数を黒にするため。
    int v_num;  // 1枚の画像の中の水平方向のパッチ数
    int h_num:
}ctrl_param;
```

"""


def main_func():
    calc_parameters = CalcParameters(base_param)
    ctrl_param = calc_parameters.calc()
    patch_control = PatchControl(base_param, ctrl_param)
    # patch_control.draw()
    movie_ctrl = MovieControl(base_param, ctrl_param)
    # movie_ctrl.make_sequence()

    """
    注意。ここから先の処理は以下の処理が終わってから実施すること。
    * Davinci Resolve でのエンコード
    * YouTube へのアップロード
    * YouTube から静止画のスクショ作成
    """
    movie_ctrl.parse_sequence()
    sdr_rgb = patch_control.restore()
    lut.save_3dlut(
        lut=sdr_rgb, grid_num=base_param['grid_num'], filename=lut_file_name,
        title="YouTube HDR to SDR conversion emulation")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
