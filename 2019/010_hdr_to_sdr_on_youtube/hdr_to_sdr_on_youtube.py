#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2

# 自作ライブラリ
from CalcParameters import CalcParameters
from DrawPatch import DrawPatch
from MovieControl import MovieControl


# definition
base_param = {
    "img_width": 3840,
    "img_height": 2160,
    "patch_size": 32,
    'grid_num': 65,
    'patch_file_name': "./base_frame/frame_{:03d}_grid_{:02d}.tiff"
}

"""
# 設計

## 動作イメージ

```
img = np.zeros((2160, 3840, 3), dtype=np.float32)

calc_parameters = CalcParameters(base_param)
ctrl_param = calc_parameters.calc()

draw_patch = DrawPatch(img, ctrl_param)
draw_patch.draw()  # もりもりパッチ画像作成。10bit DPX で保存

movie_ctrl = MovieControl(base_param)
movie_ctrl.make_sequence()  # 指定の fps に合わせて基準フレームをコピー

# Davinci Resolve でエンコード
# YouTube にアップロード
# スクショを保存

movie_ctrl.parse_sequence(base_param)  # データから基準フレームを復元

lut_ctrl = LutControl()
lut_ctrl.make()  # 基準フレームから3DLUTを生成
```

## base_param

```
typedef struct{
    int img_width;  // 3840
    int img_height; // 2160
    int patch_size;  // 32 [pixel]
    int grid_num;  // 65 [grid]
    str *patch_file_name;
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
    # draw_patch = DrawPatch(base_param, ctrl_param)
    # draw_patch.draw()
    movie_ctrl = MovieControl(base_param, ctrl_param)
    movie_ctrl.make_sequence()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
