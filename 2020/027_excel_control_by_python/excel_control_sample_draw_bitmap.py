# -*- coding: utf-8 -*-
"""
sample code for excel control using python
==========================================

* draw Gawr Gura's twitter banner
"""

# import standard libraries
import os

# import third-party libraries
import cv2
import numpy as np

# import my libraries
import excel_control_utility as ecu

# banner https://twitter.com/gawrgura
IMG_DATA = "./img/1500x500.jpg"


def conv_rgb_8bit_color_to_excel_24bit_format(rgb):
    """
    convert from 8bit rgb array to 24bit color value for excel.

    Parameters
    ----------
    rgb : ndarray
        8 bit image data.

    Returns
    -------
    ndarray
        24bit color values for excel.

    Examples
    --------
    >>> img = np.array(
    ...     [[[255, 0, 0], [0, 255, 0]],
    ...      [[0, 0, 255], [64, 128, 192]]])
    >>> conv_rgb_8bit_color_to_excel_24bit_format(rgb=img)

    """
    # return r * 0x00000001 + g * 0x00000100 + b * 0x00010000
    return rgb[..., 0] * 0x00000001 + np.uint32(rgb[..., 1]) * 0x0000100\
        + rgb[..., 2] * 0x00010000


def load_8bit_image(fname=IMG_DATA):
    img = cv2.imread(IMG_DATA)[..., ::-1]

    return img


def crop_and_resize_image(img):
    img_crop = img[:, 327:327+1045, :]
    scale_factor = 4
    after_width = img_crop.shape[1] // scale_factor
    after_height = img_crop.shape[0] // scale_factor
    out_img = cv2.resize(
        img_crop, dsize=(after_width, after_height))

    return out_img


def draw_banner(ws):
    """
    draw the twitter banner on the Excel.

    Parameters
    ----------
    ws : win32com.client.CDispatch
        excel worksheet
    """
    # 画像の準備
    img_org = load_8bit_image(fname=IMG_DATA)
    img = crop_and_resize_image(img_org)
    img_width = img.shape[1]
    img_height = img.shape[0]
    img_color = conv_rgb_8bit_color_to_excel_24bit_format(img)

    # セルが正方形となるようにサイズ変更
    ecu.change_col_width(ws, 0.5, st_pos_col=0, col_num=img_width)
    ecu.change_row_height(ws, 5.5, st_pos_row=0, row_num=img_height)

    # 描画 (Interior.Color は配列アクセスできなかった🥺🥺🥺)
    for row_idx in range(img_height):
        for col_idx in range(img_width):
            ws.Cells(row_idx + 1, col_idx + 1).Interior.Color\
                = img_color[row_idx, col_idx]


def main_func():
    # エクセルアプリ起動
    excel_app = ecu.launch_excel_app()

    # 最大化
    ecu.maximize_excel_window(excel_app)

    # エクセルファイル新規作成
    wb = ecu.create_excel_file(excel_app=excel_app)

    # シート1で画像を描画
    ws = wb.Worksheets(1)
    ws.Name = "DRAW"
    draw_banner(ws)

    # 保存
    ecu.save_excel_file(
        filename="./draw_sample.xlsx", wb=wb, excel_app=excel_app)

    # # # ファイルを閉じる
    # ecu.close_excel_file(wb)

    # # # エクセルアプリの終了
    # ecu.quit_excel_app(excel_app=excel_app)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
