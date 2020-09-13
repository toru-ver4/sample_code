# -*- coding: utf-8 -*-
"""
sample code for excel control using python
==========================================
"""

# import standard libraries
import os
from pathlib import WindowsPath
from random import randint

# import third-party libraries
import win32com.client

# import my libraries
import excel_control_utility as ecu


def rgb_8bit_to_24bit(r=192, g=128, b=255):
    return r * 0x00000001 + g * 0x00000100 + b * 0x00010000


def change_row_height(ws, height, st_pos_row, ed_pos_row):
    """
    Row の 高さを調整

    Parameters
    ----------
    ws : win32com.client.CDispatch
        excel worksheet
    height : int
        height of the raw.
    st_pos_row : int
        start position of row.
    ed_pos_row : int
        end position of row. It is not included.
    """
    range = f"{st_pos_row}:{ed_pos_row - 1}"
    ws.Rows(range).RowHeight = height


def change_col_width(ws, width, st_pos_col, ed_pos_col):
    """
    Column の 幅を調整

    Parameters
    ----------
    ws : win32com.client.CDispatch
        excel worksheet
    height : int
        height of the raw.
    st_pos_col : int
        start position of row.
    ed_pos_col : int
        end position of colmun. It is not included.
    """
    # range = f"{st_pos_col}:{ed_pos_col - 1}"
    st_cell = ws.Cells(1, st_pos_col)
    ed_cell = ws.Cells(1, ed_pos_col - 1)
    ws.Range(st_cell, ed_cell).ColumnWidth = width


def create_color_table_1d(
        ws, cell_num=24, st_pos_row=2, st_pos_col=3):
    """
    乱数で RGB値を作成し、その色でセルを塗りつぶす

    Parameters
    ----------
    ws : win32com.client.CDispatch
        excel worksheet
    cell_num : int
        the number of cells that are filled with random color.
    st_pos_row : int
        start position of row.
    st_pos_col : int
        start position of colmun.
    """
    pos_col = st_pos_col
    min_color_value = 0
    max_color_value = 255
    for idx in range(cell_num):
        r = randint(min_color_value, max_color_value)
        g = randint(min_color_value, max_color_value)
        b = randint(min_color_value, max_color_value)
        color_value = rgb_8bit_to_24bit(r, g, b)

        pos_raw = st_pos_row + idx
        cell = ws.Cells(pos_raw, pos_col)
        cell.Interior.Color = color_value


def create_color_table_2d(
        ws, cell_row_num=9, cell_col_num=16, st_pos_row=2, st_pos_col=3):
    """
    乱数で RGB値を作成し、その色でセルを塗りつぶす。2次元版。

    Parameters
    ----------
    ws : win32com.client.CDispatch
        excel worksheet
    cell_row_num : int
        the row number of cells that are filled with random color.
    cell_col_num : int
        the colmun number of cells that are filled with random color.
    st_pos_row : int
        start position of row.
    st_pos_col : int
        start position of colmun.
    """
    change_row_height(ws, 64, st_pos_row, st_pos_row + cell_row_num)
    change_col_width(ws, 10, st_pos_col, st_pos_col + cell_col_num)
    pos_col = st_pos_col
    min_color_value = 0
    max_color_value = 255
    for r_idx in range(cell_row_num):
        pos_raw = st_pos_row + r_idx
        for c_idx in range(cell_col_num):
            r = randint(min_color_value, max_color_value)
            g = randint(min_color_value, max_color_value)
            b = randint(min_color_value, max_color_value)
            color_value = rgb_8bit_to_24bit(r, g, b)

            pos_col = st_pos_col + c_idx
            cell = ws.Cells(pos_raw, pos_col)
            cell.Interior.Color = color_value


def sequential_process(ws, st_pos_row=2, st_pos_col=3):
    """
    適当にセルを範囲選択して wirte/read してみる。
    """
    sample_num = 56

    # とりあえず Gamma=2.4 のデータを書いてみる
    # まずは Index の書き込み
    pos_col = st_pos_col
    for idx in range(sample_num):
        pos_row = st_pos_row + idx
        cell = ws.Cells(pos_row, pos_col)
        cell.Value = idx

        linear_val = (idx / sample_num) ** 2.4
        cell = ws.Cells(pos_row, pos_col + 1)
        cell.Value = linear_val

    # 高速版のIndex 書き込み
    st_cel = ws.Cells(st_pos_row, st_pos_col + 2)
    ed_cel = ws.Cells(st_pos_row + sample_num - 1, st_pos_col + 2)
    ws.Range(st_cel, ed_cel).Value = [[x] for x in range(sample_num)]

    # 高速版のEOTF値 書き込み
    st_cel = ws.Cells(st_pos_row, st_pos_col + 3)
    ed_cel = ws.Cells(st_pos_row + sample_num - 1, st_pos_col + 3)
    ws.Range(st_cel, ed_cel).Value = [
        [(x / sample_num) ** 2.4] for x in range(sample_num)]


def main_func():
    # エクセルアプリ起動
    excel_app = ecu.launch_excel_app()

    # 最大化
    excel_app.WindowState = 1

    # エクセルファイル新規作成
    wb = ecu.create_excel_file(excel_app=excel_app)

    # シート1 に 1次元の塗りつぶしを実行
    ws = wb.Worksheets(1)
    ws.Name = "1D"
    create_color_table_1d(ws, cell_num=24, st_pos_row=2, st_pos_col=3)

    # シート2 に 2次元の塗りつぶしを実行
    wb.Sheets.Add(Before=ws)
    ws = wb.Worksheets(1)
    ws.Name = "2D"
    create_color_table_2d(
        ws, cell_row_num=9, cell_col_num=16, st_pos_row=2, st_pos_col=3)

    # シート3 でセルの連続選択＆書き込みの実験
    wb.Sheets.Add(Before=ws)
    ws = wb.Worksheets(1)
    ws.Name = "Gamma2.4"
    sequential_process(ws=ws, st_pos_row=2, st_pos_col=3)

    # 保存
    ecu.save_excel_file(filename="./sample.xlsx", wb=wb, excel_app=excel_app)

    # # ファイルを閉じる
    ecu.close_excel_file(wb)

    # # エクセルアプリの終了
    ecu.quit_excel_app(excel_app=excel_app)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
