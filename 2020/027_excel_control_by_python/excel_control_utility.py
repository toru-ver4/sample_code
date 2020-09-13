# -*- coding: utf-8 -*-
"""
Control a excel file using pywin32 module.
==========================================
"""

# import standard libraries
import os
from pathlib import WindowsPath

# import third-party libraries
import win32com.client

# import my libraries


def convert_from_relative_path_to_abs_path(path):
    """
    ファイルパスを相対パスから絶対パスに変更。
    win32com 経由でファイルを指定するには絶対パスである必要があるので実装。

    Parameters
    ----------
    path : str
        A relative file path.

    Returns
    -------
    str
        A abs file path.

    Examples
    --------
    >>> convert_from_relative_path_to_abs_path("./sample.xlsx")
    'C:\\Users\\toruv\\OneDrive\\work\\sample_code\\sample.xlsx'
    """
    windows_abs_path = os.path.abspath(path)
    abs_path = str(WindowsPath(windows_abs_path).resolve())

    return abs_path


def launch_excel_app(visible=True):
    """
    Excel アプリの起動
    なんか COM で云々しているようだが詳細は不明。

    Parameters
    ----------
    visible : bool
        whether to display the excel window.

    Returns
    -------
    win32com.client.CDispatch
        excel app.

    Examples
    --------
    >>> excel_app = launch_excel_app()
    >>> wb = open_excel_file(filename="./sample3.xlsx", excel_app=excel_app)
    >>> ws = wb.Worksheets(1)
    >>> ws.Name = 'Python'
    >>> save_excel_file(filename="./sample3.xlsx", wb=wb)
    >>> close_excel_file(wb)
    >>> quit_excel_app(excel_app=excel_app)

    Reference
    ---------
    * https://stackoverflow.com/questions/18648933/using-pywin32-what-is-the-difference-between-dispatch-and-dispatchex

    """
    excel_app = win32com.client.Dispatch("Excel.Application")
    excel_app.Visible = visible

    return excel_app


def quit_excel_app(excel_app):
    """
    エクセルアプリを終了する。

    Parameters
    ----------
    filename : str
        the file path of the .xlsx file.
    excel_app : win32com.client.CDispatch
        excel app.

    Examples
    --------
    >>> excel_app = launch_excel_app()
    >>> wb = open_excel_file(filename="./sample3.xlsx", excel_app=excel_app)
    >>> ws = wb.Worksheets(1)
    >>> ws.Name = 'Python'
    >>> save_excel_file(filename="./sample3.xlsx", wb=wb)
    >>> close_excel_file(wb)
    >>> quit_excel_app(excel_app=excel_app)
    """
    excel_app.Quit()


def open_excel_file(filename, excel_app):
    """
    エクセルファイルを開く。
    最近の自動保存で保存前の中間状態が保存されるのが怖いので
    内部的には別ファイルを新規作成して、そっちにデータをコピーしている。

    Parameters
    ----------
    filename : str
        the file path of the .xlsx file.
    excel_app : win32com.client.CDispatch
        excel app.

    Returns
    -------
    win32com.client.CDispatch
        excel workbook

    Examples
    --------
    >>> excel_app = launch_excel_app()
    >>> wb = open_excel_file(filename="./sample3.xlsx", excel_app=excel_app)
    >>> ws = wb.Worksheets(1)
    >>> ws.Name = 'Python'
    >>> save_excel_file(filename="./sample3.xlsx", wb=wb)
    >>> close_excel_file(wb)
    >>> quit_excel_app(excel_app=excel_app)
    """
    visible_backup = excel_app.Visible
    # エクセルのファイルを開くには絶対パスである必要があるらしい。
    # なので pathlib.Path を使って変換している。
    abs_file_name = convert_from_relative_path_to_abs_path(filename)

    excel_app.Visible = False

    # Excel の自動保存でファイルが上書きされる可能性があるため、
    # 作業用の wb は事前にコピーしたものを使う
    wb_read_only = excel_app.Workbooks.Open(abs_file_name, None, True)
    wb = create_excel_file(excel_app)
    sheet_number = wb_read_only.sheets.Count

    for sheet_idx in range(sheet_number)[::-1]:
        wb_read_only.Worksheets(sheet_idx + 1).Copy(Before=wb.Worksheets(1))

    # read only のシートを閉じる
    close_excel_file(wb_read_only)

    # 標準で作られれる "Sheet1" を削除
    wb.Worksheets(sheet_number + 1).Delete()

    excel_app.Visible = visible_backup

    return wb


def close_excel_file(wb):
    """
    エクセルファイルを閉じる

    Parameters
    ----------
    wb : win32com.client.CDispatch
        A excel workbook.

    Returns
    -------
    None

    Examples
    --------
    >>> excel_app = launch_excel_app()
    >>> wb = open_excel_file(filename="./sample3.xlsx", excel_app=excel_app)
    >>> ws = wb.Worksheets(1)
    >>> ws.Name = 'Python'
    >>> save_excel_file(filename="./sample3.xlsx", wb=wb)
    >>> close_excel_file(wb)
    >>> quit_excel_app(excel_app=excel_app)
    """
    wb.Close()


def create_excel_file(excel_app):
    """
    エクセルファイルを新規作成する。

    Parameters
    ----------
    excel_app : win32com.client.CDispatch
        excel app.

    Returns
    -------
    win32com.client.CDispatch
        excel workbook

    Examples
    --------
    >>> excel_app = launch_excel_app()
    >>> wb = create_excel_file(excel_app=excel_app)
    >>> ws = wb.Worksheets(1)
    >>> ws.Name = 'Python'
    >>> save_excel_file(filename="./sample3.xlsx", wb=wb)
    >>> close_excel_file(wb)
    >>> quit_excel_app(excel_app=excel_app)
    """
    wb = excel_app.Workbooks.Add()

    return wb


def save_excel_file(filename, wb, excel_app):
    """
    エクセルファイルを保存する。

    Parameters
    ----------
    filename : str
        the file path of the .xlsx file.
    wb : win32com.client.CDispatch
        A excel workbook.
    excel_app : win32com.client.CDispatch
        excel app.

    Returns
    -------
    None

    Examples
    --------
    >>> excel_app = launch_excel_app()
    >>> wb = open_excel_file(filename="./sample3.xlsx", excel_app=excel_app)
    >>> ws = wb.Worksheets(1)
    >>> ws.Name = 'Python'
    >>> save_excel_file(filename="./sample3.xlsx", wb=wb)
    >>> close_excel_file(wb)
    >>> quit_excel_app(excel_app=excel_app)
    """
    abs_file_name = convert_from_relative_path_to_abs_path(filename)

    # 上書き確認のダイアログがうざいので表示されないようにする
    display_alerts_backup = excel_app.DisplayAlerts
    excel_app.DisplayAlerts = False
    wb.SaveAs(abs_file_name)

    # 終わったら元の値に戻しておく。
    excel_app.DisplayAlerts = display_alerts_backup


def main_func():
    """
    メインの関数
    """
    # 変数定義
    # input_file_name = "./sample3.xlsx"
    input_file_name = None
    output_file_name = "./sample4.xlsx"

    # エクセルアプリ起動
    excel_app = launch_excel_app()

    # ファイルを開く
    if input_file_name is not None:
        wb = open_excel_file(filename=input_file_name, excel_app=excel_app)
    else:
        wb = create_excel_file(excel_app=excel_app)

    # シートを選択
    ws = wb.Worksheets(1)
    # ws = wb.Worksheets("Sheet1")  # <-- このやり方でも良い

    # シート名変更。任意のセルに値を代入
    ws.Name = 'Python'
    ws.Cells(1, 1).Value = "Hello Excel"

    # 保存
    save_excel_file(filename=output_file_name, wb=wb, excel_app=excel_app)

    # ファイルを閉じる
    close_excel_file(wb)

    # エクセルアプリの終了
    quit_excel_app(excel_app=excel_app)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
