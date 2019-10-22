#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
## 概要

* 1DLUT/3DLUT の Write/Read をするモジュール
* 異なる形式への変換は…なんかビビったので止めます…。

## 仕様

### サポート形式

1DLUT では以下の形式をサポートする

* .cube (Adobe)
* .spi1d


3DLUT では以下の形式をサポートする

* .3dl (Lustre)
* .cube (Adobe)
* .spi3d (SPI)

### データの保持について

ライブラリ内部では以下の仕様でデータを持つ

### 1DLUT

* dtype: double
* array: [idx][3]

### 3DLUT

* dtype: double
* array: [idx][3]
  * idx は R -> G -> B の順に増加する
  * cube形式と同じ順序

## 使い方

本ソースコードのmainを参照。

* 3DLUTの新規作成
* 3DLUTの形式変換
* 1DLUTの新規作成
* 1DLUTの形式変換

## references
[3dl](http://download.autodesk.com/us/systemdocs/help/2009/lustre_ext1/index.html?url=WSc4e151a45a3b785a24c3d9a411df9298473-7ffd.htm,topicNumber=d0e8061)
[cube](http://wwwimages.adobe.com/www.adobe.com/content/dam/acom/en/products/speedgrade/cc/pdfs/cube-lut-specification-1.0.pdf)

"""

import os
import numpy as np
import math

AUTHOR_INFORMATION = "This 3DLUT data was created by TY-LUT creation tool"
LUT_BIT_DEPTH_3DL = 16


def get_3d_grid_cube_format(grid_num=4):
    """
    .cubeフォーマットに準じた3DLUTの格子データを出力する。
    grid_num=3 の場の例を以下に示す。

    ```
    [[ 0.   0.   0. ]
     [ 0.5  0.   0. ]
     [ 1.   0.   0. ]
     [ 0.   0.5  0. ]
     [ 0.5  0.5  0. ]
     [ 1.   0.5  0. ]
     [ 0.   1.   0. ]
     [ 0.5  1.   0. ]
     [ 1.   1.   0. ]
     [ 0.   0.   0.5]

     中略

     [ 1.   0.5  1. ]
     [ 0.   1.   1. ]
     [ 0.5  1.   1. ]
     [ 1.   1.   1. ]]
     ```

    Parameters
    ----------
    grid_num : int
        grid number of the 3dlut.

    Returns
    -------
    array_like
        3DLUT grid data with cube format.
    """

    base = np.linspace(0, 1, grid_num)
    ones_x = np.ones((grid_num, grid_num, 1))
    ones_y = np.ones((grid_num, 1, grid_num))
    ones_z = np.ones((1, grid_num, grid_num))
    r_3d = base[np.newaxis, np.newaxis, :] * ones_x
    g_3d = base[np.newaxis, :, np.newaxis] * ones_y
    b_3d = base[:, np.newaxis, np.newaxis] * ones_z
    r_3d = r_3d.flatten()
    g_3d = g_3d.flatten()
    b_3d = b_3d.flatten()

    grid = np.dstack((r_3d, g_3d, b_3d))
    grid = grid.reshape((grid_num ** 3, 3))

    return grid


def _convert_3dlut_from_cube_to_3dl(lut, grid_num):
    """
    cube形式(R -> G -> B 順で増加) のデータを
    3dl形式(B -> G -> R) に変換

    Parameters
    ----------
    lut : array_like
        3DLUT data with cube format.
    grid_num : int
        grid number of the 3dlut.

    Returns
    -------
    array_like
        3DLUT data with 3dl format.
    """

    out_lut = lut.reshape((grid_num, grid_num, grid_num, 3), order="F")
    out_lut = out_lut.reshape((grid_num ** 3, 3))

    return out_lut


def _convert_3dlut_from_3dl_to_cube(lut, grid_num):
    """
    3dl形式(B -> G -> R) のデータを
    cube形式(R -> G -> B 順で増加) に変換

    Parameters
    ----------
    lut : array_like
        3DLUT data with 3dl format.
    grid_num : int
        grid number of the 3dlut.

    Returns
    -------
    array_like
        3DLUT data with cube format.
    """

    out_lut = _convert_3dlut_from_cube_to_3dl(lut, grid_num)

    return out_lut


def _is_float_expression(s):
    """
    copied from
    https://qiita.com/agnelloxy/items/137cbc8651ff4931258f
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def _is_int_expression(s):
    """
    copied from
    https://qiita.com/agnelloxy/items/137cbc8651ff4931258f
    """
    try:
        int(s)
        return True
    except ValueError:
        return False


def save_3dlut(lut, grid_num, filename="./data/lut_sample/cube.cube",
               title=None, min=0.0, max=1.0):
    """
    3DLUTデータをファイルに保存する。
    形式の判定はファイル名の拡張子で行う。

    Parameters
    ----------
    filename : str
        file name.
    lut : array_like
        3dlut data.
    grid_num : int
        grid number.
    title : str
        title of the 3dlut data. It is for header information.
    min : int or float
        minimum value of the 3dlut
    max : int or float
        maximum value of the 3dlut
    """

    root, ext = os.path.splitext(filename)

    if ext == ".cube":
        save_3dlut_cube_format(lut, grid_num, filename=filename,
                               title="cube_test", min=min, max=max)
    elif ext == ".3dl":
        save_3dlut_3dl_format(lut, grid_num, filename=filename,
                              title="cube_test", min=min, max=max)
    elif ext == ".spi3d":
        save_3dlut_spi_format(lut, grid_num, filename=filename,
                              title="spi3d_test", min=min, max=max)
    else:
        raise IOError('extension "{:s}" is not supported.'.format(ext))


def load_3dlut(filename="./data/lut_sample/cube.cube"):
    """
    3DLUTデータをファイルにファイルから読み込む。
    形式の判定はファイル名の拡張子で行う。

    Parameters
    ----------
    filename : str
        file name.

    Returns
    -------
    lut : array_like
        3DLUT grid data with cube format.
    grid_num : int
        grid number of the 3DLUT.
    """

    root, ext = os.path.splitext(filename)

    if ext == ".cube":
        lut, grid_num, title, min, max\
            = load_3dlut_cube_format(filename=filename)
    elif ext == ".3dl":
        lut, grid_num, title\
            = load_3dlut_3dl_format(filename=filename)
    elif ext == ".spi3d":
        lut, version, dimension, grid_num\
            = load_3dlut_spi_format(filename=filename)
    else:
        raise IOError('extension "{:s}" is not supported.'.format(ext))

    return lut, grid_num


def save_3dlut_cube_format(lut, grid_num, filename,
                           title=None, min=0.0, max=1.0):
    """
    CUBE形式で3DLUTデータをファイルに保存する。

    Parameters
    ----------
    filename : str
        file name.
    lut : array_like
        3dlut data.
    grid_num : int
        grid number.
    title : str
        title of the 3dlut data. It is for header information.
    min : int or float
        minimum value of the 3dlut
    max : int or float
        maximum value of the 3dlut
    """

    # ヘッダ情報の作成
    # ------------------------
    header = ""
    header += '# ' + AUTHOR_INFORMATION + '\n'

    if title:
        header += 'TITLE "{:s}"\n'.format(title)
    header += 'DOMAIN_MIN {0:} {0:} {0:}\n'.format(min)
    header += 'DOMAIN_MAX {0:} {0:} {0:}\n'.format(max)
    header += 'LUT_3D_SIZE {:}\n'.format(grid_num)
    header += '\n'

    # ファイルにデータを書き込む
    # ------------------------
    out_str = '{:.10e} {:.10e} {:.10e}\n'
    with open(filename, "w") as file:
        file.write(header)
        for line in lut:
            file.write(out_str.format(line[0], line[1], line[2]))


def load_3dlut_cube_format(filename):
    """
    CUBE形式の3DLUTデータをファイルから読み込む。

    Parameters
    ----------
    filename : str
        file name.

    Returns
    -------
    lut : array_like
        3DLUT data with cube format.
    grid_num : int
        grid number.
    title : str
        title of the 3dlut data.
    min : double
        minium value of the 3dlut data.
    max : double
        maximum value of the 3dlut data.
    """

    # ヘッダ情報を読みつつ、データ開始位置を探る
    # --------------------------------------
    data_start_idx = 0
    title = None
    min = 0.0
    max = 1.0
    grid_num = None
    with open(filename, "r") as file:
        for line_idx, line in enumerate(file):
            line = line.rstrip()
            if line == '':  # 空行は飛ばす
                continue
            key_value = line.split()[0]
            if key_value == 'TITLE':
                title = line.split()[1]
            if key_value == 'DOMAIN_MIN':
                min = float(line.split()[1])
            if key_value == 'DOMAIN_MAX':
                max = float(line.split()[1])
            if key_value == 'LUT_3D_SIZE':
                grid_num = int(line.split()[1])
            if _is_float_expression(line.split()[0]):
                data_start_idx = line_idx
                break

    # 3DLUTデータを読む
    # --------------------------------------
    lut = np.loadtxt(filename, delimiter=' ', skiprows=data_start_idx)

    # 得られたデータを返す
    # --------------------------------------
    return lut, grid_num, title, min, max


def save_3dlut_spi_format(lut, grid_num, filename,
                          title=None, min=0.0, max=1.0):
    """
    spi3d形式で3DLUTデータをファイルに保存する。

    Parameters
    ----------
    filename : str
        file name.
    lut : array_like
        3dlut data.
    grid_num : int
        grid number.
    title : str
        title of the 3dlut data. It is for header information.
    min : int or float
        minimum value of the 3dlut
    max : int or float
        maximum value of the 3dlut
    """

    # 3dl形式へLUTデータの並べ替えをする
    # --------------------------------
    out_lut = _convert_3dlut_from_cube_to_3dl(lut, grid_num)

    # ヘッダ情報の作成
    # ------------------------
    header = ""
    header += 'SPILUT 1.0\n'
    header += '{:d} {:d}\n'.format(3, 3)  # 数値の意味は不明
    header += '{0:d} {0:d} {0:d}\n'.format(grid_num)

    # ファイルにデータを書き込む
    # ------------------------
    line_index = 0
    out_str = '{:d} {:d} {:d} {:.10e} {:.10e} {:.10e}\n'
    with open(filename, "w") as file:
        file.write(header)
        for line in out_lut:
            r_idx, g_idx, b_idx\
                = _get_rgb_index_for_spi3d_output(line_index, grid_num)
            file.write(out_str.format(r_idx, g_idx, b_idx,
                                      line[0], line[1], line[2]))
            line_index += 1


def load_3dlut_spi_format(filename):
    """
    spi3d形式の3DLUTデータをファイルから読み込む。

    Parameters
    ----------
    filename : str
        file name.

    Returns
    -------
    lut : array_like
        3DLUT data with spi format.
    version : double
        version of the spilut
    demension : int
        it is fixed 3?
    grid_num : int
        grid number.
    """

    # ヘッダ情報を読みつつ、データ開始位置を探る
    # --------------------------------------
    data_start_idx = 0
    grid_num = None
    with open(filename, "r") as file:
        for line_idx, line in enumerate(file):
            line = line.rstrip()
            if line == '':  # 空行は飛ばす
                continue
            key_value = line.split()[0]
            if key_value == 'SPILUT':
                version = float(line.split()[1])
                continue
            if len(line.split()) == 2:
                dimension = int(line.split()[0])
            if len(line.split()) == 3:
                grid_num = int(line.split()[0])
            if len(line.split()) == 6:
                data_start_idx = line_idx
                break

    # 3DLUTデータを読む
    # --------------------------------------
    lut = np.loadtxt(filename, delimiter=' ', skiprows=data_start_idx,
                     usecols=(3, 4, 5))
    lut = _convert_3dlut_from_3dl_to_cube(lut, grid_num)

    # 得られたデータを返す
    # --------------------------------------
    return lut, version, dimension, grid_num


def _get_rgb_index_for_spi3d_output(line_index, grid_num):
    """
    3DLUT Data の行番号から、当該行に該当する r_idx, g_idx, b_idx を
    算出する。

    Parameters
    ----------
    line_index : int
        line number.
    grid_num : int
        grid number.

    Returns
    -------
    int, int, int
        grid index of each color.
    """

    r_idx = (line_index // (grid_num ** 2)) % grid_num
    g_idx = (line_index // (grid_num ** 1)) % grid_num
    b_idx = (line_index // (grid_num ** 0)) % grid_num

    return r_idx, g_idx, b_idx


def save_3dlut_3dl_format(lut, grid_num, filename,
                          title=None, min=0.0, max=1.0):
    """
    3DL形式で3DLUTデータをファイルに保存する。

    Parameters
    ----------
    filename : str
        file name.
    lut : array_like
        3dlut data.
    grid_num : int
        grid number.
    title : str
        title of the 3dlut data. It is for header information.
    min : int or float
        minimum value of the 3dlut
    max : int or float
        maximum value of the 3dlut
    """

    # 3dl形式へLUTデータの並べ替えをする
    # --------------------------------
    out_lut = _convert_3dlut_from_cube_to_3dl(lut, grid_num)

    # ヘッダ情報の作成
    # ------------------------
    header = ""
    exponent = round(math.log2(grid_num - 1))
    bit_depth = LUT_BIT_DEPTH_3DL

    if title:
        header += '# TITLE "{:s}"\n'.format(title)
    header += '# ' + AUTHOR_INFORMATION + '\n'
    header += '\n'
    header += '3DMESH\n'
    header += 'Mesh {:d} {:d}\n'.format(exponent, bit_depth)
    header += '\n'

    # データを出力bit精度に変換
    # ------------------------
    out_lut = np.uint32(np.round(out_lut * ((2 ** bit_depth) - 1)))

    # ファイルにデータを書き込む
    # ------------------------
    out_str = '{:d} {:d} {:d}\n'
    with open(filename, "w") as file:
        file.write(header)
        for line in out_lut:
            file.write(out_str.format(line[0], line[1], line[2]))


def load_3dlut_3dl_format(filename):
    """
    3DL形式の3DLUTデータをファイルから読み込む。

    Parameters
    ----------
    filename : str
        file name.

    Returns
    -------
    lut : array_like
        3DLUT data with 3dl format.
    grid_num : int
        grid number.
    title : str
        title of the 3dlut.
    """

    # ヘッダ情報を読みつつ、データ開始位置を探る
    # --------------------------------------
    data_start_idx = 0
    title = None
    grid_num = None
    with open(filename, "r") as file:
        for line_idx, line in enumerate(file):
            line = line.rstrip()
            if line == '':  # 空行は飛ばす
                continue
            if len(line.split()) < 2:
                continue
            key_value0, key_value1 = line.split()[0], line.split()[1]
            if key_value0 == "#" and key_value1 == 'TITLE':
                title = line.split()[1]
                continue
            if key_value0 == "Mesh":
                grid_num = (2 ** int(line.split()[1])) + 1
                max_value = (2 ** int(line.split()[2])) - 1
                print(max_value)
                continue
            if len(line.split()) == 3 and _is_int_expression(line.split()[0]):
                data_start_idx = line_idx
                break

    # 3DLUTデータを読む
    # --------------------------------------
    lut = np.loadtxt(filename, delimiter=' ', skiprows=data_start_idx)
    lut = _convert_3dlut_from_3dl_to_cube(lut, grid_num)
    lut = lut / max_value

    # 得られたデータを返す
    # --------------------------------------
    return lut, grid_num, title


def save_1dlut(lut, filename='./data/lut_sample/hoge.spi1d',
               title=None, min=0.0, max=1.0):
    """
    1DLUTデータをファイルに保存する。
    形式の判定はファイル名の拡張子で行う。

    Parameters
    ----------
    filename : str
        file name.
    lut : array_like
        1dlut data.
    title : str
        title of the 3dlut data. It is for header information.
    min : int or float
        minimum value of the 3dlut
    max : int or float
        maximum value of the 3dlut
    """

    root, ext = os.path.splitext(filename)

    if ext == ".cube":
        save_1dlut_cube_format(lut, filename=filename,
                               title=title, min=min, max=max)
    elif ext == ".spi1d":
        save_1dlut_spi_format(lut, filename=filename,
                              title=title, min=min, max=max)
    else:
        raise IOError('extension "{:s}" is not supported.'.format(ext))


def save_1dlut_cube_format(lut, filename, title=None, min=0.0, max=1.0):
    """
    CUBE形式で1DLUTデータをファイルに保存する。

    Parameters
    ----------
    filename : str
        file name.
    lut : array_like
        1dlut data.
    title : str
        title of the 1dlut data. It is for header information.
    min : int or float
        minimum value of the 1dlut
    max : int or float
        maximum value of the 1dlut
    """

    # ヘッダ情報の作成
    # ------------------------
    header = ""
    header += '# ' + AUTHOR_INFORMATION + '\n'

    if title:
        header += 'TITLE "{:s}"\n'.format(title)
    header += 'DOMAIN_MIN {0:} {0:} {0:}\n'.format(min)
    header += 'DOMAIN_MAX {0:} {0:} {0:}\n'.format(max)
    header += 'LUT_1D_SIZE {:}\n'.format(len(lut))
    header += '\n'

    # ファイルにデータを書き込む
    # ------------------------
    out_str = '{0:.10e} {0:.10e} {0:.10e}\n'
    with open(filename, "w") as file:
        file.write(header)
        for line in lut:
            file.write(out_str.format(line))


def load_1dlut_cube_format(filename):
    """
    CUBE形式の1DLUTデータをファイルから読み込む。

    Parameters
    ----------
    filename : str
        file name.

    Returns
    -------
    lut : array_like
        1dlut data.
    """

    # ヘッダ情報を読みつつ、データ開始位置を探る
    # --------------------------------------
    data_start_idx = 0
    # title = None
    # min = 0.0
    # max = 1.0
    with open(filename, "r") as file:
        for line_idx, line in enumerate(file):
            line = line.rstrip()
            if line == '':  # 空行は飛ばす
                continue
            key_value = line.split()[0]
            if key_value == 'TITLE':
                continue
                # title = line.split()[1]
            if key_value == 'DOMAIN_MIN':
                continue
                # min = float(line.split()[1])
            if key_value == 'DOMAIN_MAX':
                continue
                # max = float(line.split()[1])
            if key_value == 'LUT_1D_SIZE':
                continue
                # grid_num = int(line.split()[1])
            if _is_float_expression(line.split()[0]):
                data_start_idx = line_idx
                break

    # 1DLUTデータを読む
    # --------------------------------------
    lut = np.loadtxt(filename, delimiter=' ', skiprows=data_start_idx)
    lut = lut[:, 0]

    # 得られたデータを返す
    # --------------------------------------
    return lut


def save_1dlut_spi_format(lut, filename,
                          title=None, min=0.0, max=1.0):
    """
    spi1d形式で3DLUTデータをファイルに保存する。

    Parameters
    ----------
    filename : str
        file name.
    lut : array_like
        1dlut data.
    title : str
        title of the 1dlut data. It is for header information.
    min : int or float
        minimum value of the input range
    max : int or float
        maximum value of the input range
    """

    # ヘッダ情報の作成
    # ------------------------
    header = ""
    header += 'Version 1\n'
    header += 'From {:f} {:f}\n'.format(min, max)
    header += 'Length {:d}\n'.format(len(lut))
    header += 'Components 1\n'
    header += '{\n'
    footer = '}\n'

    # ファイルにデータを書き込む
    # ------------------------
    out_str = '         {:.10e}\n'
    with open(filename, "w") as file:
        file.write(header)
        for line in lut:
            file.write(out_str.format(line))
        file.write(footer)


def load_1dlut_spi_format(filename):
    """
    SPI1D形式の1DLUTデータをファイルから読み込む。

    Parameters
    ----------
    filename : str
        file name.

    Returns
    -------
    lut : array_like
        1dlut data.
    """

    # ヘッダ情報を読みつつ、データ開始位置を探る
    # --------------------------------------
    data_start_idx = 0
    # length = 0
    # title = None
    # min = 0.0
    # max = 1.0
    with open(filename, "r") as file:
        for line_idx, line in enumerate(file):
            line = line.rstrip()
            if line == '':  # 空行は飛ばす
                continue
            key_value = line.split()[0]
            if key_value == 'Version':
                continue
                # version = int(line.split()[1])
            if key_value == 'From':
                continue
                # min = float(line.split()[1])
                # max = float(line.split()[2])
            if key_value == 'Length':
                # length = int(line.split()[1])
                continue
            if _is_float_expression(line.split()[0]):
                data_start_idx = line_idx
                break

    # 1DLUTデータを読む
    # --------------------------------------
    lut = np.loadtxt(filename, delimiter=None, skiprows=data_start_idx,
                     comments="}")

    # 得られたデータを返す
    # --------------------------------------
    return lut


def load_1dlut(filename='./data/lut_sample/hoge.spi1d'):
    """
    1DLUTデータをファイルから読み込む。
    形式の判定はファイル名の拡張子で行う。

    Parameters
    ----------
    filename : str
        file name.

    Returns
    -------
    lut : array_like
        3DLUT data with 3dl format.
    """

    root, ext = os.path.splitext(filename)

    if ext == ".cube":
        lut = load_1dlut_cube_format(filename=filename)
    elif ext == ".spi1d":
        lut = load_1dlut_spi_format(filename=filename)
    else:
        raise IOError('extension "{:s}" is not supported.'.format(ext))

    return lut


def make_3dlut_grid(grid_num=33):
    """
    3DLUTの格子点データを作成

    Parameters
    ----------
    grid_num : integer
        A number of grid points.

    Returns
    -------
    ndarray
        An Array of the grid points.
        The shape is (1, grid_num ** 3, 3).

    Examples
    --------
    >>> make_3dlut_grid(grid_num=3)
    array([[[0. , 0. , 0. ],
            [0.5, 0. , 0. ],
            [1. , 0. , 0. ],
            [0. , 0.5, 0. ],
            [0.5, 0.5, 0. ],
            [1. , 0.5, 0. ],
            [0. , 1. , 0. ],
            [0.5, 1. , 0. ],
            [1. , 1. , 0. ],
            [0. , 0. , 0.5],
            [0.5, 0. , 0.5],
            [1. , 0. , 0.5],
            [0. , 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [1. , 0.5, 0.5],
            [0. , 1. , 0.5],
            [0.5, 1. , 0.5],
            [1. , 1. , 0.5],
            [0. , 0. , 1. ],
            [0.5, 0. , 1. ],
            [1. , 0. , 1. ],
            [0. , 0.5, 1. ],
            [0.5, 0.5, 1. ],
            [1. , 0.5, 1. ],
            [0. , 1. , 1. ],
            [0.5, 1. , 1. ],
            [1. , 1. , 1. ]]])
    """
    # np.meshgrid を使って 3次元の格子点座標を生成
    x = np.linspace(0, 1, grid_num)
    rgb_mesh_array = np.meshgrid(x, x, x)

    # 後の処理を行いやすくするため shape を変える
    rgb_mesh_array = [x.reshape(1, grid_num ** 3, 1) for x in rgb_mesh_array]

    # 格子点のデータ増加が R, G, B の順となるように配列を並べ替えてから
    # np.dstack を使って結合する
    rgb_grid = np.dstack(
        (rgb_mesh_array[2], rgb_mesh_array[0], rgb_mesh_array[1]))

    return rgb_grid


def _test_3dlut():
    g_num = 17
    lut = get_3d_grid_cube_format(grid_num=g_num)
    save_3dlut(lut, g_num, filename="./data/lut_sample/hoge.fuga.3dl")
    save_3dlut(lut, g_num, filename="./data/lut_sample/hoge.fuga.spi3d")
    save_3dlut(lut, g_num, filename="./data/lut_sample/hoge.fuga.spi1d")
    sample_arri_cube = "./data/lut_sample/AlexaV3_EI0800_LogC2Video_Rec709_LL_aftereffects3d.cube"
    sample_aces_spi3d = "C:/home/sip/OpenColorIO/aces_1.0.3/luts/Log2_48_nits_Shaper.RRT.DCDM.spi3d"
    sample_3dl = "./data/lut_sample/hoge.fuga.3dl"
    lut, grid_num, title, min, max = load_3dlut_cube_format(sample_arri_cube)
    print(lut)
    print(grid_num, title, min, max)
    lut, version, dimension, grid_num\
        = load_3dlut_spi_format(sample_aces_spi3d)
    print(lut)
    print(version, dimension, grid_num)
    lut, grid_num, title = load_3dlut_3dl_format(sample_3dl)
    print(lut)
    print(grid_num, title)

    lut, grid_num = load_3dlut(sample_aces_spi3d)
    print(lut)
    print(grid_num)

    lut, grid_num = load_3dlut(sample_arri_cube)
    print(lut)
    print(grid_num)

    lut, grid_num = load_3dlut(sample_3dl)
    print(lut)
    print(grid_num)


def _test_1dlut():
    sample_num = 2 ** 10
    x = np.linspace(0, 1, sample_num)

    save_1dlut(x, filename='./data/lut_sample/hoge.1dlut.cube',
               title="Puri_Chan", min=-0.1, max=1.4)
    save_1dlut(x, filename='./data/lut_sample/hoge.1dlut.spi1d',
               title="Puri_Chan", min=-0.1, max=1.4)

    lut = load_1dlut(filename='./data/lut_sample/hoge.1dlut.cube')
    # print(lut)
    lut = load_1dlut(filename='./data/lut_sample/hoge.1dlut.spi1d')
    print(lut)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # _test_3dlut()
    _test_1dlut()
