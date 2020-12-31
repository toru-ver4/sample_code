# -*- coding: utf-8 -*-
"""
parse mp4 'colr' lazily
=======================

"""

# import standard libraries
import os
from struct import unpack, pack
from pathlib import Path
import shutil

# import third-party libraries
from numpy.testing._private.utils import assert_equal

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def read_size(fp):
    """
    Parameters
    ----------
    fp : BufferedReader
        file instance? the position is the begining of the box.
    """
    size = unpack('>I', fp.read(4))[0]
    if size == 0:
        raise ValueError('extended size is not supported')
    else:
        return size


def read_box_type(fp):
    """
    Parameters
    ----------
    fp : BufferedReader
        file instance?
        the position is 4 byte from the beginning of the box.
    """
    box_type = unpack('4s', fp.read(4))[0].decode()
    return box_type


def seek_fp_type_end_to_box_start(fp):
    fp.seek(-8, 1)


def read_box_type_and_type(fp):
    """
    Parameters
    ----------
    fp : BufferedReader
        file instance? the position is the begining of the box.

    Returns
    -------
    size : int
        a box size
    type : strings
        a box type
    """
    size = read_size(fp)
    box_type = read_box_type(fp)
    seek_fp_type_end_to_box_start(fp)

    return size, box_type


def is_end_of_file(fp):
    """
    check if it is the end of file.
    """
    read_size = 8
    dummy = fp.read(8)
    if len(dummy) != read_size:
        return True
    else:
        fp.seek(-read_size, 1)
        return False


def get_moov_data(fp, address, size):
    fp.seek(address)
    data = fp.read(size)

    return data


def get_moov_address_and_size(fp):
    """
    Parameters
    ----------
    fp : BufferedReader
        file instance.
    """
    fp.seek(0)
    size = 0
    address = None
    for idx in range(10):
        fp.seek(size, 1)
        if is_end_of_file(fp):
            print("Warning: 'moov' is not found.")
            break
        size, box_type = read_box_type_and_type(fp)

        if box_type == 'moov':
            address = fp.tell()
            print(f"'moov' is found at 0x{address:016X}")
            break

    return address, size


def search_colour_info_start_address(fp):
    """
    search the start address of the `colour_primaries` in the 'colr' box.

    Parameters
    ----------
    fp : BufferedReader
        file instance.
    """
    fp.seek(0)
    moov_address, moov_size = get_moov_address_and_size(fp)
    moov_data = get_moov_data(fp, moov_address, moov_size)
    colour_tag_address = moov_data.find('colr'.encode())
    colour_type_address = moov_data.find('nclx'.encode())
    if (colour_type_address - colour_tag_address) != 4:
        print("Error: 'colr' or 'nclx' is not found.")
        return None

    colour_info_start_address = colour_type_address + 4 + moov_address
    print(f"'colr' info address is 0x{colour_info_start_address:08X}")
    return colour_info_start_address


def get_colour_characteristics(fp):
    colour_info_address = search_colour_info_start_address(fp)
    fp.seek(colour_info_address)
    colour_binary_data = fp.read(6)
    colour_primaries, transfer_characteristics, matrix_coefficients\
        = unpack('>HHH', colour_binary_data)

    return colour_primaries, transfer_characteristics, matrix_coefficients


def set_colour_characteristics(
        fp, colour_primaries, transfer_characteristics, matrix_coefficients):
    colour_info_address = search_colour_info_start_address(fp)
    fp.seek(colour_info_address)
    data = pack(
        '>HHH', colour_primaries, transfer_characteristics, matrix_coefficients
    )
    fp.write(data)


def make_dst_fname(src_fname, suffix="9_16_9"):
    """
    Examples
    --------
    >>> make_dst_fname(src_fname="./video/hoge.mp4", suffix="9_16_9")
    "./video/hoge_9_16_9.mp4"
    """
    src_path = Path(src_fname)
    parent = str(src_path.parent)
    ext = src_path.suffix
    stem = src_path.stem
    dst_path_str = parent + '/' + stem + "_" + suffix + ext

    return dst_path_str


def rewrite_colr_box(
        src_fname, colour_primaries, transfer_characteristics,
        matrix_coefficients):
    """
    rewrite the 'colr' box parametes.

    Paramters
    ---------
    src_fname : strings
        source file name.
    colour_primaries : int
        1: BT.709, 9: BT.2020
    transfer_characteristics : int
        1: BT.709, 16: ST 2084, 17: ARIB STD-B67
    matrix_coefficients : int
        1: BT.709, 9: BT.2020
    """
    suffix = f"colr_{colour_primaries}_{transfer_characteristics}"
    suffix += f"_{matrix_coefficients}"
    dst_fname = make_dst_fname(src_fname=src_fname, suffix=suffix)
    shutil.copyfile(src_fname, dst_fname)
    fp = open(dst_fname, 'rb+')
    set_colour_characteristics(
        fp, colour_primaries, transfer_characteristics,
        matrix_coefficients)
    fp.close()


def test_rewrite_colour_parameters():
    src_fname = "./video/BT2020-ST2084_H264.mp4"
    dst_fname = make_dst_fname(src_fname, suffix='colr_10_17_10')
    print(dst_fname)
    shutil.copyfile(src_fname, dst_fname)
    fp = open(dst_fname, 'rb+')
    colour_primaries, transfer_characteristics, matrix_coefficients\
        = get_colour_characteristics(fp)
    print(colour_primaries, transfer_characteristics, matrix_coefficients)
    set_colour_characteristics(fp, 10, 17, 10)


def test_get_colour_characteristics():
    src_fname = "./video/BT2020-ST2084_H264.mp4"
    fp = open(src_fname, 'rb')
    colour_primaries, transfer_characteristics, matrix_coefficients\
        = get_colour_characteristics(fp)
    print(colour_primaries, transfer_characteristics, matrix_coefficients)
    assert_equal(colour_primaries, 9)
    assert_equal(transfer_characteristics, 16)
    assert_equal(matrix_coefficients, 9)


def test_set_colour_characteristics():
    colour_primaries_i = 10
    transfer_characteristics_i = 15
    matrix_coefficients_i = 8
    src_fname = "./video/BT2020-ST2084_H264.mp4"
    dst_fname = make_dst_fname(src_fname, suffix='colr_9_15_8')
    shutil.copyfile(src_fname, dst_fname)
    fp = open(dst_fname, 'rb+')
    set_colour_characteristics(
        fp, colour_primaries_i, transfer_characteristics_i,
        matrix_coefficients_i)
    fp.close()
    fp = open(dst_fname, 'rb')
    colour_primaries_o, transfer_characteristics_o, matrix_coefficients_o\
        = get_colour_characteristics(fp)
    print(
        colour_primaries_o, transfer_characteristics_o, matrix_coefficients_o)
    assert_equal(colour_primaries_i, colour_primaries_o)
    assert_equal(transfer_characteristics_i, transfer_characteristics_o)
    assert_equal(matrix_coefficients_i, matrix_coefficients_o)


def lazy_tests():
    test_get_colour_characteristics()
    test_set_colour_characteristics()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # lazy_tests()
    # test_rewrite_colour_parameters()
    # rewrite_colr_box(
    #     src_fname='./video/BT2020-ST2084_H264_Resolve.mp4',
    #     colour_primaries=9, transfer_characteristics=16, matrix_coefficients=9)
    fname_list = [
        # './video/BT2020-ST2084_H264_command.mp4',
        # './video/src_grad_tp_1920x1080_b-size_64_DV17_HDR10.mp4',
        './video/src_grad_tp_1920x1080_b-size_64_ffmpeg_HDR10.mp4'
    ]
    for fname in fname_list:
        rewrite_colr_box(
            src_fname=fname,
            colour_primaries=1, transfer_characteristics=1,
            matrix_coefficients=1)
