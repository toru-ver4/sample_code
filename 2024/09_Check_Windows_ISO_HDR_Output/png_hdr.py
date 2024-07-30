# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
import struct

# import third-party libraries
import numpy as np
from PIL import Image, PngImagePlugin

# import my libraries
import color_space as cs

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def read_png_chunks(png_file):
    chunks = {}
    with open(png_file, 'rb') as f:
        # Verify that the file is a valid PNG file
        assert f.read(8) == b'\x89PNG\r\n\x1a\n'

        while True:
            # Read the chunk length and type
            chunk_header = f.read(8)
            if len(chunk_header) == 0:
                break

            chunk_length, chunk_type = struct.unpack('>I4s', chunk_header)

            # Read the chunk data and CRC
            chunk_data = f.read(chunk_length)
            _ = f.read(4)

            chunk_type = chunk_type.decode('ascii')
            if chunk_type in ['cICP', 'mDCv', 'cLLi', 'sRGB', 'iCCP']:
                chunks[chunk_type] = chunk_data

    return chunks


def print_chunk_data(chunk_type, data):
    print(f"Chunk Type: {chunk_type}")
    print(f"Chunk Data (hex): {data.hex()}")
    print(f"Chunk Data (ascii): {data.decode('ascii', errors='replace')}")
    print()


def check_chunk_main(png_file):
    chunks = read_png_chunks(png_file)

    for chunk_type, data in chunks.items():
        print_chunk_data(chunk_type, data)


def create_cicp_chunk_data(color_space_name):
    if color_space_name == cs.BT709:
        cicp = [1, 1, 1, 1]
    elif color_space_name == cs.P3_D65:
        cicp = [12, 16, 0, 1]
    elif color_space_name == cs.BT2020:
        cicp = [9, 16, 9, 1]
    else:
        raise ValueError("invalid color space name for cicp")
    return cicp


def create_mDCv_chunk_data(
        color_space_name=cs.BT2020, white_point=cs.D65,
        mastering_min_luminance=0.001, mastering_max_luminance=1000):
    primaries_divisor_value = 0.00002
    white_point_divisor_value = 0.00002
    luminance_divisor_value = 0.0001
    primaries = cs.get_primaries(color_space_name=color_space_name)
    primaries_decimal = np.round(primaries / primaries_divisor_value)\
        .astype(np.uint16).flatten().tolist()

    white_point_decimal = np.round(white_point / white_point_divisor_value)\
        .astype(np.uint16).flatten().tolist()

    max_min_luminance = np.array(
        [mastering_max_luminance, mastering_min_luminance])
    max_min_luminance = max_min_luminance / luminance_divisor_value
    max_min_luminance_decimal = np.round(max_min_luminance).astype(np.uint32)\
        .tolist()

    return primaries_decimal, white_point_decimal, max_min_luminance_decimal


def create_cLLi_chunk_data(maxcll=1000, maxfall=400):
    luminance_divisor_value = 0.0001
    clli_luminance = np.array([maxcll, maxfall])
    clli_luminance = clli_luminance / luminance_divisor_value
    clli_luminance_decimal = np.round(clli_luminance).astype(np.uint32)\
        .tolist()

    return clli_luminance_decimal


def putchunk_hook(fp, cid, *data):
    if cid == b"haxx":
        cid = b"cICP"
    elif cid == b"haxb":
        cid = b"mDCv"
    elif cid == b"haxc":
        cid = b"cLLi"
    return PngImagePlugin.putchunk(fp, cid, *data)


def add_chunks_to_png(
        input_file, output_file,
        file_container_color_space_name=cs.BT2020,
        mastering_display_color_space_name=cs.BT2020,
        mastering_display_white_point=cs.D65,
        mastering_display_max_min_lumiannce=[1000, 0.001],
        maxcll_maxfall=[1000, 400]
):
    # create chunk data
    cicp = create_cicp_chunk_data(
        color_space_name=file_container_color_space_name)

    primaries, white_point, max_min_luminance = create_mDCv_chunk_data(
        color_space_name=mastering_display_color_space_name,
        white_point=mastering_display_white_point,
        mastering_max_luminance=mastering_display_max_min_lumiannce[0],
        mastering_min_luminance=mastering_display_max_min_lumiannce[1]
    )

    clli_list = create_cLLi_chunk_data(
        maxcll=maxcll_maxfall[0], maxfall=maxcll_maxfall[1]
    )

    # Open the image
    with Image.open(input_file) as im:

        im.convert("I;16")
        # Create a PngInfo object
        pnginfo = PngImagePlugin.PngInfo()

        # cICP chunk data
        cicp_data = bytes(cicp)
        pnginfo.add(b"haxx", cicp_data)

        # # mDCv chunk data (example data, big-endian 2-byte integers
        # # for chromaticity and 4-byte integers for luminance)
        # mdcv_data = struct.pack(
        #     '>6H2H2I',
        #     *primaries,  # Display primaries
        #     *white_point,  # White point
        #     *max_min_luminance)  # Max, Min luminance
        # pnginfo.add(b"haxb", mdcv_data)

        # # cLLi chunk data
        # clli_data = struct.pack('>2I', *clli_list)  # MaxCLL, MaxFALL
        # pnginfo.add(b"haxc", clli_data)

        im.encoderinfo = {"pnginfo": pnginfo}

        print(f"save {output_file}")
        with open(output_file, 'wb') as om:
            PngImagePlugin._save(im, om, output_file, chunk=putchunk_hook)


def debug_add_chunks():
    src_fname = "./debug/src_tp/HDR_TP_3840x960.png"
    dst_fname = "./debug/src_tp/HDR_TP_3840x960_cicp.png"
    add_chunks_to_png(
        input_file=src_fname,
        output_file=dst_fname,
        file_container_color_space_name=cs.BT2020,
        mastering_display_color_space_name=cs.BT2020,
        mastering_display_white_point=cs.D65,
        mastering_display_max_min_lumiannce=[1000, 0.001],
        maxcll_maxfall=[1000, 400]
    )


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # check_chunk_main(png_file="./debug/src_tp/HDR_TP_3840x960_cicp.png")
    # check_chunk_main(png_file="./debug/png/TP_Rec2100-PQ.png")
    debug_add_chunks()
