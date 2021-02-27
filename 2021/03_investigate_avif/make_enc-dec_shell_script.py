# -*- coding: utf-8 -*-
"""
decode
======
"""

# import standard libraries
import os

# import third-party libraries

# import my libraries


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def make_avif_name(
        encoder, bit_depth, chroma_subsampling,
        gamut, trc, mtx_coef, range):
    fname = f"./avif/{encoder}_{bit_depth}-bit_"
    fname += f"yuv{chroma_subsampling}_cicp-{gamut}-{trc}-{mtx_coef}_"
    fname += f"{range}-range.avif"

    return fname


def make_decoded_png_name(
        encoder, decoder, bit_depth, chroma_subsampling,
        gamut, trc, mtx_coef, range):
    fname = f"./decoded_png/{encoder}-{decoder}_{bit_depth}-bit_"
    fname += f"yuv{chroma_subsampling}_cicp-{gamut}-{trc}-{mtx_coef}_"
    fname += f"{range}-range.png"

    return fname


def make_script():
    script_name = "encode_decode_avif.sh"
    encoder_list = ['aom', 'rav1e', 'svt']
    decoder_list = ['aom', 'dav1d', 'libgav1']
    bit_depth = 12
    chroma_subsampling = 444
    gamut = 9
    trc = 16
    mtx_coef = 9
    range = 'full'
    src_img = "./png/SMPTE_ST2084_ITU-R_BT.2020_D65_1920x1080_rev04_type1.png"
    encode_ops = f"-d {bit_depth} -y {chroma_subsampling} "
    encode_ops += f"--cicp {gamut}/{trc}/{mtx_coef} -r {range} --min 0 --max 0"

    out_buf = "#!/bin/bash\n"

    # encode
    for encoder in encoder_list:
        avif_name = make_avif_name(
            encoder, bit_depth, chroma_subsampling,
            gamut, trc, mtx_coef, range)
        command = f"avifenc_{encoder} {src_img} {encode_ops} {avif_name}\n"
        print(command)
        out_buf += 'echo " "\n'
        out_buf += f'echo "{command[:-1]}"\n'
        out_buf += command

    # decode
    for encoder in encoder_list:
        for decoder in decoder_list:
            avif_name = make_avif_name(
                encoder, bit_depth, chroma_subsampling,
                gamut, trc, mtx_coef, range)
            png_name = make_decoded_png_name(
                encoder, decoder, bit_depth, chroma_subsampling,
                gamut, trc, mtx_coef, range)
            command = f"avifdec_{decoder} {avif_name} {png_name}\n"
            print(command)
            out_buf += 'echo " "\n'
            out_buf += f'echo "{command[:-1]}"\n'
            out_buf += command

    with open(script_name, 'wt', newline="\n") as f:
        f.write(out_buf)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    make_script()
