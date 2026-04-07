# -*- coding: utf-8 -*-
import os
from pathlib import Path

GRAY_PATCH_SIZE = 32
COLOR_PATCH_SIZE = 64

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080


WIN_ENCODE_PRESET_LIST = [
    "./resolve_encode_preset/H.265_NVENC_Main10.xml",
    "./resolve_encode_preset/AV1_NVENC_Main10.xml",
]

SRC_IMAGE_LIST = [
    "./img/src_img.dpx",
    "./img/src_img.png",
    "./img/src_img.tif"
]


def make_encode_output_fname(src_image, encode_preset):
    # output file settings
    encode_preset_stem = Path(encode_preset).stem
    dir_path = Path("./encode_data/Resolve") / "pre_resolve_test"
    dir_path.mkdir(parents=True, exist_ok=True)
    basename = f"{(Path(src_image).suffix[1:]).upper()}_{encode_preset_stem}"
    output_fname = str(dir_path / basename)

    return output_fname


def make_decode_output_fname(src_image, encode_preset, encode_app):
    # output file settings
    encode_preset_stem = Path(encode_preset).stem
    if encode_app == "resolve":
        dir_path = Path("./decode_data/Resolve") / "enc_resolve"
    elif encode_app == 'ffmpeg':
        dir_path = Path("./decode_data/Resolve") / "enc_ffmpeg"
    else:
        raise ValueError("Invalid encode_app parameter")
    dir_path.mkdir(parents=True, exist_ok=True)
    basename = f"{(Path(src_image).suffix[1:]).upper()}_{encode_preset_stem}"
    decoded_image = str(dir_path / basename)

    return decoded_image


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
