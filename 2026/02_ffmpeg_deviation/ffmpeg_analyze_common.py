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
    "./img/src_img.tif",
    "./img/src_img.exr",
]


def make_encode_output_fname(src_image, encode_preset, encode_app):
    # output file settings
    encode_preset_stem = Path(encode_preset).stem
    if encode_app == 'resolve':
        dir_path = Path("./encode_data/Resolve") / "pre_resolve_test"
    elif encode_app == 'ffmpeg':
        dir_path = Path("./encode_data/FFmpeg") / "pre_resolve_test"
    else:
        raise ValueError("Invalid encode_app parameter")
    dir_path.mkdir(parents=True, exist_ok=True)
    basename = f"{(Path(src_image).suffix[1:]).upper()}_{encode_preset_stem}"
    output_fname = str(dir_path / basename)

    return output_fname


def make_decode_output_fname(src_image, encode_preset, encode_app, decode_app):
    # output file settings
    encode_preset_stem = Path(encode_preset).stem
    if decode_app == 'resolve':
        dir_path = Path("./decode_data/Resolve") / f"enc_{encode_app}"
    elif decode_app == 'ffmpeg':
        dir_path = Path("./decode_data/FFmpeg") / f"enc_{encode_app}"
    else:
        raise ValueError("Invalid decode parameter.")
    dir_path.mkdir(parents=True, exist_ok=True)
    basename = f"{(Path(src_image).suffix[1:]).upper()}_{encode_preset_stem}"
    decoded_image = str(dir_path / basename)

    return decoded_image


def make_raw_yuv_encoded_name(encoder):
    if encoder == "x265":
        target_dir = Path("./encode_data/x265")
        target_dir.mkdir(parents=True, exist_ok=True)
        fname = str(target_dir / "raw_1920x1080_I010_encoded.hevc")
    elif encoder == 'ffmpeg':
        target_dir = Path("./encode_data/FFmpeg")
        target_dir.mkdir(parents=True, exist_ok=True)
        fname = str(target_dir / "raw_1920x1080_I010_encoded.hevc")
    elif encoder == 'resolve':
        target_dir = Path("./encode_data/Resolve")
        target_dir.mkdir(parents=True, exist_ok=True)
        fname = str(target_dir / "raw_1920x1080_I010_encoded.hevc")
    else:
        raise ValueError("Invalid encoder name.")

    return fname


def make_raw_yuv_name():
    target_dir = Path("./raw")
    target_dir.mkdir(parents=True, exist_ok=True)
    fname = str(target_dir / "src_1920x1080_I010.yuv")

    return fname


def make_n_bit_yuv420_name(bit_depth, gamut):
    target_dir = Path("./raw")
    target_dir.mkdir(parents=True, exist_ok=True)
    fname = str(target_dir / f"ref_1920x1080_yuv420p{bit_depth}le_{gamut}.yuv")

    return fname


def make_raw_yuv_mp4_fname():
    return "./encode_data/x265/raw_I010_x265_hevc.mp4"


def make_raw_yuv_mp4_resolve_decoded_fname():
    return "./decode_data/Resolve/enc_x265/raw_I010_x265_hevc_"


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
