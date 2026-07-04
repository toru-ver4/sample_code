# -*- coding: utf-8 -*-

# import standard libraries
import sys
import os
from pathlib import Path
import subprocess

# import third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from colour import write_image, read_image
from colour.algebra import vecmul
from scipy import linalg

# import my libraries
import test_pattern_generator2 as tpg
from create_src_test_pattern import get_10bit_ramp_from_img, calc_rgb_to_ycbcr_matrix
from ffmpeg_analyze_common import (
    WIN_ENCODE_PRESET_LIST,
    SRC_IMAGE_LIST,
    make_decode_output_fname,
    make_encode_output_fname,
    make_raw_yuv_name,
    make_raw_yuv_encoded_name,
    make_raw_yuv_mp4_fname,
    make_raw_yuv_mp4_resolve_decoded_fname
)
import plot_utility as pu


def run_command(args):
    print(" ".join(args))
    subprocess.run(args, check=True)


RAW_YUV_REFERENCE_STREAM_STEM = "DPX_H.265_NVENC_Main10"
RAW_YUV_DE265_OUTPUT_STEM = "dst_1920x1080_I010"
RAW_YUV_RESOLVE_OUTPUT_STEM = "dst_resolve_encoded_1920x1080_I010"


def _make_raw_yuv_analysis_dir(part):
    dir_path = Path("./decode_data/de265/raw_yuv") / part
    dir_path.mkdir(parents=True, exist_ok=True)

    return dir_path


def make_raw_yuv_reference_mp4_fname(encode_app):
    if encode_app == "resolve":
        dir_path = Path("./encode_data/Resolve/pre_resolve_test")
    elif encode_app == "ffmpeg":
        dir_path = Path("./encode_data/FFmpeg/pre_resolve_test")
    else:
        raise ValueError("Invalid encode_app parameter.")

    return str(dir_path / f"{RAW_YUV_REFERENCE_STREAM_STEM}.mp4")


def make_raw_yuv_reference_hevc_fname(encode_app):
    if encode_app == "resolve":
        dir_path = Path("./encode_data/Resolve/bitstream")
    elif encode_app == "ffmpeg":
        dir_path = Path("./encode_data/FFmpeg/bitstream")
    else:
        raise ValueError("Invalid encode_app parameter.")
    dir_path.mkdir(parents=True, exist_ok=True)

    return str(dir_path / f"{RAW_YUV_REFERENCE_STREAM_STEM}.hevc")


def make_raw_yuv_de265_output_fname(encoded_by, suffix):
    valid_encoded_by = {"ffmpeg", "x265", "resolve"}
    if encoded_by not in valid_encoded_by:
        raise ValueError(f"Invalid encoded_by parameter: {encoded_by}")

    dir_path = _make_raw_yuv_analysis_dir(f"enc_{encoded_by}")

    return str(dir_path / f"{RAW_YUV_DE265_OUTPUT_STEM}.{suffix}")


def make_raw_yuv_resolve_output_fname():
    dir_path = Path("./raw")
    dir_path.mkdir(parents=True, exist_ok=True)

    return str(dir_path / f"{RAW_YUV_RESOLVE_OUTPUT_STEM}.yuv")


def check_davinci_resolve_encode_decode_data_core(decoded_img_fname, title, graph_fname):
    print(f"input: {decoded_img_fname}, output: {graph_fname}")
    data = read_image(decoded_img_fname)
    rgb_10bit = np.round(get_10bit_ramp_from_img(img=data) * 1023).astype(np.int16)
    x = np.arange(1024, dtype=np.int16)
    ref_rgb10_bit = np.repeat(x[..., np.newaxis], 3, axis=-1)

    diff = rgb_10bit - ref_rgb10_bit
    channel_colors = [pu.RED, pu.GREEN, pu.BLUE]
    channel_labels = ["R", "G", "B"]
    x_ticks = [x * 64 for x in range(16)] + [1023]
    y_ticks_upper = [x * 64 for x in range(16)] + [1023]
    diff_y_ticks = list(range(-4, 5))

    fig, axes = plt.subplots(
        nrows=2, ncols=1, figsize=(10, 6), sharex=True,
        gridspec_kw={"height_ratios": [6.5, 3.5]}
    )
    ax1, ax2 = axes

    for c_idx, (color, label) in enumerate(zip(channel_colors, channel_labels)):
        ax1.plot(
            x, rgb_10bit[..., c_idx],
            '-', color=color, lw=1.5, label=f"Decoded {label}"
        )
        ax2.plot(
            x, diff[..., c_idx],
            '-o', color=color, lw=1.5, ms=4, label=label
        )

    for c_idx, (color, label) in enumerate(zip(channel_colors, channel_labels)):
        ax1.plot(
            x, ref_rgb10_bit[..., c_idx],
            '--', color=color, alpha=0.5, lw=2, label=f"Reference {label}"
        )


    title = title
    ax1.set_title(title)
    ax1.set_ylabel("Decoded Code Value")
    ax1.set_xlim(0, 1023)
    ax1.set_ylim(-5, 1028)
    ax1.set_xticks(x_ticks)
    ax1.set_yticks(y_ticks_upper)
    ax1.grid(True, which='major', color="#B0B0B0", linestyle='-')
    ax1.legend(loc='upper left', ncol=2)

    ax2.axhline(0, color='k', lw=1, alpha=0.6)
    ax2.set_xlabel("Reference Code Value")
    ax2.set_ylabel("Difference from Reference")
    ax2.set_xlim(0, 1023)
    ax2.set_xticks(x_ticks)
    ax2.set_ylim(-4.2, 4.2)
    ax2.set_yticks(diff_y_ticks)
    ax2.grid(True, which='major', color="#B0B0B0", linestyle='-')
    ax2.legend(loc='upper left', ncol=3)

    save_fname = graph_fname
    fig.tight_layout()
    plt.savefig(save_fname)
    plt.close(fig)


def check_decode_data(encode_app, decode_app):
    encode_preset_list = WIN_ENCODE_PRESET_LIST
    src_image_list = SRC_IMAGE_LIST

    for encode_preset in encode_preset_list:
        for src_image in src_image_list:
            fname_base = make_decode_output_fname(
                src_image=src_image,
                encode_preset=encode_preset,
                encode_app=encode_app,
                decode_app=decode_app
            )
            fname = fname_base + "_00086400.png"
            print(fname)
            title = str(Path(fname).stem)
            print(f"TITLE = {title}")
            graph_fname = f"./debug/enc-{encode_app}_dec-{decode_app}_{title}.png"
            check_davinci_resolve_encode_decode_data_core(
                decoded_img_fname=fname,
                title=title,
                graph_fname=graph_fname
            )
        #     break
        # break


def ffmpeg_decode_to_single_image_core(mp4_fname, decoded_fname):
        cmd = 'ffmpeg'
        ops = [
            "-hide_banner",
            "-i", mp4_fname,
            "-map", "0:v:0",
            "-frames:v", "1",
            "-vf", "scale=in_range=limited:out_range=full",
            "-pix_fmt", "rgb48be",
            "-update", "1",
            decoded_fname,
            "-y"
        ]
        args = [cmd] + ops
        run_command(args)


def check_enc_davinci_dec_ffmpeg_data():
    encode_preset_list = WIN_ENCODE_PRESET_LIST
    src_image_list = SRC_IMAGE_LIST
    decode_dir = "./decode_data/FFmpeg/DaVinci_Enc/"

    for encode_preset in encode_preset_list:
        for src_image in src_image_list:
            fname_base = make_encode_output_fname(
                src_image=src_image, encode_preset=encode_preset
            )
            mp4_fname = fname_base + ".mp4"
            decoded_fname = decode_dir + str(Path(mp4_fname).stem) + ".png"
            title = str(Path(fname_base).stem)
            graph_fname = f"./debug/enc-resolve_dec-ffmpeg_{title}.png"
            check_davinci_resolve_encode_decode_data_core(
                decoded_img_fname=decoded_fname,
                title=title,
                graph_fname=graph_fname
            )


def encode_ffmpeg(src_image, encoder, output_fname):
    cmd = "ffmpeg"
    ops = [
        '-hide_banner',
        '-loop', '1',
        '-color_primaries', 'bt709',
        '-color_trc', 'bt709',
        '-colorspace', 'bt709',
        '-framerate', "24",
        '-t', "5",
        '-i', src_image,
        '-c:v', encoder,
        '-pix_fmt', 'yuv420p10le',
        '-color_primaries', 'bt709',
        '-color_trc', 'bt709',
        '-colorspace', 'bt709',
        '-qp', '0',
        output_fname, '-y',
    ]
    args = [cmd] + ops
    run_command(args)


def encode_with_ffmpeg():
    encode_preset_list = WIN_ENCODE_PRESET_LIST
    src_image_list = SRC_IMAGE_LIST

    for encode_preset in encode_preset_list:
        for src_image in src_image_list:
            mp4_fname_base = make_encode_output_fname(
                src_image=src_image, encode_preset=encode_preset, encode_app='ffmpeg'
            )
            mp4_fname = mp4_fname_base + ".mp4"
            if "H.265" in encode_preset:
                encoder = "libx265"
            elif "AV1" in encode_preset:
                encoder = "libsvtav1"
            else:
                raise ValueError("Invalid encoder parameters.")
            encode_ffmpeg(
                src_image=src_image, encoder=encoder, output_fname=mp4_fname
            )


def decode_mp4_with_ffmpeg(encode_app):
    encode_preset_list = WIN_ENCODE_PRESET_LIST
    src_image_list = SRC_IMAGE_LIST
    if encode_app == 'resolve':
        decode_dir = "./decode_data/FFmpeg/enc_resolve/"
    elif encode_app == 'ffmpeg':
        decode_dir = "./decode_data/FFmpeg/enc_ffmpeg/"
    else:
        raise ValueError("Invalid encode_app parameter")
    Path(decode_dir).mkdir(parents=True, exist_ok=True)

    for encode_preset in encode_preset_list:
        for src_image in src_image_list:
            fname_base = make_encode_output_fname(
                src_image=src_image, encode_preset=encode_preset, encode_app=encode_app
            )
            mp4_fname = fname_base + ".mp4"
            decoded_fname = decode_dir + str(Path(mp4_fname).stem) + "_00086400.png"
            ffmpeg_decode_to_single_image_core(
                mp4_fname=mp4_fname, decoded_fname=decoded_fname
            )
        #     break
        # break


def encode_raw_yuv_to_hevc_with_x265():
    output_fname = make_raw_yuv_encoded_name(encoder="x265")
    cmd = "x265cli.exe"
    ops = [
        "--input", make_raw_yuv_name(),
        "--input-res", "1920x1080",
        "--fps", "24",
        "--frames", "120",
        "--input-depth", "10",
        "--input-csp", "i420",
        "--profile", "main10",
        "--lossless",
        "--output", output_fname
    ]
    args = [cmd] + ops
    run_command(args)


def hevc_to_mp4(output_fname):
    input_fname = make_raw_yuv_encoded_name(encoder="x265")
    cmd = "ffmpeg.exe"
    ops = [
        '-hide_banner',
        "-i", input_fname,
        output_fname,
        '-y'
    ]
    args = [cmd] + ops
    run_command(args)


def decode_hevc_with_resolve_to_raw_yuv():
    input_fname = make_raw_yuv_reference_hevc_fname(encode_app="resolve")
    output_fname = make_raw_yuv_resolve_output_fname()
    cmd = "x265cli.exe"
    ops = [
        "--input", input_fname,
        "--input-res", "1920x1080",
        "--fps", "24",
        "--frames", "120",
        "--input-depth", "10",
        "--input-csp", "i420",
        "--profile", "main10",
        "--lossless",
        "--output", output_fname
    ]
    args = [cmd] + ops
    run_command(args)


def extract_hevc_bitstream(in_fname, out_fname):
    cmd = "ffmpeg.exe"
    ops = [
        '-hide_banner',
        "-i", in_fname,
        "-map", "0:v:0",
        "-c:v", "copy",
        "-bsf:v", "hevc_mp4toannexb",
        "-f", "hevc",
        out_fname,
        '-y'
    ]
    args = [cmd] + ops
    run_command(args)


def extract_ffmpeg_resolve_hevc_bitstream():
    # resolve
    input_fname = make_raw_yuv_reference_mp4_fname(encode_app="resolve")
    output_fname = make_raw_yuv_reference_hevc_fname(encode_app="resolve")
    extract_hevc_bitstream(in_fname=input_fname, out_fname=output_fname)

    # ffmpeg
    input_fname = make_raw_yuv_reference_mp4_fname(encode_app="ffmpeg")
    output_fname = make_raw_yuv_reference_hevc_fname(encode_app="ffmpeg")
    extract_hevc_bitstream(in_fname=input_fname, out_fname=output_fname)


def decode_with_de265(in_fname, out_fname):
    cmd = "dec265"
    ops = [
        "-o", out_fname,
        "-f", "1",
        "-v", in_fname
    ]
    args = [cmd] + ops
    run_command(args)


def decode_yuv420p10le_1frame(in_fname, out_fname, width=1920, height=1080):
    print(f"input = {in_fname}, output = {out_fname}")
    y_size = width * height
    uv_width = width // 2
    uv_height = height // 2
    uv_size = uv_width * uv_height
    total_samples = y_size + uv_size * 2

    frame = np.fromfile(in_fname, dtype='<u2', count=total_samples)
    if frame.size != total_samples:
        raise ValueError(
            f"Invalid frame size. expected={total_samples} samples, "
            f"actual={frame.size} samples."
        )

    y = frame[:y_size].reshape((height, width))
    u_420 = frame[y_size:y_size + uv_size].reshape((uv_height, uv_width))
    v_420 = frame[y_size + uv_size:].reshape((uv_height, uv_width))

    # 4:2:0 chroma planes are expanded back to luma resolution with NN.
    u = np.repeat(np.repeat(u_420, 2, axis=0), 2, axis=1)
    v = np.repeat(np.repeat(v_420, 2, axis=0), 2, axis=1)

    y = (y.astype(np.int16) - 64) / (219 * 4)
    u = (u.astype(np.int16) - 512) / (224 * 4)
    v = (v.astype(np.int16) - 512) / (224 * 4)

    mtx = linalg.inv(calc_rgb_to_ycbcr_matrix(gamut='bt.709'))
    yuv = np.dstack([y, u, v])
    rgb = vecmul(mtx, yuv)

    write_image(rgb, out_fname, bit_depth='uint16')


def decode_x265_ffmpeg_resolve_x265_encoded_data_with_de265():
    # ffmpeg
    input_fname = make_raw_yuv_reference_hevc_fname(encode_app="ffmpeg")
    output_fname = make_raw_yuv_de265_output_fname(encoded_by="ffmpeg", suffix="yuv")
    decode_with_de265(in_fname=input_fname, out_fname=output_fname)

    # x265
    input_fname = make_raw_yuv_encoded_name(encoder='x265')
    output_fname = make_raw_yuv_de265_output_fname(encoded_by="x265", suffix="yuv")
    decode_with_de265(in_fname=input_fname, out_fname=output_fname)

    # resolve
    input_fname = make_raw_yuv_reference_hevc_fname(encode_app="resolve")
    output_fname = make_raw_yuv_de265_output_fname(encoded_by="resolve", suffix="yuv")
    decode_with_de265(in_fname=input_fname, out_fname=output_fname)


def yuv_to_png_x265_ffmpeg_encoded_data_with_de265():
    # ffmpeg
    input_fname = make_raw_yuv_de265_output_fname(encoded_by="ffmpeg", suffix="yuv")
    output_fname = make_raw_yuv_de265_output_fname(encoded_by="ffmpeg", suffix="png")
    decode_yuv420p10le_1frame(in_fname=input_fname, out_fname=output_fname)

    # x265
    input_fname = make_raw_yuv_de265_output_fname(encoded_by="x265", suffix="yuv")
    output_fname = make_raw_yuv_de265_output_fname(encoded_by="x265", suffix="png")
    decode_yuv420p10le_1frame(in_fname=input_fname, out_fname=output_fname)

    # resolve
    input_fname = make_raw_yuv_de265_output_fname(encoded_by="resolve", suffix="yuv")
    output_fname = make_raw_yuv_de265_output_fname(encoded_by="resolve", suffix="png")
    decode_yuv420p10le_1frame(in_fname=input_fname, out_fname=output_fname)


def make_raw_yuv_title_from_fname(img_fname):
    return f"RAW_YUV {str(Path(img_fname).stem)}"


def make_raw_yuv_graph_fname(img_fname, enc_app="", dec_app="de265"):
    return f"./debug/enc-{enc_app}_dec-{dec_app}_{Path(img_fname).stem}.png"


def plot_de265_decoded_image():
    encode_app_list = ['resolve', 'ffmpeg', 'x265']

    for encode_app in encode_app_list:
        img_fname = make_raw_yuv_de265_output_fname(encoded_by=encode_app, suffix="png")
        title = make_raw_yuv_title_from_fname(img_fname)
        graph_fname = make_raw_yuv_graph_fname(img_fname=img_fname, enc_app=encode_app, dec_app="de265")
        check_davinci_resolve_encode_decode_data_core(
            decoded_img_fname=img_fname, title=title, graph_fname=graph_fname
        )


def check_raw_yuv_10bit_data_encode_ffmpeg():
    # # --------------------------
    # # raw yuv
    # # --------------------------
    # encode_raw_yuv_to_hevc_with_x265()
    # extract_ffmpeg_resolve_hevc_bitstream()

    # # --------------------------
    # # decode x265 encoded data with davinci resolve
    # # --------------------------
    # # convert from .hevc to .mp4
    # mp4_for_resolve_fname = make_raw_yuv_mp4_fname()
    # hevc_to_mp4(output_fname=mp4_for_resolve_fname) 
    # # open encode_decode_with_resolve.py and run `decode_core`

    # # --------------------------
    # # decode x265 and FFmpeg encoded data with de265
    # # --------------------------
    # decode_x265_ffmpeg_resolve_x265_encoded_data_with_de265()

    # # ----------------------------------
    # # yuv to png
    # # ----------------------------------
    # yuv_to_png_x265_ffmpeg_encoded_data_with_de265()

    # # --------------------------
    # # plot decoded image data
    # # --------------------------
    # plot_de265_decoded_image()
    pass


def encode_raw_yuv_using_ffmpeg(src_yuv, output_fname):
    cmd = "ffmpeg"
    ops = [
        '-hide_banner',
        '-color_primaries', 'bt709',
        '-color_trc', 'bt709',
        '-colorspace', 'bt709',
        '-f', 'rawvideo',
        '-pix_fmt', 'yuv420p10le',
        '-video_size', '1920x1080',
        '-framerate', "24",
        '-i', src_yuv,
        '-c:v', "libx265",
        '-pix_fmt', 'yuv420p10le',
        '-color_primaries', 'bt709',
        '-color_trc', 'bt709',
        '-colorspace', 'bt709',
        '-qp', '0',
        output_fname, '-y',
    ]
    args = [cmd] + ops
    run_command(args)


def encode_raw_yuv_using_ffmpeg_x265_davinci_resolve():
    # ffmpeg
    input_fname = make_raw_yuv_name()
    output_fname = make_raw_yuv_encoded_name(encoder='ffmpeg')
    encode_raw_yuv_using_ffmpeg(src_yuv=input_fname, output_fname=output_fname)

    # x265
    encode_raw_yuv_to_hevc_with_x265()


def make_output_fname_for_encoded_from_i010_and_decode_de265(enc_app, ext):
    target_dir = Path("./decode_data/de265") / "encoded_from_raw_yuv" / f"enc_{enc_app}"
    target_dir.mkdir(parents=True, exist_ok=True)

    return str(target_dir / f"dst_1920x1080_I010.{ext}")


def decode_hevc_with_de265():
    # ffmpeg
    input_fname = make_raw_yuv_encoded_name(encoder='ffmpeg')
    output_fname = make_output_fname_for_encoded_from_i010_and_decode_de265(enc_app='ffmpeg', ext='yuv')
    decode_with_de265(in_fname=input_fname, out_fname=output_fname)

    # x265
    input_fname = make_raw_yuv_encoded_name(encoder='x265')
    output_fname = make_output_fname_for_encoded_from_i010_and_decode_de265(enc_app='x265', ext='yuv')
    decode_with_de265(in_fname=input_fname, out_fname=output_fname)


def yuv_to_png_for_encoded_from_i010_and_decode_de265():
    # ffmpeg
    input_fname = make_output_fname_for_encoded_from_i010_and_decode_de265(enc_app='ffmpeg', ext='yuv')
    output_fname = make_output_fname_for_encoded_from_i010_and_decode_de265(enc_app='ffmpeg', ext='png')
    decode_yuv420p10le_1frame(in_fname=input_fname, out_fname=output_fname)

    # x265
    input_fname = make_output_fname_for_encoded_from_i010_and_decode_de265(enc_app='x265', ext='yuv')
    output_fname = make_output_fname_for_encoded_from_i010_and_decode_de265(enc_app='x265', ext='png')
    decode_yuv420p10le_1frame(in_fname=input_fname, out_fname=output_fname)


def plot_encoded_from_i010_and_decode_de265_data():
    # ffmpeg
    img_fname = make_output_fname_for_encoded_from_i010_and_decode_de265(enc_app='ffmpeg', ext='png')
    title = "Encoded HEVC with FFmpeg from I010 file"
    graph_fname = "./debug/enc-ffmepg_dec_de265_from_i010.png"
    check_davinci_resolve_encode_decode_data_core(
        decoded_img_fname=img_fname, title=title, graph_fname=graph_fname
    )

    # x265
    img_fname = make_output_fname_for_encoded_from_i010_and_decode_de265(enc_app='x265', ext='png')
    title = "Encoded HEVC with x265 from I010 file"
    graph_fname = "./debug/enc-x265_dec_de265_from_i010.png"
    check_davinci_resolve_encode_decode_data_core(
        decoded_img_fname=img_fname, title=title, graph_fname=graph_fname
    )


def check_raw_yuv_10bit_data_decode_ffmpeg():
    # encode_raw_yuv_using_ffmpeg_x265_davinci_resolve()
    # decode_hevc_with_de265()
    # yuv_to_png_for_encoded_from_i010_and_decode_de265()
    plot_encoded_from_i010_and_decode_de265_data()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # encode_with_ffmpeg()

    # decode_mp4_with_ffmpeg(encode_app='resolve')
    # decode_mp4_with_ffmpeg(encode_app='ffmpeg')

    # check_decode_data(encode_app='resolve', decode_app='resolve')
    # check_decode_data(encode_app='resolve', decode_app='ffmpeg')
    # check_decode_data(encode_app='ffmpeg', decode_app='resolve')
    # check_decode_data(encode_app='ffmpeg', decode_app='ffmpeg')

    check_raw_yuv_10bit_data_encode_ffmpeg()
    # check_raw_yuv_10bit_data_decode_ffmpeg()
