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

# import my libraries
import test_pattern_generator2 as tpg
from create_src_test_pattern import get_10bit_ramp_from_img
from encode_decode_with_resolve import make_decode_output_fname, make_encode_output_fname
import plot_utility as pu


def make_encode_with_ffmpeg_output_fname(src_image, encode_preset):
    encode_preset_stem = Path(encode_preset).stem
    dir_path = Path("./encode_data/FFmpeg") / "pre_resolve_test"
    dir_path.mkdir(parents=True, exist_ok=True)
    basename = f"{(Path(src_image).suffix[1:]).upper()}_{encode_preset_stem}"
    output_fname = str(dir_path / basename)

    return output_fname


def check_davinci_resolve_encode_decode_data_core(decoded_img_fname, title, graph_fname):
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


def check_davinci_resolve_encode_decode_data():
    encode_preset_list = [
        "./resolve_encode_preset/H.265_NVENC_Main10.xml",
        "./resolve_encode_preset/AV1_NVENC_Main10.xml",
    ]
    src_image_list = [
        "./img/src_img.dpx",
        "./img/src_img.png",
        "./img/src_img.tif"
    ]

    for encode_preset in encode_preset_list:
        for src_image in src_image_list:
            fname_base = make_decode_output_fname(
                src_image=src_image, encode_preset=encode_preset
            )
            fname = fname_base + "_00086400.png"
            print(fname)
            title = str(Path(fname_base).stem)
            graph_fname = f"./debug/enc-resolve_dec-resolve_{title}.png"
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
        print(" ".join(args))
        subprocess.run(args)


def decode_davinci_mp4_with_ffmpeg():
    encode_preset_list = [
        "./resolve_encode_preset/H.265_NVENC_Main10.xml",
        "./resolve_encode_preset/AV1_NVENC_Main10.xml",
    ]
    src_image_list = [
        "./img/src_img.dpx",
        "./img/src_img.png",
        "./img/src_img.tif"
    ]
    decode_dir = "./decode_data/FFmpeg/DaVinci_Enc/"

    for encode_preset in encode_preset_list:
        for src_image in src_image_list:
            fname_base = make_encode_output_fname(
                src_image=src_image, encode_preset=encode_preset
            )
            mp4_fname = fname_base + ".mp4"
            decoded_fname = decode_dir + str(Path(mp4_fname).stem) + ".png"
            ffmpeg_decode_to_single_image_core(
                mp4_fname=mp4_fname, decoded_fname=decoded_fname
            )
        #     break
        # break


def check_enc_davinci_dec_ffmpeg_data():
    encode_preset_list = [
        "./resolve_encode_preset/H.265_NVENC_Main10.xml",
        "./resolve_encode_preset/AV1_NVENC_Main10.xml",
    ]
    src_image_list = [
        "./img/src_img.dpx",
        "./img/src_img.png",
        "./img/src_img.tif"
    ]
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
    print(" ".join(args))
    subprocess.run(args)


def encode_with_ffmpeg():
    encode_preset_list = [
        "./resolve_encode_preset/H.265_NVENC_Main10.xml",
        "./resolve_encode_preset/AV1_NVENC_Main10.xml",
    ]
    src_image_list = [
        "./img/src_img.dpx",
        "./img/src_img.png",
        "./img/src_img.tif"
    ]

    for encode_preset in encode_preset_list:
        for src_image in src_image_list:
            mp4_fname_base = make_encode_with_ffmpeg_output_fname(
                src_image=src_image, encode_preset=encode_preset
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


def decode_ffmpeg_mp4_with_ffmpeg():
    encode_preset_list = [
        "./resolve_encode_preset/H.265_NVENC_Main10.xml",
        "./resolve_encode_preset/AV1_NVENC_Main10.xml",
    ]
    src_image_list = [
        "./img/src_img.dpx",
        "./img/src_img.png",
        "./img/src_img.tif"
    ]

    decode_dir = "./decode_data/FFmpeg/FFmpeg_Enc/"

    for encode_preset in encode_preset_list:
        for src_image in src_image_list:
            mp4_fname_base = make_encode_with_ffmpeg_output_fname(
                src_image=src_image, encode_preset=encode_preset
            )
            mp4_fname = mp4_fname_base + ".mp4"
            decoded_fname = decode_dir + str(Path(mp4_fname).stem) + ".png"
            ffmpeg_decode_to_single_image_core(
                mp4_fname=mp4_fname, decoded_fname=decoded_fname
            )
        #     break
        # break


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # check_davinci_resolve_encode_decode_data()

    # decode_davinci_mp4_with_ffmpeg()
    # check_enc_davinci_dec_ffmpeg_data()

    # encode_with_ffmpeg()
    # decode_ffmpeg_mp4_with_ffmpeg()

    # DaVinci でデコードするやつ
    # DaVinci でデコードしたやつを解析プロットするやつ
    # FFmpeg でデコードしたやつを解析プロットするやつ
