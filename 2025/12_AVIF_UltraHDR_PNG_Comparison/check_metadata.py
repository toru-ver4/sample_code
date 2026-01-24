from pathlib import Path
import os
import sys
import subprocess
from itertools import product

import numpy as np

THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from create_hdr_media import (
    MDCV_PRIMARIES_LIST,
    MDCV_LUMINANCE_LIST,
    CLLI_LUMINANCE_LIST,
    KIND_AV1,
    KIND_AVIF,
    KIND_HEVC,
    make_media_file_name_without_ext
)

def dump_container_with_mp4box():
    input_file_list = []
    kind_list = [KIND_HEVC, KIND_AV1, KIND_AVIF]
    extension_list = ['.mp4', '.mov', '.avif']
    mdcv_primaries_list = MDCV_PRIMARIES_LIST + [None]
    mdcv_luminance_list = MDCV_LUMINANCE_LIST + [None]
    for mdcv_primaries, mdcv_luminance, clli_luminance, kind, extension\
        in product(mdcv_primaries_list, mdcv_luminance_list, CLLI_LUMINANCE_LIST, kind_list, extension_list):

        input_file_without_ext = make_media_file_name_without_ext(
            kind=kind,
            suffix=None,
            mdcv_primaries=mdcv_primaries,
            mdcv_luminance=mdcv_luminance,
            clli_luminance=clli_luminance
        )
        input_file = input_file_without_ext + extension
        if os.path.exists(input_file):
            input_file_list.append(input_file)
    for input_file in input_file_list:
        dump_container_with_mp4box_core(input_file=input_file)


def convert_xml_to_json(xml_file, json_file):
    with open(json_file, "w", encoding="utf-8") as f:
        cmd = ["dasel"]
        ops = ['-f', xml_file, '-r', 'xml', '-w', "json"]
        args = cmd + ops
        print(" ".join(args) + f" > {json_file}")
        subprocess.run(args, stdout=f)


def dump_container_with_mp4box_core(input_file):
    output_xml_file = f"./data/{Path(input_file).name}.xml"
    output_json_file = f"./data/{Path(input_file).name}.json"
    with open(output_xml_file, "w", encoding="utf-8") as f:
        cmd = ["MP4Box"]
        ops = ["-stdb", "-dxml", input_file]
        args = cmd + ops
        print(" ".join(args) + f" > {output_xml_file}")
        subprocess.run(args, stdout=f)

    convert_xml_to_json(xml_file=output_xml_file, json_file=output_json_file)
    Path(output_xml_file).unlink(missing_ok=True)


def dump_bitstream_with_gpac():
    input_file_list = []
    kind_list = [KIND_HEVC, KIND_AV1, KIND_AVIF]
    extension_list = ['.obu', '.h265']
    mdcv_primaries_list = MDCV_PRIMARIES_LIST + [None]
    mdcv_luminance_list = MDCV_LUMINANCE_LIST + [None]
    for mdcv_primaries, mdcv_luminance, clli_luminance, kind, extension\
        in product(mdcv_primaries_list, mdcv_luminance_list, CLLI_LUMINANCE_LIST, kind_list, extension_list):

        input_file_without_ext = make_media_file_name_without_ext(
            kind=kind,
            suffix=None,
            mdcv_primaries=mdcv_primaries,
            mdcv_luminance=mdcv_luminance,
            clli_luminance=clli_luminance
        )
        input_file = input_file_without_ext + extension
        if os.path.exists(input_file):
            input_file_list.append(input_file)

    for input_file in input_file_list:
        dump_bitstream_with_gpac_core(input_file=input_file)


def dump_bitstream_with_gpac_core(input_file):
    output_xml_file = f"./data/{Path(input_file).name}.xml"
    output_json_file = f"./data/{Path(input_file).name}.json"
    with open(output_xml_file, "w", encoding="utf-8") as f:
        cmd = ["gpac"]
        ops = ["-i", input_file, "inspect:deep:analyze=on"]
        args = cmd + ops
        print(" ".join(args) + f" > {output_xml_file}")
        subprocess.run(args, stdout=f)

    convert_xml_to_json(xml_file=output_xml_file, json_file=output_json_file)
    Path(output_xml_file).unlink(missing_ok=True)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    dump_container_with_mp4box()
    dump_bitstream_with_gpac()
