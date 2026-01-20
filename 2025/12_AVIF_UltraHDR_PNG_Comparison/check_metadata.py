from pathlib import Path
import os
import sys
import subprocess

import numpy as np


def dump_container_with_mp4box():
    input_file_list = [
        "./video/test_10000-nits_h265.mp4",
        "./video/test_10000-nits_h265.mov",
        "./video/test_10000-nits_av1.mp4",
        "./video/test_10000-nits_av1.mov",
    ]
    for input_file in input_file_list:
        dump_container_with_mp4box_core(input_file=input_file)


def dump_container_with_mp4box_core(input_file):
    output_xml_file = f"./data/{Path(input_file).name}.xml"
    output_json_file = f"./data/{Path(input_file).name}.json"
    with open(output_xml_file, "w", encoding="utf-8") as f:
        cmd = ["MP4Box"]
        ops = ["-stdb", "-dxml", input_file]
        args = cmd + ops
        print(" ".join(args) + f" > {output_xml_file}")
        subprocess.run(args, stdout=f)

    with open(output_json_file, "w", encoding="utf-8") as f:
        cmd = ["xq"]
        ops = [output_xml_file, '-j']
        args = cmd + ops
        print(" ".join(args) + f" > {output_json_file}")
        subprocess.run(args, stdout=f)


def dump_bitstream_with_gpac():
    input_file_list = [
        "./video/test_10000-nits_h265.h265",
        "./video/test_10000-nits_av1.obu"
    ]
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

    with open(output_json_file, "w", encoding="utf-8") as f:
        cmd = ["xq"]
        ops = [output_xml_file, '-j']
        args = cmd + ops
        print(" ".join(args) + f" > {output_json_file}")
        subprocess.run(args, stdout=f)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    dump_container_with_mp4box()
    dump_bitstream_with_gpac()
