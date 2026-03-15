from pathlib import Path
import os
import sys
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
    KIND_PNG,
    make_media_file_name_without_ext
)


def dump_param_file_list():
    mdcv_primaries_list = MDCV_PRIMARIES_LIST
    mdcv_luminance_list = MDCV_LUMINANCE_LIST
    clli_luminance_list = CLLI_LUMINANCE_LIST
    kind_ext_list = [
        [KIND_AV1, ".mp4"],
        [KIND_HEVC, ".mp4"],
        [KIND_AVIF, ".avif"],
        [KIND_PNG, ".png"]
    ]
    cnt = 0
    print("No, mdcv_primaries, mdcv_luminance, clli_luminance, AV1, HEVC, AVIF, PNG",)
    for mdcv_primaries, mdcv_luminance, clli_luminance\
        in product(mdcv_primaries_list, mdcv_luminance_list, clli_luminance_list):
        if (mdcv_primaries is None) and (mdcv_luminance is not None):
            continue
        if (mdcv_primaries is not None) and (mdcv_luminance is None):
            continue
        print(f"{cnt+1}, ", end="")
        print(f"{mdcv_primaries}, ", end="")
        print(f"{mdcv_luminance}, ", end="")
        print(f"{clli_luminance}, ", end="")
        for idx, (kind, ext) in enumerate(kind_ext_list):
            file_name_without_ext = make_media_file_name_without_ext(
                kind=kind,
                suffix=None,
                mdcv_primaries=mdcv_primaries,
                mdcv_luminance=mdcv_luminance,
                clli_luminance=clli_luminance
            )
            file_name = file_name_without_ext + ext
            if os.path.exists(file_name):
                file_name = "./metadata_img/" + Path(file_name).name
            else:
                file_name = None
            if idx < len(kind_ext_list) - 1:
                print(f"{file_name}, ", end="")
            else:
                print(f"{file_name}")
        cnt += 1


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    dump_param_file_list()
