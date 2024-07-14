# -*- coding: utf-8 -*-
"""

==========

"""

# import standard libraries
import os
from pathlib import Path
import subprocess

# import third-party libraries
import numpy as np
import color_space as cs

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def add_metadata_to_src_mov(
        input_fname=None,
        output_fname=None,
        color_matrix=9,
        color_range=1,
        transfer_function=16,
        color_primaries=9,
        chromaticity_coordinates=[0.68, 0.32, 0.265, 0.690, 0.15, 0.06],
        white_point=[0.3127, 0.3290],
        max_cll=1000,
        max_fall=400,
        max_luminance=1000,
        min_luminance=0.01
):
    """
     ./src/mkvmerge \
      -o output.mkv\
      --colour-matrix 0:9 \
      --colour-range 0:1 \
      --colour-transfer-characteristics 0:16 \
      --colour-primaries 0:9 \
      --max-content-light 0:1000 \
      --max-frame-light 0:300 \
      --max-luminance 0:1000 \
      --min-luminance 0:0.01 \
      --chromaticity-coordinates 0:0.68,0.32,0.265,0.690,0.15,0.06 \
      --white-colour-coordinates 0:0.3127,0.3290 \
      input.mov
    """
    chromaticity_coord_str = [f"{x}" for x in chromaticity_coordinates]
    cmd = "./debug/src_tp/mkvmerge.exe"
    ops = [
        "-o", output_fname,
        "--colour-matrix", f"0:{color_matrix}",
        "--colour-range", f"0:{color_range}",
        "--colour-transfer-characteristics", f"0:{transfer_function}",
        "--colour-primaries", f"0:{color_primaries}",
        "--max-content-light", f"0:{max_cll}",
        "--max-frame-light", f"0:{max_fall}",
        "--max-luminance", f"0:{max_luminance}",
        "--min-luminance", f"0:{min_luminance}",
        "--chromaticity-coordinates", f"0:{','.join(chromaticity_coord_str)}",
        "--white-colour-coordinates", f"0:{white_point[0]},{white_point[1]}",
        input_fname
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


def make_output_fname(
        color_space_name, max_cll, max_fall, max_luminance, min_luminance):
    fname = "./debug/src_tp/HDR_TP_"
    fname += f"max_cll_fall-{max_cll}-{max_fall}_"
    fname += f"max_min_luminance-{max_luminance}-{min_luminance}_"
    fname += f"primaries-{color_space_name.replace(' ', '_')}"
    fname += ".mkv"

    return fname


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # fixed parameters
    # input_fname = "./debug/src_tp/HDR_TP_ST2086.mov"
    input_fname = "./debug/src_tp/Src_HDR_TP_for_ST2086_HEVC.mov"
    color_matrix = 9
    color_range = 1
    transfer_function = 16
    color_primaries = 9
    white_point = [0.3127, 0.3290]

    # variable parameters
    max_cll = 400
    max_fall = 200
    max_luminance = 600
    min_luminance = 0.05
    color_space_name = cs.BT709
    chromaticity_coordinates = cs.get_primaries(color_space_name).flatten()
    output_fname = make_output_fname(
        color_space_name=color_space_name,
        max_cll=max_cll,
        max_fall=max_fall,
        max_luminance=max_luminance,
        min_luminance=min_luminance
    )

    # create .mkv file
    add_metadata_to_src_mov(
        input_fname=input_fname,
        output_fname=output_fname,
        color_matrix=color_matrix,
        color_range=color_range,
        transfer_function=transfer_function,
        color_primaries=color_primaries,
        chromaticity_coordinates=chromaticity_coordinates,
        white_point=white_point,
        max_cll=max_cll,
        max_fall=max_fall,
        max_luminance=max_luminance,
        min_luminance=min_luminance
    )

    # variable parameters
    max_cll = 10000
    max_fall = 10000
    max_luminance = 10000
    min_luminance = 0
    color_space_name = cs.BT2020
    chromaticity_coordinates = cs.get_primaries(color_space_name).flatten()
    output_fname = make_output_fname(
        color_space_name=color_space_name,
        max_cll=max_cll,
        max_fall=max_fall,
        max_luminance=max_luminance,
        min_luminance=min_luminance
    )

    # create .mkv file
    add_metadata_to_src_mov(
        input_fname=input_fname,
        output_fname=output_fname,
        color_matrix=color_matrix,
        color_range=color_range,
        transfer_function=transfer_function,
        color_primaries=color_primaries,
        chromaticity_coordinates=chromaticity_coordinates,
        white_point=white_point,
        max_cll=max_cll,
        max_fall=max_fall,
        max_luminance=max_luminance,
        min_luminance=min_luminance
    )
