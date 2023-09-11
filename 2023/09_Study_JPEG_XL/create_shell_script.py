# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2023 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def create_shell_script():
    """
    -x color_space=RGB_D65_202_Rel_PeQ
    """
    script_name = "./png_to_jxl.sh"
    cmd = "cjxl"
    quarity_ops = "-q 100"
    color_space_rgb = "RGB"
    white_point_d65 = "D65"
    primaries_srgb = "SRG"
    primaries_bt2020 = "202"
    primaries_p3 = "DCI"
    tf_709 = "709"
    tf_srgb = "SRG"
    tf_pq = "PeQ"
    tf_hlg = "HLG"
    render_intent_relative = "Rel"

    param_list = [
        [primaries_bt2020, tf_hlg],
        [primaries_bt2020, tf_pq],
        [primaries_p3, tf_pq],
        [primaries_srgb, tf_srgb],
        [primaries_srgb, tf_709]
    ]

    out_buf = []
    out_buf.append("#!/bin/sh")
    for param in param_list:
        primary = param[0]
        tf = param[1]
        white_point = white_point_d65
        rendering_intent = render_intent_relative

        src = f"./img/{white_point_d65}_{primary}_{tf}.png"
        dst = f"./img/{white_point_d65}_{primary}_{tf}.jxl"
        ops = f"{quarity_ops} -x color_space="
        ops += f"{color_space_rgb}_{white_point}_{primary}_"
        ops += f"{rendering_intent}_{tf}"
        # out_buf.append(f'echo "{cmd} {src} {dst} {ops}"')
        out_buf.append(f"{cmd} {src} {dst} {ops}")

    with open(script_name, 'w') as f:
        f.write("\n".join(out_buf) + "\n")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    create_shell_script()
