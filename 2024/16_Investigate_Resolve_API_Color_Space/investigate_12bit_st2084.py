# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
from pathlib import Path

# import third-party libraries

# import my libraries
import resolve_wrapper as drw
import resolve_constants as drc


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def main_func():
    width = 4096
    height = 2160
    framerate = 24
    gamut = drc.PRJ_COLOR_SPACE_REC709
    gamma = drc.PRJ_GAMMA_STR_ST2084

    drw.refresh_lut_list()

    project_name = "ST 2084 12-bit"
    video_monitor_format = drw.make_videoMonitorFormat_str(
        width=width, height=height, framerate=framerate
    )
    project_settings_params = {
        "timelineResolutionWidth": f"{width}",
        "timelineResolutionHeight": f"{height}",
        "videoMonitorFormat": video_monitor_format,
        "timelineFrameRate": f"{framerate}",
        "videoMonitorUse444SDI": "0",
        "videoMonitorSDIConfiguration": "single_link",
        "videoDataLevels": "Video",
        "videoMonitorUseHDROverHDMI": "1",
        "colorScienceMode": "davinciYRGBColorManagedv2",
        "isAutoColorManage": "0",
        "rcmPresetMode": "Custom",
        "separateColorSpaceAndGamma": "1",
        "colorSpaceInput": f"{gamut}",
        "colorSpaceInputGamma": f"{gamma}",
        "colorSpaceTimeline": f"{gamut}",
        "colorSpaceTimelineGamma": f"{gamma}",
        "colorSpaceOutput": f"{gamut}",
        "colorSpaceOutputGamma": f"{gamma}",
        "timelineWorkingLuminance": "10000",
        "timelineWorkingLuminanceMode": "Custom",
        "inputDRT": "None",
        "outputDRT": "None",
        "hdrMasteringLuminanceMax": "1000",
        "hdrMasteringOn": "1",
    }

    # control the project
    drw.close_current_project()
    drw.delete_project(project_name=project_name)
    project = drw.create_project(project_name=project_name)

    # set up the project settings
    drw.setup_project_settings(params=project_settings_params)

    # create timelines
    timeline = drw.create_empty_timeline(name="My_Timeline")

    # create the composition
    _, comp = drw.append_fusion_composition_to_timeline(
                num_of_frame=2,
                pos_timecode="01:00:00:00"
    )

    comp.Lock()

    # add st 2084 ramp of the 12-bit precision
    bg1 = drw.add_comp_tool(comp=comp, name="Background", pos=(0, 1))
    drw.set_tool_topleft_color(tool=bg1, rgba=[0.18, 0.18, 0.18, 1.0])
    dctl = drw.create_dctl_comp(
        comp=comp, dctl_path="TY_DCTL\draw_st2084_12bit_ramp.dctl",
        base_pos=(1, 1)
    )
    media_out = drw.get_comp_tool_by_name(comp=comp, name="MediaOut1")

    drw.connect_dctl(dctl=dctl, source=bg1)
    drw.connect_tool(dctl, media_out)

    comp.Unlock()

    # encode
    codec = drc.CODEC_DPX_RGB_12_BITS
    format_extension = drc.OUT_FILE_EXTENSTION_DPX
    output_fname = "./render_out/st2084_12-bit" + "." + format_extension
    target_dir = str(Path(output_fname).resolve().parent)
    custom_name = str(Path(output_fname).resolve().name)

    render_settings = {
        "TargetDir": target_dir,
        "CustomName": custom_name,
    }
    drw.set_render_format_codec_settings(format=format_extension, codec=codec)
    drw.set_render_settings(setting_dict=render_settings)
    drw.run_rendering_and_wait_until_finish(project=project)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
