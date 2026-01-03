# -*- coding: utf-8 -*-

# import standard libraries
from pathlib import Path
import sys
import os
from pprint import pprint
import shutil

# import third-party libraries
import numpy as np
from colour.models import RGB_COLOURSPACE_BT2020

# import my libraries
import ty_davinci_constants as drc
import ty_davinci_control_lib_2 as dcl
import transfer_functions as tf
from test_pattern_generator2 import generate_color_checker_rgb_value

REVISION = 0  # Development verion
REVISION = 1  # Initial Release

#####################
# Debug
#####################
def dump_tool_input_value(tool):
    print("=" * 80)
    print(f" {tool.Name} InputValue List")
    print("=" * 80)
    for value in tool.GetInputList().values():
        print(f"{value.ID} = {tool.GetInput(value.ID)}")


def compare_tool_input_value(aa, bb):
    print("=" * 80)
    print(f" {aa.Name} {bb.Name} Compare")
    print("=" * 80)
    aa_input = []
    bb_input = []
    for key, value in aa.GetInputList().items():
        aa_input.append({"name": value.ID, "value": aa.GetInput(value.ID)})

    for key, value in bb.GetInputList().items():
        bb_input.append({"name": value.ID, "value": bb.GetInput(value.ID)})

    for idx in range(len(aa_input)):
        if aa_input[idx]["value"] != bb_input[idx]["value"]:
            msg = f"{aa_input[idx]['name']}: "
            msg += f"{aa_input[idx]['value']}, "
            msg += f"{bb_input[idx]['value']}, "
            print(msg)


def dump_tool_main_input_value(tool):
    print("=" * 80)
    print(f" {tool.Name} MainInput List")
    print("=" * 80)
    idx = 1
    while(True):
        input_tool = tool.FindMainInput(idx)
        if input_tool is None:
            break
        print(f"{idx}: Name = {input_tool.Name}, ID = {input_tool.ID}")
        idx += 1


def dump_tool_list(comp):
    print("=" * 80)
    print(" Tool List")
    print("=" * 80)
    for value in comp.GetToolList().values():
        print(f"tool id = {value.ID}, tool name = {value.Name}")


def debug_resolve():
    # print(get_project_setting("videoMonitorFormat"))
    pprint(dcl.get_project_setting(name=None))
    project = dcl.get_current_project()
    format_list = project.GetRenderFormats()
    buf = ""
    for render_format_name, ext in format_list.items():
        codecs = project.GetRenderCodecs(ext)
        buf += f"=== {ext} ===\n"
        for key, value in codecs.items():
            buf += f"{key}: {value}\n"
        buf += "\n"
        print(f"=== {ext} ===")
        print(codecs)
        print('')
    sys.exit(0)


def debug_fusion():
    target_track_name = "dummy_video_1920x1080_24P.mp4"
    timeline = dcl.get_current_timeline()
    timeline_item_list = dcl.get_timeline_items_in_track(
        timeline=timeline, track_type="video", track_idx=1
    )
    
    for timeline_item in timeline_item_list:
        if timeline_item.GetName() == target_track_name:
            break

    comp = timeline_item.GetFusionCompByIndex(1)
    merge_tool = dcl.get_comp_tool_by_name(comp=comp, name="Merge1")
    media_out = dcl.get_comp_tool_by_name(comp=comp, name="MediaOut1")
    print(merge_tool)
    dump_tool_input_value(tool=merge_tool)
    dump_tool_main_input_value(tool=merge_tool)
    # transform = dcl.get_comp_tool_by_name(comp=comp, name="Transform1")
    # dump_tool_main_input_value(tool=transform)
    # dump_tool_input_value(tool=media_out)

    # rec56 = dcl.get_comp_tool_by_name(comp=comp, name="Text4")
    # rec_mask = dcl.add_comp_tool(comp=comp, name="TextPlus", pos=(20, 20))
    # compare_tool_input_value(aa=rec56, bb=rec_mask)

    # dump_tool_list(comp=fusion_comp)

    rectangle1 = dcl.get_comp_tool_by_name(comp=comp, name="Rectangle1")
    dump_tool_input_value(tool=rectangle1)

    import sys
    sys.exit(0)


#####################
# Logic
#####################
class HDBasedMaskBorderSize:
    def __init__(self, px):
        canvas_width, _ = dcl.get_project_resolution()
        val = px / 1920
        current_canvas_px = int(round(val * canvas_width))
        self._size = current_canvas_px / (canvas_width)

    @property
    def size(self):
        return self._size


class HdPixelBasedSize:
    def __init__(self, verical_px, hv_same=False, inverse=False):
        """
        Calculate the size parameters based on Full HD vertical pixel units.

        Parameters
        ----------
        size: float
            Full HD based size. unit is pixel (0 to 1080).
        hv_same : bool
            If true, return `h_size` as the `v_size`
        height : int
            Canvas height (720, 1080, 1440, 2160, ...)
        inverse: bool
            If true, calculate `v_size` based on horizontal size.
        resolution : list or tuple
            [width, height] or (width, height)
        """
        _, height = dcl.get_project_resolution()
        if height != 1080:
            val = verical_px / (1080.0)
            current_canvas_pixel = self.to_even(int(round(val * height)))
            current_canvas_val = current_canvas_pixel / height
        else:
            val = self.to_even(int(verical_px)) / (1080.0)
            current_canvas_val = val

        self.height_based_size\
            = HeightBasedSize(current_canvas_val, hv_same=hv_same, inverse=inverse)

    @property
    def v_size(self):
        return self.height_based_size.v_size

    @property
    def h_size(self):
        return self.height_based_size.h_size
    
    def to_even(self, n: int) -> int:
        return n - (n % 2)


class HeightBasedSize:
    _cached_resolution = None  

    @classmethod
    def _get_resolution(cls):
        if cls._cached_resolution is None:
            cls._cached_resolution = dcl.get_project_resolution()
        return cls._cached_resolution

    def __init__(self, size, hv_same=False, inverse=False):
        """
        Calculate the size parameters based on vertical relative parmaeters.

        Parameters
        ----------
        size : float
            A size parameter based on vertical size (0.0 to 1.0)
        hv_same : bool
            If true, return `h_size` as the `v_size`
        inverse : bool
            If true, calculate `v_size` based on horizontal size.
        """
        width, height = self._get_resolution()
        if not inverse:
            self._v_size = size
            self._h_size = size if hv_same else size * height / width
        else:
            self._h_size = size
            self._v_size = size if hv_same else size * width / height

    @property
    def v_size(self):
        return self._v_size

    @property
    def h_size(self):
        return self._h_size


class FusionParams:
    def __init__(self):
        """
        Parameters
        ----------
        fps: float
            framerate
        width: int
            project canvas size (h)
        height: int
            project canvas size (v)
        """
        # basic parameters
        self.fps = float(dcl.get_project_setting(name="timelineFrameRate"))
        self.fps_int = int(round(self.fps))
        self.width, self.height = dcl.get_project_resolution()

        self.bg_color = [0.01, 0.01, 0.01, 1.0]  # RGBA

        self.info_area_height = HdPixelBasedSize(38).v_size
        self.info_font_color = [0.5, 0.5, 0.5, 1.0]
        self.info_font_size = 0.021
        self.info_vanchor = 2.3

        # border
        self.base_bg_border_width = round(self.height / 1080)

        # scale
        scale_dynamic_range_factor = 4
        scale_each_dynamic_range_num_of_setps = 16
        self.scale_num_of_element = scale_dynamic_range_factor * scale_each_dynamic_range_num_of_setps + 1
        each_scale_width = 28  # please specify int value to reduce rounding error
        scale_width_rate = (self.scale_num_of_element * each_scale_width) / 1920
        scale_h_margin_int = self.to_even(self.width * (1.0 - scale_width_rate) / 2.0)
        self.scale_h_margin = HdPixelBasedSize(scale_h_margin_int).h_size
        self.scale_v_margin = self.info_area_height * 2
        scale_hh_int = self.to_even(1920 * scale_width_rate / self.scale_num_of_element)
        scale_vv_int = scale_hh_int * 0.8
        self.scale_hh = HdPixelBasedSize(scale_hh_int).h_size
        self.scale_vv = HdPixelBasedSize(scale_vv_int).v_size
        self.scale_text_size = 0.02
        self.scale_text_h_anchor = -0.1
        self.scale_text_v_anchor = -1.0

        # color checker
        self.num_of_cc_patch_h = 6
        self.num_of_cc_patch_v = 4
        self.cc_corner_radius = 0.05
        cc_width_int = 700  # HD Based Size
        cc_margin_int = 10  # HD Based Size
        cc_patch_size_int\
            = (cc_width_int - cc_margin_int * (self.num_of_cc_patch_h + 1)) / self.num_of_cc_patch_h
        cc_height_int = (cc_margin_int + cc_patch_size_int) * self.num_of_cc_patch_v + cc_margin_int
        self.cc_width = HdPixelBasedSize(cc_width_int).h_size
        self.cc_height = HdPixelBasedSize(cc_height_int).v_size
        self.cc_margin_h = HdPixelBasedSize(cc_margin_int).h_size
        self.cc_margin_v = HdPixelBasedSize(cc_margin_int).v_size
        self.pp_hh = HdPixelBasedSize(cc_patch_size_int).h_size
        self.pp_vv = HdPixelBasedSize(cc_patch_size_int).v_size
        self.cc_pos = [
            1.0 - (1 - scale_width_rate) / 2.0 - (self.cc_width / 2.0) + self.scale_hh / 2.0,
            0.68
        ]
        self.cc_rgb = generate_color_checker_rgb_value(color_space=RGB_COLOURSPACE_BT2020)
        pseudo_cc_center = [
            (self.pp_hh + self.cc_margin_h) * (self.num_of_cc_patch_h // 2) + self.cc_margin_h / 2.0,
            (self.pp_vv + self.cc_margin_v) * (self.num_of_cc_patch_v // 2) + self.cc_margin_v / 2.0
        ]
        self.cc_center_offset = [
            0.5 - pseudo_cc_center[0],
            0.5 - pseudo_cc_center[1],
        ]
    
        # ramp
        self.ramp_width = HdPixelBasedSize(1024).h_size
        self.ramp_height = 0.06
        self.ramp_border_width = 1
        ramp_pos_v = 0.414
        self.ramp_bit_depth_info_text_pos = [
            [self.scale_h_margin, (1 - ramp_pos_v) - self.ramp_height * 0.5],
            [self.scale_h_margin, (1 - ramp_pos_v) - (self.ramp_height * 1.5)]
        ]
        self.ramp_pos = [
            0.063,
            ramp_pos_v,
        ]
        self.ramp_bottom_text_pos_v = (1 - self.ramp_pos[1]) - self.ramp_height * 2 - 0.02
        self.ramp_top_text_pos_v = (1 - self.ramp_pos[1]) + 0.02

    def to_even(self, n: int|float) -> int:
        n_int = int(round(n))
        return n_int - (n_int % 2)


def add_and_change_attribute():

    dummy_video_path = "./img/SMPTE ST2084_ITU-R BT.2020_D65_1920x1080_rev07_type1.png"
    dummy_video_full_path = str(Path(dummy_video_path).resolve())
    clip = dcl.add_file_to_media_pool(file_path=dummy_video_full_path)
    from pprint import pprint
    pprint(clip.GetClipProperty())
    timeline_item = dcl.append_clip_to_timeline(
        clip=clip,
        media_type=1,
        start_frame=0,
        end_frame=1,  # specify the frame length
        pos_frame_idx=60*60*24
    )

    print(timeline_item)


def create_adaptive_hdr_tp(
        width, height, framerate, gamut, gamma
):
    ##################
    # Project Settings
    ##################
    dcl.refresh_lut_list()

    project_name = f"Adaptive_HDR_TP_REV{REVISION:02d}"
    video_monitor_format = dcl.make_videoMonitorFormat_str(
        width=width, height=height, framerate=framerate
    )
    project_settings_params = {
        "timelineResolutionWidth": f"{width}",
        "timelineResolutionHeight": f"{height}",
        "videoMonitorFormat": video_monitor_format,
        "timelineFrameRate": f"{framerate}",
        "videoMonitorUse444SDI": drc.PRJ_PARAM_DISABLE,
        "videoMonitorSDIConfiguration": drc.PRJ_SDI_SINGLE_LINK,
        "videoDataLevels": drc.PRJ_VIDEO_DATA_LEVEL_FULL,
        "videoMonitorUseHDROverHDMI": drc.PRJ_PARAM_ENABLE,
        "colorScienceMode": drc.PRJ_COLOR_SCIENCE_MODE_RCM_ON,
        "isAutoColorManage": drc.PRJ_PARAM_DISABLE,
        "rcmPresetMode": drc.PRJ_PRESET_MODE_CUSTOM,
        "separateColorSpaceAndGamma": drc.PRJ_PARAM_ENABLE,
        "colorSpaceInput": f"{gamut}",
        "colorSpaceInputGamma": f"{gamma}",
        "colorSpaceTimeline": drc.PRJ_COLOR_SPACE_REC2020,
        "colorSpaceTimelineGamma": drc.PRJ_GAMMA_STR_ST2084,
        "colorSpaceOutput": f"{gamut}",
        "colorSpaceOutputGamma": f"{gamma}",
        "timelineWorkingLuminance": "10000",
        "timelineWorkingLuminanceMode": "Custom",
        "inputDRT": drc.PRJ_PARAM_NONE,
        "outputDRT": drc.PRJ_PARAM_NONE,
        "hdrMasteringLuminanceMax": "1000",
        "hdrMasteringOn": "1",
    }
    start_time_code = "01:00:00:00"
    start_frame = dcl.timecode_to_frame_index(
        timecode=start_time_code, fps_float=framerate
    )

    # control the project
    dcl.close_current_project()
    dcl.delete_project(project_name=project_name)
    project = dcl.create_project(project_name=project_name)

    # set up the project settings
    dcl.setup_project_settings(params=project_settings_params)

    ###########################
    # Add files to the timeline
    ###########################
    # create timelines
    timeline = dcl.create_empty_timeline(name="My_Timeline")

    ####################################################
    # Temporarily commented out because it is slow...
    ####################################################
    # dcl.set_timeline_settings(timeline=timeline, params=project_settings_params)

    ###################
    # Core Function
    ###################
    add_and_change_attribute()

    dcl.set_current_timecode(timecode=start_time_code)


def main():
    # debug_resolve()
    # debug_fusion()

    from itertools import product
    resolution_list = [
        # "1280x720",
        "1920x1080",
        # "2048x1080",
        # "2560x1440",
        # "3840x2160",
        # "4096x2160",
    ]
    framerate_list = [
        24,
    ]
    gamut_list = [
        drc.PRJ_COLOR_SPACE_REC709,
        # drc.PRJ_COLOR_SPACE_SRGB,
        # drc.PRJ_COLOR_SPACE_P3D65,
        # drc.PRJ_COLOR_SPACE_REC2020
    ]
    gamma_list = [
        drc.PRJ_GAMMA_STR_SRGB,
        # drc.PRJ_GAMMA_STR_GAMMA24,
        # drc.PRJ_GAMMA_STR_ST2084
    ]

    for resolution, framerate, gamut, gamma in product(
        resolution_list, framerate_list, gamut_list, gamma_list
    ):
        width, height = resolution.split("x")
        create_adaptive_hdr_tp(
            width=width, height=height, framerate=framerate,
            gamut=gamut, gamma=gamma
        )


def debug():
    x = np.linspace(0, 4, 33)
    y = 10 ** x
    for x, y in zip(x, y):
        print(f"{x:.2f}, {y:.2f}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
    # debug()
