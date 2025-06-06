# -*- coding: utf-8 -*-

# import standard libraries
import sys
import os
from pathlib import Path
from pprint import pprint
import copy
import shutil
import subprocess
import psutil
import csv
import time

# import third-party libraries
import numpy as np

# import my libraries
import ty_davinci_constants as drc
import ty_davinci_control_lib_2 as dcl
import transfer_functions as tf

REVISION = 0  # Development verion

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
        width, height = dcl.get_project_resolution()
        if not inverse:
            self._v_size = size
            if hv_same:
                self._h_size = size
            else:    
                self._h_size = self._v_size * height / width
        else:
            self._h_size = size
            if hv_same:
                self._v_size = size
            else:
                self._v_size = self._h_size * width / height

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

        self.frame_marker_v_pos = HdPixelBasedSize(140).v_size
        self.frame_marker_v_pos2 = HdPixelBasedSize(108).v_size
        self.motion_blur_radius = HeightBasedSize(0.2).h_size
        self.motion_blur_mask_size = HeightBasedSize(0.075)


def create_background_comp(
        comp, ppp: FusionParams, base_pos=[0, 0]
):
    x_pos = base_pos[0]
    y_pos = base_pos[1]
    transparent_bg = dcl.add_transparent_background(comp=comp, pos=(x_pos, y_pos))

    x_pos += 1
    y_pos += 0
    output_merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=(x_pos, y_pos)
    )

    x_pos += 0
    y_pos += 0
    bg = dcl.add_comp_tool(
        comp=comp, name="Background", pos=(x_pos, y_pos-1)
    )
    dcl.set_tool_topleft_color(tool=bg, rgba=ppp.bg_color)
    dcl.connect_merge_tool(merge_tool=output_merge, bg_tool=transparent_bg, fg_tool=bg)

    return output_merge, x_pos


def create_info_comp(
        comp, ppp: FusionParams, base_pos=[0, 0]
):
    x_pos = base_pos[0]
    y_pos = base_pos[1]

    transparent_bg = dcl.add_transparent_background(comp=comp, pos=(x_pos, y_pos))

    # add rectangle
    x_pos += 1
    rectangle_mask = dcl.add_comp_tool(
        comp=comp, name="RectangleMask", pos=(x_pos, y_pos-2)
    )
    rectangle_mask_input = {
        "Center": {1: 0.5, 2: ppp.info_area_height/2.0, 3: 0.0},
        "Width": 1.0,
        "Height": ppp.info_area_height,
    }
    dcl.set_multiple_tool_input(
        tool=rectangle_mask, input_dict=rectangle_mask_input
    )

    rectangle_bg = dcl.add_comp_tool(
        comp=comp, name="Background", pos=(x_pos, y_pos-1)
    )
    rectangle_bg_input = {
        "TopLeftRed": 0.0,
        "TopLeftGreen": 0.0,
        "TopLeftBlue": 0.0,
        "TopLeftAlpha": 1.0,
        "EffectMask": rectangle_mask,
    }
    dcl.set_multiple_tool_input(tool=rectangle_bg, input_dict=rectangle_bg_input)

    rectangle_merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=(x_pos, y_pos-0)
    )
    dcl.connect_merge_tool(
        merge_tool=rectangle_merge, bg_tool=transparent_bg, fg_tool=rectangle_bg
    )

    # info text
    x_pos += 1
    info_text = dcl.add_comp_tool(
        comp=comp, name="TextPlus", pos=(x_pos, y_pos-1)
    )
    info_text_merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=(x_pos, y_pos-0)
    )
    font_family = "Noto Sans"
    font_weight = "Regular"
    gamut = dcl.get_project_setting("colorSpaceOutput")
    gamma = dcl.get_project_setting("colorSpaceOutputGamma")
    project_width, project_height = dcl.get_project_resolution()
    info_text_str = f"  TP for Adaptive HDR, {project_width}x{project_height}, "
    info_text_str += f"{gamma}, {gamut}"
    print(f"info_text = {info_text}")
    info_text_input = {
        "Center": {1: 0.0, 2: 0.0, 3: 0.0},
        "StyledText": info_text_str,
        "Font": font_family,
        "Style": font_weight,
        "Size": ppp.info_font_size,
        "Red1": ppp.info_font_color[0],
        "Green1": ppp.info_font_color[1],
        "Blue1": ppp.info_font_color[2],
        "VerticalTopCenterBottom": ppp.info_vanchor,
        "HorizontalLeftCenterRight": -1.0,
        "AdvancedFontControls": 1.0,
    }
    dcl.set_multiple_tool_input(tool=info_text, input_dict=info_text_input)
    dcl.connect_merge_tool(
        merge_tool=info_text_merge, bg_tool=rectangle_merge, fg_tool=info_text
    )

    # info text
    x_pos += 1
    rev_text = dcl.add_comp_tool(
        comp=comp, name="TextPlus", pos=(x_pos, y_pos-1)
    )
    rev_text_merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=(x_pos, y_pos-0)
    )
    rev_text_input = {
        "Center": {1: 1.0, 2: 0.0, 3: 0.0},
        "StyledText": f"Revision {REVISION:02d}  ",
        "Font": font_family,
        "Style": font_weight,
        "Size": ppp.info_font_size,
        "Red1": ppp.info_font_color[0],
        "Green1": ppp.info_font_color[1],
        "Blue1": ppp.info_font_color[2],
        "VerticalTopCenterBottom": ppp.info_vanchor,
        "HorizontalLeftCenterRight": 1.0,
        "AdvancedFontControls": 1.0,
    }
    dcl.set_multiple_tool_input(tool=rev_text, input_dict=rev_text_input)
    dcl.connect_merge_tool(
        merge_tool=rev_text_merge, bg_tool=info_text_merge, fg_tool=rev_text
    )

    output_merge = rev_text_merge
    return output_merge, x_pos


def create_adaptive_htr_tp_comp():
    tl_item_fusion_comp, comp = \
        dcl.append_fusion_composition_to_timeline(
            num_of_frame=1,
            pos_frame_idx=dcl.sec_to_frame_idx(60 * 60)
        )
    ppp = FusionParams()

    comp.Lock()

    x_pos = 1
    y_pos = 7  # y_pos is fixed this value
    pseudo_bg = dcl.add_transparent_background(comp=comp, pos=(x_pos, y_pos))

    # base background
    x_pos += 2
    # x_pos will be overwritten in the following function
    background, x_pos = create_background_comp(comp=comp, ppp=ppp, base_pos=(x_pos, y_pos-1))
    background_merge = dcl.add_comp_tool(comp=comp, name="Merge", pos=(x_pos, y_pos))
    dcl.connect_merge_tool(
        merge_tool=background_merge, bg_tool=pseudo_bg, fg_tool=background
    )
    
    # infomation
    x_pos += 2
    info, x_pos = create_info_comp(comp=comp, ppp=ppp, base_pos=(x_pos, y_pos-1))
    info_merge = dcl.add_comp_tool(comp=comp, name="Merge", pos=(x_pos, y_pos))
    dcl.connect_merge_tool(
        merge_tool=info_merge, bg_tool=background_merge, fg_tool=info
    )

    # media out
    x_pos += 2
    media_out = dcl.get_comp_tool_by_name(comp=comp, name="MediaOut1")
    dcl.set_tool_position(comp=comp, tool=media_out, pos=(x_pos, y_pos))

    dcl.connect_mediaout(source=info_merge, mediaout=media_out)

    comp.Unlock()


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
        "colorSpaceTimeline": drc.PRJ_COLOR_SPACE_REC709,
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
    create_adaptive_htr_tp_comp()

    dcl.set_current_timecode(timecode=start_time_code)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
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
        # drc.PRJ_COLOR_SPACE_REC709,
        # drc.PRJ_COLOR_SPACE_P3D65,
        drc.PRJ_COLOR_SPACE_REC2020
    ]
    gamma_list = [
        # drc.PRJ_GAMMA_STR_GAMMA24,
        drc.PRJ_GAMMA_STR_ST2084
    ]

    for resolution, framerate, gamut, gamma in product(
        resolution_list, framerate_list, gamut_list, gamma_list
    ):
        width, height = resolution.split("x")
        create_adaptive_hdr_tp(
            width=width, height=height, framerate=framerate,
            gamut=gamut, gamma=gamma
        )
