# -*- coding: utf-8 -*-

# import standard libraries
import sys
import os
from pathlib import Path
from pprint import pprint
import copy
import shutil
import subprocess

# import third-party libraries
import numpy as np

# import my libraries
import ty_davinci_constants as drc
import ty_davinci_control_lib_2 as dcl
import transfer_functions as tf

REVISION = 2

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
            msg = f"{aa_input[idx]["name"]}: "
            msg += f"{aa_input[idx]["value"]}, "
            msg += f"{bb_input[idx]["value"]}, "
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
    transform = dcl.get_comp_tool_by_name(comp=comp, name="Transform1")
    dump_tool_main_input_value(tool=transform)
    # dump_tool_input_value(tool=media_out)

    rec56 = dcl.get_comp_tool_by_name(comp=comp, name="Text4")
    rec_mask = dcl.add_comp_tool(comp=comp, name="TextPlus", pos=(20, 20))
    # compare_tool_input_value(aa=rec56, bb=rec_mask)

    # transform = dcl.get_comp_tool_by_name(comp=comp, name="Transform1")
    # circle_angle_splineout = transform["CircularAngle"].GetConnectedOutput()
    # if circle_angle_splineout:
    #     spline = circle_angle_splineout.GetTool()
    #     splinedata = spline.GetKeyFrames()
    #     print(f"splinedata type = {type(splinedata)}")
    #     print(f"splinedata = {splinedata}")

    # print(type(transform["CircularAngle"]))
    # print(dir(transform["CircularAngle"]))

    # dump_tool_list(comp=fusion_comp)

    rectangle61 = dcl.get_comp_tool_by_name(comp=comp, name="Rectangle61")
    dump_tool_input_value(tool=rectangle61)

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
    def __init__(self, px, hv_same=False, inverse=False):
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
            val = px / (1080.0)
            current_canvas_pixel = self.to_even(int(round(val * height)))
            current_canvas_val = current_canvas_pixel / height
        else:
            val = self.to_even(int(px)) / (1080.0)
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
    def __init__(self, fps, width, height):
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
        self.fps = fps
        self.fps_int = int(round(fps))
        self.width = width
        self.height = height

        # basic background parameters
        self.base_bg_color = [0.18, 0.18, 0.18, 1.0]
        self.cd_circle_ll = HeightBasedSize(0.58).h_size
        self.cd_circle_mm = HeightBasedSize(0.515).h_size
        self.cd_circle_ss = HeightBasedSize(0.495).h_size
        self.cd_line_width = HdPixelBasedSize(4).v_size
        self.cd_line_color = [0.0, 0.0, 0.0, 1.0]
        self.cd_font_size = HeightBasedSize(0.85).h_size
        self.cross_line_width = self.cd_line_width
        self.gray90 = 0.8
        self.gray80 = 0.7
        self.gray40 = 0.4
        self.gray30 = 0.3
        self.gray28 = 0.28
        self.cross_line_color = [self.gray90, self.gray90, self.gray90, 1.0]
        self.info_font_size = 0.021
        self.info_vanchor = 2.3
        self.deg45_line_color = (72/255) ** 2.4
        self.deg45_line_margin = 192
        self.deg45_line_width = 1
        self.audio_font_size = HeightBasedSize(0.33).h_size
        self.audio_pos_h = 0.19
        self.audio_pos_v = 0.27
        self.audio_text_list = {
            8: ["L", "L", "", ""],
            7: ["", "", "R", "R"],
            6: ["L", "L", "", ""],
            5: ["", "", "R", "R"],
            4: ["L", "L", "", ""],
            3: ["", "", "R", "R"],
            2: ["C", "C", "C", "C"],
            1: ["", "", "", ""],
        }
        self.audio_pos = [
            {1: 0.5 - self.audio_pos_h, 2: 1 - self.audio_pos_v, 3: 0.0},
            {1: 0.5 - self.audio_pos_h, 2: self.audio_pos_v, 3: 0.0},
            {1: 0.5 + self.audio_pos_h, 2: 1 - self.audio_pos_v, 3: 0.0},
            {1: 0.5 + self.audio_pos_h, 2: self.audio_pos_v, 3: 0.0},
        ]

        # frame marker parameters
        frame_marker_h_st_pos = 0.07
        frame_marker_h_ed_pos = 1 - frame_marker_h_st_pos
        self.frame_marker_h_pos\
            = self.linspace(
                frame_marker_h_st_pos, frame_marker_h_ed_pos,
                self.fps_int + 1, width=self.width
            )
        self.frame_marker_v_pos = HdPixelBasedSize(140).v_size
        self.frame_marker_v_pos2 = HdPixelBasedSize(108).v_size
        frame_marker_width\
            = (frame_marker_h_ed_pos - frame_marker_h_st_pos) / (self.fps_int * 2 + 1)
        self.frame_marker_width = self.conv_to_width_base_pixel_size(frame_marker_width)
        self.frame_marker_height = HdPixelBasedSize(140-108).v_size

        frame_marker_outline_width = self.calc_frame_marker_outline_width(
            h_pos_list=self.frame_marker_h_pos,
            each_marker_width=self.frame_marker_width
        )
        self.frame_marker_outline_width\
            = self.conv_to_width_base_pixel_size(frame_marker_outline_width)

        self.frame_marker_outline_height = self.frame_marker_height * 3
        self.frame_marker_outline_v_pos\
            = (self.frame_marker_v_pos - self.frame_marker_v_pos2) / 2.0\
            + self.frame_marker_v_pos2
        self.frame_marker_outline_line_width = HDBasedMaskBorderSize(6).size

        # ramp pattern parameters
        self.ramp_height = 0.09
        self.lumi_text_v_pos = 0.829
        self.cv_text_v_pos = 0.968
        self.ramp_border_width = int(round(4 / 1080.0 * self.height))

        # motion blur parameters
        self.motion_blur_radius = HeightBasedSize(0.2).h_size
        self.motion_blur_mask_size = HeightBasedSize(0.075)
        self.motion_blur_mask_corner_radius = 0.6
        self.motion_blur_bg_color = (192/255.0) ** 2.4
        self.motion_blur_text_color = (64/255.0) ** 2.4
        self.motion_blur_color_mask = [
            [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
        ]
        self.motion_blur_text = ["X", "X", "X", "X"]
        self.motion_blur_text_size = 0.06
        circle_center_x = round(0.15 * self.width) / self.width
        circle_center_y = round(0.33 * self.height) / self.height
        self.motion_blur_circle_center_list = [
            { 1: 1 - circle_center_x, 2: 1 - circle_center_y, 3: 0.0 },
            { 1: circle_center_x, 2: 1 - circle_center_y, 3: 0.0 },
            { 1: circle_center_x, 2: circle_center_y, 3: 0.0 },
            { 1: 1 - circle_center_x, 2: circle_center_y, 3: 0.0 },
        ]
        self.motion_blur_angle_st_ed_list = [
            [-180, 180, 540.0],
            [540.0, 180.0, -180],
            [540.0, 180.0, -180],
            [-180, 180, 540.0],
        ]
        self.motion_blur_magic_number = 1.1
        self.motion_blur_line_length\
            = self.conv_to_width_base_pixel_size(
                self.motion_blur_radius * 1.15 * self.motion_blur_magic_number)
        self.motion_blur_line_width = HdPixelBasedSize(4).v_size
        self.motion_blur_line_mask_size = HeightBasedSize(
            self.motion_blur_radius * self.motion_blur_magic_number
            - (self.motion_blur_line_length - self.motion_blur_radius * self.motion_blur_magic_number),
            inverse=True
        )
        self.motion_blur_circle_line_width = HDBasedMaskBorderSize(4).size
        self.motion_blur_line_color = [0.0, 0.0, 0.0, 1.0]

    def calc_frame_marker_outline_width(self, h_pos_list, each_marker_width):
        margin = h_pos_list[1] - h_pos_list[0]
        st_pos = h_pos_list[0] - margin + (each_marker_width / 2)
        ed_pos = h_pos_list[-1] + margin - (each_marker_width / 2)

        return ed_pos - st_pos

    def linspace(self, start, stop, num, width):
        if num == 1:
            return [start]
        step = (stop - start) / (num - 1)
        return [int((start + step * i) * width + 0.5) / width for i in range(num)]
    
    def conv_to_width_base_pixel_size(self, val):
        val2 = int(round(val * self.width))
        val3 = val2 if val2 % 2 == 0 else val2 - 1

        return val3 / self.width


def create_background_circle(
        comp, bg_rgba=[0.0, 0.0, 0.0, 1.0],
        size=[0.45, 0.45], merge_pos=(1, 1)
    ):
    """
    Returns
    -------
    Merge
        A output merge tool
    """
    circle_mask = dcl.add_comp_tool(
        comp=comp, name="EllipseMask", pos=(merge_pos[0], merge_pos[1] - 2)
    )
    circle_mask_input = {
        "Width": size[0],
        "Height": size[1],
    }
    dcl.set_multiple_tool_input(tool=circle_mask, input_dict=circle_mask_input)

    circle_bg = dcl.add_comp_tool(
        comp=comp, name="Background", pos=(merge_pos[0], merge_pos[1] - 1)
    )
    circle_bg_input = {
        "TopLeftRed": bg_rgba[0],
        "TopLeftGreen": bg_rgba[1],
        "TopLeftBlue": bg_rgba[2],
        "TopLeftAlpha": bg_rgba[3],
        "EffectMask": circle_mask,
    }
    dcl.set_multiple_tool_input(tool=circle_bg, input_dict=circle_bg_input)

    merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=(merge_pos[0], merge_pos[1] + 0)
    )
    dcl.connect_merge_tool(merge_tool=merge, bg_tool=None, fg_tool=circle_bg)

    return merge


def draw_info_comp(
        comp, font_size, vanchor, bg_rgba, fg_rgba, height, base_pos=[0, 0]):
    x_pos = base_pos[0]
    y_pos = base_pos[1]

    rectangle_mask = dcl.add_comp_tool(
        comp=comp, name="RectangleMask", pos=(x_pos+0, y_pos-2)
    )
    rectangle_mask_input = {
        "Center": {1: 0.5, 2: height/2.0, 3: 0.0},
        "Width": 1.0,
        "Height": height,
    }
    dcl.set_multiple_tool_input(
        tool=rectangle_mask, input_dict=rectangle_mask_input
    )
    rectangle_fg = dcl.add_comp_tool(
        comp=comp, name="Background", pos=(x_pos+0, y_pos-1)
    )
    dcl.set_tool_topleft_color(tool=rectangle_fg, rgba=bg_rgba)
    dcl.set_tool_input(tool=rectangle_fg, name="EffectMask", value=rectangle_mask)
    rectangle_merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=(x_pos+0, y_pos+0)
    )
    dcl.connect_merge_tool(
        merge_tool=rectangle_merge, bg_tool=None, fg_tool=rectangle_fg
    )

    # info text
    info_text = dcl.add_comp_tool(
        comp=comp, name="TextPlus", pos=(x_pos+1, y_pos-1)
    )
    info_text_merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=(x_pos+1, y_pos-0)
    )
    font_family = "Noto Sans"
    font_weight = "Regular"
    fps = float(dcl.get_project_setting("timelineFrameRate"))
    fps = int(fps) if fps.is_integer() else fps
    gamut = dcl.get_project_setting("colorSpaceOutput")
    gamma = dcl.get_project_setting("colorSpaceOutputGamma")
    project_width, project_height = dcl.get_project_resolution()
    info_text_str = f"  Countdown v2, {project_width}x{project_height}, "
    info_text_str += f"{fps} fps, {gamma}, {gamut}"
    print(f"info_text = {info_text}")
    info_text_input = {
        "Center": {1: 0.0, 2: 0.0, 3: 0.0},
        "StyledText": info_text_str,
        "Font": font_family,
        "Style": font_weight,
        "Size": font_size,
        "Red1": fg_rgba[0],
        "Green1": fg_rgba[1],
        "Blue1": fg_rgba[2],
        "VerticalTopCenterBottom": vanchor,
        "HorizontalLeftCenterRight": -1.0,
        "AdvancedFontControls": 1.0,
    }
    dcl.set_multiple_tool_input(tool=info_text, input_dict=info_text_input)
    dcl.connect_merge_tool(
        merge_tool=info_text_merge, bg_tool=rectangle_merge, fg_tool=info_text
    )

    # rev text
    rev_text = dcl.add_comp_tool(
        comp=comp, name="TextPlus", pos=(x_pos+2, y_pos-1)
    )
    rev_text_merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=(x_pos+2, y_pos-0)
    )
    rev_text_input = {
        "Center": {1: 1.0, 2: 0.0, 3: 0.0},
        "StyledText": "Revision 02  ",
        "Font": font_family,
        "Style": font_weight,
        "Size": font_size,
        "Red1": fg_rgba[0],
        "Green1": fg_rgba[1],
        "Blue1": fg_rgba[2],
        "VerticalTopCenterBottom": vanchor,
        "HorizontalLeftCenterRight": 1.0,
        "AdvancedFontControls": 1.0,
    }
    dcl.set_multiple_tool_input(tool=rev_text, input_dict=rev_text_input)
    dcl.connect_merge_tool(
        merge_tool=rev_text_merge, bg_tool=info_text_merge, fg_tool=rev_text
    )

    in_merge = rectangle_merge
    out_merge = rev_text_merge

    return in_merge, out_merge


def create_still_background_comp(comp, ppp: FusionParams, tool_pos):
    """
    Parameters
    ----------
    comp : Composition
        A fusion Composition instance
    ppp : FusionParams
        A parameter set for fusion composition
    tool_pos : list
        [h_pos, v_pos] of the base tool (lower left)
    """
    x_pos = tool_pos[0]
    y_pos = tool_pos[1]

    bg1 = dcl.add_comp_tool(
        comp=comp, name="Background", pos=(x_pos+0, y_pos-1)
    )
    dcl.set_tool_topleft_color(
        tool=bg1, rgba=ppp.base_bg_color)

    draw_45deg_line_dctl = dcl.add_dctl_comp(
        comp=comp, dctl_path="TY_DCTL/draw_45deg_lines.dctl",
        base_pos=[x_pos+1, y_pos-1],
        option={
            "sliderIntParam0": ppp.deg45_line_margin,
            "sliderIntParam1": ppp.deg45_line_width,
            "sliderFloatParam0": ppp.deg45_line_color,
        }
    )

    cross_h_line_merge = dcl.add_line_comp(
        comp=comp, rgba=ppp.cross_line_color, width=1.0, angle=0,
        height=ppp.cross_line_width, pos=[x_pos+2, y_pos-1],
        connect_fg=True
    )
    cross_v_line_merge = dcl.add_line_comp(
        comp=comp, rgba=ppp.cross_line_color, width=1.0, angle=90,
        height=ppp.cross_line_width, pos=[x_pos+3, y_pos-1],
        connect_fg=True
    )
    large_white_circle_merge = create_background_circle(
        comp=comp, bg_rgba=[ppp.gray80, ppp.gray80, ppp.gray80, 1.0],
        size=[ppp.cd_circle_ll, ppp.cd_circle_ll],
        merge_pos=[x_pos+4, y_pos-1]
    )
    middle_black_circle_merge = create_background_circle(
        comp=comp, bg_rgba=[0.0, 0.0, 0.0, 1.0],
        size=[ppp.cd_circle_mm, ppp.cd_circle_mm],
        merge_pos=[x_pos+5, y_pos-1]
    )
    small_grey_circle_merge = create_background_circle(
        comp=comp, bg_rgba=[0.18, 0.18, 0.18, 1.0],
        size=[ppp.cd_circle_ss, ppp.cd_circle_ss],
        merge_pos=[x_pos+6, y_pos-1]
    )
    h_line_merge = dcl.add_line_comp(
        comp=comp, rgba=ppp.cd_line_color, angle=0,
        width=ppp.cd_circle_ll,
        height=ppp.cd_line_width, pos=[x_pos+7, y_pos-1],
        connect_fg=True
    )
    v_line_merge = dcl.add_line_comp(
        comp=comp, rgba=ppp.cd_line_color, angle=90,
        width=ppp.cd_circle_ll,
        height=ppp.cd_line_width, pos=[x_pos+8, y_pos-1],
        connect_fg=True
    )
    info_in_merge, info_out_merge = draw_info_comp(
        comp=comp, font_size=ppp.info_font_size, vanchor=ppp.info_vanchor,
        bg_rgba=[0.0, 0.0, 0.0, 1.0], fg_rgba=[0.5, 0.5, 0.5, 1.0],
        height=0.035, base_pos=[x_pos+9, y_pos-1])
    border_dctl = dcl.add_dctl_comp(
        comp=comp, dctl_path="TY_DCTL/draw_countdown_border.dctl",
        base_pos=[x_pos+12, y_pos-1]
    )
    output_merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=(x_pos+12, y_pos)
    )

    dcl.connect_dctl(dctl=draw_45deg_line_dctl, source=bg1)

    dcl.connect_merge_tool(
        merge_tool=cross_h_line_merge,
        bg_tool=draw_45deg_line_dctl, fg_tool=None
    )
    dcl.connect_merge_tool(
        merge_tool=cross_v_line_merge,
        bg_tool=cross_h_line_merge, fg_tool=None
    )
    dcl.connect_merge_tool(
        merge_tool=large_white_circle_merge,
        bg_tool=cross_v_line_merge, fg_tool=None
    )
    dcl.connect_merge_tool(
        merge_tool=middle_black_circle_merge,
        bg_tool=large_white_circle_merge, fg_tool=None
    )
    dcl.connect_merge_tool(
        merge_tool=small_grey_circle_merge,
        bg_tool=middle_black_circle_merge, fg_tool=None
    )
    dcl.connect_merge_tool(
        merge_tool=h_line_merge,
        bg_tool=small_grey_circle_merge, fg_tool=None
    )
    dcl.connect_merge_tool(
        merge_tool=v_line_merge,
        bg_tool=h_line_merge, fg_tool=None
    )
    dcl.connect_merge_tool(
        merge_tool=info_in_merge,
        bg_tool=v_line_merge, fg_tool=None
    )
    dcl.connect_dctl(dctl=border_dctl, source=info_out_merge)
    dcl.connect_merge_tool(merge_tool=output_merge, bg_tool=None, fg_tool=border_dctl)
    
    return output_merge


def create_countdown_animation_comp(
    comp, ppp: FusionParams, count_str, tool_pos
):
    """
    Parameters
    ----------
    comp : Composition
        A fusion Composition instance
    ppp : FusionParams
        A parameter set for fusion composition
    count_str : int
        A number indicate the countdown
    fps : int
        framerate
    tool_pos : list
        [h_pos, v_pos] of the base tool (lower left)
    """
    x_pos = tool_pos[0]
    y_pos = tool_pos[1]

    base_bg = dcl.add_transparent_background(
        comp=comp, pos=(x_pos+0, y_pos-1)
    )
    radial_wipe = dcl.add_comp_tool(
        comp=comp, name="EllipseMask", pos=(x_pos+1, y_pos-4)
    )
    wipe_circle_mask = dcl.add_comp_tool(
        comp=comp, name="EllipseMask", pos=(x_pos+1, y_pos-3)
    )
    wipe_circle_fg = dcl.add_comp_tool(
        comp=comp, name="Background", pos=(x_pos+1, y_pos-2)
    )
    wipe_circle_merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=(x_pos+1, y_pos-1)
    )
    countdown_text = dcl.add_comp_tool(
        comp=comp, name="TextPlus", pos=(x_pos+2, y_pos-2)
    )
    countdown_text_merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=(x_pos+2, y_pos-1)
    )

    audio_channel_text_merge_list = []
    audio_channel_text_list = []
    for t_idx in range(4):
        audio_channel_text_merge = dcl.add_comp_tool(
            comp=comp, name="Merge", pos=(x_pos+3+t_idx, y_pos-1)
        )
        audio_channel_text_merge_list.append(audio_channel_text_merge)
        audio_channel_text = dcl.add_comp_tool(
            comp=comp, name="TextPlus", pos=(x_pos+3+t_idx, y_pos-2)
        )
        audio_channel_text_list.append(audio_channel_text)
    output_merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=(x_pos+3+3, y_pos-0)
    )

    # wipe animation settings
    radial_wipe_input = {
        "Invert": 1.0,
        "BorderWidth": 1.0,
        "Solid": 0.0,
        "CapStyle": 0.0,
        "Width": 1.0,
        "Height": 1.0,
        "Angle": 90,
    }
    dcl.set_multiple_tool_input(tool=radial_wipe, input_dict=radial_wipe_input)
    radial_wipe["WriteLength"] = comp.BezierSpline()
    radial_wipe["WriteLength"][0] = 1.0
    radial_wipe["WriteLength"][ppp.fps_int] = 0.0

    # mask settings for wipe animation
    wipe_circle_mask_input = {
        "Invert": 1.0,
        "Width": ppp.cd_circle_mm,
        "Height": ppp.cd_circle_mm,
        "PaintMode": "Subtract",
        "EffectMask": radial_wipe,
    }
    dcl.set_multiple_tool_input(
        tool=wipe_circle_mask, input_dict=wipe_circle_mask_input
    )

    # color settings for wipe animation
    wipe_circle_fg_input = {
        "TopLeftRed": 0.0,
        "TopLeftGreen": 0.0,
        "TopLeftBlue": 0.0,
        "TopLeftAlpha": 1.0,
        "EffectMask": wipe_circle_mask,
    }
    dcl.set_multiple_tool_input(
        tool=wipe_circle_fg, input_dict=wipe_circle_fg_input
    )

    font_family = "Noto Sans Mono"
    font_weight = "Black"
    countdown_text_input = {
        "StyledText": f"{count_str}" if count_str != 1 else "",
        "Font": font_family,
        "Style": font_weight,
        "Size": ppp.cd_font_size,
        "Red1": ppp.gray80,
        "Green1": ppp.gray80,
        "Blue1": ppp.gray80,
    }
    dcl.set_multiple_tool_input(
        tool=countdown_text, input_dict=countdown_text_input
    )

    audio_channel_text_input_base = {
        "Font": font_family,
        "Style": font_weight,
        "Size": ppp.audio_font_size,
        "Red1": ppp.gray28,
        "Green1": ppp.gray28,
        "Blue1": ppp.gray28,
    }

    for t_idx in range(4):
        audio_channel_text_input = copy.deepcopy(audio_channel_text_input_base)
        audio_channel_text_input["Center"] = ppp.audio_pos[t_idx]
        audio_channel_text_input["StyledText"] = ppp.audio_text_list[count_str][t_idx]
        dcl.set_multiple_tool_input(
            tool=audio_channel_text_list[t_idx], input_dict=audio_channel_text_input
        )
        audio_channel_text_list[t_idx]["Opacity1"] = comp.BezierSpline()
        audio_channel_text_list[t_idx]["Opacity1"][0] = 0.0
        audio_channel_text_list[t_idx]["Opacity1"][(ppp.fps_int//2)-1] = 0.0
        audio_channel_text_list[t_idx]["Opacity1"][ppp.fps_int//2] = 1.0

    # connect
    dcl.connect_merge_tool(
        merge_tool=wipe_circle_merge, bg_tool=base_bg, fg_tool=wipe_circle_fg
    )
    dcl.connect_merge_tool(
        merge_tool=countdown_text_merge,
        bg_tool=wipe_circle_merge, fg_tool=countdown_text
    )
    dcl.connect_merge_tool(
        merge_tool=audio_channel_text_merge_list[0],
        bg_tool=countdown_text_merge, fg_tool=audio_channel_text_list[0]
    )
    dcl.connect_merge_tool(
        merge_tool=audio_channel_text_merge_list[1],
        bg_tool=audio_channel_text_merge_list[0], fg_tool=audio_channel_text_list[1]
    )
    dcl.connect_merge_tool(
        merge_tool=audio_channel_text_merge_list[2],
        bg_tool=audio_channel_text_merge_list[1], fg_tool=audio_channel_text_list[2]
    )
    dcl.connect_merge_tool(
        merge_tool=audio_channel_text_merge_list[3],
        bg_tool=audio_channel_text_merge_list[2], fg_tool=audio_channel_text_list[3]
    )
    dcl.connect_merge_tool(
        merge_tool=output_merge,
        bg_tool=None, fg_tool=audio_channel_text_merge_list[3]
    )

    return output_merge


def create_frame_marker_core(comp, ppp, idx, fps, tool_pos=(1, 3)):
    x_pos = tool_pos[0]
    y_pos = tool_pos[1]

    bg = dcl.add_comp_tool(
        comp=comp, name="Background", pos=(x_pos+0, y_pos-1)
    )
    bg_mask = dcl.add_comp_tool(
        comp=comp, name="RectangleMask", pos=(x_pos+0, y_pos-2)
    )
    bg_merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=(x_pos+0, y_pos)
    )

    inv_bg = dcl.add_comp_tool(
        comp=comp, name="Background", pos=(x_pos+1, y_pos-1)
    )
    inv_bg_mask = dcl.add_comp_tool(
        comp=comp, name="RectangleMask", pos=(x_pos+1, y_pos-2)
    )
    inv_bg_merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=(x_pos+1, y_pos)
    )

    bg_mask_input = {
        "Filter": "Box",
        "CapStyle": 0.0,
        "Center": {
            1: ppp.frame_marker_h_pos[idx],
            2: ppp.frame_marker_v_pos, 3: 0.0
        },
        "Width": ppp.frame_marker_width,
        "Height": ppp.frame_marker_height,
    }
    bg_input = {
        "TopLeftRed": 0.0,
        "TopLeftGreen": 0.0,
        "TopLeftBlue": 0.0,
        "TopLeftAlpha": 1.0,
        "EffectMask": bg_mask,
    }

    inv_bg_mask_input = {
        "Filter": "Box",
        "CapStyle": 0.0,
        "Center": {
            1: ppp.frame_marker_h_pos[idx],
            2: ppp.frame_marker_v_pos2, 3: 0.0
        },
        "Width": ppp.frame_marker_width,
        "Height": ppp.frame_marker_height,
    }

    inv_bg_input = {
        "TopLeftRed": 0.0,
        "TopLeftGreen": 0.0,
        "TopLeftBlue": 0.0,
        "TopLeftAlpha": 1.0,
        "EffectMask": inv_bg_mask,
    }

    dcl.set_multiple_tool_input(tool=bg, input_dict=bg_input)
    dcl.set_multiple_tool_input(tool=bg_mask, input_dict=bg_mask_input)
    dcl.set_multiple_tool_input(tool=inv_bg, input_dict=inv_bg_input)
    dcl.set_multiple_tool_input(tool=inv_bg_mask, input_dict=inv_bg_mask_input)

    # set keyframe
    color_list = ["TopLeftRed", "TopLeftGreen", "TopLeftBlue"]
    for color in color_list:
        base_idx = (idx + fps//2) % fps
        bg[color] = comp.BezierSpline()
        bg[color][base_idx] = ppp.gray80
        bg[color][base_idx + 1] = 0.0
        bg[color][base_idx - 1] = 0.0

        inv_bg[color] = comp.BezierSpline()
        inv_idx = (fps - 0) - idx
        inv_base_idx = (inv_idx + fps//2) % fps
        inv_bg[color][inv_base_idx] = ppp.gray80
        inv_bg[color][inv_base_idx + 1] = 0.0
        inv_bg[color][inv_base_idx - 1] = 0.0

    dcl.connect_merge_tool(merge_tool=bg_merge, bg_tool=None, fg_tool=bg)
    dcl.connect_merge_tool(merge_tool=inv_bg_merge, bg_tool=None, fg_tool=inv_bg)
    dcl.connect_merge_tool(merge_tool=inv_bg_merge, bg_tool=bg_merge, fg_tool=None)

    return bg_merge, inv_bg_merge


def create_frame_marker(comp, ppp: FusionParams, tool_pos=(1, 3)):
    x_pos = tool_pos[0]
    y_pos = tool_pos[1]
    fps = ppp.fps_int

    base_bg = dcl.add_transparent_background(comp=comp, pos=(x_pos, y_pos-1))

    # Add aditional background to mask the deg45 lines.
    frame_bg_merge = dcl.add_comp_tool(comp=comp, name="Merge", pos=[x_pos+1, y_pos-1])
    frame_bg = dcl.add_comp_tool(comp=comp, name="Background", pos=[x_pos+1, y_pos-2])
    frame_bg_mask = dcl.add_comp_tool(
        comp=comp, name="RectangleMask", pos=[x_pos+1, y_pos-3]
    )
    frame_bg_input = {
        "TopLeftRed": ppp.base_bg_color[0],
        "TopLeftGreen": ppp.base_bg_color[1],
        "TopLeftBlue": ppp.base_bg_color[2],
        "TopLeftAlpha": ppp.base_bg_color[3],
        "EffectMask": frame_bg_mask,
    }
    dcl.set_multiple_tool_input(tool=frame_bg, input_dict=frame_bg_input)
    frame_bg_mask_input = {
        "Center": {1: 0.5, 2: ppp.frame_marker_outline_v_pos, 3: 0.0},
        "Width": ppp.frame_marker_outline_width,
        "Height": ppp.frame_marker_outline_height,
    }
    dcl.set_multiple_tool_input(tool=frame_bg_mask, input_dict=frame_bg_mask_input)
    dcl.connect_merge_tool(merge_tool=frame_bg_merge, bg_tool=base_bg, fg_tool=frame_bg)

    merge_list = []
    for idx in range(fps+1):
        bg_merge, inv_bg_merge = create_frame_marker_core(
            comp=comp, ppp=ppp, idx=idx, fps=fps,
            tool_pos=(x_pos+2*idx+2, y_pos-1)
        )
        merge_list.append([bg_merge, inv_bg_merge])

    for idx in range(1, fps+1):
        dcl.connect_merge_tool(
            merge_tool=merge_list[idx][0],
            bg_tool=merge_list[idx-1][1], fg_tool=None
        )
    dcl.connect_merge_tool(
        merge_tool=merge_list[0][0], bg_tool=frame_bg_merge, fg_tool=None
    )

    outline_rect = dcl.add_comp_tool(
        comp=comp, name="RectangleMask", pos=(x_pos+2*(fps+1)+2, y_pos-3)
    )
    outline_bg = dcl.add_comp_tool(
        comp=comp, name="Background", pos=((x_pos+2*(fps+1)+2, y_pos-2))
    )
    outline_merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=((x_pos+2*(fps+1)+2, y_pos-1))
    )
    output_merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=((x_pos+2*(fps+1)+2, y_pos-0))
    )

    outline_bg_input = {
        "TopLeftRed": 0.0,
        "TopLeftGreen": 0.0,
        "TopLeftBlue": 0.0,
        "TopLeftAlpha": 1.0,
        "EffectMask": outline_rect,
    }
    dcl.set_multiple_tool_input(tool=outline_bg, input_dict=outline_bg_input)

    outline_rect_input = {
        "BorderWidth": ppp.frame_marker_outline_line_width,
        "Solid": 0.0,
        "Center": {1: 0.5, 2: ppp.frame_marker_outline_v_pos, 3: 0.0},
        "Width": ppp.frame_marker_outline_width,
        "Height": ppp.frame_marker_outline_height,
    }
    dcl.set_multiple_tool_input(tool=outline_rect, input_dict=outline_rect_input)
    dcl.connect_merge_tool(
        merge_tool=outline_merge,
        bg_tool=merge_list[-1][1], fg_tool=outline_bg
    )
    dcl.connect_merge_tool(
        merge_tool=output_merge,
        bg_tool=None, fg_tool=outline_merge
    )

    return output_merge


def add_ramp_info_text(
        comp, ppp, t_idx, luminance, st2084_cv, st_pos, ramp_width, x_pos, y_pos
    ):
    x_pos_offset = 1 + t_idx * 2
    lumi_text = dcl.add_comp_tool(
        comp=comp, name="TextPlus", pos=(x_pos+x_pos_offset, y_pos-1)
    )
    lumi_text_merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=(x_pos+x_pos_offset, y_pos-0)
    )
    cv_text = dcl.add_comp_tool(
        comp=comp, name="TextPlus", pos=(x_pos+x_pos_offset+1, y_pos-1)
    )
    cv_text_merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=(x_pos+x_pos_offset+1, y_pos-0)
    )
    text_input_base = {
        "Center": {1: 0.949, 2: 0.975, 3: 0.0},
        "StyledText": "dummy",
        "Font": "Noto Sans",
        "Style": "Regular",
        "Size": 0.022,
        "Red1": ppp.gray80,
        "Green1": ppp.gray80,
        "Blue1": ppp.gray80,
        "Enabled2": 1,
        "Thickness2": 0.12,
        "Red2": 0.0,
        "Green2": 0.0,
        "Blue2": 0.0,
    }
    text_center_pos = st_pos + ramp_width * st2084_cv
    lumi_text_input = copy.deepcopy(text_input_base)
    lumi_text_input["Center"] = {1: text_center_pos, 2: ppp.lumi_text_v_pos, 3: 0.0}
    lumi_text_input["StyledText"] = f"{luminance}"
    dcl.set_multiple_tool_input(tool=lumi_text, input_dict=lumi_text_input)
    cv_text_input = copy.deepcopy(text_input_base)
    cv_text_input["Center"] = {1: text_center_pos, 2: ppp.cv_text_v_pos, 3: 0.0}
    cv_text_input["StyledText"] = str(int(round(1023 * st2084_cv)))
    dcl.set_multiple_tool_input(tool=cv_text, input_dict=cv_text_input)
    dcl.connect_merge_tool(
        merge_tool=cv_text_merge,
        bg_tool=lumi_text_merge, fg_tool=cv_text
    )
    dcl.connect_merge_tool(
        merge_tool=lumi_text_merge,
        bg_tool=None, fg_tool=lumi_text
    )
    st_merge = lumi_text_merge
    ed_merge = cv_text_merge

    return st_merge, ed_merge


def create_ramp(comp, ppp: FusionParams, tool_pos=(1, 3)):
    x_pos = tool_pos[0]
    y_pos = tool_pos[1]

    ramp_dctl = dcl.add_dctl_comp(
        comp=comp, dctl_path="TY_DCTL/draw_countdown_ramp.dctl", base_pos=[x_pos, y_pos-1],
        option={
            "sliderFloatParam0": ppp.frame_marker_outline_width,
            "sliderFloatParam1": ppp.ramp_height * 0.93,
            "sliderIntParam0": ppp.ramp_border_width
        }
    )

    # info text
    st_merge = None
    ed_merge = None
    prev_ed_merge = None
    luminance_list = [0, 0.1, 1, 10, 100, 1000, 10000]
    st2084_cv_list = tf.oetf_from_luminance(np.array(luminance_list), tf.ST2084)
    ramp_width = ppp.frame_marker_outline_width
    st_pos = (1 - ppp.frame_marker_outline_width) / 2.0
    for t_idx, st2084_cv in enumerate(st2084_cv_list):
        st_merge_temp, ed_merge_temp = add_ramp_info_text(
            comp=comp, ppp=ppp, t_idx=t_idx,
            luminance=luminance_list[t_idx], st2084_cv=st2084_cv,
            st_pos=st_pos, ramp_width=ramp_width, x_pos=x_pos, y_pos=y_pos-1
        )
        ed_merge = ed_merge_temp
        if st_merge is None:
            st_merge = st_merge_temp

        if prev_ed_merge is not None:
            dcl.connect_merge_tool(
                merge_tool=st_merge_temp, bg_tool=prev_ed_merge, fg_tool=None
            )
        prev_ed_merge = ed_merge_temp
    dcl.connect_merge_tool(merge_tool=st_merge, bg_tool=ramp_dctl, fg_tool=None)

    output_merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=(x_pos+len(luminance_list)*2, y_pos)
    )
    dcl.connect_merge_tool(merge_tool=output_merge, bg_tool=None, fg_tool=ed_merge)

    return ramp_dctl, output_merge


def create_motion_blur_animation_core(comp, c_idx, ppp: FusionParams, tool_pos=(1, 3)):
    x_pos = tool_pos[0]
    y_pos = tool_pos[1]

    h_line_merge = dcl.add_line_comp(
        comp=comp, rgba=ppp.cd_line_color, angle=0,
        width=ppp.motion_blur_line_length,
        height=ppp.motion_blur_line_width,
        center=ppp.motion_blur_circle_center_list[c_idx],
        pos=[x_pos+0, y_pos-3],
        connect_fg=False
    )
    v_line_merge = dcl.add_line_comp(
        comp=comp, rgba=ppp.cd_line_color, angle=90,
        width=ppp.motion_blur_line_length,
        height=ppp.motion_blur_line_width,
        center=ppp.motion_blur_circle_center_list[c_idx],
        pos=[x_pos+1, y_pos-3],
    )
    dummy_bg = dcl.add_transparent_background(comp=comp, pos=(x_pos, y_pos-2))
    cross_line_mask_merge = dcl.add_comp_tool(comp=comp, name="Merge", pos=(x_pos+1, y_pos-2))
    cross_line_mask = dcl.add_comp_tool(comp=comp, name="RectangleMask", pos=(x_pos+1, y_pos-1))
    circle_merge = dcl.add_comp_tool(comp=comp, name="Merge", pos=(x_pos+2, y_pos-2))
    circle_bg = dcl.add_comp_tool(comp=comp, name="Background", pos=(x_pos+2, y_pos-4))
    circle_mask = dcl.add_comp_tool(comp=comp, name="EllipseMask", pos=(x_pos+2, y_pos-5))

    output_merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=(x_pos+3, y_pos-0)
    )
    second_last_merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=(x_pos+3, y_pos-2)
    )
    transform = dcl.add_comp_tool(
        comp=comp, name="Transform", pos=(x_pos+3, y_pos-3)
    )
    text_merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=(x_pos+3, y_pos-4)
    )
    text = dcl.add_comp_tool(
        comp=comp, name="TextPlus", pos=(x_pos+4, y_pos-4)
    )
    bg = dcl.add_comp_tool(
        comp=comp, name="Background", pos=(x_pos+3, y_pos-5)
    )
    mask = dcl.add_comp_tool(
        comp=comp, name="RectangleMask", pos=(x_pos+3, y_pos-6)
    )

    # input
    cross_line_mask_input = {
        "Filter": "Box",
        "CapStyle": 0.0,
        "Invert": 1,
        "Center": ppp.motion_blur_circle_center_list[c_idx],
        "Width": ppp.motion_blur_line_mask_size.h_size,
        "Height": ppp.motion_blur_line_mask_size.v_size,
    }
    circle_mask_input = {
        "Filter": "Box",
        "CapStyle": 0.0,
        "Solid": 0,
        "BorderWidth": ppp.motion_blur_circle_line_width,
        "Center": ppp.motion_blur_circle_center_list[c_idx],
        "Width": ppp.motion_blur_radius * 1.1,
        "Height": ppp.motion_blur_radius * 1.1,
    }
    circle_bg_input = {
        "TopLeftRed": ppp.motion_blur_line_color[0],
        "TopLeftGreen": ppp.motion_blur_line_color[1],
        "TopLeftBlue": ppp.motion_blur_line_color[2],
        "TopLeftAlpha": ppp.motion_blur_line_color[3],
        "EffectMask": circle_mask,
    }
    cross_line_mask_merge_input = {
        "EffectMask": cross_line_mask
    }

    mask_input = {
        "Width": ppp.motion_blur_mask_size.h_size,
        "Height": ppp.motion_blur_mask_size.v_size,
        "CornerRadius": ppp.motion_blur_mask_corner_radius,
    }
    bg_input = {
        "TopLeftRed": ppp.motion_blur_bg_color * ppp.motion_blur_color_mask[c_idx][0],
        "TopLeftGreen": ppp.motion_blur_bg_color * ppp.motion_blur_color_mask[c_idx][1],
        "TopLeftBlue": ppp.motion_blur_bg_color * ppp.motion_blur_color_mask[c_idx][2],
        "TopLeftAlpha": 1.0,
        "EffectMask": mask,
    }
    text_input = {
        "Center": {1: 0.5, 2: 0.5, 3: 0.0},
        "StyledText": ppp.motion_blur_text[c_idx],
        "Font": "Noto Sans",
        "Style": "Bold",
        "Size": ppp.motion_blur_text_size,
        "Red1": ppp.motion_blur_text_color,
        "Green1": ppp.motion_blur_text_color,
        "Blue1": ppp.motion_blur_text_color,
    }
    transform_input = {
        "CircleCenter": ppp.motion_blur_circle_center_list[c_idx],
        "Radius": ppp.motion_blur_radius
    }

    user_control = dict(
        CircleC=dict(
            ICS_ControlPage="Controls",
            INPID_PreviewControl="PointControl",
            LINKID_DataType="Point",
            LINKS_Name="CircleCenter",
        ),
        CircleCenter=dict(
            INPID_PreviewControl="PointControl",
            LINKID_DataType="Point",
            ICS_ControlPage="Controls",
            INPID_InputControl="OffsetControl",
            LINKS_Name="CircleCenter",
        ),
        CircularAngle=dict(
            ICS_ControlPage="Controls",
            INPID_PreviewControl="AngleControl",
            INP_SplineType="Default",
            LINKID_DataType="Number",
            INPID_InputControl="SliderControl",
            INP_Integer=False,
            INP_MaxScale=720,
            LINKS_Name="CircularAngle",
        ),
        Radius=dict(
            INP_Integer=False,
            INPID_InputControl="SliderControl",
            INP_SplineType="Default",
            PC_ControlID=0,
            INPID_PreviewControl="EllipseControl",
            LINKID_DataType="Number",
            PC_ControlGroup=3,
            ICS_ControlPage="Controls",
            LINKS_Name="Radius",
        ),
    )

    transform.UserControls = user_control
    transform = transform.Refresh()

    # set input
    dcl.set_multiple_tool_input(tool=cross_line_mask, input_dict=cross_line_mask_input)
    dcl.set_multiple_tool_input(tool=circle_mask, input_dict=circle_mask_input)
    dcl.set_multiple_tool_input(tool=circle_bg, input_dict=circle_bg_input)
    dcl.set_multiple_tool_input(
        tool=cross_line_mask_merge, input_dict=cross_line_mask_merge_input
    )

    dcl.set_multiple_tool_input(tool=mask, input_dict=mask_input)
    dcl.set_multiple_tool_input(tool=bg, input_dict=bg_input)
    dcl.set_multiple_tool_input(tool=text, input_dict=text_input)
    dcl.set_multiple_tool_input(tool=transform, input_dict=transform_input)
    expression = (
        "Point("
        "Radius * comp:GetPrefs(\"Comp.FrameFormat.Height\") / "
        "comp:GetPrefs(\"Comp.FrameFormat.Width\") * sin(CircularAngle/180*pi) + "
        "CircleCenter.X, "
        "Radius * cos(CircularAngle/180*pi) + CircleCenter.Y"
        ")"
    )
    transform["Center"].SetExpression(expression)

    bezier_spline = comp.BezierSpline()
    fps = ppp.fps_int
    key_frame = {
        -fps//2: {
            1: ppp.motion_blur_angle_st_ed_list[c_idx][0],
            'LH': {1: -fps/2.0, 2: 0.0},
            'RH': {1: fps/2.0, 2: 0.0}
        },
        fps//2: {
            1: ppp.motion_blur_angle_st_ed_list[c_idx][1],
            'LH': {1: -fps/2.0, 2: 0.0},
            'RH': {1: fps/2.0, 2: 0.0}
        },
        fps + (fps//2): {
            1: ppp.motion_blur_angle_st_ed_list[c_idx][2],
            'LH': {1: -fps/2.0, 2: 0.0},
            'RH': {1: fps/2.0, 2: 0.0}
        }
    }
    bezier_spline.SetKeyFrames(key_frame)
    transform["CircularAngle"] = bezier_spline

    # connect
    dcl.connect_merge_tool(merge_tool=v_line_merge, bg_tool=h_line_merge, fg_tool=None)
    dcl.connect_merge_tool(
        merge_tool=cross_line_mask_merge, bg_tool=dummy_bg, fg_tool=v_line_merge
    )
    dcl.connect_merge_tool(
        merge_tool=circle_merge, bg_tool=cross_line_mask_merge, fg_tool=circle_bg
    )

    dcl.connect_merge_tool(merge_tool=text_merge, bg_tool=bg, fg_tool=text)
    dcl.connect_tool(text_merge, transform)
    dcl.connect_merge_tool(
        merge_tool=second_last_merge, bg_tool=circle_merge, fg_tool=transform
    )
    dcl.connect_merge_tool(
        merge_tool=output_merge, bg_tool=None, fg_tool=second_last_merge
    )

    return output_merge


def create_motion_blur_animation(comp, ppp: FusionParams, tool_pos=(1, 3)):
    x_pos = tool_pos[0]
    y_pos = tool_pos[1]

    input_merge = None
    pre_merge = None
    num_of_blur_obj = 4

    base_bg = dcl.add_transparent_background(comp=comp, pos=(x_pos, y_pos-1))

    output_merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=(x_pos+5*(num_of_blur_obj)-2, y_pos))

    for c_idx in range(num_of_blur_obj):
        merge = create_motion_blur_animation_core(
            comp=comp, c_idx=c_idx, ppp=ppp, tool_pos=(x_pos+5*c_idx, y_pos-1)
        )
        if input_merge is None:
            input_merge = merge
        if c_idx > 0:
            dcl.connect_merge_tool(merge_tool=merge, bg_tool=pre_merge, fg_tool=None)
        pre_merge = merge

    dcl.connect_merge_tool(merge_tool=input_merge, bg_tool=base_bg, fg_tool=None)
    dcl.connect_merge_tool(merge_tool=output_merge, bg_tool=None, fg_tool=pre_merge)

    return output_merge


def add_beep_sound():
    fps = float(dcl.get_project_setting(name="timelineFrameRate"))
    if fps.is_integer():
        wav_file_path = str(Path("./wav/countdown.wav").resolve())
    else:
        wav_file_path = str(Path("./wav/countdown_ntsc.wav").resolve())
    clip = dcl.add_file_to_media_pool(file_path=wav_file_path)
    dcl.append_clip_to_timeline(
        clip=clip, pos_frame_idx=dcl.sec_to_frame_idx(sec=60*60),
        media_type=2
    )

def create_countdown_comp():
    fps = float(dcl.get_project_setting(name="timelineFrameRate"))
    width, height = dcl.get_project_resolution()
    ppp = FusionParams(fps=fps, width=width, height=height)
    for idx, countdown_str in enumerate([4, 3, 2, 1]):
        start_frame = dcl.sec_to_frame_idx(60 * 60 + idx)
        tl_item_fusion_comp, comp =\
            dcl.append_fusion_composition_to_timeline(
                num_of_frame=ppp.fps_int,
                pos_frame_idx=start_frame
            )
        create_countdown_comp_each_sec(
            comp=comp, ppp=ppp, fps=ppp.fps_int, count_str=countdown_str)
        dcl.force_rcm_update_via_page_switch()
        # break

    add_beep_sound()


def create_countdown_comp_each_sec(comp, ppp, fps=24, count_str=3):
    """
    Parameters
    ----------
    comp : Composition
        A fusion Composition instance
    ppp : FusionParams
        A parameter set for fusion composition
    fps : int
        framerate
    count_str : int
        A character indicate the number of the count down.
    """
    comp.Lock()

    x_pos = 0
    y_pos = 3
    pseudo_bg = dcl.add_transparent_background(comp=comp, pos=(x_pos, y_pos))

    # basic background
    x_pos = 1
    y_pos += 0
    still_background_merge = create_still_background_comp(
        comp=comp, ppp=ppp, tool_pos=(x_pos, y_pos)
    )

    # countdown animation
    x_pos += 14
    y_pos += 0
    cntdown_anime_output_merge\
        = create_countdown_animation_comp(
            comp=comp, ppp=ppp, count_str=count_str,
            tool_pos=(x_pos, y_pos)
        )
    
    # frame marker
    x_pos += 8
    y_pos += 0
    frame_marker_output_merge\
        = create_frame_marker(
            comp=comp, ppp=ppp, tool_pos=(x_pos, y_pos)
        )
    
    # ramp pattern
    x_pos += 2*(fps+1) + 4
    y_pos += 0
    ramp_dctl, ramp_output_merge\
        = create_ramp(comp, ppp=ppp, tool_pos=(x_pos, y_pos))

    # motion blur animation
    x_pos += 2 * 7 + 2
    y_pos += 0
    motion_blur_output_merge\
        = create_motion_blur_animation(comp=comp, ppp=ppp, tool_pos=(x_pos, y_pos))

    # motion blur animation
    x_pos += 5 * 4
    y_pos += 0
    color_gain\
        = dcl.add_comp_tool(comp=comp, name="ColorGain", pos=[x_pos, y_pos])
    if count_str == 1:
        color_gain["GainRed"] = comp.BezierSpline()
        color_gain["GainGreen"] = comp.BezierSpline()
        color_gain["GainBlue"] = comp.BezierSpline()
        color_gain["GainRed"][0] = 1.0
        color_gain["GainGreen"][0] = 1.0
        color_gain["GainBlue"][0] = 1.0
        color_gain["GainRed"][1] = 0.0
        color_gain["GainGreen"][1] = 0.0
        color_gain["GainBlue"][1] = 0.0

    media_out = dcl.get_comp_tool_by_name(comp=comp, name="MediaOut1")
    x_pos += 3
    y_pos += 0
    dcl.set_tool_position(comp=comp, tool=media_out, pos=(x_pos, y_pos))

    # connect
    dcl.connect_merge_tool(
        merge_tool=still_background_merge, bg_tool=pseudo_bg, fg_tool=None
    )
    dcl.connect_merge_tool(
        merge_tool=cntdown_anime_output_merge,
        bg_tool=still_background_merge, fg_tool=None
    )
    dcl.connect_merge_tool(
        merge_tool=frame_marker_output_merge,
        bg_tool=cntdown_anime_output_merge, fg_tool=None
    )
    dcl.connect_dctl(dctl=ramp_dctl, source=frame_marker_output_merge)
    dcl.connect_merge_tool(
        merge_tool=ramp_output_merge, bg_tool=frame_marker_output_merge, fg_tool=None
    )
    dcl.connect_merge_tool(
        merge_tool=motion_blur_output_merge, bg_tool=ramp_output_merge, fg_tool=None
    )
    dcl.connect_tool(a=motion_blur_output_merge, b=color_gain)
    dcl.connect_mediaout(source=color_gain, mediaout=media_out)

    comp.Unlock()


def encode_hevc_using_ffmpeg(png_fname, fps: float, seq_file_ext, gamut, gamma, start_frame):
    print(png_fname)
    in_fname_ffmpeg = str(Path(png_fname + "_%08d." + seq_file_ext))
    target_dir = str(Path(png_fname).resolve().parent.parent)
    target_name = str(Path(png_fname).name)
    out_fname_444 = str(Path(target_dir) / Path(f"{target_name}_yub444p12le_rev{REVISION:02d}.mp4"))
    out_fname_420 = str(Path(target_dir) / Path(f"{target_name}_yub420p10le_rev{REVISION:02d}.mp4"))
    pix_fmt_444 = "yuv444p12le"
    pix_fmt_420 = "yuv420p10le"
    lossless_on = 'lossless=1'
    lossless_off = 'lossless=1'
    print(in_fname_ffmpeg)
    print(out_fname_444)
    cmd = "ffmpeg"
    codec = "libx265"
    # codec = "librav1e"
    # codec = "libvvenc"

    if gamma == drc.PRJ_GAMMA_STR_GAMMA24:
        color_trc = "bt709"
    elif gamma == drc.PRJ_GAMMA_STR_ST2084:
        color_trc = "smpte2084"
    else:
        raise ValueError("invalid gamma parameter")

    if gamut == drc.PRJ_COLOR_SPACE_REC709:
        color_primaries = "bt709"
        color_space = "bt709"
    elif gamut == drc.PRJ_COLOR_SPACE_REC2020:
        color_primaries = "bt2020"
        color_space = "bt2020nc"
    elif gamut == drc.PRJ_COLOR_SPACE_P3D65:
        color_primaries = "smpte432"
        color_space = "bt709"
    else:
        raise ValueError("invalid gamut parameter")

    if fps.is_integer():
        wav_file = "./wav/countdown.wav"
    else:
        wav_file = "./wav/countdown_ntsc.wav"

    for out_fname, pix_fmt in zip([out_fname_444, out_fname_420], [pix_fmt_444, pix_fmt_420]):
        if out_fname == out_fname_444:
            lossless = lossless_on
        else:
            lossless = lossless_off
        ops = [
            '-start_number', f"{start_frame}",
            '-color_primaries', color_primaries, '-color_trc', color_trc,
            '-colorspace', color_space,
            '-r', f"{fps}", '-i', in_fname_ffmpeg, '-i', wav_file,
            '-c:v', codec,
            # '-profile:v', 'main444-12',
            '-pix_fmt', pix_fmt,
            '-x265-params', lossless,
            '-c:a', 'aac', '-b:a', '128k',
            '-color_primaries', color_primaries, '-color_trc', color_trc,
            '-colorspace', color_space,
            str(out_fname), '-y'
        ]
        args = [cmd] + ops
        print(" ".join(args))
        subprocess.run(args)


def create_countdown_video_each_spec(
        width, height, framerate, gamut, gamma):
    ##################
    # Project Settings
    ##################
    dcl.refresh_lut_list()

    project_name = "Countdown_v2_Rev01"
    video_monitor_format = dcl.make_videoMonitorFormat_str(
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
        "colorSpaceTimeline": drc.PRJ_COLOR_SPACE_REC709,
        "colorSpaceTimelineGamma": drc.PRJ_GAMMA_STR_ST2084,
        "colorSpaceOutput": f"{gamut}",
        "colorSpaceOutputGamma": f"{gamma}",
        "timelineWorkingLuminance": "10000",
        "timelineWorkingLuminanceMode": "Custom",
        "inputDRT": "None",
        "outputDRT": "None",
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

    # add files to the media storage
    relative_file_list = [
        "./videos/countdown_HDR_24fps_hevc_yuv420p10le.mov",
        "./videos/countdown_SDR_24fps_hevc_yuv420p10le.mov",
        "./videos/countdown_SDR_60P_%04d.png",
        "./videos/countdown.wav",
    ]
    file_path_list = [
        str(Path(x).resolve()) for x in relative_file_list
    ]
    print(file_path_list)

    ###################
    # Core Function
    ###################
    create_countdown_comp()

    dcl.set_current_timecode(timecode=start_time_code)

    dcl.open_page(page_name=drc.FUSION_PAGE_STR)

    ###################
    # encode
    ###################
    # preset_path = str(
    #     Path("./render_presets/H265_Main10_444_10-bit Render.xml").resolve()
    # )
    preset_path = None

    # format_extension = drc.OUT_FILE_EXTENSTION_MOV
    format_extension = drc.OUT_FILE_EXTENSTION_PNG
    # codec = drc.CODEC_H265_NVIDIA
    # codec = drc.CODEC_H264_NVIDIA
    # codec = drc.CODEC_APPLE_PRORES_4444
    # format_extension = drc.OUT_FILE_EXTENSTION_EXR
    # codec = drc.CODEC_EXR_RGB_HALF
    codec = drc.CODEC_DPX_RGB_16_BITS
    basename = f"{width}x{height}_{framerate}P_{gamma}_{gamut}"
    # output_fname = f"./render_out/{basename}" + "." + format_extension

    if sys.platform == "darwin":  # macOS
        dir_path = Path("/Volumes/My Passport/Countdown/temp_seq")
        output_fname = str(dir_path / basename)
    else:
        output_fname = f"D:\\abuse\\Countdown\\temp_seq\\{basename}"
        dir_path = Path(r"D:\abuse\Countdown\temp_seq")
    shutil.rmtree(dir_path, ignore_errors=True)
    dir_path.mkdir(parents=True, exist_ok=True)
    if format_extension is drc.OUT_FILE_EXTENSTION_PNG:
        output_fname = output_fname
    else:
        output_fname = output_fname + "." + format_extension
    target_dir = str(Path(output_fname).resolve().parent)
    custom_name = str(Path(output_fname).resolve().name)

    render_settings = {
        "TargetDir": target_dir,
        "CustomName": custom_name,
    }

    if preset_path is not None:
        dcl.import_render_preset(preset_path=preset_path)
    else:
        dcl.set_render_format_codec_settings(format=format_extension, codec=codec)

    dcl.set_render_settings(setting_dict=render_settings)
    dcl.run_rendering_and_wait_until_finish(project=project)

    encode_hevc_using_ffmpeg(
        png_fname=output_fname, fps=framerate, seq_file_ext=format_extension,
        gamma=gamma, gamut=gamut, start_frame=start_frame)


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
        23.976,
        24,
        25,
        29.97,
        30,
        50,
        59.94,
        60
    ]
    gamut_list = [
        drc.PRJ_COLOR_SPACE_REC709,
        drc.PRJ_COLOR_SPACE_P3D65,
        drc.PRJ_COLOR_SPACE_REC2020
    ]
    gamma_list = [
        drc.PRJ_GAMMA_STR_GAMMA24,
        drc.PRJ_GAMMA_STR_ST2084
    ]

    for resolution, framerate, gamut, gamma in product(
        resolution_list, framerate_list, gamut_list, gamma_list
    ):
        width, height = resolution.split("x")
        create_countdown_video_each_spec(
            width=width, height=height, framerate=framerate,
            gamut=gamut, gamma=gamma
        )
