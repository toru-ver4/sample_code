# -*- coding: utf-8 -*-

# import standard libraries
import sys
import os
from pathlib import Path
from pprint import pprint
import copy
from collections import OrderedDict

# import third-party libraries
import numpy as np

# import my libraries
import ty_davinci_constants as drc
import ty_davinci_control_lib_2 as dcl
import transfer_functions as tf


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
    dcl.refresh_lut_list()
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

    fusion_comp = timeline_item.GetFusionCompByIndex(1)
    merge_tool = dcl.get_comp_tool_by_name(comp=fusion_comp, name="Merge1")
    media_out = dcl.get_comp_tool_by_name(comp=fusion_comp, name="MediaOut1")
    print(merge_tool)
    dump_tool_input_value(tool=merge_tool)
    dump_tool_main_input_value(tool=merge_tool)
    transform = dcl.get_comp_tool_by_name(comp=fusion_comp, name="Transform1")
    dump_tool_main_input_value(tool=transform)
    # dump_tool_input_value(tool=media_out)

    rec56 = dcl.get_comp_tool_by_name(comp=fusion_comp, name="Text4")
    rec_mask = dcl.add_comp_tool(comp=fusion_comp, name="TextPlus", pos=(20, 20))
    # compare_tool_input_value(aa=rec56, bb=rec_mask)

    transform = dcl.get_comp_tool_by_name(comp=fusion_comp, name="Transform1")
    pprint(transform.UserControls)

    # is_font_available(family="Noto Sans Mono", font_weight="Black")
    
    # dump_tool_list(comp=fusion_comp)

    import sys
    sys.exit(0)


#####################
# Logic
#####################
class HeightBasedSize:
    def __init__(self, size, hv_same=False, resolution=None):
        """
        Parameters
        ----------
        size: float
            A size parameter
        resolution : list or tuple
            [width, height] or (width, height)
        """
        if resolution is None:
            width, height = dcl.get_project_resolution()
        else:
            width, height = resolution
        self._v_size = size

        if hv_same:
            self._h_size = size
        else:    
            self._h_size = (self._v_size * height) / width

    @property
    def v_size(self):
        return self._v_size

    @property
    def h_size(self):
        return self._h_size


class FusionParams:
    def __init__(self, fps):
        """
        Parameters
        ----------
        resolution : list or tuple
            [width, height] or (width, height)
        """
        self.cd_circle_ll = HeightBasedSize(0.58)
        self.cd_circle_mm = HeightBasedSize(0.515)
        self.cd_circle_ss = HeightBasedSize(0.495)
        self.cd_line_width = HeightBasedSize(0.005)
        self.cd_line_color = [0.0, 0.0, 0.0, 1.0]
        self.cd_font_size = HeightBasedSize(0.85)
        self.cross_line_width = self.cd_line_width
        self.gray90 = 0.8
        self.gray80 = 0.7
        self.cross_line_color = [self.gray90, self.gray90, self.gray90, 1.0]
        self.info_area_height = HeightBasedSize(0.1)

        frame_marker_h_st_pos = 0.07
        frame_marker_h_ed_pos = 1 - frame_marker_h_st_pos
        self.frame_marker_h_pos\
            = self.linspace(frame_marker_h_st_pos, frame_marker_h_ed_pos, fps + 1)
        self.frame_marker_v_pos = 0.1295 + 0.005
        self.frame_marker_v_pos2 = 0.0998 + 0.005
        self.frame_marker_width\
            = (frame_marker_h_ed_pos - frame_marker_h_st_pos) / (fps * 2 + 1)
        self.frame_marker_height = 0.03
        self.frame_marker_outline_width = self.calc_frame_marker_outline_width(
            h_pos_list=self.frame_marker_h_pos,
            each_marker_width=self.frame_marker_width
        )
        self.frame_marker_outline_height = self.frame_marker_height * 3
        self.frame_marker_outline_v_pos\
            = (self.frame_marker_v_pos - self.frame_marker_v_pos2) / 2.0\
            + self.frame_marker_v_pos2
        self.frame_marker_outline_line_width = 0.003

        self.ramp_height = 0.09
        self.lumi_text_v_pos = 0.829
        self.cv_text_v_pos = 0.968

        self.motion_blur_mask_size = HeightBasedSize(0.075)
        self.motion_blur_bg_color = (192/255.0) ** 2.4
        self.motion_blur_text_color = (64/255.0) ** 2.4
        self.motion_blur_color_mask = [
            [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
        ]
        self.motion_blur_text = ["@", "@", "@", "@"]
        self.motion_blur_text_size = 0.06

    def calc_frame_marker_outline_width(self, h_pos_list, each_marker_width):
        margin = h_pos_list[1] - h_pos_list[0]
        st_pos = h_pos_list[0] - margin + (each_marker_width / 2)
        ed_pos = h_pos_list[-1] + margin - (each_marker_width / 2)

        return ed_pos - st_pos

    def linspace(self, start, stop, num):
        if num == 1:
            return [start]
        step = (stop - start) / (num - 1)
        return [start + step * i for i in range(num)]


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


def draw_line_comp(comp, rgba, width, height, angle=0, base_pos=[0, 0]):
    x_pos = base_pos[0]
    y_pos = base_pos[1]

    line = dcl.add_comp_tool(comp=comp, name="RectangleMask", pos=[x_pos, y_pos-2])
    line_fg = dcl.add_comp_tool(comp=comp, name="Background", pos=[x_pos, y_pos-1])
    line_merge = dcl.add_comp_tool(comp=comp, name="Merge", pos=[x_pos, y_pos+0])

    line_input = {
        "Width": width,
        "Height": height,
        "Angle": angle,
    }
    dcl.set_multiple_tool_input(tool=line, input_dict=line_input)
    line_fg_input = {
        "TopLeftRed": rgba[0],
        "TopLeftGreen": rgba[1],
        "TopLeftBlue": rgba[2],
        "TopLeftAlpha": rgba[3],
        "EffectMask": line,
    }
    dcl.set_multiple_tool_input(tool=line_fg, input_dict=line_fg_input)

    dcl.connect_merge_tool(
        merge_tool=line_merge,
        bg_tool=None, fg_tool=line_fg
    )

    return line_merge


def draw_info_comp(
        comp, font_size, bg_rgba, fg_rgba, height, base_pos=[0, 0]):
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
    fps = int(dcl.get_project_setting("timelineFrameRate"))
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
        "VerticalTopCenterBottom": 1.75,
        "HorizontalLeftCenterRight": -1.0,
        "AdvancedFontControls": 1.0,
    }
    dcl.is_font_available(family=font_family, font_weight=font_weight)
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
        "StyledText": "Revision 00  ",
        "Font": font_family,
        "Style": font_weight,
        "Size": font_size,
        "Red1": fg_rgba[0],
        "Green1": fg_rgba[1],
        "Blue1": fg_rgba[2],
        "VerticalTopCenterBottom": 1.75,
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
        comp=comp, name="Background", pos=(x_pos+0, y_pos+0)
    )
    dcl.set_tool_topleft_color(tool=bg1, rgba=[0.18, 0.18, 0.18, 1.0])

    cross_h_line_merge = draw_line_comp(
        comp=comp, rgba=ppp.cross_line_color, width=1.0, angle=0,
        height=ppp.cross_line_width.h_size, base_pos=[x_pos+1, y_pos]
    )
    cross_v_line_merge = draw_line_comp(
        comp=comp, rgba=ppp.cross_line_color, width=1.0, angle=90,
        height=ppp.cross_line_width.h_size, base_pos=[x_pos+2, y_pos]
    )
    large_white_circle_merge = create_background_circle(
        comp=comp, bg_rgba=[ppp.gray80, ppp.gray80, ppp.gray80, 1.0],
        size=[ppp.cd_circle_ll.h_size, ppp.cd_circle_ll.h_size],
        merge_pos=[x_pos+3, y_pos]
    )
    middle_black_circle_merge = create_background_circle(
        comp=comp, bg_rgba=[0.0, 0.0, 0.0, 1.0],
        size=[ppp.cd_circle_mm.h_size, ppp.cd_circle_mm.h_size],
        merge_pos=[x_pos+4, y_pos]
    )
    small_grey_circle_merge = create_background_circle(
        comp=comp, bg_rgba=[0.18, 0.18, 0.18, 1.0],
        size=[ppp.cd_circle_ss.h_size, ppp.cd_circle_ss.h_size],
        merge_pos=[x_pos+5, y_pos]
    )
    h_line_merge = draw_line_comp(
        comp=comp, rgba=ppp.cd_line_color, angle=0,
        width=ppp.cd_circle_ll.h_size,
        height=ppp.cd_line_width.h_size, base_pos=[x_pos+6, y_pos]
    )
    v_line_merge = draw_line_comp(
        comp=comp, rgba=ppp.cd_line_color, angle=90,
        width=ppp.cd_circle_ll.h_size,
        height=ppp.cd_line_width.h_size, base_pos=[x_pos+7, y_pos]
    )
    info_in_merge, info_out_merge = draw_info_comp(
        comp=comp, font_size=0.025, bg_rgba=[0.0, 0.0, 0.0, 1.0],
        fg_rgba=[0.5, 0.5, 0.5, 1.0], height=0.035, base_pos=[x_pos+8, y_pos])
    border_dctl = dcl.add_dctl_comp(
        comp=comp, dctl_path="TY_DCTL/draw_countdown_border.dctl", base_pos=[x_pos+11, y_pos]
    )

    dcl.connect_merge_tool(
        merge_tool=cross_h_line_merge,
        bg_tool=bg1, fg_tool=None
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
    
    out_tool = border_dctl

    return out_tool


def create_countdown_animation_comp(
    comp, ppp: FusionParams, count_str, fps, tool_pos
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

    radial_wipe = dcl.add_comp_tool(
        comp=comp, name="EllipseMask", pos=(x_pos+0, y_pos-3)
    )
    wipe_circle_mask = dcl.add_comp_tool(
        comp=comp, name="EllipseMask", pos=(x_pos+0, y_pos-2)
    )
    wipe_circle_fg = dcl.add_comp_tool(
        comp=comp, name="Background", pos=(x_pos+0, y_pos-1)
    )
    wipe_circle_merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=(x_pos+0, y_pos-0)
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
    radial_wipe["WriteLength"][fps] = 0.0

    # mask settings for wipe animation
    wipe_circle_mask_input = {
        "Invert": 1.0,
        "Width": ppp.cd_circle_mm.h_size,
        "Height": ppp.cd_circle_mm.h_size,
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

    # text
    countdown_text = dcl.add_comp_tool(
        comp=comp, name="TextPlus", pos=(x_pos+1, y_pos-1)
    )
    countdown_text_merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=(x_pos+1, y_pos-0)
    )

    font_family = "Noto Sans Mono"
    font_weight = "Black"
    countdown_text_input = {
        "StyledText": f"{count_str}",
        "Font": font_family,
        "Style": font_weight,
        "Size": ppp.cd_font_size.h_size,
        "Red1": ppp.gray80,
        "Green1": ppp.gray80,
        "Blue1": ppp.gray80,
    }
    dcl.is_font_available(family=font_family, font_weight=font_weight)
    dcl.set_multiple_tool_input(
        tool=countdown_text, input_dict=countdown_text_input
    )

    # connect
    dcl.connect_merge_tool(
        merge_tool=countdown_text_merge,
        bg_tool=wipe_circle_merge, fg_tool=countdown_text
    )
    dcl.connect_merge_tool(
        merge_tool=wipe_circle_merge,
        bg_tool=None, fg_tool=wipe_circle_fg
    )

    # output
    input_merge = wipe_circle_merge
    output_merge = countdown_text_merge

    return input_merge, output_merge


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


def create_frame_marker(comp, ppp, fps, tool_pos=(1, 3)):
    x_pos = tool_pos[0]
    y_pos = tool_pos[1]
    merge_list = []
    for idx in range(fps+1):
        bg_merge, inv_bg_merge = create_frame_marker_core(
            comp=comp, ppp=ppp, idx=idx, fps=fps, tool_pos=(x_pos+2*idx, y_pos)
        )
        merge_list.append([bg_merge, inv_bg_merge])

    for idx in range(1, fps+1):
        dcl.connect_merge_tool(
            merge_tool=merge_list[idx][0],
            bg_tool=merge_list[idx-1][1], fg_tool=None
        )

    outline_rect = dcl.add_comp_tool(
        comp=comp, name="RectangleMask", pos=(x_pos+2*(fps+1), y_pos-2)
    )
    outline_bg = dcl.add_comp_tool(
        comp=comp, name="Background", pos=((x_pos+2*(fps+1), y_pos-1))
    )
    outline_merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=((x_pos+2*(fps+1), y_pos-0))
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

    return merge_list[0][0], outline_merge


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
        comp=comp, dctl_path="TY_DCTL/draw_countdown_ramp.dctl", base_pos=[x_pos, y_pos],
        option={
            "sliderFloatParam0": ppp.frame_marker_outline_width,
            "sliderFloatParam1": ppp.ramp_height * 0.93,
            "sliderIntParam0": 4
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
            st_pos=st_pos, ramp_width=ramp_width, x_pos=x_pos, y_pos=y_pos
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

    return ramp_dctl, ed_merge


def create_motion_blur_animation_core(comp, c_idx, ppp: FusionParams, tool_pos=(1, 3)):
    x_pos = tool_pos[0]
    y_pos = tool_pos[1]

    output_merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=(x_pos+0, y_pos-0)
    )
    transform = dcl.add_comp_tool(
        comp=comp, name="Transform", pos=(x_pos+0, y_pos-1)
    )
    text_merge = dcl.add_comp_tool(
        comp=comp, name="Merge", pos=(x_pos+0, y_pos-2)
    )
    text = dcl.add_comp_tool(
        comp=comp, name="TextPlus", pos=(x_pos+1, y_pos-2)
    )
    bg = dcl.add_comp_tool(
        comp=comp, name="Background", pos=(x_pos+0, y_pos-3)
    )
    mask = dcl.add_comp_tool(
        comp=comp, name="RectangleMask", pos=(x_pos+0, y_pos-4)
    )

    # input
    mask_input = {
        "Width": ppp.motion_blur_mask_size.h_size,
        "Height": ppp.motion_blur_mask_size.v_size,
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

    user_control = OrderedDict()
    user_control["CircleC"] = {
        "ICS_ControlPage": "Controls",
        "INPID_PreviewControl": "PointControl",
        "LINKID_DataType": "Point",
        "LINKS_Name": "CircleCenter",
    }
    user_control["CircleCenter"] = {
        "INPID_PreviewControl": "PointControl",
        "LINKID_DataType": "Point",
        "ICS_ControlPage": "Controls",
        "INPID_InputControl": "OffsetControl",
        "LINKS_Name": "CircleCenter",
    }
    user_control["CircleAngle"] = {
        "ICS_ControlPage": "Controls",
        "INPID_PreviewControl": "AngleControl",
        "INP_SplineType": "Default",
        "LINKID_DataType": "Number",
        "INPID_InputControl": "SliderControl",
        "INP_Integer": False,
        "INP_MaxScale": 360,
        "LINKS_Name": "CircleAngle",
    }
    user_control["Radius"] = {
        "INP_Integer": False,
        "INPID_InputControl": "SliderControl",
        "INP_SplineType": "Default",
        "PC_ControlID": 0,
        "INPID_PreviewControl": "EllipseControl",
        "LINKID_DataType": "Number",
        "PC_ControlGroup": 3,
        "ICS_ControlPage": "Controls",
        "LINKS_Name": "Radius",
    }

    ctrl = {}
    ctrl['RENAME2'] = {
        'ICD_Width': 0.5,
        'INP_Default': 0,
        'ICS_ControlPage': "File",
        'BTNCS_Execute': " tool:SetAttrs({TOOLS_Name = 'w_' .. tool.Input:GetConnectedOutput():GetTool():GetAttrs()['TOOLS_Name']}) ",
        'INPID_InputControl': "ButtonControl",
        'LINKID_DataType': "Number",
        'LINKS_Name': "Rename2"
    }

    transform.UserControls = user_control
    transform = transform.Refresh()

    # set input
    dcl.set_multiple_tool_input(tool=mask, input_dict=mask_input)
    dcl.set_multiple_tool_input(tool=bg, input_dict=bg_input)
    dcl.set_multiple_tool_input(tool=text, input_dict=text_input)

    # connect
    dcl.connect_merge_tool(merge_tool=output_merge, bg_tool=None, fg_tool=transform)
    dcl.connect_merge_tool(merge_tool=text_merge, bg_tool=bg, fg_tool=text)
    dcl.connect_tool(text_merge, transform)

    return output_merge


def create_motion_blur_animation(comp, ppp: FusionParams, tool_pos=(1, 3)):
    x_pos = tool_pos[0]
    y_pos = tool_pos[1]

    input_merge = None
    output_merge = None
    pre_merge = None

    for c_idx in range(4):
        merge = create_motion_blur_animation_core(
            comp=comp, c_idx=c_idx, ppp=ppp, tool_pos=(x_pos + 2 * c_idx, y_pos)
        )
        if input_merge is None:
            input_merge = merge
        if c_idx > 0:
            dcl.connect_merge_tool(merge_tool=merge, bg_tool=pre_merge, fg_tool=None)
        pre_merge = merge
        output_merge = merge

    return input_merge, output_merge


def create_countdown_comp():
    fps = int(dcl.get_project_setting(name="timelineFrameRate"))
    ppp = FusionParams(fps=fps)
    for idx, countdown_str in enumerate([4, 3, 2, 1]):
        tl_item_fusion_comp, comp =\
            dcl.append_fusion_composition_to_timeline(
                num_of_frame=fps,
                pos_timecode=f"01:00:{idx:02d}:00"
            )
        create_countdown_comp_each_sec(
            comp=comp, ppp=ppp, fps=fps, count_str=countdown_str)
        break


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

    # basic background
    x_pos = 1
    y_pos = 3
    still_bg_tool = create_still_background_comp(
        comp=comp, ppp=ppp, tool_pos=(x_pos, y_pos)
    )

    # countdown animation
    x_pos = 13
    y_pos += 4
    cntdown_anime_input_merge, cntdown_anime_output_merge\
        = create_countdown_animation_comp(
            comp=comp, ppp=ppp, count_str=count_str, fps=fps,
            tool_pos=(x_pos, y_pos)
        )
    
    # frame marker
    x_pos = 15
    y_pos += 4
    frame_marker_input_merge, frame_marker_output_merge\
        = create_frame_marker(
            comp=comp, ppp=ppp, fps=fps, tool_pos=(x_pos, y_pos)
        )
    
    # ramp pattern
    x_pos += 2*(fps+1) + 1
    y_pos += 2
    st_ramp_dctl, ed_ramp_dctl\
        = create_ramp(comp, ppp=ppp, tool_pos=(x_pos, y_pos))

    # motion blur animation
    x_pos += 2 * 7 + 1
    y_pos += 0
    st_motion_blur, ed_motion_blur\
        = create_motion_blur_animation(comp=comp, ppp=ppp, tool_pos=(x_pos, y_pos))

    media_out = dcl.get_comp_tool_by_name(comp=comp, name="MediaOut1")
    x_pos += 3 * 4
    y_pos += 0
    dcl.set_tool_position(comp=comp, tool=media_out, pos=(x_pos, y_pos))

    # connect
    dcl.connect_mediaout(source=ed_motion_blur, mediaout=media_out)
    dcl.connect_merge_tool(merge_tool=st_motion_blur, bg_tool=ed_ramp_dctl, fg_tool=None)
    dcl.connect_dctl(dctl=st_ramp_dctl, source=frame_marker_output_merge)
    dcl.connect_merge_tool(
        merge_tool=cntdown_anime_input_merge,
        bg_tool=still_bg_tool, fg_tool=None
    )
    dcl.connect_merge_tool(
        merge_tool=frame_marker_input_merge,
        bg_tool=cntdown_anime_output_merge, fg_tool=None
    )

    comp.Unlock()


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
    # set_timeline_settings(timeline=timeline, params=project_settings_params)

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

    create_countdown_comp()
    dcl.set_current_timecode(timecode="01:00:00:00")

    dcl.open_page(page_name=drc.FUSION_PAGE_STR)

    # ###################
    # # encode
    # ###################
    # # preset_path = str(
    # #     Path("./render_presets/h265_main10_444_qp-0.xml").resolve()
    # # )
    # preset_path = None

    # format_extension = drc.OUT_FILE_EXTENSTION_MOV
    # # codec = drc.CODEC_H265_NVIDIA
    # codec = drc.CODEC_APPLE_PRORES_4444
    # # format_extension = drc.OUT_FILE_EXTENSTION_EXR
    # # codec = drc.CODEC_EXR_RGB_HALF
    # basename = f"{width}x{height}_{framerate}_{gamma}_{gamut}"
    # output_fname = f"./render_out/{basename}" + "." + format_extension
    # target_dir = str(Path(output_fname).resolve().parent)
    # custom_name = str(Path(output_fname).resolve().name)

    # render_settings = {
    #     "TargetDir": target_dir,
    #     "CustomName": custom_name,
    # }

    # if preset_path is not None:
    #     dcl.import_render_preset(preset_path=preset_path)
    # else:
    #     dcl.set_render_format_codec_settings(format=format_extension, codec=codec)

    # dcl.set_render_settings(setting_dict=render_settings)
    # # run_rendering_and_wait_until_finish(project=project)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # debug_resolve()
    # debug_fusion()

    from itertools import product
    resolution_list = [
        "1920x1080",
        # "2048x1080",
        # "3840x2160",
        # "4096x2160",
    ]
    framerate_list = [
        24,
        # 25,
        # 30,
        # 50,
        # 60
    ]
    gamut_list = [
        drc.PRJ_COLOR_SPACE_REC709,
        # drc.PRJ_COLOR_SPACE_P3D65,
        # drc.PRJ_COLOR_SPACE_REC2020
    ]
    gamma_list = [
        drc.PRJ_GAMMA_STR_GAMMA24,
        # drc.PRJ_GAMMA_STR_ST2084
    ]

    for resolution, framerate, gamut, gamma in product(
        resolution_list, framerate_list, gamut_list, gamma_list
    ):
        width, height = resolution.split("x")
        create_countdown_video_each_spec(
            width=width, height=height, framerate=framerate,
            gamut=gamut, gamma=gamma
        )
