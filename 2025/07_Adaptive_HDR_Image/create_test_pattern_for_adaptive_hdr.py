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
from colour.models import RGB_COLOURSPACE_BT2020

# import my libraries
import ty_davinci_constants as drc
import ty_davinci_control_lib_2 as dcl
import transfer_functions as tf
from test_pattern_generator2 import generate_color_checker_rgb_value

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
        cc_width_int = 720  # HD Based Size
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
        self.ramp_pos = [
            self.scale_h_margin,
            0.4,
        ]
        self.ramp_text_pos_v = (1 - self.ramp_pos[1]) - self.ramp_height * 2 - 0.02

    def to_even(self, n: int|float) -> int:
        n_int = int(round(n))
        return n_int - (n_int % 2)


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
    print(f"info_text = {info_text_str}")
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


def create_scale_column(comp, ppp: FusionParams=None, base_pos=[0, 0]):
    x_pos = base_pos[0]
    y_pos = base_pos[1]

    base_bg = dcl.add_transparent_background(comp=comp, pos=(x_pos, y_pos))

    bg_tool = base_bg
    fg_tool = None
    rgba_color_list = [
        [0, 0, 1, 1],
        [1, 0, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]
    num_of_color = len(rgba_color_list)

    for c_idx in range(num_of_color):
        x_pos += 1
        center_y = ppp.scale_v_margin + ppp.scale_vv / 2.0 + ppp.scale_vv * c_idx
        merge = dcl.add_comp_tool(comp=comp, name="Merge", pos=(x_pos, y_pos))
        rectangle = dcl.add_rectangle_comp(
            comp=comp,
            rgba_color=rgba_color_list[c_idx],
            center=[0.5, center_y],
            width=ppp.scale_hh,
            height=ppp.scale_vv,
            base_pos=[x_pos, y_pos-1]
        )
        fg_tool = rectangle
        dcl.connect_merge_tool(merge_tool=merge, bg_tool=bg_tool, fg_tool=fg_tool)
        bg_tool = merge
    last_merge = bg_tool

    return last_merge, x_pos


def create_scale_info_text(
        comp, ppp: FusionParams=None, center_y=0.6, luminance=10, base_pos=[0, 0]):
    x_pos = base_pos[0]
    y_pos = base_pos[1]

    info_text = dcl.add_comp_tool(comp=comp, name="TextPlus", pos=(x_pos, y_pos))
    font_family = "Noto Sans Mono"
    font_weight = "Medium"
    if luminance < 100:
        info_text_str = f"{luminance:5.1f} nits"
    else:
        info_text_str = f"{luminance:5.0f} nits"
    info_text_input = {
        "Center": {1: 0.5, 2: center_y, 3: 0.0},
        "StyledText": info_text_str,
        "Font": font_family,
        "Style": font_weight,
        "Size": ppp.scale_text_size,
        "Red1": ppp.info_font_color[0],
        "Green1": ppp.info_font_color[1],
        "Blue1": ppp.info_font_color[2],
        "VerticalTopCenterBottom": ppp.scale_text_v_anchor,
        "HorizontalLeftCenterRight": ppp.scale_text_h_anchor,
        "AdvancedFontControls": 0.0,
		"SelectTransform": 2,  # Line Rotation
		"LineAngleZ": 90,
    }
    dcl.set_multiple_tool_input(tool=info_text, input_dict=info_text_input)

    return info_text


def create_scale_comp(comp, ppp: FusionParams=None, base_pos=[0, 0]):
    x_pos = base_pos[0]
    y_pos = base_pos[1]

    base_bg = dcl.add_transparent_background(comp=comp, pos=(x_pos, y_pos))

    bg_tool = base_bg
    x_pos += 1
    scale, x_pos = create_scale_column(comp=comp, ppp=ppp, base_pos=[x_pos, y_pos-1])

    text_pos_y = ppp.scale_v_margin + ppp.scale_vv * 12
    x = np.linspace(0, 4, ppp.scale_num_of_element)
    luminance_list = 10 ** x

    x_pos += 2
    for h_idx in range(ppp.scale_num_of_element):
        color_gain = dcl.add_comp_tool(comp=comp, name="ColorGain", pos=(x_pos, y_pos-1))
        color_gain_input = {
			"LockRGB": 1,
			"GainRed": luminance_list[h_idx] / 100.0,
        }
        dcl.set_multiple_tool_input(tool=color_gain, input_dict=color_gain_input)
        dcl.connect_tool(scale, color_gain)

        text_merge = dcl.add_comp_tool(comp=comp, name="Merge", pos=(x_pos+1, y_pos-1))
        luminance_text = create_scale_info_text(
            comp=comp, ppp=ppp, luminance=luminance_list[h_idx], center_y=text_pos_y,
            base_pos=(x_pos+1, y_pos-2)
        )
        dcl.connect_merge_tool(
            merge_tool=text_merge, bg_tool=color_gain, fg_tool=luminance_text
        )

        transform = dcl.add_comp_tool(comp=comp, name="Transform", pos=(x_pos+2, y_pos-1))
        transform_input = {
            "Center": {
                1: ppp.scale_h_margin + ppp.scale_hh / 2.0 + ppp.scale_hh * h_idx,
                2: 0.5,
                3: 0.0
            },
        }
        dcl.set_multiple_tool_input(tool=transform, input_dict=transform_input)
        dcl.connect_tool(text_merge, transform)

        offset = 0 if h_idx == 0 else -1
        output_merge = dcl.add_comp_tool(comp=comp, name="Merge", pos=(x_pos+3, y_pos+offset))
        dcl.connect_merge_tool(
            merge_tool=output_merge, bg_tool=bg_tool, fg_tool=transform
        )
        bg_tool = output_merge
        y_pos += 4 if h_idx == 0 else 2
    last_merge = bg_tool

    return last_merge, x_pos+3


def create_color_checker_background(comp, ppp: FusionParams=None, base_pos=[0, 0]):
    x_pos = base_pos[0]
    y_pos = base_pos[1]

    bg = dcl.add_comp_tool(comp=comp, name="Background", pos=(x_pos, y_pos-1))
    bg_mask = dcl.add_comp_tool(
        comp=comp, name="RectangleMask", pos=(x_pos, y_pos-2)
    )
    bg_mask_input = {
        "Center": {1: 0.5, 2: 0.5, 3: 0.0},
        "Width": ppp.cc_width,
        "Height": ppp.cc_height,
        "CornerRadius": ppp.cc_corner_radius,
    }
    dcl.set_multiple_tool_input(tool=bg_mask, input_dict=bg_mask_input)
    bg_input = {
        "TopLeftRed": 0.00,
        "TopLeftGreen": 0.00,
        "TopLeftBlue": 0.00,
        "TopLeftAlpha": 1.0,
        "EffectMask": bg_mask,
    }
    dcl.set_multiple_tool_input(tool=bg, input_dict=bg_input)

    return bg


def create_color_checker_patch_vertical(
        comp, ppp: FusionParams=None, h_idx=0, base_pos=[0, 0]):
    x_pos = base_pos[0]
    y_pos = base_pos[1]

    base_bg = dcl.add_transparent_background(comp=comp, pos=(x_pos, y_pos))
    bg_tool = base_bg

    x_pos += 1
    for v_idx in range(ppp.num_of_cc_patch_v):
        cc_idx = v_idx * ppp.num_of_cc_patch_h + h_idx
        rectangle = dcl.add_rectangle_comp(
            comp=comp,
            rgba_color=[
                ppp.cc_rgb[cc_idx, 0],
                ppp.cc_rgb[cc_idx, 1],
                ppp.cc_rgb[cc_idx, 2],
                1
            ],
            center=[
                ppp.cc_margin_h + ppp.pp_hh / 2.0 + (ppp.pp_hh + ppp.cc_margin_h) * h_idx + ppp.cc_center_offset[0],
                ppp.cc_margin_v + ppp.pp_vv / 2.0 + (ppp.pp_vv + ppp.cc_margin_v) * (3 - v_idx) + ppp.cc_center_offset[1],
            ],
            width=ppp.pp_hh,
            height=ppp.pp_vv,
            base_pos=[x_pos, y_pos-1],
        )

        y_pos_offset = 0 if v_idx == 0 else -1
        merge = dcl.add_comp_tool(
            comp=comp, name="Merge", pos=(x_pos+1, y_pos + y_pos_offset)
        )
        dcl.connect_merge_tool(
            merge_tool=merge, bg_tool=bg_tool, fg_tool=rectangle
        )
        bg_tool = merge
        y_pos_offset = 5 if v_idx == 0 else 2
        y_pos += y_pos_offset

    return bg_tool, x_pos+1


def create_color_checker_comp(comp, ppp: FusionParams=None, base_pos=[0, 0]):
    x_pos = base_pos[0]
    y_pos = base_pos[1]

    base_bg = dcl.add_transparent_background(comp=comp, pos=(x_pos, y_pos))

    x_pos += 1
    bg = create_color_checker_background(comp=comp, ppp=ppp, base_pos=(x_pos, y_pos-1))
    bg_merge = dcl.add_comp_tool(comp=comp, name="Merge", pos=(x_pos, y_pos))
    dcl.connect_merge_tool(merge_tool=bg_merge, bg_tool=base_bg, fg_tool=bg)
    bg_tool = bg_merge

    for h_idx in range(ppp.num_of_cc_patch_h):
        x_pos += 1
        vertical_patches, x_pos = create_color_checker_patch_vertical(
            comp=comp, ppp=ppp, h_idx=h_idx, base_pos=(x_pos, y_pos-1)
        )
        x_pos += 1
        cc_merge = dcl.add_comp_tool(comp=comp, name="Merge", pos=(x_pos, y_pos))
        dcl.connect_merge_tool(
            merge_tool=cc_merge, bg_tool=bg_tool, fg_tool=vertical_patches
        )
        bg_tool = cc_merge

    x_pos += 1
    transform = dcl.add_comp_tool(comp=comp, name="Transform", pos=(x_pos, y_pos))
    transform_input = {
        "Center": {
            1: ppp.cc_pos[0],
            2: ppp.cc_pos[1],
            3: 0.0
        },
    }
    dcl.set_multiple_tool_input(tool=transform, input_dict=transform_input)
    dcl.connect_tool(bg_tool, transform)

    return transform, x_pos


def create_ramp_info_text(
        comp, ppp: FusionParams=None, center_pos=[0, 1], luminance=10, base_pos=[0, 0]):
    x_pos = base_pos[0]
    y_pos = base_pos[1]

    info_text = dcl.add_comp_tool(comp=comp, name="TextPlus", pos=(x_pos, y_pos))
    font_family = "Noto Sans Mono"
    font_weight = "Medium"
    if luminance < 1000:
        info_text_str = f"{luminance}"
    else:
        info_text_str = f"{int(luminance/1000)}K"
    info_text_input = {
        "Center": {1: center_pos[0], 2: center_pos[1], 3: 0.0},
        "StyledText": info_text_str,
        "Font": font_family,
        "Style": font_weight,
        "Size": ppp.scale_text_size,
        "Red1": ppp.info_font_color[0],
        "Green1": ppp.info_font_color[1],
        "Blue1": ppp.info_font_color[2],
        "VerticalTopCenterBottom": 0,
        "HorizontalLeftCenterRight": 0,
    }
    dcl.set_multiple_tool_input(tool=info_text, input_dict=info_text_input)

    return info_text


def create_ramp_pattern_comp(comp, ppp: FusionParams=None, base_pos=[0, 0]):
    x_pos = base_pos[0]
    y_pos = base_pos[1]

    base_bg = dcl.add_transparent_background(comp=comp, pos=(x_pos, y_pos))

    x_pos += 1
    upper_ramp_dctl = dcl.add_dctl_comp(
        comp=comp, dctl_path="TY_DCTL/draw_8bit_10bit_ramp.dctl",
        option={
            "sliderIntParam0": ppp.ramp_border_width,
            "sliderFloatParam0": ppp.ramp_height,  # ramp height
            "sliderFloatParam1": ppp.ramp_pos[0],  # ramp st_pos_h
            "sliderFloatParam2": ppp.ramp_pos[1],  # ramp st_pos_v
            "sliderFloatParam3": 1,  # show top scale
            "sliderFloatParam4": 0,  # show bottom scale
            "checkBoxParam0": 0,  # 8-bit ramp
        },
        base_pos=[x_pos, y_pos]
    )
    dcl.connect_dctl(upper_ramp_dctl, base_bg)

    x_pos += 1
    lower_ramp_dctl = dcl.add_dctl_comp(
        comp=comp, dctl_path="TY_DCTL/draw_8bit_10bit_ramp.dctl",
        option={
            "sliderIntParam0": ppp.ramp_border_width,
            "sliderFloatParam0": ppp.ramp_height,  # ramp height
            "sliderFloatParam1": ppp.ramp_pos[0],  # ramp st_pos_h
            "sliderFloatParam2": ppp.ramp_pos[1] + ppp.ramp_height,  # ramp st_pos_v
            "sliderFloatParam3": 0,  # show top scale
            "sliderFloatParam4": 1,  # show bottom scale
            "checkBoxParam0": 1,  # 8-bit ramp
        },
        base_pos=[x_pos, y_pos]
    )
    dcl.connect_dctl(lower_ramp_dctl, upper_ramp_dctl)

    luminance_list = [0, 0.1, 1, 10, 100, 1000, 10000]
    st2084_cv_list = tf.oetf_from_luminance(np.array(luminance_list), tf.ST2084)
    ramp_width = ppp.ramp_width
    st_pos_h = ppp.ramp_pos[0]
    st_pos_v = ppp.ramp_text_pos_v

    bg_tool = lower_ramp_dctl
    for st2084_cv, luminance in zip(st2084_cv_list, luminance_list):
        x_pos += 1
        center_pos = [st_pos_h + ramp_width * st2084_cv, st_pos_v]
        ramp_text = create_ramp_info_text(
            comp=comp, ppp=ppp, center_pos=center_pos, luminance=luminance,
            base_pos=[x_pos, y_pos]
        )
        x_pos += 1
        text_merge = dcl.add_comp_tool(comp=comp, name="Merge", pos=[x_pos, y_pos])
        dcl.connect_merge_tool(merge_tool=text_merge, bg_tool=bg_tool, fg_tool=ramp_text)
        bg_tool = text_merge

    return bg_tool, x_pos


def create_adaptive_htr_tp_comp():
    tl_item_fusion_comp, comp = \
        dcl.append_fusion_composition_to_timeline(
            num_of_frame=1,
            pos_frame_idx=dcl.sec_to_frame_idx(60 * 60)
        )
    ppp = FusionParams()
    margin_between_modules = 2

    comp.Lock()

    x_pos = 1
    y_pos = 7  # y_pos is fixed this value
    pseudo_bg = dcl.add_transparent_background(comp=comp, pos=(x_pos, y_pos))
    bg_tool = pseudo_bg

    # base background
    x_pos += margin_between_modules
    # x_pos will be overwritten in the following function
    background, x_pos = create_background_comp(comp=comp, ppp=ppp, base_pos=(x_pos, y_pos-1))
    background_merge = dcl.add_comp_tool(comp=comp, name="Merge", pos=(x_pos, y_pos))
    dcl.connect_merge_tool(
        merge_tool=background_merge, bg_tool=bg_tool, fg_tool=background
    )
    bg_tool = background_merge
    
    # infomation
    x_pos += margin_between_modules
    info, x_pos = create_info_comp(comp=comp, ppp=ppp, base_pos=(x_pos, y_pos-1))
    info_merge = dcl.add_comp_tool(comp=comp, name="Merge", pos=(x_pos, y_pos))
    dcl.connect_merge_tool(
        merge_tool=info_merge, bg_tool=bg_tool, fg_tool=info
    )
    bg_tool = info_merge

    # border
    x_pos += margin_between_modules
    border_dctl = dcl.add_dctl_comp(
        comp=comp, dctl_path="TY_DCTL/draw_countdown_border.dctl",
        option={"sliderIntParam0": ppp.base_bg_border_width},
        base_pos=[x_pos, y_pos]
    )
    dcl.connect_dctl(border_dctl, bg_tool)
    bg_tool = border_dctl

    # scale
    x_pos += margin_between_modules
    scale, x_pos = create_scale_comp(comp=comp, ppp=ppp, base_pos=(x_pos, y_pos-1))
    x_pos += 2
    scale_merge = dcl.add_comp_tool(comp=comp, name="Merge", pos=(x_pos, y_pos))
    dcl.connect_merge_tool(
        merge_tool=scale_merge, bg_tool=bg_tool, fg_tool=scale
    )
    bg_tool = scale_merge

    # color checker
    x_pos += margin_between_modules
    color_checker, x_pos = create_color_checker_comp(
        comp=comp, ppp=ppp, base_pos=(x_pos, y_pos-1)
    )
    cc_merge = dcl.add_comp_tool(comp=comp, name="Merge", pos=(x_pos, y_pos))
    dcl.connect_merge_tool(
        merge_tool=cc_merge, bg_tool=bg_tool, fg_tool=color_checker
    )
    bg_tool = cc_merge

    # 8-bit / 10-bit ramp pattern
    x_pos += margin_between_modules
    ramp, x_pos = create_ramp_pattern_comp(
        comp=comp, ppp=ppp, base_pos=(x_pos, y_pos-1)
    )
    ramp_merge = dcl.add_comp_tool(comp=comp, name="Merge", pos=(x_pos, y_pos))
    dcl.connect_merge_tool(
        merge_tool=ramp_merge, bg_tool=bg_tool, fg_tool=ramp
    )
    bg_tool = ramp_merge

    # media out
    x_pos += margin_between_modules
    media_out = dcl.get_comp_tool_by_name(comp=comp, name="MediaOut1")
    dcl.set_tool_position(comp=comp, tool=media_out, pos=(x_pos, y_pos))

    dcl.connect_mediaout(source=bg_tool, mediaout=media_out)

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
    create_adaptive_htr_tp_comp()

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
