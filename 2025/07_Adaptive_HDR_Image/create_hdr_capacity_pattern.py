# -*- coding: utf-8 -*-

# import standard libraries
from pathlib import Path
import sys
import os
from pprint import pprint
import time

# import third-party libraries
import numpy as np
from colour.models import RGB_COLOURSPACE_BT2020

# import my libraries
import ty_davinci_constants as drc
import ty_davinci_control_lib_2 as dcl
import transfer_functions as tf
from test_pattern_generator2 import generate_color_checker_rgb_value, img_wirte_float_as_16bit_int

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


class Hd720pPixelBasedSize:
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
        val = self.to_even(int(verical_px)) / (720.0)
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

        self.info_area_height = Hd720pPixelBasedSize(38).v_size
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
        self.scale_h_margin = Hd720pPixelBasedSize(scale_h_margin_int).h_size
        self.scale_v_margin = self.info_area_height * 2
        scale_hh_int = self.to_even(1920 * scale_width_rate / self.scale_num_of_element)
        scale_vv_int = scale_hh_int * 0.8
        self.scale_hh = Hd720pPixelBasedSize(scale_hh_int).h_size
        self.scale_vv = Hd720pPixelBasedSize(scale_vv_int).v_size
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
        self.cc_width = Hd720pPixelBasedSize(cc_width_int).h_size
        self.cc_height = Hd720pPixelBasedSize(cc_height_int).v_size
        self.cc_margin_h = Hd720pPixelBasedSize(cc_margin_int).h_size
        self.cc_margin_v = Hd720pPixelBasedSize(cc_margin_int).v_size
        self.pp_hh = Hd720pPixelBasedSize(cc_patch_size_int).h_size
        self.pp_vv = Hd720pPixelBasedSize(cc_patch_size_int).v_size
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
        self.ramp_width = Hd720pPixelBasedSize(1024).h_size
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
    info_text_str += f"Color Gamut: {gamma}, Transfer Characteristics: {gamut}"
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


def create_capacity_text(comp, hdr_capacity, base_pos=[0, 0]):
    x_pos = base_pos[0]
    y_pos = base_pos[1]

    base_bg = dcl.add_comp_tool(
        comp=comp, name="Background", pos=(x_pos, y_pos)
    )
    fg_cv = np.round(tf.oetf_from_luminance(200, tf.ST2084)*1023)
    bg_cv = fg_cv - 32
    bg_luminance = tf.eotf_to_luminance(bg_cv/1023, tf.ST2084) / 100
    fg_luminance = tf.eotf_to_luminance(fg_cv/1023, tf.ST2084) / 100
    base_bg_input = {
        "TopLeftRed": bg_luminance,
        "TopLeftGreen": bg_luminance,
        "TopLeftBlue": bg_luminance,
        "TopLeftAlpha": 1.0,
    }
    dcl.set_multiple_tool_input(
        tool=base_bg, input_dict=base_bg_input
    )
    bg_tool = base_bg
    x_pos += 1

    if hdr_capacity is not None:
        sdr_luminance_list = [80, 100, 203]
        hdr_luminance_list = []
        for sdr_luminance in sdr_luminance_list:
            hdr_luminance = sdr_luminance * (2 ** hdr_capacity)
            hdr_luminance_list.append(hdr_luminance)

        info_text_str = f"HDR Capacity = {hdr_capacity:.3f}\n"
        for sdr_luminance, hdr_luminance in zip(sdr_luminance_list, hdr_luminance_list):
            suffix = "" if sdr_luminance == sdr_luminance_list[-1] else "\n"
            info_text_str += f"log2({hdr_luminance:.1f} nit / {sdr_luminance} nit){suffix}"
    else:
        info_text_str = "SDR Alternative Image"

    info_text = dcl.add_comp_tool(comp=comp, name="TextPlus", pos=(x_pos, y_pos-1))
    font_family = "Noto Sans"
    font_weight = "Bold"
    info_text_input = {
        "Center": {1: 0.595, 2: 0.5, 3: 0.0},
        "StyledText": info_text_str,
        "Font": font_family,
        "Style": font_weight,
        "Size": 0.105,
        "Red1": fg_luminance if hdr_capacity is not None else bg_luminance / 2.03,
        "Green1": fg_luminance if hdr_capacity is not None else 0.1,
        "Blue1": fg_luminance if hdr_capacity is not None else 0.1,
        "VerticalTopCenterBottom": 0.0,
        "HorizontalLeftCenterRight": 0.0,
        "AdvancedFontControls": 0.0,
        "LineSpacing": 1.85,
    }
    dcl.set_multiple_tool_input(tool=info_text, input_dict=info_text_input)

    output_merge = dcl.add_comp_tool(comp=comp, name="Merge", pos=(x_pos, y_pos))
    dcl.connect_merge_tool(
        merge_tool=output_merge, bg_tool=bg_tool, fg_tool=info_text
    )
    bg_tool = output_merge
    last_merge = bg_tool

    return last_merge, x_pos


def create_rectangles(comp, is_sdr, base_pos=[0, 0]):
    x_pos = base_pos[0]
    y_pos = base_pos[1]

    base_bg = dcl.add_transparent_background(comp=comp, pos=(x_pos, y_pos))

    bg_tool = base_bg
    fg_tool = None

    if is_sdr:
        rgba_color_list = [
            [1.0, 1.0, 0.1, 1.0],
            [0.1, 1.0, 1.0, 1.0],
            [0.1, 1.0, 0.1, 1.0],
            [1.0, 0.1, 1.0, 1.0],
            [1.0, 0.1, 0.1, 1.0],
            [0.1, 0.1, 1.0, 1.0],
        ]
    else:
        rgba_color_list = [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]

    num_of_color = len(rgba_color_list)

    rect_size = 120
    rect_size_h = Hd720pPixelBasedSize(rect_size).h_size
    rect_size_v = Hd720pPixelBasedSize(rect_size).v_size
    rect_margin_v = 0
    rect_base_pos = [
        Hd720pPixelBasedSize(rect_size//2).h_size,
        1 - Hd720pPixelBasedSize(rect_size//2).v_size
    ]

    for c_idx in range(num_of_color):
        x_pos += 1
        left_rect_pos = [
            rect_base_pos[0],
            rect_base_pos[1] - (rect_size_v + rect_margin_v) * c_idx,
        ]
        right_rect_pos = [
            rect_base_pos[0] + rect_size_h,
            left_rect_pos[1],
        ]

        left_rectangle = dcl.add_rectangle_comp(
            comp=comp,
            rgba_color=rgba_color_list[c_idx],
            center=left_rect_pos,
            width=rect_size_h,
            height=rect_size_v,
            base_pos=[x_pos, y_pos-1]
        )
        left_merge = dcl.add_comp_tool(comp=comp, name="Merge", pos=(x_pos, y_pos))
        fg_tool = left_rectangle
        dcl.connect_merge_tool(merge_tool=left_merge, bg_tool=bg_tool, fg_tool=fg_tool)
        bg_tool = left_merge

        x_pos += 1
        right_rectangle = dcl.add_rectangle_comp(
            comp=comp,
            rgba_color=[1, 1, 1, 1],
            center=right_rect_pos,
            width=rect_size_h,
            height=rect_size_v,
            base_pos=[x_pos, y_pos-1]
        )
        right_merge = dcl.add_comp_tool(comp=comp, name="Merge", pos=(x_pos, y_pos))
        fg_tool = right_rectangle
        dcl.connect_merge_tool(merge_tool=right_merge, bg_tool=bg_tool, fg_tool=fg_tool)
        bg_tool = right_merge

    last_merge = bg_tool

    return last_merge, x_pos


def create_htr_capacity_tp_comp(hdr_capacity=2.3, is_sdr=False):
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
    
    # capacity text
    x_pos += margin_between_modules
    hdr_capacity_text, x_pos = create_capacity_text(
        comp=comp, hdr_capacity=hdr_capacity, base_pos=(x_pos, y_pos-1)
    )
    capacity_text_merge = dcl.add_comp_tool(comp=comp, name="Merge", pos=(x_pos, y_pos))
    dcl.connect_merge_tool(
        merge_tool=capacity_text_merge, bg_tool=bg_tool, fg_tool=hdr_capacity_text
    )
    bg_tool = capacity_text_merge

    # rectangles
    x_pos += margin_between_modules
    rectangles, x_pos = create_rectangles(comp=comp, is_sdr=is_sdr, base_pos=(x_pos, y_pos-1))
    rectangle_merge = dcl.add_comp_tool(comp=comp, name="Merge", pos=(x_pos, y_pos))
    dcl.connect_merge_tool(
        merge_tool=rectangle_merge, bg_tool=bg_tool, fg_tool=rectangles
    )
    bg_tool = rectangle_merge

    # half gain for sdr
    if is_sdr:
        x_pos += margin_between_modules
        color_gain = dcl.add_comp_tool(comp=comp, name="ColorGain", pos=(x_pos, y_pos))
        color_gain_input = {
			"LockRGB": 1,
			"GainRed": 100/203,
        }
        dcl.set_multiple_tool_input(tool=color_gain, input_dict=color_gain_input)
        dcl.connect_tool(bg_tool, color_gain)
        bg_tool = color_gain

    # media out
    x_pos += margin_between_modules
    media_out = dcl.get_comp_tool_by_name(comp=comp, name="MediaOut1")
    dcl.set_tool_position(comp=comp, tool=media_out, pos=(x_pos, y_pos))

    dcl.connect_mediaout(source=bg_tool, mediaout=media_out)

    comp.Unlock()


def create_hdr_capacity_tp(
        width, height, framerate, gamut, gamma, hdr_capacity=2.3
):
    ##################
    # Project Settings
    ##################
    dcl.refresh_lut_list()

    project_name = f"HDR_Capacity_TP_Rev{REVISION:02d}"
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
    if gamma != drc.PRJ_GAMMA_STR_SRGB:
        create_htr_capacity_tp_comp(hdr_capacity=hdr_capacity)
    else:
        create_htr_capacity_tp_comp(hdr_capacity=hdr_capacity, is_sdr=True)
    dcl.force_rcm_update_via_page_switch()

    dcl.set_current_timecode(timecode=start_time_code)


    ###################
    # Encode
    ###################
    preset_path = None

    if preset_path is None:
        format_extension = drc.OUT_FILE_EXTENSTION_PNG
        codec = drc.CODEC_PNG_RGB_16_BITS

        if sys.platform == "darwin":  # macOS
            dir_path = Path("/Volumes/My Passport/Countdown/temp_seq")
        elif sys.platform == "win32":  # Windows
            dir_path = Path(r"C:\Users\toruv\OneDrive\work\sample_code\2025\07_Adaptive_HDR_Image\src_img")
        else:
            pass

        if hdr_capacity is not None:
            basename = f"HDR_Capacity_{hdr_capacity:.3f}_{width}x{height}"
        else:
            basename = f"HDR_Capacity_SDR_{width}x{height}"

        output_fname = str(dir_path / basename)
        if format_extension in drc.STILL_SEQ_FILE_EXTENTION_LIST:
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
    
    output_full_fname_1 = str(dir_path / basename) + "00086400" + "." + format_extension
    output_full_fname_2 = str(dir_path / basename) + "_00086400" + "." + format_extension
    if os.path.exists(output_full_fname_1):
        output_full_fname = output_full_fname_1
    elif os.path.exists(output_full_fname_2):
        output_full_fname = output_full_fname_2
    else:
        ValueError("file not found")
    output_new_full_fname = str(dir_path / basename) + "." + format_extension
    if os.path.exists(output_new_full_fname):
        os.remove(output_new_full_fname)
    os.rename(output_full_fname, output_new_full_fname)


def main():
    # debug_resolve()
    # debug_fusion()

    # hdr_capacity_list = [
    #     0.000, 0.563, 0.978, 1.300, 1.563, 1.978, 2.300, 2.563, 2.885, 3.300, 3.622, 3.885, 4.300, 5.622, 6.965
    # ]
    hdr_capacity_list = [
        2.300
    ]

    create_hdr_capacity_tp(
        width=1280, height=720, framerate=24,
        gamut=drc.PRJ_COLOR_SPACE_REC2020, gamma=drc.PRJ_GAMMA_STR_SRGB,
        hdr_capacity=None
    )

    for hdr_capacity in hdr_capacity_list:
        create_hdr_capacity_tp(
            width=1280, height=720, framerate=24,
            gamut=drc.PRJ_COLOR_SPACE_REC2020, gamma=drc.PRJ_GAMMA_STR_ST2084,
            hdr_capacity=hdr_capacity
        )


def make_sdr_image():
    rng = np.random.default_rng(1341293)
    mono_img = rng.random(size=(720, 1280))
    img = np.dstack((mono_img, mono_img, mono_img))
    img_wirte_float_as_16bit_int("./src_img/HDR_Capacity_SDR_1280x720.png", img)


def make_hdr_capacity_list():
    base_luminance = 203
    target_lumiannce_list = [
        203, 300, 400, 500, 600, 800, 1000, 1200, 1500, 2000, 2500, 3000, 4000, 10000
    ]
    hdr_capacity_list = [
        np.log2(x/base_luminance) for x in target_lumiannce_list
    ]
    for hdr_capacity in hdr_capacity_list:
        print(f"{hdr_capacity:.3f} ", end="")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
    # make_sdr_image()
    # make_hdr_capacity_list()
