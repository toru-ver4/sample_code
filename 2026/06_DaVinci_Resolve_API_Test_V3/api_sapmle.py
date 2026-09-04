#!/usr/bin/env python
# import standard libraries
import os
import time
from pathlib import Path
import pprint

# import my libraries
import ty_davinci_resolve as tdr

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2026 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def close_and_delete_project_if_exists(session, project_name):
    """Close the named project if loaded, then delete it if it exists.

    Parameters
    ----------
    session
        Connected DaVinci Resolve session.
    project_name
        Name of the project to delete.

    Returns
    -------
    None

    Examples
    --------
    >>> session = tdr.ResolveSession.connect()  # doctest: +SKIP
    >>> close_and_delete_project_if_exists(  # doctest: +SKIP
    ...     session, "sample_project"
    ... )
    """
    current_project = session.project_manager.GetCurrentProject()
    if (
        current_project is not None
        and current_project.GetName() == project_name
    ):
        tdr.close_project(session, project=current_project)

    if project_name in tdr.list_projects(session):
        tdr.delete_project(session=session, name=project_name)


def create_project_sample():
    project_name = "sample_create_project"
    session = tdr.ResolveSession.connect()
    close_and_delete_project_if_exists(session, project_name)
    project = tdr.create_project(session, name=project_name)
    time.sleep(2)
    tdr.close_project(session)


def get_project_settings_sample():
    project_name = "sample_get_project_settings"
    session = tdr.ResolveSession.connect()
    close_and_delete_project_if_exists(session, project_name)
    project = tdr.create_project(session, name=project_name)
    pprint.pprint(project.GetSetting())


def get_current_timeline_settings_sample():
    session = tdr.ResolveSession.connect()
    project = tdr.get_current_project(session=session)
    timeline = tdr.get_timeline(project=project, index=1)
    pprint.pprint(timeline.GetSetting())


def project_settings_sample():
    project_name = "sample_create_project"
    session = tdr.ResolveSession.connect()
    close_and_delete_project_if_exists(session, project_name)
    project = tdr.create_project(session, name=project_name)

    project_settings_params2 = {
        tdr.ProjectSetting.TIMELINE_RESOLUTION_WIDTH: "3840",
        tdr.ProjectSetting.TIMELINE_RESOLUTION_HEIGHT: "2160",
        tdr.ProjectSetting.TIMELINE_FRAME_RATE: tdr.FrameRate.FPS_23_976,
        tdr.ProjectSetting.VIDEO_MONITOR_FORMAT: tdr.make_video_monitor_format(3840, 2160, 23.976),
        tdr.ProjectSetting.VIDEO_MONITOR_USE_444_SDI: tdr.SettingToggle.ENABLED,
        tdr.ProjectSetting.VIDEO_MONITOR_SDI_CONFIGURATION: tdr.SDIConfiguration.SINGLE_LINK,
        tdr.ProjectSetting.VIDEO_DATA_LEVELS: tdr.VideoDataLevel.FULL,
        tdr.ProjectSetting.VIDEO_MONITOR_USE_HDR_OVER_HDMI: tdr.SettingToggle.ENABLED,

        tdr.ProjectSetting.COLOR_SCIENCE_MODE: tdr.ColorScienceMode.DAVINCI_YRGB_COLOR_MANAGED,
        tdr.ProjectSetting.AUTO_COLOR_MANAGEMENT: tdr.SettingToggle.DISABLED,
        tdr.ProjectSetting.RCM_PRESET_MODE: tdr.ProjectPresetMode.CUSTOM,
        tdr.ProjectSetting.SEPARATE_COLOR_SPACE_AND_GAMMA: tdr.SettingToggle.ENABLED,
        tdr.ProjectSetting.COLOR_SPACE_INPUT: tdr.ColorSpace.REC_2020,
        tdr.ProjectSetting.COLOR_SPACE_INPUT_GAMMA: tdr.Gamma.ST2084,
        tdr.ProjectSetting.COLOR_SPACE_TIMELINE: tdr.ColorSpace.P3_D65,
        tdr.ProjectSetting.COLOR_SPACE_TIMELINE_GAMMA: tdr.Gamma.ST2084,
        tdr.ProjectSetting.COLOR_SPACE_OUTPUT: tdr.ColorSpace.P3_D65,
        tdr.ProjectSetting.COLOR_SPACE_OUTPUT_GAMMA: tdr.Gamma.ST2084,
        tdr.ProjectSetting.TIMELINE_WORKING_LUMINANCE_MODE: tdr.ProjectPresetMode.CUSTOM,
        tdr.ProjectSetting.TIMELINE_WORKING_LUMINANCE: "10000",
        tdr.ProjectSetting.INPUT_DRT: tdr.DynamicRangeTransform.NONE,
        tdr.ProjectSetting.OUTPUT_DRT: tdr.DynamicRangeTransform.NONE,
        tdr.ProjectSetting.GRAPHICS_WHITE_LEVEL: "100",
        tdr.ProjectSetting.HDR_MASTERING_LUMINANCE_MAX: "10000",
        tdr.ProjectSetting.HDR_MASTERING_ON: tdr.SettingToggle.ENABLED,
    }

    tdr.set_settings(
        project,
        settings=project_settings_params2
    )


def timeline_settings_sample():
    project_name = "sample_timeline_settings"
    session = tdr.ResolveSession.connect()
    close_and_delete_project_if_exists(session, project_name)
    project = tdr.create_project(session, name=project_name)

    project_settings_params2 = {
        tdr.ProjectSetting.TIMELINE_RESOLUTION_WIDTH: "3840",
        tdr.ProjectSetting.TIMELINE_RESOLUTION_HEIGHT: "2160",
        tdr.ProjectSetting.TIMELINE_FRAME_RATE: tdr.FrameRate.FPS_59_94,
        tdr.ProjectSetting.VIDEO_MONITOR_FORMAT: tdr.make_video_monitor_format(3840, 2160, 59.94),
        tdr.ProjectSetting.VIDEO_MONITOR_USE_444_SDI: tdr.SettingToggle.DISABLED,
        tdr.ProjectSetting.VIDEO_MONITOR_SDI_CONFIGURATION: tdr.SDIConfiguration.SINGLE_LINK,
        tdr.ProjectSetting.VIDEO_DATA_LEVELS: tdr.VideoDataLevel.FULL,
        tdr.ProjectSetting.VIDEO_MONITOR_USE_HDR_OVER_HDMI: tdr.SettingToggle.ENABLED,

        tdr.ProjectSetting.COLOR_SCIENCE_MODE: tdr.ColorScienceMode.DAVINCI_YRGB_COLOR_MANAGED,
        tdr.ProjectSetting.AUTO_COLOR_MANAGEMENT: tdr.SettingToggle.DISABLED,
        tdr.ProjectSetting.RCM_PRESET_MODE: tdr.ProjectPresetMode.CUSTOM,
        tdr.ProjectSetting.SEPARATE_COLOR_SPACE_AND_GAMMA: tdr.SettingToggle.ENABLED,
        tdr.ProjectSetting.COLOR_SPACE_INPUT: tdr.ColorSpace.REC_2020,
        tdr.ProjectSetting.COLOR_SPACE_INPUT_GAMMA: tdr.Gamma.ST2084,
        tdr.ProjectSetting.COLOR_SPACE_TIMELINE: tdr.ColorSpace.P3_D65,
        tdr.ProjectSetting.COLOR_SPACE_TIMELINE_GAMMA: tdr.Gamma.ST2084,
        tdr.ProjectSetting.COLOR_SPACE_OUTPUT: tdr.ColorSpace.P3_D65,
        tdr.ProjectSetting.COLOR_SPACE_OUTPUT_GAMMA: tdr.Gamma.ST2084,
        tdr.ProjectSetting.TIMELINE_WORKING_LUMINANCE_MODE: tdr.ProjectPresetMode.CUSTOM,
        tdr.ProjectSetting.TIMELINE_WORKING_LUMINANCE: "10000",
        tdr.ProjectSetting.INPUT_DRT: tdr.DynamicRangeTransform.NONE,
        tdr.ProjectSetting.OUTPUT_DRT: tdr.DynamicRangeTransform.NONE,
        tdr.ProjectSetting.GRAPHICS_WHITE_LEVEL: "100",
        tdr.ProjectSetting.HDR_MASTERING_LUMINANCE_MAX: "10000",
        tdr.ProjectSetting.HDR_MASTERING_ON: tdr.SettingToggle.ENABLED,
    }

    timeline_settings_param = {
        tdr.ProjectSetting.TIMELINE_RESOLUTION_WIDTH: "1920",
        tdr.ProjectSetting.TIMELINE_RESOLUTION_HEIGHT: "1080",

        ##########################################################################
        # DO NOT SET TIMELINE FRAME RATE IN THE **TIMELINE SETTINGS**.
        # INSTEAD, PLEASE SET THIS VALUE IN THE **PROJECT SETTINGS**.
        # ------------------------------------------------------------------------
        # tdr.ProjectSetting.TIMELINE_FRAME_RATE: tdr.FrameRate.FPS_59_94,
        ##########################################################################

        tdr.ProjectSetting.VIDEO_MONITOR_FORMAT: tdr.make_video_monitor_format(1920, 1080, 59.94),
        tdr.ProjectSetting.VIDEO_MONITOR_USE_444_SDI: tdr.SettingToggle.ENABLED,
        tdr.ProjectSetting.VIDEO_MONITOR_SDI_CONFIGURATION: tdr.SDIConfiguration.SINGLE_LINK,
        tdr.ProjectSetting.VIDEO_DATA_LEVELS: tdr.VideoDataLevel.VIDEO,
        tdr.ProjectSetting.VIDEO_MONITOR_USE_HDR_OVER_HDMI: tdr.SettingToggle.ENABLED,

        tdr.ProjectSetting.COLOR_SCIENCE_MODE: tdr.ColorScienceMode.DAVINCI_YRGB_COLOR_MANAGED,
        tdr.ProjectSetting.AUTO_COLOR_MANAGEMENT: tdr.SettingToggle.DISABLED,
        tdr.ProjectSetting.RCM_PRESET_MODE: tdr.ProjectPresetMode.CUSTOM,
        tdr.ProjectSetting.SEPARATE_COLOR_SPACE_AND_GAMMA: tdr.SettingToggle.ENABLED,
        # tdr.ProjectSetting.COLOR_SPACE_INPUT: tdr.ColorSpace.REC_709,  # DO NOT SET
        # tdr.ProjectSetting.COLOR_SPACE_INPUT_GAMMA: tdr.Gamma.GAMMA_2_4,  # DO NOT SET
        tdr.ProjectSetting.COLOR_SPACE_TIMELINE: tdr.ColorSpace.DAVINCI_WG,
        tdr.ProjectSetting.COLOR_SPACE_TIMELINE_GAMMA: tdr.Gamma.DAVINCI_INTERMEDIATE,
        tdr.ProjectSetting.COLOR_SPACE_OUTPUT: tdr.ColorSpace.REC_709,
        tdr.ProjectSetting.COLOR_SPACE_OUTPUT_GAMMA: tdr.Gamma.GAMMA_2_4,
        tdr.ProjectSetting.TIMELINE_WORKING_LUMINANCE_MODE: tdr.ProjectPresetMode.CUSTOM,
        tdr.ProjectSetting.TIMELINE_WORKING_LUMINANCE: "10000",
        tdr.ProjectSetting.INPUT_DRT: tdr.DynamicRangeTransform.NONE,
        tdr.ProjectSetting.OUTPUT_DRT: tdr.DynamicRangeTransform.NONE,
        tdr.ProjectSetting.GRAPHICS_WHITE_LEVEL: "100",
        tdr.ProjectSetting.HDR_MASTERING_LUMINANCE_MAX: "10000",
        tdr.ProjectSetting.HDR_MASTERING_ON: tdr.SettingToggle.ENABLED,
    }

    tdr.set_settings(
        project=project,
        settings=project_settings_params2
    )
    media_pool = tdr.get_media_pool(project=project)
    timeline = tdr.create_empty_timeline(media_pool=media_pool, name="Test3")
    tdr.set_timeline_settings(timeline=timeline, settings=timeline_settings_param)


def encode_test():
    project_name = "sample_encode"
    session = tdr.ResolveSession.connect()
    close_and_delete_project_if_exists(session, project_name)
    project = tdr.create_project(session, name=project_name)

    project_settings_params2 = {
        tdr.ProjectSetting.TIMELINE_RESOLUTION_WIDTH: "3840",
        tdr.ProjectSetting.TIMELINE_RESOLUTION_HEIGHT: "2160",
        tdr.ProjectSetting.TIMELINE_FRAME_RATE: tdr.FrameRate.FPS_59_94,
        tdr.ProjectSetting.VIDEO_MONITOR_FORMAT: tdr.make_video_monitor_format(3840, 2160, 59.94),
        tdr.ProjectSetting.VIDEO_MONITOR_USE_444_SDI: tdr.SettingToggle.DISABLED,
        tdr.ProjectSetting.VIDEO_MONITOR_SDI_CONFIGURATION: tdr.SDIConfiguration.SINGLE_LINK,
        tdr.ProjectSetting.VIDEO_DATA_LEVELS: tdr.VideoDataLevel.FULL,
        tdr.ProjectSetting.VIDEO_MONITOR_USE_HDR_OVER_HDMI: tdr.SettingToggle.ENABLED,

        tdr.ProjectSetting.COLOR_SCIENCE_MODE: tdr.ColorScienceMode.DAVINCI_YRGB_COLOR_MANAGED,
        tdr.ProjectSetting.AUTO_COLOR_MANAGEMENT: tdr.SettingToggle.DISABLED,
        tdr.ProjectSetting.RCM_PRESET_MODE: tdr.ProjectPresetMode.CUSTOM,
        tdr.ProjectSetting.SEPARATE_COLOR_SPACE_AND_GAMMA: tdr.SettingToggle.ENABLED,
        tdr.ProjectSetting.COLOR_SPACE_INPUT: tdr.ColorSpace.REC_2020,
        tdr.ProjectSetting.COLOR_SPACE_INPUT_GAMMA: tdr.Gamma.ST2084,
        tdr.ProjectSetting.COLOR_SPACE_TIMELINE: tdr.ColorSpace.P3_D65,
        tdr.ProjectSetting.COLOR_SPACE_TIMELINE_GAMMA: tdr.Gamma.ST2084,
        tdr.ProjectSetting.COLOR_SPACE_OUTPUT: tdr.ColorSpace.P3_D65,
        tdr.ProjectSetting.COLOR_SPACE_OUTPUT_GAMMA: tdr.Gamma.ST2084,
        tdr.ProjectSetting.TIMELINE_WORKING_LUMINANCE_MODE: tdr.ProjectPresetMode.CUSTOM,
        tdr.ProjectSetting.TIMELINE_WORKING_LUMINANCE: "10000",
        tdr.ProjectSetting.INPUT_DRT: tdr.DynamicRangeTransform.NONE,
        tdr.ProjectSetting.OUTPUT_DRT: tdr.DynamicRangeTransform.NONE,
        tdr.ProjectSetting.GRAPHICS_WHITE_LEVEL: "100",
        tdr.ProjectSetting.HDR_MASTERING_LUMINANCE_MAX: "10000",
        tdr.ProjectSetting.HDR_MASTERING_ON: tdr.SettingToggle.ENABLED,
    }

    timeline_settings_param = {
        tdr.ProjectSetting.TIMELINE_RESOLUTION_WIDTH: "1920",
        tdr.ProjectSetting.TIMELINE_RESOLUTION_HEIGHT: "1080",

        ##########################################################################
        # DO NOT SET TIMELINE FRAME RATE IN THE **TIMELINE SETTINGS**.
        # INSTEAD, PLEASE SET THIS VALUE IN THE **PROJECT SETTINGS**.
        # ------------------------------------------------------------------------
        # tdr.ProjectSetting.TIMELINE_FRAME_RATE: tdr.FrameRate.FPS_59_94,
        ##########################################################################

        tdr.ProjectSetting.VIDEO_MONITOR_FORMAT: tdr.make_video_monitor_format(1920, 1080, 59.94),
        tdr.ProjectSetting.VIDEO_MONITOR_USE_444_SDI: tdr.SettingToggle.ENABLED,
        tdr.ProjectSetting.VIDEO_MONITOR_SDI_CONFIGURATION: tdr.SDIConfiguration.SINGLE_LINK,
        tdr.ProjectSetting.VIDEO_DATA_LEVELS: tdr.VideoDataLevel.VIDEO,
        tdr.ProjectSetting.VIDEO_MONITOR_USE_HDR_OVER_HDMI: tdr.SettingToggle.ENABLED,

        tdr.ProjectSetting.COLOR_SCIENCE_MODE: tdr.ColorScienceMode.DAVINCI_YRGB_COLOR_MANAGED,
        tdr.ProjectSetting.AUTO_COLOR_MANAGEMENT: tdr.SettingToggle.DISABLED,
        tdr.ProjectSetting.RCM_PRESET_MODE: tdr.ProjectPresetMode.CUSTOM,
        tdr.ProjectSetting.SEPARATE_COLOR_SPACE_AND_GAMMA: tdr.SettingToggle.ENABLED,
        # tdr.ProjectSetting.COLOR_SPACE_INPUT: tdr.ColorSpace.REC_709,  # DO NOT SET
        # tdr.ProjectSetting.COLOR_SPACE_INPUT_GAMMA: tdr.Gamma.GAMMA_2_4,  # DO NOT SET
        tdr.ProjectSetting.COLOR_SPACE_TIMELINE: tdr.ColorSpace.DAVINCI_WG,
        tdr.ProjectSetting.COLOR_SPACE_TIMELINE_GAMMA: tdr.Gamma.DAVINCI_INTERMEDIATE,
        tdr.ProjectSetting.COLOR_SPACE_OUTPUT: tdr.ColorSpace.REC_709,
        tdr.ProjectSetting.COLOR_SPACE_OUTPUT_GAMMA: tdr.Gamma.GAMMA_2_4,
        tdr.ProjectSetting.TIMELINE_WORKING_LUMINANCE_MODE: tdr.ProjectPresetMode.CUSTOM,
        tdr.ProjectSetting.TIMELINE_WORKING_LUMINANCE: "10000",
        tdr.ProjectSetting.INPUT_DRT: tdr.DynamicRangeTransform.NONE,
        tdr.ProjectSetting.OUTPUT_DRT: tdr.DynamicRangeTransform.NONE,
        tdr.ProjectSetting.GRAPHICS_WHITE_LEVEL: "100",
        tdr.ProjectSetting.HDR_MASTERING_LUMINANCE_MAX: "10000",
        tdr.ProjectSetting.HDR_MASTERING_ON: tdr.SettingToggle.ENABLED,
    }

    tdr.set_settings(
        project=project,
        settings=project_settings_params2
    )
    media_pool = tdr.get_media_pool(project=project)
    timeline = tdr.create_empty_timeline(media_pool=media_pool, name="Test3")
    tdr.set_timeline_settings(timeline=timeline, settings=timeline_settings_param)

    fps_int = 60
    fusion_frame_sec = 5
    fusion_frames = fps_int * fusion_frame_sec
    
    fusion_item, comp = tdr.append_fusion_composition(
        timeline=timeline,
        duration_frames=fusion_frames,
        record_frame=tdr.timecode_to_frames("01:00:00:00", fps_int),
        media_pool=media_pool
    )

    rectangle_mask = tdr.add_tool(
        comp=comp,
        tool_type=tdr.FusionTool.RECTANGLE_MASK,
        position=(2, 0)
    )
    rectangle_mask.Center = {1: 0.5, 2: 0.5, 3: 0.0}
    rectangle_mask.Width = 0.0
    rectangle_mask.Height = 0.5

    rectangle_bg = tdr.add_tool(
        comp=comp,
        tool_type=tdr.FusionTool.BACKGROUND,
        position=(2, 1)
    )
    rectangle_bg.TopLeftRed = 1.0
    rectangle_bg.TopLeftGreen = 1.0
    rectangle_bg.TopLeftBlue = 1.0
    rectangle_bg.TopLeftAlpha = 1.0
    rectangle_bg.EffectMask = rectangle_mask

    rectangle_mask.Width = comp.BezierSpline()
    rectangle_mask.Width[0] = 0.0
    rectangle_mask.Width[fusion_frames] = 1.0

    media_out = tdr.get_tool(comp=comp, name="MediaOut1")
    tdr.connect_default_output(source=rectangle_bg, target=media_out)

    tdr.open_page(session=session, page=tdr.Page.DELIVER)

    render_format = tdr.RenderFormat.QUICKTIME
    codec = tdr.VideoCodec.PRORES_422_HQ
    tdr.set_render_format_codec(
        project=project,
        render_format=render_format,
        codec=codec
    )
    tdr.set_render_settings(
        project=project,
        settings={
            tdr.RenderSetting.TARGET_DIR: str(Path.home() / "Downloads"),
            tdr.RenderSetting.CUSTOM_NAME: "Encode_Test_ProRes422HQ.mov",
            tdr.RenderSetting.EXPORT_AUDIO: False
        }
    )

    tdr.render_current_settings(project=project)


def fusion_key_frame_test():
    project_name = "fusion_key_frame"
    session = tdr.ResolveSession.connect()
    close_and_delete_project_if_exists(session, project_name)
    project = tdr.create_project(session, name=project_name)

    project_settings_params2 = {
        tdr.ProjectSetting.TIMELINE_RESOLUTION_WIDTH: "3840",
        tdr.ProjectSetting.TIMELINE_RESOLUTION_HEIGHT: "2160",
        tdr.ProjectSetting.TIMELINE_FRAME_RATE: tdr.FrameRate.FPS_59_94,
        tdr.ProjectSetting.VIDEO_MONITOR_FORMAT: tdr.make_video_monitor_format(3840, 2160, 59.94),
        tdr.ProjectSetting.VIDEO_MONITOR_USE_444_SDI: tdr.SettingToggle.DISABLED,
        tdr.ProjectSetting.VIDEO_MONITOR_SDI_CONFIGURATION: tdr.SDIConfiguration.SINGLE_LINK,
        tdr.ProjectSetting.VIDEO_DATA_LEVELS: tdr.VideoDataLevel.FULL,
        tdr.ProjectSetting.VIDEO_MONITOR_USE_HDR_OVER_HDMI: tdr.SettingToggle.ENABLED,

        tdr.ProjectSetting.COLOR_SCIENCE_MODE: tdr.ColorScienceMode.DAVINCI_YRGB_COLOR_MANAGED,
        tdr.ProjectSetting.AUTO_COLOR_MANAGEMENT: tdr.SettingToggle.DISABLED,
        tdr.ProjectSetting.RCM_PRESET_MODE: tdr.ProjectPresetMode.CUSTOM,
        tdr.ProjectSetting.SEPARATE_COLOR_SPACE_AND_GAMMA: tdr.SettingToggle.ENABLED,
        tdr.ProjectSetting.COLOR_SPACE_INPUT: tdr.ColorSpace.REC_2020,
        tdr.ProjectSetting.COLOR_SPACE_INPUT_GAMMA: tdr.Gamma.ST2084,
        tdr.ProjectSetting.COLOR_SPACE_TIMELINE: tdr.ColorSpace.P3_D65,
        tdr.ProjectSetting.COLOR_SPACE_TIMELINE_GAMMA: tdr.Gamma.ST2084,
        tdr.ProjectSetting.COLOR_SPACE_OUTPUT: tdr.ColorSpace.P3_D65,
        tdr.ProjectSetting.COLOR_SPACE_OUTPUT_GAMMA: tdr.Gamma.ST2084,
        tdr.ProjectSetting.TIMELINE_WORKING_LUMINANCE_MODE: tdr.ProjectPresetMode.CUSTOM,
        tdr.ProjectSetting.TIMELINE_WORKING_LUMINANCE: "10000",
        tdr.ProjectSetting.INPUT_DRT: tdr.DynamicRangeTransform.NONE,
        tdr.ProjectSetting.OUTPUT_DRT: tdr.DynamicRangeTransform.NONE,
        tdr.ProjectSetting.GRAPHICS_WHITE_LEVEL: "100",
        tdr.ProjectSetting.HDR_MASTERING_LUMINANCE_MAX: "10000",
        tdr.ProjectSetting.HDR_MASTERING_ON: tdr.SettingToggle.ENABLED,
    }

    timeline_settings_param = {
        tdr.ProjectSetting.TIMELINE_RESOLUTION_WIDTH: "1920",
        tdr.ProjectSetting.TIMELINE_RESOLUTION_HEIGHT: "1080",

        ##########################################################################
        # DO NOT SET TIMELINE FRAME RATE IN THE **TIMELINE SETTINGS**.
        # INSTEAD, PLEASE SET THIS VALUE IN THE **PROJECT SETTINGS**.
        # ------------------------------------------------------------------------
        # tdr.ProjectSetting.TIMELINE_FRAME_RATE: tdr.FrameRate.FPS_59_94,
        ##########################################################################

        tdr.ProjectSetting.VIDEO_MONITOR_FORMAT: tdr.make_video_monitor_format(1920, 1080, 59.94),
        tdr.ProjectSetting.VIDEO_MONITOR_USE_444_SDI: tdr.SettingToggle.ENABLED,
        tdr.ProjectSetting.VIDEO_MONITOR_SDI_CONFIGURATION: tdr.SDIConfiguration.SINGLE_LINK,
        tdr.ProjectSetting.VIDEO_DATA_LEVELS: tdr.VideoDataLevel.VIDEO,
        tdr.ProjectSetting.VIDEO_MONITOR_USE_HDR_OVER_HDMI: tdr.SettingToggle.ENABLED,

        tdr.ProjectSetting.COLOR_SCIENCE_MODE: tdr.ColorScienceMode.DAVINCI_YRGB_COLOR_MANAGED,
        tdr.ProjectSetting.AUTO_COLOR_MANAGEMENT: tdr.SettingToggle.DISABLED,
        tdr.ProjectSetting.RCM_PRESET_MODE: tdr.ProjectPresetMode.CUSTOM,
        tdr.ProjectSetting.SEPARATE_COLOR_SPACE_AND_GAMMA: tdr.SettingToggle.ENABLED,
        # tdr.ProjectSetting.COLOR_SPACE_INPUT: tdr.ColorSpace.REC_709,  # DO NOT SET
        # tdr.ProjectSetting.COLOR_SPACE_INPUT_GAMMA: tdr.Gamma.GAMMA_2_4,  # DO NOT SET
        tdr.ProjectSetting.COLOR_SPACE_TIMELINE: tdr.ColorSpace.DAVINCI_WG,
        tdr.ProjectSetting.COLOR_SPACE_TIMELINE_GAMMA: tdr.Gamma.DAVINCI_INTERMEDIATE,
        tdr.ProjectSetting.COLOR_SPACE_OUTPUT: tdr.ColorSpace.REC_709,
        tdr.ProjectSetting.COLOR_SPACE_OUTPUT_GAMMA: tdr.Gamma.GAMMA_2_4,
        tdr.ProjectSetting.TIMELINE_WORKING_LUMINANCE_MODE: tdr.ProjectPresetMode.CUSTOM,
        tdr.ProjectSetting.TIMELINE_WORKING_LUMINANCE: "10000",
        tdr.ProjectSetting.INPUT_DRT: tdr.DynamicRangeTransform.NONE,
        tdr.ProjectSetting.OUTPUT_DRT: tdr.DynamicRangeTransform.NONE,
        tdr.ProjectSetting.GRAPHICS_WHITE_LEVEL: "100",
        tdr.ProjectSetting.HDR_MASTERING_LUMINANCE_MAX: "10000",
        tdr.ProjectSetting.HDR_MASTERING_ON: tdr.SettingToggle.ENABLED,
    }

    tdr.set_settings(
        project=project,
        settings=project_settings_params2
    )
    media_pool = tdr.get_media_pool(project=project)
    timeline = tdr.create_empty_timeline(media_pool=media_pool, name="Test3")
    tdr.set_timeline_settings(timeline=timeline, settings=timeline_settings_param)

    fps_int = 60
    fusion_frame_sec = 2
    fusion_frames = fps_int * fusion_frame_sec
    
    fusion_item, comp = tdr.append_fusion_composition(
        timeline=timeline,
        duration_frames=fusion_frames,
        record_frame=tdr.timecode_to_frames("01:00:00:00", fps_int),
        media_pool=media_pool
    )

    tdr.open_page(session=session, page=tdr.Page.FUSION)

    rectangle_mask = tdr.add_tool(
        comp=comp,
        tool_type=tdr.FusionTool.RECTANGLE_MASK,
        position=(2, 0)
    )
    rectangle_mask.Center = {1: 0.5, 2: 0.5, 3: 0.0}
    rectangle_mask.Width = 0.0
    rectangle_mask.Height = 1.0

    rectangle_bg = tdr.add_tool(
        comp=comp,
        tool_type=tdr.FusionTool.BACKGROUND,
        position=(2, 1)
    )
    rectangle_bg.TopLeftRed = 1.0
    rectangle_bg.TopLeftGreen = 1.0
    rectangle_bg.TopLeftBlue = 1.0
    rectangle_bg.TopLeftAlpha = 1.0
    rectangle_bg.EffectMask = rectangle_mask

    key_frames = {
        0: { 1: 0.0, 'LH': {1: 0.0, 2: 0.0}, 'RH': {1: fusion_frames/2, 2: 0.0} },
        fusion_frames-1: { 1: 1.0, 'LH': {1: -fusion_frames/2, 2: 0.0}, 'RH': {1: 0.0, 2: 0.0} }
    }
    bezier_spline = comp.BezierSpline()
    bezier_spline.SetKeyFrames(key_frames)
    rectangle_mask.Width = bezier_spline

    merge = tdr.add_tool(
        comp=comp,
        tool_type=tdr.FusionTool.MERGE,
        position=(3, 1)
    )

    circle_mask = tdr.add_tool(
        comp=comp,
        tool_type = tdr.FusionTool.ELLIPSE_MASK,
        position = (3, -1)
    )
    circle_mask.Center = {1: 0.5, 2: 0.0, 3: 0.0}
    circle_mask.Width = 100/1920
    circle_mask.Height = 100/1920

    circle_bg = tdr.add_tool(
        comp=comp,
        tool_type = tdr.FusionTool.BACKGROUND,
        position = (3, 0)
    )
    circle_bg.TopLeftRed = 0.0
    circle_bg.TopLeftGreen = 1.0
    circle_bg.TopLeftBlue = 1.0
    circle_bg.TopLeftAlpha = 1.0
    circle_bg.EffectMask = circle_mask

    circle_xy_path = tdr.add_modifier(
        comp=comp,
        modifier_type=tdr.FusionModifier.XY_PATH,
    )

    x_spline = tdr.add_modifier(comp=comp, modifier_type=tdr.FusionModifier.BEZIER_SPLINE)
    y_spline = tdr.add_modifier(comp=comp, modifier_type=tdr.FusionModifier.BEZIER_SPLINE)

    y_spline.SetKeyFrames({
        0: {
            1: 0.0,
            "LH": {1: 0.0, 2: 0.1},
            "RH": {1: fusion_frames//4, 2: 0.0},
        },
        fusion_frames//2: {
            1: 1.0,
            "LH": {1: -fusion_frames//4, 2: 0.0},
            "RH": {1: fusion_frames//4, 2: 0.0},
        },
        fusion_frames-1: {
            1: 0.0,
            "LH": {1: -fusion_frames//4, 2: 0.0},
            "RH": {1: 0.0, 2: 0.0}
        },
    })
    x_spline.SetKeyFrames({
        int((fusion_frames//4)*0): {
            1: 0.5,
            "LH": {1: 0.0, 2: 0.0},
            "RH": {1: fusion_frames//8, 2: 0.0},
        },
        int((fusion_frames//4)*1): {
            1: 0.75,
            "LH": {1: -fusion_frames//8, 2: 0.0},
            "RH": {1: fusion_frames//8, 2: 0.0},
        },
        int((fusion_frames//4)*2): {
            1: 0.5,
            "LH": {1: 0.0, 2: 0.0},
            "RH": {1: 0.0, 2: 0.0},
        },
        int((fusion_frames//4)*3): {
            1: 0.25,
            "LH": {1: -fusion_frames//8, 2: 0.0},
            "RH": {1: fusion_frames//8, 2: 0.0},
        },
        fusion_frames-1: {
            1: 0.5,
            "LH": {1: -fusion_frames//8, 2: 0.0},
            "RH": {1: fusion_frames//8, 2: 0.0},
        },
    })

    circle_xy_path.X = x_spline
    circle_xy_path.Y = y_spline
    circle_mask.Center = circle_xy_path

    tdr.connect_merge(
        merge=merge,
        background=rectangle_bg,
        foreground=circle_bg
    )

    media_out = tdr.get_tool(comp=comp, name="MediaOut1")
    tdr.connect_default_output(source=merge, target=media_out)


def draw_sharp_edge_rectangle_using_fusion_sample():
    project_name = "sharp_edge_rectangle_sample"
    session = tdr.ResolveSession.connect()
    close_and_delete_project_if_exists(session, project_name)
    project = tdr.create_project(session, name=project_name)

    project_settings_params = {
        tdr.ProjectSetting.TIMELINE_RESOLUTION_WIDTH: "1920",
        tdr.ProjectSetting.TIMELINE_RESOLUTION_HEIGHT: "1080",
        tdr.ProjectSetting.TIMELINE_FRAME_RATE: tdr.FrameRate.FPS_24,
        tdr.ProjectSetting.VIDEO_MONITOR_FORMAT: tdr.make_video_monitor_format(1920, 1080, 24),
        tdr.ProjectSetting.VIDEO_MONITOR_USE_444_SDI: tdr.SettingToggle.ENABLED,
        tdr.ProjectSetting.VIDEO_MONITOR_SDI_CONFIGURATION: tdr.SDIConfiguration.SINGLE_LINK,
        tdr.ProjectSetting.VIDEO_DATA_LEVELS: tdr.VideoDataLevel.FULL,
        tdr.ProjectSetting.VIDEO_MONITOR_USE_HDR_OVER_HDMI: tdr.SettingToggle.DISABLED,

        tdr.ProjectSetting.COLOR_SCIENCE_MODE: tdr.ColorScienceMode.DAVINCI_YRGB,
    }

    tdr.set_settings(project=project, settings=project_settings_params)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_project_sample()
    # get_project_settings_sample()
    # project_settings_sample()
    # timeline_settings_sample()
    # get_current_timeline_settings_sample()
    # encode_test()
    # fusion_key_frame_test()
    draw_sharp_edge_rectangle_using_fusion_sample()
