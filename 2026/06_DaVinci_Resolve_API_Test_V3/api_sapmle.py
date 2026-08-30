#!/usr/bin/env python
# import standard libraries
import os
import time
from pathlib import Path
import pprint

# import third-party libraries

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

    project_settings_params3 = {
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
    tdr.set_timeline_settings(timeline=timeline, settings=project_settings_params3)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_project_sample()
    # get_project_settings_sample()
    # project_settings_sample()
    timeline_settings_sample()
    # get_current_timeline_settings_sample()
