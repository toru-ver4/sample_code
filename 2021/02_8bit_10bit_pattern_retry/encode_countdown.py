# -*- coding: utf-8 -*-
"""
Encode test videos using Davinci17
==================================

"""

# import standard libraries
import os
import sys
from pathlib import Path
import imp

# import third-party libraries
import DaVinciResolveScript as dvr_script

# import my libraries
import ty_davinci_control_lib as dcl
imp.reload(dcl)

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

H265_CQP0_PRESET_NAME = "H265_yuv444p12le_cqp-0"
DNxHR_PRESET_NAME = "DNxHR_444_12bit"

"""
import sys
sys.path.append("C:/Users/toruv/OneDrive/work/sample_code/2021/02_8bit_10bit_pattern_retry")
import encode_countdown as ecd
import imp
imp.reload(ecd)
ecd.encode_tp()
"""


PROJECT_PARAMS_BASE = dict(
    colorScienceMode="davinciYRGBColorManagedv2",
    rcmPresetMode="Custom",
    separateColorSpaceAndGamma="1",
    inputDRT="None",
    outputDRT="None",
)

PROJECT_PARAMS_1920x1080_24P = dict(
    timelineResolutionHeight="1080",
    timelineResolutionWidth="1920",
    videoMonitorFormat="HD 1080p 24",
    timelineFrameRate="24.000",
    timelinePlaybackFrameRate="24",
)

PROJECT_PARAMS_1920x1080_30P = dict(
    timelineResolutionHeight="1080",
    timelineResolutionWidth="1920",
    videoMonitorFormat="HD 1080p 30",
    timelineFrameRate="30.000",
    timelinePlaybackFrameRate="30",
)

PROJECT_PARAMS_1920x1080_60P = dict(
    timelineResolutionHeight="1080",
    timelineResolutionWidth="1920",
    videoMonitorFormat="HD 1080p 60",
    timelineFrameRate="60.000",
    timelinePlaybackFrameRate="60",
)

PROJECT_PARAMS_3840x2160_24P = dict(
    timelineResolutionHeight="3840",
    timelineResolutionWidth="2160",
    videoMonitorFormat="UHD 1080p 24",
    timelineFrameRate="24.000",
    timelinePlaybackFrameRate="24",
)

PROJECT_PARAMS_3840x2160_30P = dict(
    timelineResolutionHeight="3840",
    timelineResolutionWidth="2160",
    videoMonitorFormat="UHD 1080p 30",
    timelineFrameRate="30.000",
    timelinePlaybackFrameRate="30",
)

PROJECT_PARAMS_3840x2160_60P = dict(
    timelineResolutionHeight="3840",
    timelineResolutionWidth="2160",
    videoMonitorFormat="UHD 1080p 60",
    timelineFrameRate="60.000",
    timelinePlaybackFrameRate="60",
)

PROJECT_PARAMS_BT709 = dict(
    colorSpaceInput="Rec.709",
    colorSpaceInputGamma="Gamma 2.4",
    colorSpaceTimeline="Rec.709",
    colorSpaceTimelineGamma="Gamma 2.4",
    colorSpaceOutput="Rec.709",
    colorSpaceOutputGamma="Gamma 2.4",
)

PROJECT_PARAMS_BT2100 = dict(
    colorSpaceInput="Rec.2020",
    colorSpaceInputGamma="ST2084",
    colorSpaceTimeline="Rec.2020",
    colorSpaceTimelineGamma="ST2084",
    colorSpaceOutput="Rec.2020",
    colorSpaceOutputGamma="ST2084",
)

PROJECT_PRAMS_RESOLUTION_LIST = [
    PROJECT_PARAMS_1920x1080_24P,
    PROJECT_PARAMS_1920x1080_30P,
    PROJECT_PARAMS_1920x1080_60P,
    PROJECT_PARAMS_3840x2160_24P,
    PROJECT_PARAMS_3840x2160_30P,
    PROJECT_PARAMS_3840x2160_60P
]

PROJECT_PARAMS_COLOR_LIST = [
    PROJECT_PARAMS_BT709, PROJECT_PARAMS_BT2100
]


def create_clip_name_base(project_param):
    width = project_param['timelineResolutionWidth']
    height = project_param['timelineResolutionHeight']
    fps = project_param['timelinePlaybackFrameRate']
    if project_param['colorSpaceOutputGamma'] == "ST2084":
        dr = "HDR"
    elif project_param['colorSpaceOutputGamma'] == "Gamma 2.4":
        dr = "SDR"
    else:
        print("Error: invalid Dynamic Range")
        dr = "SDR"
    clip_name = f"movie_{dr}_{width}x{height}_{fps}fps_"

    return clip_name


def get_project_name(project_param):
    fps = project_param['timelinePlaybackFrameRate']
    return f"{fps}P"


def encode_tp(close_current_project=True):
    for project_resolution_param in PROJECT_PRAMS_RESOLUTION_LIST:
        for project_color_param in PROJECT_PARAMS_COLOR_LIST:
            project_param = {}
            project_param.update(
                **PROJECT_PARAMS_BASE,
                **project_resolution_param,
                **project_color_param)

            print(project_param)

            encode_tp_core(project_param=project_param)
            break
        break


def encode_tp_core(project_param, close_current_project=True):
    # parameter definition
    media_video_path = Path('D:/abuse/2020/005_make_countdown_movie/movie_seq')
    out_path = Path(
        'D:/Resolve/render_out/Countdown_Rev06')
    format_str = None
    codec = None
    preset_name = H265_CQP0_PRESET_NAME
    clip_name_base = create_clip_name_base(project_param)
    audio_bit_depth = 24
    db_value = 0

    # main process
    print("script start")
    resolve, project_manager = dcl.init_davinci17(
        close_current_project=close_current_project)
    project_manager.OpenFolder("Countdown")
    project = dcl.prepare_project(
        project_manager=project_manager,
        project_name=get_project_name(project_param))

    dcl.load_encode_preset(project, preset_name)
    dcl.set_project_settings_from_dict(project, project_param)

    print("add media to pool")
    dcl.add_clips_to_media_pool(resolve, project, media_video_path)
    clip_list, clip_name_list\
        = dcl.get_media_pool_clip_list_and_clip_name_list(project)

    # add video
    selected_video_clip_list = []
    for clip_obj, clip_name in zip(clip_list, clip_name_list):
        print(f"clip_name = {clip_name}")
        if clip_name_base in clip_name:
            print(f"{clip_name_base} is found!")
            selected_video_clip_list.append(clip_obj)
            break

    dcl.add_clips_to_the_current_timeline(
        resolve, project, selected_video_clip_list)

    # timeline = project.GetCurrentTimeline()
    # video = timeline.GetItemListInTrack('video', 1)[0]
    # print("dir(video_track)")
    # print(dir(video))

    # dcl.create_timeline_from_clip(
    #     resolve, project, selected_video_clip_list, timeline_name="video")

    # compound
    # all_clip = get_video_and_audio_timeline_items(timeline)
    # timeline.CreateCompoundClip(all_clip)
    # video_clip = timeline.GetItemListInTrack('video', 1)[0]
    # print(video_clip)
    # fusion_comp = video_clip.AddFusionComp()
    # print(dir(fusion_comp))

    # out_path = out_path.joinpath("ababa")
    # dcl.encode(resolve, project, out_path, format_str, codec, preset_name)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
