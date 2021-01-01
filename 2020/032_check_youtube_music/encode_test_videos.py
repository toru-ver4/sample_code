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


FREQ_MAGNIFICATION_LIST = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
BASE_FREQ_LIST = [
    261.626, 293.665, 329.628, 349.228, 391.995, 440.000, 493.883]

"""
import sys
sys.path.append("C:/Users/toruv/OneDrive/work/sample_code/2020/032_check_youtube_music")
import encode_test_videos as etv
import imp
imp.reload(etv)
etv.encode_tp()
"""


PROJECT_PARAMS = dict(
    timelineResolutionHeight="1080",
    timelineResolutionWidth="1920",
    videoMonitorFormat="HD 1080p 24",
    timelineFrameRate="24.000",
    timelinePlaybackFrameRate="24",
    colorScienceMode="davinciYRGBColorManagedv2",
    rcmPresetMode="Custom",
    separateColorSpaceAndGamma="1",
    colorSpaceInput="Rec.709",
    colorSpaceInputGamma="Gamma 2.4",
    colorSpaceTimeline="Rec.709",
    colorSpaceTimelineGamma="Gamma 2.4",
    colorSpaceOutput="Rec.709",
    colorSpaceOutputGamma="Gamma 2.4",
    inputDRT="None",
    outputDRT="None",
)


def make_audio_fname(freq=440.03, db_value=-3, bit_depth=24):
    fname = f"freq_{int(freq):05d}Hz_db{db_value}_{bit_depth}bit.wav"

    return fname


def make_audio_fname_list(audio_bit_depth=24, db_value=0):
    fname_list = []
    for freq_magnification in FREQ_MAGNIFICATION_LIST:
        for base_freq in BASE_FREQ_LIST:
            freq = base_freq * freq_magnification
            fname = make_audio_fname(
                freq=freq, db_value=db_value, bit_depth=audio_bit_depth)
            fname_list.append(fname)

    return fname_list


def get_video_and_audio_timeline_items(timeline):
    audio_items = timeline.GetItemListInTrack('audio', 1)
    video_items = timeline.GetItemListInTrack('video', 1)

    return video_items + audio_items


def encode_tp(close_current_project=True):
    # parameter definition
    media_audio_path = Path('C:/Users/toruv/OneDrive/work/sample_code/2020/032_check_youtube_music/audio')
    media_video_path = Path('C:/Users/toruv/OneDrive/work/sample_code/2020/032_check_youtube_music/image')
    out_path = Path(
        'D:/abuse/2020/031_cms_for_video_playback/mp4/')
    format_str = "mp4"
    codec = "H264_NVIDIA"
    preset_name = "H264_lossless"
    dummy_image_name = 'dummy_background.png'
    audio_bit_depth = 24
    db_value = 0
    project_params = PROJECT_PARAMS

    # main process
    print("script start")
    resolve, project_manager = dcl.init_davinci17(
        close_current_project=close_current_project)
    project = dcl.prepare_project(project_manager)

    dcl.set_project_settings_from_dict(project, project_params)

    print("add media to pool")
    dcl.add_clips_to_media_pool(resolve, project, media_audio_path)
    dcl.add_clips_to_media_pool(resolve, project, media_video_path)
    clip_list, clip_name_list\
        = dcl.get_media_pool_clip_list_and_clip_name_list(project)

    print("create file liset")
    fname_list = make_audio_fname_list(
        audio_bit_depth=audio_bit_depth, db_value=db_value)

    # add audio
    selected_audio_clip_list = []
    for fname in fname_list:
        print(f"fname = {fname}")
        for clip_obj, clip_name in zip(clip_list, clip_name_list):
            print(f"clip_name = {clip_name}")
            if clip_name == fname:
                print(f"{fname} is found!")
                selected_audio_clip_list.append(clip_obj)
                break
    timeline = dcl.create_timeline_from_clip(
        resolve, project, selected_audio_clip_list, timeline_name="main")

    # add video
    selected_video_clip_list = []
    for clip_obj, clip_name in zip(clip_list, clip_name_list):
        print(f"clip_name = {clip_name}")
        if clip_name == dummy_image_name:
            print(f"{dummy_image_name} is found!")
            selected_video_clip_list.append(clip_obj)
            break

    dcl.add_clips_to_the_current_timeline(
        resolve, project, selected_video_clip_list)

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
