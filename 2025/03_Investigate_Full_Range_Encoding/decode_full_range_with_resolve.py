# -*- coding: utf-8 -*-

# import standard libraries
import sys
import os
from pathlib import Path

# import third-party libraries
import numpy as np
import matplotlib.pyplot as plt

# import my libraries
import ty_davinci_constants as drc
import ty_davinci_control_lib_2 as dcl


#####################
# Logic
#####################
def decode_full_range(
        width, height, framerate, gamut, gamma, ffmpeg_encode_dir):
    
    ##################
    # Project Settings
    ##################
    dcl.refresh_lut_list()

    encode_name = Path(ffmpeg_encode_dir).stem
    dir_path = Path(ffmpeg_encode_dir)

    project_name = f"Resolve_Full_Range_Decode_{encode_name}"
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
    if "MOV" in encode_name:
        ext_str = ".mov"
    elif "MP4" in encode_name:
        ext_str = ".mp4"
    else:
        raise ValueError("Invalid Encode Extension")
    encoded_video = str(dir_path / encode_name)
    encoded_video += ext_str
    print(encoded_video)
    relative_file_list = [
        encoded_video,
    ]
    file_path_list = [
        str(Path(x).resolve()) for x in relative_file_list
    ]
    print(file_path_list)

    clip = dcl.add_file_to_media_pool(file_path=file_path_list[0])
    dcl.append_clip_to_timeline(clip=clip)

    # clip.SetClipProperty("Data Level", "Full")
    # clip_property = clip.GetClipProperty()
    # print(clip_property)

    ###################
    # decode
    ###################
    preset_path = str(Path("./resolve_encode_preset/PNG_16-bit.xml").resolve())
    dcl.import_render_preset(preset_path=preset_path)

    # output file settings
    target_dir = str(Path(encoded_video).resolve().parent)
    custom_name = str(Path(encoded_video).resolve().stem)
    render_settings = {
        "TargetDir": target_dir,
        "CustomName": custom_name,
    }
    dcl.set_render_settings(setting_dict=render_settings)
    dcl.run_rendering_and_wait_until_finish(project=project)

    dcl.save_project()


#####################
# Main
#####################
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    resolution = "1920x1080"
    framerate = 24
    gamut = drc.PRJ_COLOR_SPACE_REC709
    gamma = drc.PRJ_GAMMA_STR_REC709
    width, height = resolution.split("x")

    ffmpeg_encode_dir_list = [
        "./encode_data/FFmpeg/DNxHR_MOV_HQX_10-bit_Full",
        "./encode_data/FFmpeg/H265_MOV_Main10_Full",
        "./encode_data/FFmpeg/H265_MOV_Main10_Limited",
        "./encode_data/FFmpeg/H265_MP4_Main10_Full",
        "./encode_data/FFmpeg/H265_NVENC_MOV_Main10_Full",
        "./encode_data/FFmpeg/H265_NVENC_MP4_Main10_Full",
        "./encode_data/FFmpeg/ProRes_MOV_422HQ_Full",
    ]
    
    for ffmpeg_encode_dir in ffmpeg_encode_dir_list:
        decode_full_range(
            width=width, height=height, framerate=framerate,
            gamut=gamut, gamma=gamma, ffmpeg_encode_dir=ffmpeg_encode_dir
        )
