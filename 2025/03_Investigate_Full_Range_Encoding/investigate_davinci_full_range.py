# -*- coding: utf-8 -*-

# import standard libraries
import sys
import os
from pathlib import Path
import copy
import shutil
import subprocess

# import third-party libraries
import numpy as np

# import my libraries
import ty_davinci_constants as drc
import ty_davinci_control_lib_2 as dcl


#####################
# Logic
#####################
def encode_decode_seq(
        width, height, framerate, gamut, gamma, encode_param):
    
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

    if preset_path is None:
        # format_extension = drc.OUT_FILE_EXTENSTION_MOV
        format_extension = drc.OUT_FILE_EXTENSTION_PNG
        # format_extension = drc.OUT_FILE_EXTENSTION_TIFF
        # format_extension = drc.OUT_FILE_EXTENSTION_EXR
        # format_extension = drc.OUT_FILE_EXTENSTION_DPX

        # codec = drc.CODEC_H265_NVIDIA
        # codec = drc.CODEC_H264_NVIDIA
        # codec = drc.CODEC_APPLE_PRORES_4444
        # codec = drc.CODEC_EXR_RGB_HALF
        # codec = drc.CODEC_DPX_RGB_10_BITS
        codec = drc.CODEC_PNG_RGB_16_BITS
        # codec = drc.CODEC_TIF_RGB_16_BITS

        if sys.platform == "darwin":  # macOS
            dir_path = Path("/Volumes/My Passport/Countdown/temp_seq")
        elif sys.platform == "win32":  # Windows
            dir_path = Path(r"D:\abuse\Countdown\temp_seq")
        else:
            pass

        basename = f"{width}x{height}_{framerate}P_{gamma}_{gamut}"
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

        shutil.rmtree(dir_path, ignore_errors=True)
        dir_path.mkdir(parents=True, exist_ok=True)
    if preset_path is not None:
        dcl.import_render_preset(preset_path=preset_path)
    else:
        dcl.set_render_format_codec_settings(format=format_extension, codec=codec)

    dcl.set_render_settings(setting_dict=render_settings)
    # dcl.run_rendering_and_wait_until_finish(project=project)


#####################
# Main
#####################
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    resolution = "1920x1080"
    framerate = 24
    gamut = drc.PRJ_COLOR_SPACE_REC709
    gamma = drc.PRJ_GAMMA_STR_REC709

    encode_param_list = [
        [drc.CODEC_APPLE_PRORES_422_HQ, drc.OUT_FILE_EXTENSTION_MOV, "./preset/hoge.xml"],
    ]

    width, height = resolution.split("x")

    for encode_param in encode_param_list:
        encode_decode_seq(
            width=width, height=height, framerate=framerate,
            gamut=gamut, gamma=gamma, encode_param=encode_param
        )
