# -*- coding: utf-8 -*-

# import standard libraries
import os
from pathlib import Path

# import my libraries
import ty_davinci_constants as drc
import ty_davinci_control_lib_2 as dcl

from ffmpeg_analyze_common import (
    WIN_ENCODE_PRESET_LIST,
    SRC_IMAGE_LIST,
    make_encode_output_fname,
    make_decode_output_fname
)


#####################
# Logic
#####################

def encode_core_with_resolve(
        width, height, framerate, gamut, gamma, src_image, encode_preset_list, encode_app):
    
    ##################
    # Project Settings
    ##################
    dcl.refresh_lut_list()

    project_name = "Resolve_10bit_Encoding"
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
        "colorSpaceTimelineGamma": drc.PRJ_COLOR_SPACE_REC709,
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
    # dcl.set_timeline_settings(timeline=timeline, params=project_settings_params)

    # add files to the media storage
    relative_file_list = [
        src_image,
    ]
    file_path_list = [
        str(Path(x).resolve()) for x in relative_file_list
    ]
    print(file_path_list)

    clip = dcl.add_file_to_media_pool(file_path=file_path_list[0])
    dcl.append_clip_to_timeline(clip=clip)

    ###################
    # encode
    ###################
    for encode_preset in encode_preset_list:
        preset_path = str(Path(encode_preset).resolve())
        dcl.import_render_preset(preset_path=preset_path)

        output_fname = make_encode_output_fname(
            src_image=src_image, encode_preset=encode_preset, encode_app=encode_app
        )
        target_dir = str(Path(output_fname).resolve().parent)
        custom_name = str(Path(output_fname).resolve().name)
        render_settings = {
            "TargetDir": target_dir,
            "CustomName": custom_name,
        }

        dcl.set_render_settings(setting_dict=render_settings)

        dcl.run_rendering_and_wait_until_finish(project=project)

    dcl.save_project()


def decode_core(
        width, height, framerate, gamut, gamma, src_image, encode_preset, encode_app):
    
    ##################
    # Project Settings
    ##################
    dcl.refresh_lut_list()

    encode_preset_stem = Path(encode_preset).stem
    project_name = f"Resolve_10bit_Decode_{encode_preset_stem}"
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
        "colorSpaceTimelineGamma": drc.PRJ_GAMMA_STR_REC709,
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
    ext_str = ".mp4"
    decoded_image = make_encode_output_fname(
        src_image=src_image, encode_preset=encode_preset, encode_app=encode_app
    )
    decoded_image += ext_str
    print(decoded_image)
    relative_file_list = [
        decoded_image,
    ]
    file_path_list = [
        str(Path(x).resolve()) for x in relative_file_list
    ]
    print(file_path_list)

    clip = dcl.add_file_to_media_pool(file_path=file_path_list[0])
    dcl.append_clip_to_timeline(clip=clip, start_frame=start_frame, end_frame=start_frame+1)

    # clip.SetClipProperty("Data Level", "Full")
    # clip_property = clip.GetClipProperty()
    # print(clip_property)

    ###################
    # decode
    ###################
    preset_path = str(Path("./resolve_encode_preset/PNG_16bit.xml").resolve())
    dcl.import_render_preset(preset_path=preset_path)

    decoded_image = make_decode_output_fname(
        src_image=src_image, encode_preset=encode_preset,
        encode_app=encode_app, decode_app='resolve'
    )
    target_dir = str(Path(decoded_image).resolve().parent)
    custom_name = str(Path(decoded_image).resolve().name)
    render_settings = {
        "TargetDir": target_dir,
        "CustomName": custom_name,
    }
    dcl.set_render_settings(setting_dict=render_settings)
    dcl.run_rendering_and_wait_until_finish(project=project)

    dcl.save_project()


def encode_and_decode_with_ffmpeg(
        width, height, framerate, gamut, gamma, src_image, encode_preset_list, encode_app):
    encode_core_with_resolve(
        width=width, height=height, framerate=framerate,
        gamut=gamut, gamma=gamma, src_image=src_image,
        encode_preset_list=encode_preset_list,
        encode_app=encode_app
    )

    for encode_preset in encode_preset_list:
        decode_core(
            width=width, height=height, framerate=framerate,
            gamut=gamut, gamma=gamma, src_image=src_image,
            encode_preset=encode_preset, encode_app=encode_app
        )


def decode_ffmpeg_enc_data(
        width, height, framerate, gamut, gamma, src_image, encode_preset_list, encode_app):
    for encode_preset in encode_preset_list:
        decode_core(
            width=width, height=height, framerate=framerate,
            gamut=gamut, gamma=gamma, src_image=src_image,
            encode_preset=encode_preset, encode_app=encode_app
        )


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

    win_encode_preset_list = WIN_ENCODE_PRESET_LIST
    src_image_list = SRC_IMAGE_LIST

    for src_image in src_image_list:
        # encode_and_decode_with_ffmpeg(
        #     width=width, height=height, framerate=framerate, gamut=gamut, gamma=gamma,
        #     src_image=src_image,
        #     encode_preset_list=win_encode_preset_list,
        #     encode_app="resolve"
        # )
        decode_ffmpeg_enc_data(
            width=width, height=height, framerate=framerate, gamut=gamut, gamma=gamma,
            src_image=src_image,
            encode_preset_list=win_encode_preset_list,
            encode_app="ffmpeg"
        )
