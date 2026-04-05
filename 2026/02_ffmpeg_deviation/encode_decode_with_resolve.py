# -*- coding: utf-8 -*-

# import standard libraries
import os
from pathlib import Path

# import my libraries
import ty_davinci_constants as drc
import ty_davinci_control_lib_2 as dcl


#####################
# Logic
#####################
def make_encode_output_fname(src_image, encode_preset):
    preset_path = str(Path(encode_preset).resolve())
    dcl.import_render_preset(preset_path=preset_path)
    # output file settings
    encode_preset_stem = Path(encode_preset).stem
    dir_path = Path("./encode_data/Resolve") / "pre_resolve_test"
    dir_path.mkdir(parents=True, exist_ok=True)
    basename = f"{(Path(src_image).suffix[1:]).upper()}_{encode_preset_stem}"
    output_fname = str(dir_path / basename)

    return output_fname


def make_decode_output_fname(src_image, encode_preset):
    preset_path = str(Path("./resolve_encode_preset/PNG_16bit.xml").resolve())
    dcl.import_render_preset(preset_path=preset_path)

    # output file settings
    encode_preset_stem = Path(encode_preset).stem
    dir_path = Path("./decode_data/Resolve") / "pre_resolve_test"
    dir_path.mkdir(parents=True, exist_ok=True)
    basename = f"{(Path(src_image).suffix[1:]).upper()}_{encode_preset_stem}"
    decoded_image = str(dir_path / basename)

    return decoded_image


def encode_range(
        width, height, framerate, gamut, gamma, src_image, encode_preset_list):
    
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
        output_fname = make_encode_output_fname(src_image=src_image, encode_preset=encode_preset)
        target_dir = str(Path(output_fname).resolve().parent)
        custom_name = str(Path(output_fname).resolve().name)
        render_settings = {
            "TargetDir": target_dir,
            "CustomName": custom_name,
        }

        dcl.set_render_settings(setting_dict=render_settings)

        dcl.run_rendering_and_wait_until_finish(project=project)

    dcl.save_project()


def decode_range(
        width, height, framerate, gamut, gamma, src_image, encode_preset):
    
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
    encoded_video = make_encode_output_fname(src_image=src_image, encode_preset=encode_preset)
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
    dcl.append_clip_to_timeline(clip=clip, start_frame=start_frame, end_frame=start_frame+1)

    # clip.SetClipProperty("Data Level", "Full")
    # clip_property = clip.GetClipProperty()
    # print(clip_property)

    ###################
    # decode
    ###################
    encoded_video = make_decode_output_fname(src_image=src_image, encode_preset=encode_preset)
    target_dir = str(Path(encoded_video).resolve().parent)
    custom_name = str(Path(encoded_video).resolve().name)
    render_settings = {
        "TargetDir": target_dir,
        "CustomName": custom_name,
    }
    dcl.set_render_settings(setting_dict=render_settings)
    dcl.run_rendering_and_wait_until_finish(project=project)

    dcl.save_project()


def encode_and_decode(width, height, framerate, gamut, gamma, src_image, encode_preset_list):
    encode_range(
        width=width, height=height, framerate=framerate,
        gamut=gamut, gamma=gamma, src_image=src_image, encode_preset_list=encode_preset_list
    )

    for encode_preset in encode_preset_list:
        decode_range(
            width=width, height=height, framerate=framerate,
            gamut=gamut, gamma=gamma, src_image=src_image, encode_preset=encode_preset
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

    win_encode_preset_list = [
        "./resolve_encode_preset/H.265_NVENC_Main10.xml",
        "./resolve_encode_preset/AV1_NVENC_Main10.xml",
    ]

    src_image_list = [
        "./img/src_img.dpx",
        "./img/src_img.png",
        "./img/src_img.tif"
    ]

    for src_image in src_image_list:
        encode_and_decode(
            width=width, height=height, framerate=framerate, gamut=gamut, gamma=gamma,
            src_image=src_image,
            encode_preset_list=win_encode_preset_list
        )
