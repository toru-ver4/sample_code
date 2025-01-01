# -*- coding: utf-8 -*-

"""
方針: 
"""

import sys
import os
from pathlib import Path
import functools
import resolve_constants as drc
from pprint import pprint

if sys.platform == "darwin":  # macOS
    resolve_script_api = (
        "/Library/Application Support/Blackmagic Design/DaVinci Resolve/"
        "Developer/Scripting"
    )
    sys.path.append(os.path.join(resolve_script_api, "Modules"))
elif sys.platform == "win32":  # Windows
    resolve_script_api = os.path.expandvars(
        r"%PROGRAMDATA%\Blackmagic Design\DaVinci Resolve\Support"
        r"\Developer\Scripting"
    )
    sys.path.append(os.path.join(resolve_script_api, "Modules"))
elif sys.platform == "linux":  # Linux
    resolve_script_api = "/opt/resolve/Developer/Scripting"
    sys.path.append(os.path.join(resolve_script_api, "Modules"))

import DaVinciResolveScript as dvr_script


resolve = dvr_script.scriptapp("Resolve")
if resolve is None:
    raise ConnectionError("The DaVinci Resolve app is not running.")

fusion = resolve.Fusion()


class TyResolveModuleError(Exception):
    """
    Raised when a error occured inside the TyResolve module.
    """
    def __init__(self, value, message):
        self.value = value
        self.message = message
        super().__init__(f"{message}")


# =====================
# for Debug
# =====================
DEBUG_ON = True
# DEBUG_ON = False


def log_return_value(func):
    """
    デバッグモードが有効な場合のみ、関数名と戻り値をログ出力するデコレーター。
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if DEBUG_ON:
            print(f"[Debug] {func.__name__}() -> {result}")
        return result
    return wrapper


def get_project_manager():
    """
    Returns
    -------
    ProjectManager (BlackmagicFusion.PyRemoteObject)
        The project manager instance

    """
    project_manager = resolve.GetProjectManager()

    return project_manager


def _close_project(project):
    """
    Parameters
    ----------
    project : Project (BlackmagicFusion.PyRemoteObject)
        A Project instance of the DaVinci Resolve

    Returns
    -------
    Bool
        Returns True if successful, and False otherwise.
    """
    project_manager = get_project_manager()
    ret_val = project_manager.CloseProject(project)

    return ret_val


@log_return_value
def close_current_project():
    """
    Returns
    -------
    Bool
        Returns True if successful, and False otherwise.    
    """
    project = get_current_project()
    ret_val = _close_project(project=project)

    return ret_val


@log_return_value
def delete_project(project_name):
    """
    Parameters
    ----------
    project_name : str
        The project name

    Returns
    -------
    Bool
        Returns True if successful, and False otherwise.
    """
    project_manager = get_project_manager()
    ret_val = project_manager.DeleteProject(project_name)

    return ret_val


@log_return_value
def create_project(project_name):
    """
    Parameters
    ----------
    project_name : str
    A project name

    Returns
    -------
    Project (BlackmagicFusion.PyRemoteObject)
        If project does not already exist
    """
    project_manager = get_project_manager()
    project = project_manager.CreateProject(project_name)

    if project is None:
        msg = f'"{project_name}" project is already exists. '
        msg += 'Please use `load_project` function instead.'
        raise TyResolveModuleError(project, msg)

    return project


@log_return_value
def save_project():
    """
    Returns
    -------
    Bool
        Returns True if successful.
    """
    project_manager = get_project_manager()
    ret_val = project_manager.SaveProject()

    if ret_val is None:
        msg = 'Failed to save project. '
        msg += 'Please save project manually.'
        raise TyResolveModuleError(project, msg)

    return ret_val


@log_return_value
def load_project(project_name):
    """
    Parameters
    ----------
    project_name : str
    A project name

    Returns
    -------
    Project (BlackmagicFusion.PyRemoteObject)
        If project does not already exist
    """
    project_manager = get_project_manager()
    project = project_manager.LoadProject(project_name)

    if project is None:
        msg = f'Failed to load "{project_name}" project. '
        msg += 'Please verify that the `project_name` is correct.'
        raise TyResolveModuleError(project, msg)

    return project


@log_return_value
def get_current_project():
    """
    Returns
    -------
    Project (BlackmagicFusion.PyRemoteObject)
        If project does not already exist
    """
    project_manager = get_project_manager()
    project = project_manager.GetCurrentProject()

    return project


@log_return_value
def open_page(page_name="edit"):
    """
    Parameters
    ----------
    page_name : str
        A page name. You can use strings listed below.
            * "media"
            * "cut"
            * "edit"
            * 'fusion'
            * "color"
            * "deliver"
    """
    ret_val = resolve.OpenPage(page_name)

    if ret_val is not True:
        msg = f'Failed to open "{page_name}" page. '
        msg += 'Please verify that the `page_name` is correct.'
        raise TyResolveModuleError(project, msg)

    return project


@log_return_value
def get_media_pool():
    """
    Returns
    -------
    MediaPool (BlackmagicFusion.PyRemoteObject)
        A MediaPool instance
    """
    project = get_current_project()
    media_pool = project.GetMediaPool()

    return media_pool


@log_return_value
def create_empty_timeline(name="timeline_x"):
    """
    Parameters
    ----------
    name : str
        A timeline name

    Returns
    -------
    Timeline (BlackmagicFusion.PyRemoteObject)
        A empty Timeline instance
    """
    open_page(page_name='edit')
    media_pool = get_media_pool()
    timeline = media_pool.CreateEmptyTimeline(name)

    if timeline is None:
        msg = f'The Timeline "{name}" already exists. '
        msg += "Please provide a different name."
        raise TyResolveModuleError(project, msg)

    return timeline


def set_project_setting(name, value):
    """
    Parameters
    ----------
    name : str
        A project setting name
    value : str
        A project setting value

    Returns
    -------
    Returns True if successful, and False otherwise.
    """
    result = project.SetSetting(name, value)
    if result:
        print(f'    Project.SetSetting("{name}", "{value}") -> Success')
    else:
        print(f'    Project.SetSetting("{name}", "{value}") -> Failed')

    return result


@log_return_value
def get_project_setting(name):
    """
    Parameters
    ----------
    name : str
        A project setting name

    Returns
    -------
    str
        A project setting value
    """
    result = project.GetSetting(name)
    print(f'    Project.GetSetting("{name}") -> "{result}"')

    return result


@log_return_value
def setup_project_settings(params):
    """
    Parameters
    ----------
    params : dict
        A dictionary that specify the project settings.

    Returns
    -------
    Bool
        Returns True if successful.

    Examples
    --------
    >>> params = {
    ...     "timelineResolutionWidth": "1920",
    ...     "timelineResolutionHeight": "1080",
    ...     "videoMonitorFormat": "HD 1080p 24",
    ...     "timelineFrameRate": "24",
    ...     "videoMonitorUse444SDI": "0",
    ...     "videoMonitorSDIConfiguration": "single_link",
    ...     "videoDataLevels": "Video",
    ...     "videoMonitorUseHDROverHDMI": "1",
    ...     "colorScienceMode": "davinciYRGBColorManagedv2",
    ...     "rcmPresetMode": "Custom",
    ...     "separateColorSpaceAndGamma": "1",
    ...     "colorSpaceInput": "Rec.2020",
    ...     "colorSpaceInputGamma": "ST2084",
    ...     "colorSpaceTimeline": "Rec.2020",
    ...     "colorSpaceTimelineGamma": "ST2084",
    ...     "colorSpaceOutput": "Rec.2020",
    ...     "colorSpaceOutputGamma": "ST2084",
    ...     "timelineWorkingLuminance": "10000",
    ...     "timelineWorkingLuminanceMode": "Custom",
    ...     "inputDRT": "None",
    ...     "outputDRT": "None",
    ...     "hdrMasteringLuminanceMax": "1000",
    ...     "hdrMasteringOn": "1",
    ... }
    >>> setup_project_settings(params=params)
    """
    is_success = True
    for name, value in params.items():
        result = set_project_setting(name, value)
        if result is False:
            is_success = False

    if is_success is False:
        msg = 'Project.SetSetting() was failed. '
        msg += 'Please check your "params" parameters.'
        raise TyResolveModuleError(project, msg)

    return is_success


@log_return_value
def get_media_storage():
    """
    Returns
    -------
    MediaStorage (BlackmagicFusion.PyRemoteObject)
        A MediaStorage instance
    """
    media_storage = resolve.GetMediaStorage()

    return media_storage


@log_return_value
def add_file_to_media_pool(file_path, start_frame=None, end_frame=None):
    """
    Parameters
    ----------
    file_path : str
        A absolute file path.
    start_frame : int
        A start frame number.
    end_frame : int
        A end frame number.

    Returns
    -------
    MediaPoolItem
        A MediaPoolItem instance.
    """
    open_page(page_name='media')
    media_storage = get_media_storage()

    if (start_frame is None) and (end_frame is None):
        ret_value = media_storage.AddItemListToMediaPool([file_path])
    else:
        media_info = {
            "media": file_path,
            "startFrame": start_frame,
            "endFrame": end_frame,
        }
        ret_value = media_storage.AddItemListToMediaPool([media_info])

    if ret_value == []:
        msg = 'add_files_to_media_pool() was failed. '
        msg += 'Please check `file_path_list` parameter.'
        raise TyResolveModuleError(ret_value, msg)

    return ret_value[0]


@log_return_value
def add_seq_file_to_media_pool(file_path, start_idx, end_idx):
    """
    Parameters
    ----------
    file_path : str
        A sequence file path.
        example: `file_path = "/media/countdown_%04d.png"`
    start_idx : int
        A sequence file start index.
    end_idx : int
        A sequence file end index.
    """
    open_page(page_name="media")
    media_pool = get_media_pool()
    clip_info = {
        "FilePath": file_path,
        "StartIndex": start_idx,
        "EndIndex": end_idx,
    }
    ret_value = media_pool.ImportMedia([clip_info])

    if ret_value is None:
        msg = 'add_files_to_media_pool() was failed. '
        msg += 'Please check `file_path_list` parameter.'
        raise TyResolveModuleError(ret_value, msg)

    return ret_value[0]


@log_return_value
def append_clip_to_timeline(
        clip, pos_timecode=None,
        start_frame=None, end_frame=None, media_type=None, track_index=1):
    """
    Parameters
    ----------
    clip : MediaPoolItem
        clip
    pos_timecode : str
        clip start position (timecode).
        example -> `pos_timecode="01:00:00:12"`
    start_frame : int or float
        start frame number
    end_frame : int or float
        end frame number
    media_type : int
        1: video only, 2: autio only
    track_index : int
        track index

    Returns
    -------
    TimelineItem
        A TimelineItem instance
    """
    media_pool = get_media_pool()

    clip_info = {
        "mediaPoolItem": clip
    }

    if pos_timecode is not None:
        frame_idx = _timecode_to_frame_index(timecode=pos_timecode)
        clip_info.update({'recordFrame': frame_idx})
    if (start_frame is not None) and (end_frame is not None):
        clip_info.update({'startFrame': start_frame})
        clip_info.update({'endFrame': end_frame})
    if media_type is not None:
        clip_info.update({'mediaType': media_type})
    if track_index is not None:
        clip_info.update({'trackIndex': track_index})

    ret_value = media_pool.AppendToTimeline([clip_info])

    if ret_value[0] is None:
        msg = 'append_clip_to_timeline() was failed. '
        msg += 'Please check arguments.'
        raise TyResolveModuleError(ret_value, msg)

    return ret_value[0]


@log_return_value
def insert_generator_into_timeline(timeline, generator_name):
    timeline_item = timeline.InsertGeneratorIntoTimeline(generator_name)
    if timeline_item is None:
        msg = '`insert_generator_into_timeline` was failed. '
        msg += 'Please check if "generator_name" is correct.'
        raise TyResolveModuleError(False, msg)

    return timeline_item


def _frame_index_to_timecode(
        frame_index, start_timecode="01:00:00:00"):
    fps_float = get_project_setting("timelineFrameRate")
    if abs(fps_float - int(fps_float)) > 0.0:
        msg = 'Unsupported frame rate '
        msg += 'Please specify integer framerate to the project settings.'
        raise TyResolveModuleError(False, msg)

    fps = int(fps_float)

    hours, minutes, seconds, frames = map(int, start_timecode.split(':'))

    total_frames = frames + frame_index

    added_seconds = total_frames // fps
    frames = total_frames % fps
    seconds += added_seconds
    minutes += seconds // 60
    seconds %= 60
    hours += minutes // 60
    minutes %= 60

    new_timecode = f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"
    return new_timecode


def _timecode_to_frame_index(timecode: str):
    fps_float = get_project_setting("timelineFrameRate")
    if abs(fps_float - int(fps_float)) > 0.0:
        msg = 'Unsupported frame rate. '
        msg += 'Please specify integer framerate in the project settings.'
        raise TyResolveModuleError(False, msg)

    fps = int(fps_float)

    th, tm, ts, tf = map(int, timecode.split(':'))

    frame_index = ((th * 3600) + (tm * 60) + ts) * fps + tf

    return frame_index


@log_return_value
def set_current_timecode(timecode):
    timeline = get_current_project().GetCurrentTimeline()
    ret_val = timeline.SetCurrentTimecode(timecode)

    return ret_val


@log_return_value
def set_render_format_codec_settings(format='mov', codec='ProRes422HQ'):
    """
    Parameters
    ----------
    format : str
        A render format.
        Examples are 'mov', 'mp4', 'tif', 'png', ...
    codec : str
        A coding option.
        Examples are "H265_NVIDIA", "H265", "ProRes422HQ", "DNxHRHQX_12"
    """
    project = get_current_project()
    result = project.SetCurrentRenderFormatAndCodec(format, codec)

    if result:
        msg = "    Project.SetCurrentRenderFormatAndCodec("
        msg += f'"{format}", "{codec}") -> Success'
        print(msg)
    else:
        msg = "    Project.SetCurrentRenderFormatAndCodec("
        msg += f'"{format}", "{codec}") -> Failed'
        print(msg)

        msg = 'Failed to set_render_format_codec_settings() '
        msg += 'Please check if the input parameters are correct.'
        raise TyResolveModuleError(result, msg)

    return result


def set_render_setting(name, value):
    project = get_current_project()
    result = project.SetRenderSettings({name: value})

    if result:
        msg = f"    project.SetRenderSettings({{{name}: {value}}})"
        msg += '-> Sucess'
        print(msg)
    else:
        msg = f"    project.SetRenderSettings({{{name}: {value}}})"
        msg += '-> Failed'
        print(msg)

    return result


@log_return_value
def set_render_settings(setting_dict: dict):
    """
    Parameters
    ----------
    setting_dict : dict
        setting dictionary
    """
    for key, value in setting_dict.items():
        result = set_render_setting(name=key, value=value)
    if result is not True:
        msg = 'Failed to set_render_settings() '
        msg += 'Please check if the input parameters are correct.'
        raise TyResolveModuleError(result, msg)

    return result


@log_return_value
def delete_render_preset(preset_name):
    """
    preset_name : str
        A preset name.
    """
    project = get_current_project()
    result = project.DeleteRenderPreset(preset_name)

    return result


@log_return_value
def import_render_preset(preset_path):
    """
    Parameters
    ----------
    preset_path : str
        A path for the preset settigs.
    """
    preset_name = Path(preset_path).stem
    delete_render_preset(preset_name=preset_name)
    result = resolve.ImportRenderPreset(preset_path)

    if result is not True:
        msg = 'Failed to import_render_preset() '
        msg += 'Please check if the input parameters are correct.'
        raise TyResolveModuleError(result, msg)

    return result


@log_return_value
def add_fusion_comp(timeline_item):
    """
    Parameters
    ----------
    timeline_item : TimelineItem
        A TimelineItem instance

    Returns
    -------
    Composition
        A fusion composition
    """
    fusion_comp = timeline_item.AddFusionComp()
    if fusion_comp is None:
        msg = 'Failed to add_fusion_comp() '
        msg += 'Please check if the `timeline_item` is exist on the timeline.'
        raise TyResolveModuleError(fusion_comp, msg)

    return fusion_comp


@log_return_value
def append_fusion_composition_to_timeline(
        num_of_frame: int, pos_timecode: str | None = None):
    """
    Note
    ----
    This function uses abnormal workaround.
    This is because `Timeline:InsertFusionCompositionIntoTimeline()` can not
    specify the frame length.
    """
    fps_str = int(get_project_setting(name="timelineFrameRate"))
    dummy_video_path = f"./videos/dummy_video_{fps_str}P.mp4"
    dummy_video_full_path = str(Path(dummy_video_path).resolve())
    clip = add_file_to_media_pool(file_path=dummy_video_full_path)
    timeline_item = append_clip_to_timeline(
        clip=clip,
        media_type=1,
        start_frame=0,
        end_frame=num_of_frame,  # specify the frame length
        pos_timecode=pos_timecode)
    fusion_comp = add_fusion_comp(timeline_item)

    # Delete MediaIn1 (Because MediaIn is dummy data)
    get_comp_node_by_name(comp=fusion_comp, name="MediaIn1").Delete()


    return timeline_item, fusion_comp


############################################
# Fusion Page
############################################
@log_return_value
def get_comp_node_by_name(comp, name):
    """
    Parameters
    ----------
    comp : Composition
        The composition
    name : str
        The node name.
    
    Returns
    -------
    
    """
    node = None
    for _, value in comp.GetToolList().items():
        if value.Name == name:
            node = value
            pass

    if node is None:
        msg = 'Failed to get_media_out() '
        msg += 'Please check if the media_out node name is "MediaOut1".'
        raise TyResolveModuleError(node, msg)

    return node


@log_return_value
def add_fusion_node(comp, name):
    """
    Parameters
    ----------
    comp : Composition
        The composition
    name : str
        The node name.
    
    Returns
    -------
    """
    node = comp.AddTool(name)

    if node is None:
        msg = 'Failed to add_fusion_node() '
        msg += f'Please check if "name" = {name} is correct.'
        raise TyResolveModuleError(node, msg)
    
    return node


def get_media_out(comp):
    """
    Parameters
    ----------
    comp: Composition
        The composition

    Returns
    -------
    Media Out
        The media out node
    """
    return get_comp_node_by_name(comp=comp, name="MediaOut1")


def get_media_in(comp):
    """
    Parameters
    ----------
    comp: Composition
        The composition

    Returns
    -------
    Media In
        The media In node
    """
    return get_comp_node_by_name(comp=comp, name="MediaIn1")


@log_return_value
def connect_node(a, b):
    """
    Connect a to b.
    """
    result = b.Input.ConnectTo(a.Output)

    if result is not True:
        msg = 'Failed to connect_node() '
        raise TyResolveModuleError(media_out, msg)

    return result


@log_return_value
def set_topleft_color(node, rgba=[0.18, 0.18, 0.18, 1.0]):
    channels = ["Red", "Green", "Blue", "Alpha"]
    for channel, value in zip(channels, rgba):
        node.SetInput(f"TopLeft{channel}", value)
        # verify
        if node.GetInput(f"TopLeft{channel}") != value:
            msg = 'Failed to set_topleft_color()'
            raise TyResolveModuleError(False, msg)
    return True


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    ##################
    # Project Settings
    ##################
    project_name = "Hello World3"
    project_settings_params = {
        "timelineResolutionWidth": "1920",
        "timelineResolutionHeight": "1080",
        "videoMonitorFormat": "HD 1080p 24",
        "timelineFrameRate": "24",
        "videoMonitorUse444SDI": "0",
        "videoMonitorSDIConfiguration": "single_link",
        "videoDataLevels": "Video",
        "videoMonitorUseHDROverHDMI": "1",
        "colorScienceMode": "davinciYRGBColorManagedv2",
        "rcmPresetMode": "Custom",
        "separateColorSpaceAndGamma": "1",
        "colorSpaceInput": "Rec.2020",
        "colorSpaceInputGamma": "ST2084",
        "colorSpaceTimeline": "Rec.2020",
        "colorSpaceTimelineGamma": "ST2084",
        "colorSpaceOutput": "Rec.2020",
        "colorSpaceOutputGamma": "ST2084",
        "timelineWorkingLuminance": "10000",
        "timelineWorkingLuminanceMode": "Custom",
        "inputDRT": "None",
        "outputDRT": "None",
        "hdrMasteringLuminanceMax": "1000",
        "hdrMasteringOn": "1",
    }

    # control the project
    close_current_project()
    delete_project(project_name=project_name)
    project = create_project(project_name=project_name)
    save_project()
    close_current_project()
    project = load_project(project_name=project_name)

    # set up the project settings
    setup_project_settings(params=project_settings_params)

    ###########################
    # Add files to the timeline
    ###########################
    # create timelines
    timeline = create_empty_timeline(name="My_Timeline")

    # add files to the media storage
    relative_file_list = [
        "./videos/countdown_HDR_24fps_hevc_yuv420p10le.mov",
        "./videos/countdown_SDR_24fps_hevc_yuv420p10le.mov",
        "./videos/countdown_SDR_60P_%04d.png",
        "./videos/countdown.wav",
        "./videos/dummy_video_24P.mp4",
    ]
    file_path_list = [
        str(Path(x).resolve()) for x in relative_file_list
    ]
    print(file_path_list)
    clip_hdr = add_file_to_media_pool(file_path=file_path_list[0])
    clip_sdr = add_file_to_media_pool(
        file_path=file_path_list[1], start_frame=24, end_frame=71
    )
    clip_seq = add_seq_file_to_media_pool(
        file_path=file_path_list[2], start_idx=120, end_idx=179
    )
    clip_audio = add_file_to_media_pool(file_path=file_path_list[3])
    clip_black = add_file_to_media_pool(
        file_path=file_path_list[4], start_frame=0, end_frame=119
    )

    # # add clips to the timeline
    append_clip_to_timeline(clip=clip_hdr)
    append_clip_to_timeline(clip=clip_sdr)
    append_clip_to_timeline(
        clip=clip_seq, media_type=1, pos_timecode="01:00:06:00")
    tl_item_audio = append_clip_to_timeline(
        clip=clip_audio,
        media_type=2,
        start_frame=24,
        end_frame=24+60,
        pos_timecode="01:00:06:00"
    )
    solid_color = insert_generator_into_timeline(
        timeline=timeline, generator_name=drc.GENERATOR_SOLID_COLOR
    )
    tl_item_fusion_comp, fusion_comp =\
        append_fusion_composition_to_timeline(
            num_of_frame=24,
            pos_timecode="01:00:14:00"
        )

    bg1 = add_fusion_node(comp=fusion_comp, name="Background")
    set_topleft_color(node=bg1, rgba=[0.18, 0.18, 0.18, 1.0])

    circle_fg = add_fusion_node(comp=fusion_comp, name="Background")
    set_topleft_color(node=circle_fg, rgba=[0.8, 0.05, 0.05, 1.0])

    circle_merge = add_fusion_node(comp=fusion_comp, name="Merge")
    print(circle_merge)

    media_out = get_media_out(comp=fusion_comp)
    connect_node(a=bg1, b=media_out)

    print(dir(bg1))

    # ###################
    # # encode
    # ###################
    # preset_path = str(
    #     Path("./render_presets/h265_main10_444_qp-0.xml").resolve()
    # )
    # # preset_path = None

    # format_extension = drc.OUT_FILE_EXTENSTION_MOV
    # # codec = drc.CODEC_H265_NVIDIA
    # codec = drc.CODEC_APPLE_PRORES_422_HQ
    # # format_extension = drc.OUT_FILE_EXTENSTION_EXR
    # # codec = drc.CODEC_EXR_RGB_HALF
    # output_fname = "./render_out/dummy_out" + "." + format_extension
    # target_dir = str(Path(output_fname).resolve().parent)
    # custom_name = str(Path(output_fname).resolve().name)

    # render_settings = {
    #     # "SelectAllFrames": True,
    #     # "MarkIn": _timecode_to_frame_index("01:00:00:00"),
    #     # "MarkOut": _timecode_to_frame_index("01:00:08:12"),
    #     "TargetDir": target_dir,
    #     "CustomName": custom_name,
    #     # "UniqueFilenameStyle": drc.UNIQUE_FILENAME_STYLE_SUFFIX,
    #     # "ExportVideo": True,
    #     # "ExportAudio": True,
    #     # "FormatWidth": 3840,
    #     # "FormatHeight": 2160,
    #     # "FrameRate": 23.976,
    #     # "PixelAspectRatio": "square",
    #     # "VideoQuality": drc.VIDEO_QUALITY_AUTOMATIC,
    #     # "AudioCodec": drc.AUDIO_CODEC_LINEAR_PCM,
    #     # "AudioBitDepth": drc.AUDIO_BIT_DEPTH_24,
    #     # "AudioSampleRate": drc.AUDIO_SAMPLE_RATE_480,
    #     # "ColorSpaceTag": "Same as Project",
    #     # "GammaTag": "Same as Project",
    #     # "ExportAlpha": False,
    #     # "EncodingProfile": "Main10",
    #     # "MultiPassEncode": True,
    #     # "AlphaMode": 
    #     # "NetworkOptimization": True,
    #     # "ClipStartFrame": 0,
    #     # "TimelineStartTimecode": "01:00:00:00",
    #     # "ReplaceExistingFilesInPlace": True,
    # }

    # if preset_path is not None:
    #     import_render_preset(preset_path=preset_path)
    # else:
    #     set_render_format_codec_settings(format=format_extension, codec=codec)

    # set_render_settings(setting_dict=render_settings)
    # project.AddRenderJob()
    # project.StartRendering()
    # project.DeleteAllRenderJobs()
