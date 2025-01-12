# -*- coding: utf-8 -*-

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
    project_manager = get_project_manager()
    project = project_manager.GetCurrentProject()

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
    project_manager = get_project_manager()
    project = project_manager.GetCurrentProject()

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
        msg = 'setup_project_settings() was failed. '
        msg += 'Please check your "params" parameters.'
        raise TyResolveModuleError(project, msg)

    return is_success


def get_project_resolution():
    width = get_project_setting(name="timelineResolutionWidth")
    height = get_project_setting(name="timelineResolutionHeight")

    return [int(width), int(height)]


def set_timeline_settings(timeline, params):
    """
    set timeline settings from the dictionary type parameters.

    Parameters
    ----------
    timeline : Timeline
        a Timeline instance
    parames : dict
        dictionary type parameters
    """
    is_success = True
    print("Now this script is setting the timeline settings...")
    result = set_timeline_setting(
        timeline=timeline, name="useCustomSettings", value="1"
    )
    if result is False:
        is_success = False

    for name, value in params.items():
        # ignore settings for "Input" because they are not exist.
        if name == "colorSpaceInput" or name == "colorSpaceInputGamma":
            continue
        result = set_timeline_setting(
            timeline=timeline, name=name, value=value
        )
        if result is False:
            is_success = False

    if is_success is False:
        msg = 'set_timeline_settings() was failed. '
        msg += 'Please check your "params" parameters.'
        raise TyResolveModuleError(project, msg)

    return is_success


def set_timeline_setting(timeline, name, value):
    """
    Parameters
    ----------
    timeline : Timeline
        A Timeline
    name : str
        A project setting name
    value : str
        A project setting value

    Returns
    -------
    Returns True if successful, and False otherwise.
    """
    result = timeline.SetSetting(name, value)
    if result:
        print(f'    Timeline.SetSetting("{name}", "{value}") -> Success')
    else:
        print(f'    Timeline.SetSetting("{name}", "{value}") -> Failed')

    return result


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
def get_current_timeline():
    timeline = get_current_project().GetCurrentTimeline()

    return timeline


@log_return_value
def get_timeline_items_in_track(timeline, track_type="video", track_idx=1):
    """
    Parameters
    ----------
    track_type : str
        The track type.
        "video", "audio" or "subtitle".
    track_idx : int
        The track index. It starts from 1.
    """
    timeline_item_list = timeline.GetItemListInTrack(track_type, track_idx)

    if timeline_item_list is None:
        msg = 'Failed to get_timeline_items_in_track(). '
        msg += 'Please check if the input argument is correct.'
        raise TyResolveModuleError(False, msg)

    return timeline_item_list


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


def _create_dummy_video_relative_path():
    fps_str = int(get_project_setting(name="timelineFrameRate"))
    width = int(get_project_setting(name="timelineResolutionWidth"))
    height = int(get_project_setting(name="timelineResolutionHeight"))
    dummy_video_path = f"./videos/dummy_video_{width}x{height}_{fps_str}P.mp4"

    return dummy_video_path


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
    dummy_video_path = _create_dummy_video_relative_path()
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
    get_comp_tool_by_name(comp=fusion_comp, name="MediaIn1").Delete()

    return timeline_item, fusion_comp


############################################
# Fusion Page
############################################
@log_return_value
def get_comp_tool_by_name(comp, name):
    """
    Parameters
    ----------
    comp : Composition
        The composition
    name : str
        The tool name.
    
    Returns
    -------
    
    """
    tool = None
    for _, value in comp.GetToolList().items():
        if value.Name == name:
            tool = value
            pass

    if tool is None:
        msg = 'Failed to get_media_out() '
        msg += 'Please check if the media_out tool name is "MediaOut1".'
        raise TyResolveModuleError(tool, msg)

    return tool


@log_return_value
def add_comp_tool(comp, name, pos=(2, 3)):
    """
    Parameters
    ----------
    comp : Composition
        The composition
    name : str
        The tool name.
    
    Returns
    -------
    """
    tool = comp.AddTool(name, pos[0], pos[1])

    if tool is None:
        msg = 'Failed to add_comp_tool() '
        msg += f'Please check if "name" = {name} is correct.'
        raise TyResolveModuleError(tool, msg)
    
    return tool


@log_return_value
def connect_tool(a, b):
    """
    Connect a to b.
    """
    result = b.Input.ConnectTo(a.Output)

    if result is not True:
        msg = 'Failed to connect_tool() '
        raise TyResolveModuleError(result, msg)

    return result


@log_return_value
def connect_merge_tool(merge_tool, bg_tool, fg_tool):
    if bg_tool is not None:
        merge_tool.ConnectInput("Background", bg_tool)
    if fg_tool is not None:
        merge_tool.ConnectInput("Foreground", fg_tool)


@log_return_value
def set_tool_input(tool, name, value):
    tool.SetInput(name, value)

    # verify
    tolerance = 1e-6
    verify_value = tool.GetInput(name)

    if isinstance(value, float):
        if abs(value - verify_value) > tolerance:
            msg = 'Failed to set_tool_input()\n'
            msg += f"Verification failed for {name}: expected {value}, got {verify_value}"
            raise TyResolveModuleError(False, msg)
    elif isinstance(value, str) or isinstance(value, int):
        if value != verify_value:
            msg = 'Failed to set_tool_input()\n'
            msg += f"Verification failed for {name}: expected {value}, got {verify_value}"
            raise TyResolveModuleError(False, msg)
    elif (type(value).__name__ == "PyRemoteObject")\
            and (type(value).__module__ == "BlackmagicFusion"):
        pass
    else:
        msg = 'Unknown type to the set_tool_input()\n'
        msg += f"print(type({value})) = {type(value)}"
        raise TyResolveModuleError(False, msg)

    return True


def set_multiple_tool_input(tool, input_dict):
    for name, value in input_dict.items():
        set_tool_input(tool=tool, name=name, value=value)


def set_tool_topleft_color(tool, rgba=[0.18, 0.18, 0.18, 1.0]):
    channels = ["Red", "Green", "Blue", "Alpha"]
    for channel, value in zip(channels, rgba):
        set_tool_input(tool=tool, name=f"TopLeft{channel}", value=value)
    return True


@log_return_value
def set_tool_position(comp, tool, pos=(1, 1)):
    flow = comp.CurrentFrame.FlowView
    flow.SetPos(tool, pos[0], pos[1])

    # verify
    tolrerance = 0.1
    verify_pos = flow.GetPosTable(tool).values()
    is_same_value = all(abs(a - b) < tolrerance for a, b in zip(pos, verify_pos))

    if is_same_value is not True:
        msg = 'Failed to set_tool_position()'
        raise TyResolveModuleError(is_same_value, msg)

    return is_same_value


def _get_font_list():
    font_list = fusion.FontManager
    return font_list.GetFontList()


@log_return_value
def is_font_available(family, font_weight):
    """
    family : str
        A font family.
        Examples are "Nomo Sans Mono", "Barlow Condensed", and so on.
    font_weight : str
        A font weight.
        Examples are "Light", "Regular", "Bold", and so on.
    """
    fonts = _get_font_list()
    is_available = family in fonts and font_weight in fonts[family]

    if is_available is not True:
        font_path = str(Path("./fonts").resolve())
        msg = f'Required font "{family} - {font_weight}" is not found.\n'
        msg += f'Please install "{family}" font in the {font_path}'
        raise TyResolveModuleError(is_available, msg)

    return is_available


#####################
# Debug
#####################

def dump_tool_input_value(tool):
    print("=" * 80)
    print(f" {tool.Name} InputValue List")
    print("=" * 80)
    for key, value in tool.GetInputList().items():
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
            msg = f"{aa_input[idx]["name"]}: "
            msg += f"{aa_input[idx]["value"]}, "
            msg += f"{bb_input[idx]["value"]}, "
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


def debug_code():
    print("Debug Start")
    target_track_name = "dummy_video_1920x1080_24P.mp4"
    timeline = get_current_timeline()
    timeline_item_list = get_timeline_items_in_track(
        timeline=timeline, track_type="video", track_idx=1
    )
    
    for timeline_item in timeline_item_list:
        if timeline_item.GetName() == target_track_name:
            break

    fusion_comp = timeline_item.GetFusionCompByIndex(1)
    merge_tool = get_comp_tool_by_name(comp=fusion_comp, name="Merge1")
    print(merge_tool)
    dump_tool_input_value(tool=merge_tool)
    dump_tool_main_input_value(tool=merge_tool)
    text = get_comp_tool_by_name(comp=fusion_comp, name="Text1")
    dump_tool_main_input_value(tool=text)
    dump_tool_input_value(tool=merge_tool)

    text_base = add_comp_tool(comp=fusion_comp, name="TextPlus", pos=(10, 10))
    compare_tool_input_value(aa=text, bb=text_base)

    # is_font_available(family="Noto Sans Mono", font_weight="Black")

    import sys
    sys.exit(0)


#####################
# Logic
#####################
class HeightBasedSize:
    def __init__(self, size, hv_same=False, resolution=None):
        """
        Parameters
        ----------
        size: float
            A size parameter
        resolution : list or tuple
            [width, height] or (width, height)
        """
        if resolution is None:
            width, height = get_project_resolution()
        else:
            width, height = resolution
        self._v_size = size

        if hv_same:
            self._h_size = size
        else:    
            self._h_size = (self._v_size * height) / width

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
        resolution : list or tuple
            [width, height] or (width, height)
        """
        self.cd_circle_ll = HeightBasedSize(0.61)
        self.cd_circle_mm = HeightBasedSize(0.54)
        self.cd_circle_ss = HeightBasedSize(0.525)
        self.cd_line_width = HeightBasedSize(0.005)
        self.cd_font_size = HeightBasedSize(0.85)
        self.cross_line_width = self.cd_line_width
        self.cross_line_color = [235/255, 235/255, 235/255, 1.0]
        self.info_area_height = HeightBasedSize(0.1)


def create_background_circle(
        comp, bg_rgba=[0.0, 0.0, 0.0, 1.0],
        size=[0.45, 0.45], merge_pos=(1, 1)
    ):
    """
    Returns
    -------
    Merge
        A output merge tool
    """
    circle_mask = add_comp_tool(
        comp=comp, name="EllipseMask", pos=(merge_pos[0], merge_pos[1] - 2)
    )
    circle_mask_input = {
        "Width": size[0],
        "Height": size[1],
    }
    set_multiple_tool_input(tool=circle_mask, input_dict=circle_mask_input)

    circle_bg = add_comp_tool(
        comp=comp, name="Background", pos=(merge_pos[0], merge_pos[1] - 1)
    )
    circle_bg_input = {
        "TopLeftRed": bg_rgba[0],
        "TopLeftGreen": bg_rgba[1],
        "TopLeftBlue": bg_rgba[2],
        "TopLeftAlpha": bg_rgba[3],
        "EffectMask": circle_mask,
    }
    set_multiple_tool_input(tool=circle_bg, input_dict=circle_bg_input)

    merge = add_comp_tool(
        comp=comp, name="Merge", pos=(merge_pos[0], merge_pos[1] + 0)
    )
    connect_merge_tool(merge_tool=merge, bg_tool=None, fg_tool=circle_bg)

    return merge


# def draw_line(comp, color, width, height, angle=0, base_pos=[0, 0]):



def create_still_background_comp(comp, ppp: FusionParams, tool_pos):
    """
    Parameters
    ----------
    comp : Composition
        A fusion Composition instance
    ppp : FusionParams
        A parameter set for fusion composition
    tool_pos : list
        [h_pos, v_pos] of the base tool (lower left)
    """
    x_pos = tool_pos[0]
    y_pos = tool_pos[1]

    bg1 = add_comp_tool(
        comp=comp, name="Background", pos=(x_pos+0, y_pos+0)
    )
    set_tool_topleft_color(tool=bg1, rgba=[0.18, 0.18, 0.18, 1.0])

    cross_h_line = add_comp_tool(comp=comp, name="RectangleMask", pos=[x_pos+1, y_pos-2])
    cross_h_line_fg = add_comp_tool(comp=comp, name="Background", pos=[x_pos+1, y_pos-1])
    cross_h_line_merge = add_comp_tool(comp=comp, name="Merge", pos=[x_pos+1, y_pos+0])

    cross_v_line = add_comp_tool(comp=comp, name="RectangleMask", pos=[x_pos+2, y_pos-2])
    cross_v_line_fg = add_comp_tool(comp=comp, name="Background", pos=[x_pos+2, y_pos-1])
    cross_v_line_merge = add_comp_tool(comp=comp, name="Merge", pos=[x_pos+2, y_pos+0])

    large_white_circle_merge = create_background_circle(
        comp=comp, bg_rgba=[0.5, 0.5, 0.5, 1.0],
        size=[ppp.cd_circle_ll.h_size, ppp.cd_circle_ll.h_size],
        merge_pos=[x_pos+3, y_pos+0]
    )
    middle_black_circle_merge = create_background_circle(
        comp=comp, bg_rgba=[0.0, 0.0, 0.0, 1.0],
        size=[ppp.cd_circle_mm.h_size, ppp.cd_circle_mm.h_size],
        merge_pos=[x_pos+4, y_pos+0]
    )
    small_grey_circle_merge = create_background_circle(
        comp=comp, bg_rgba=[0.18, 0.18, 0.18, 1.0],
        size=[ppp.cd_circle_ss.h_size, ppp.cd_circle_ss.h_size],
        merge_pos=[x_pos+5, y_pos+0]
    )
    h_line = add_comp_tool(comp=comp, name="RectangleMask", pos=[x_pos+6, y_pos-2])
    h_line_fg = add_comp_tool(comp=comp, name="Background", pos=[x_pos+6, y_pos-1])
    h_line_merge = add_comp_tool(comp=comp, name="Merge", pos=[x_pos+6, y_pos+0])

    v_line = add_comp_tool(comp=comp, name="RectangleMask", pos=[x_pos+7, y_pos-2])
    v_line_fg = add_comp_tool(comp=comp, name="Background", pos=[x_pos+7, y_pos-1])
    v_line_merge = add_comp_tool(comp=comp, name="Merge", pos=[x_pos+7, y_pos+0])

    cross_h_line_input = {
        "Width": 1.0,
        "Height": ppp.cross_line_width.h_size,
    }
    set_multiple_tool_input(tool=cross_h_line, input_dict=cross_h_line_input)
    cross_h_line_fg_input = {
        "TopLeftRed": ppp.cross_line_color[0],
        "TopLeftGreen": ppp.cross_line_color[1],
        "TopLeftBlue": ppp.cross_line_color[2],
        "TopLeftAlpha": ppp.cross_line_color[3],
        "EffectMask": cross_h_line,
    }
    set_multiple_tool_input(tool=cross_h_line_fg, input_dict=cross_h_line_fg_input)

    cross_v_line_input = {
        "Width": 1.0,
        "Height": ppp.cross_line_width.h_size,
        "Angle": 90, 
    }
    set_multiple_tool_input(tool=cross_v_line, input_dict=cross_v_line_input)
    cross_v_line_fg_input = {
        "TopLeftRed": ppp.cross_line_color[0],
        "TopLeftGreen": ppp.cross_line_color[1],
        "TopLeftBlue": ppp.cross_line_color[2],
        "TopLeftAlpha": ppp.cross_line_color[3],
        "EffectMask": cross_v_line,
    }
    set_multiple_tool_input(tool=cross_v_line_fg, input_dict=cross_v_line_fg_input)

    h_line_input = {
        "Width": ppp.cd_circle_ll.h_size,
        "Height": ppp.cd_line_width.h_size,
    }
    set_multiple_tool_input(tool=h_line, input_dict=h_line_input)
    h_line_fg_input = {
        "TopLeftRed": 0.0,
        "TopLeftGreen": 0.0,
        "TopLeftBlue": 0.0,
        "TopLeftAlpha": 1.0,
        "EffectMask": h_line,
    }
    set_multiple_tool_input(tool=h_line_fg, input_dict=h_line_fg_input)

    v_line_input = {
        "Width": ppp.cd_circle_ll.h_size,
        "Height": ppp.cd_line_width.h_size,
        "Angle": 90,
    }
    set_multiple_tool_input(tool=v_line, input_dict=v_line_input)
    v_line_fg_input = {
        "TopLeftRed": 0.0,
        "TopLeftGreen": 0.0,
        "TopLeftBlue": 0.0,
        "TopLeftAlpha": 1.0,
        "EffectMask": v_line,
    }
    set_multiple_tool_input(tool=v_line_fg, input_dict=v_line_fg_input)

    connect_merge_tool(
        merge_tool=cross_h_line_merge,
        bg_tool=bg1, fg_tool=cross_h_line_fg
    )

    connect_merge_tool(
        merge_tool=cross_v_line_merge,
        bg_tool=cross_h_line_merge, fg_tool=cross_v_line_fg
    )

    connect_merge_tool(
        merge_tool=large_white_circle_merge,
        bg_tool=cross_v_line_merge, fg_tool=None
    )
    connect_merge_tool(
        merge_tool=middle_black_circle_merge,
        bg_tool=large_white_circle_merge, fg_tool=None
    )
    connect_merge_tool(
        merge_tool=small_grey_circle_merge,
        bg_tool=middle_black_circle_merge, fg_tool=None
    )
    connect_merge_tool(
        merge_tool=h_line_merge,
        bg_tool=small_grey_circle_merge, fg_tool=h_line_fg
    )
    connect_merge_tool(
        merge_tool=v_line_merge,
        bg_tool=h_line_merge, fg_tool=v_line_fg
    )

    out_tool = v_line_merge

    return out_tool


def create_countdown_animation_comp(
    comp, ppp: FusionParams, count_str, fps, tool_pos
):
    """
    Parameters
    ----------
    comp : Composition
        A fusion Composition instance
    ppp : FusionParams
        A parameter set for fusion composition
    count_str : int
        A number indicate the countdown
    fps : int
        framerate
    tool_pos : list
        [h_pos, v_pos] of the base tool (lower left)
    """
    x_pos = tool_pos[0]
    y_pos = tool_pos[1]

    radial_wipe = add_comp_tool(
        comp=comp, name="EllipseMask", pos=(x_pos+0, y_pos-3)
    )
    wipe_circle_mask = add_comp_tool(
        comp=comp, name="EllipseMask", pos=(x_pos+0, y_pos-2)
    )
    wipe_circle_fg = add_comp_tool(
        comp=comp, name="Background", pos=(x_pos+0, y_pos-1)
    )
    wipe_circle_merge = add_comp_tool(
        comp=comp, name="Merge", pos=(x_pos+0, y_pos-0)
    )

    # wipe animation settings
    radial_wipe_input = {
        "Invert": 1.0,
        "BorderWidth": 1.0,
        "Solid": 0.0,
        "CapStyle": 0.0,
        "Width": 1.0,
        "Height": 1.0,
        "Angle": 90,
    }
    set_multiple_tool_input(tool=radial_wipe, input_dict=radial_wipe_input)
    radial_wipe["WriteLength"] = comp.BezierSpline()
    radial_wipe["WriteLength"][0] = 1.0
    radial_wipe["WriteLength"][fps] = 0.0

    # mask settings for wipe animation
    wipe_circle_mask_input = {
        "Invert": 1.0,
        "Width": ppp.cd_circle_ss.h_size,
        "Height": ppp.cd_circle_ss.h_size,
        "PaintMode": "Subtract",
        "EffectMask": radial_wipe,
    }
    set_multiple_tool_input(
        tool=wipe_circle_mask, input_dict=wipe_circle_mask_input
    )

    # color settings for wipe animation
    wipe_circle_fg_input = {
        "TopLeftRed": 0.0,
        "TopLeftGreen": 0.0,
        "TopLeftBlue": 0.0,
        "TopLeftAlpha": 1.0,
        "EffectMask": wipe_circle_mask,
    }
    set_multiple_tool_input(
        tool=wipe_circle_fg, input_dict=wipe_circle_fg_input
    )

    # text
    countdown_text = add_comp_tool(
        comp=comp, name="TextPlus", pos=(x_pos+1, y_pos-1)
    )
    countdown_text_merge = add_comp_tool(
        comp=comp, name="Merge", pos=(x_pos+1, y_pos-0)
    )

    font_family = "Noto Sans Mono"
    font_weight = "Black"
    countdown_text_input = {
        "StyledText": f"{count_str}",
        "Font": font_family,
        "Style": font_weight,
        "Size": ppp.cd_font_size.h_size,
        "Red1": 0.5,
        "Green1": 0.5,
        "Blue1": 0.5,
    }
    is_font_available(family=font_family, font_weight=font_weight)
    set_multiple_tool_input(
        tool=countdown_text, input_dict=countdown_text_input
    )

    # connect
    connect_merge_tool(
        merge_tool=countdown_text_merge,
        bg_tool=wipe_circle_merge, fg_tool=countdown_text
    )
    connect_merge_tool(
        merge_tool=wipe_circle_merge,
        bg_tool=None, fg_tool=wipe_circle_fg
    )

    # output
    input_merge = wipe_circle_merge
    output_merge = countdown_text_merge

    return input_merge, output_merge


def create_countdown_comp():
    ppp = FusionParams()
    fps = int(get_project_setting(name="timelineFrameRate"))
    for idx, countdown_str in enumerate([3, 2, 1, 0]):
        tl_item_fusion_comp, comp =\
            append_fusion_composition_to_timeline(
                num_of_frame=fps,
                pos_timecode=f"01:00:{idx:02d}:00"
            )
        create_countdown_comp_each_sec(
            comp=comp, ppp=ppp, fps=fps, count_str=countdown_str)
        break


def create_countdown_comp_each_sec(comp, ppp, fps=24, count_str=3):
    """
    Parameters
    ----------
    comp : Composition
        A fusion Composition instance
    ppp : FusionParams
        A parameter set for fusion composition
    fps : int
        framerate
    count_str : int
        A character indicate the number of the count down.
    """
    comp.Lock()

    still_bg_tool = create_still_background_comp(
        comp=comp, ppp=ppp, tool_pos=(1, 3)
    )
    cntdown_anime_input_merge, cntdown_anime_output_merge\
        = create_countdown_animation_comp(
            comp=comp, ppp=ppp, count_str=count_str, fps=fps, tool_pos=(12, 3)
        )

    # connect
    media_out = get_comp_tool_by_name(comp=comp, name="MediaOut1")
    set_tool_position(comp=comp, tool=media_out, pos=(16, 3))

    connect_tool(cntdown_anime_output_merge, media_out)
    connect_merge_tool(
        merge_tool=cntdown_anime_input_merge,
        bg_tool=still_bg_tool, fg_tool=None
    )

    comp.Unlock()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # debug_code()

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
        "isAutoColorManage": "0",
        "rcmPresetMode": "Custom",
        "separateColorSpaceAndGamma": "1",
        "colorSpaceInput": "Rec.709",
        "colorSpaceInputGamma": "Gamma 2.4",
        "colorSpaceTimeline": "Rec.709",
        "colorSpaceTimelineGamma": "Gamma 2.4",
        "colorSpaceOutput": "Rec.709",
        "colorSpaceOutputGamma": "Gamma 2.4",
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
    # save_project()
    # close_current_project()
    # project = load_project(project_name=project_name)

    # set up the project settings
    setup_project_settings(params=project_settings_params)

    ###########################
    # Add files to the timeline
    ###########################
    # create timelines
    timeline = create_empty_timeline(name="My_Timeline")

    ####################################################
    # # Temporarily commented out because it is slow...
    ####################################################
    # set_timeline_settings(timeline=timeline, params=project_settings_params)

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

    # clip_hdr = add_file_to_media_pool(file_path=file_path_list[0])
    # clip_sdr = add_file_to_media_pool(
    #     file_path=file_path_list[1], start_frame=24, end_frame=71
    # )
    # clip_seq = add_seq_file_to_media_pool(
    #     file_path=file_path_list[2], start_idx=120, end_idx=179
    # )
    # clip_audio = add_file_to_media_pool(file_path=file_path_list[3])

    # # add clips to the timeline
    # append_clip_to_timeline(clip=clip_hdr)
    # append_clip_to_timeline(clip=clip_sdr)
    # append_clip_to_timeline(
    #     clip=clip_seq, media_type=1, pos_timecode="01:00:06:00")
    # tl_item_audio = append_clip_to_timeline(
    #     clip=clip_audio,
    #     media_type=2,
    #     start_frame=24,
    #     end_frame=24+60,
    #     pos_timecode="01:00:06:00"
    # )
    # solid_color = insert_generator_into_timeline(
    #     timeline=timeline, generator_name=drc.GENERATOR_SOLID_COLOR
    # )

    create_countdown_comp()
    set_current_timecode(timecode="01:00:00:00")

    open_page(page_name=drc.EDIT_PAGE_STR)

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
