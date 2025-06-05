# -*- coding: utf-8 -*-

import sys
import os
from pathlib import Path
import functools
import time
import subprocess

from ty_davinci_constants import EDIT_PAGE_STR, FUSION_PAGE_STR

if sys.platform == "darwin":  # macOS
    resolve_script_api = (
        "/Library/Application Support/Blackmagic Design/DaVinci Resolve/"
        "Developer/Scripting"
    )
    sys.path.append(os.path.join(resolve_script_api, "Modules"))
    resolve_lut_path = \
        "/Library/Application Support/Blackmagic Design/DaVinci Resolve/LUT/"
elif sys.platform == "win32":  # Windows
    resolve_script_api = os.path.expandvars(
        r"%PROGRAMDATA%\Blackmagic Design\DaVinci Resolve\Support"
        r"\Developer\Scripting"
    )
    sys.path.append(os.path.join(resolve_script_api, "Modules"))
    resolve_lut_path = os.path.expandvars(
        r"%PROGRAMDATA%\Blackmagic Design\DaVinci Resolve\Support\LUT"
    )
elif sys.platform == "linux":  # Linux
    resolve_script_api = "/opt/resolve/Developer/Scripting"
    sys.path.append(os.path.join(resolve_script_api, "Modules"))
    resolve_lut_path = "/home/resolve/LUT"
    ValueError("Linux platform is not supported")

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
# DEBUG_ON = True
DEBUG_ON = False


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


def reboot_resolve():
    global resolve
    global fusion
    # OS ごとのコマンド設定
    if sys.platform == "win32":  # Windows
        kill_cmd = r'Get-Process | Where-Object { $_.ProcessName -eq "Resolve" } | Stop-Process -Force'
        launch_cmd = r'C:\Program Files\Blackmagic Design\DaVinci Resolve\Resolve.exe'
    elif sys.platform == "darwin":  # macOS
        kill_cmd = "osascript -e 'tell application \"/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/MacOS/Resolve\" to quit'"
        launch_cmd = r'/Applications/DaVinci\ Resolve/DaVinci\ Resolve.app/Contents/MacOS/Resolve'
    else:
        raise OSError("Unsupported platform")

    # 1. 現在起動中の Resolve を終了する
    try:
        if sys.platform == "win32":
            subprocess.run(["powershell", "-Command", kill_cmd], check=True)
        elif sys.platform == "darwin":
            subprocess.run(kill_cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Unexpected error has occured during the launch of Resolve:", e)

    # 2. 終了の完了を待つ（約3秒）
    time.sleep(3)

    # 3. Resolve を起動する
    try:
        if sys.platform == "win32":
            subprocess.Popen(launch_cmd)
        elif sys.platform == "darwin":
            subprocess.Popen(launch_cmd, shell=True)
    except Exception as e:
        print("Failed to launch Resolve:", e)
        raise

    # 4. Resolve 起動のために約10秒待つ
    time.sleep(10)

    # 5. dvr_script を使って Resolve に接続、接続できなければリトライ（最大5回、各回2秒待ち）
    max_retries = 10
    for attempt in range(max_retries):
        try:
            resolve = dvr_script.scriptapp("Resolve")
            if resolve is None:
                raise ConnectionError("The DaVinci Resolve app is not running.")
            # 正常に接続できたらループを抜ける
            print("Successed to launch Resolve")
            fusion = resolve.Fusion()
            break
        except ConnectionError as e:
            print(f"Failed to launch Resolve ({attempt+1}/{max_retries}): {e}")
            time.sleep(2)
    else:
        # 最大リトライ回数に達した場合
        raise ConnectionError("Failed to connect to DaVinci Resolve after several retries.")


# =============================
# System
# =============================
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
        raise TyResolveModuleError(ret_val, msg)

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
def get_current_page():
    """
    Returns
    -------
    str
        A current page name.
        The return value is one of the following strings:
            * "media"
            * "cut"
            * "edit"
            * 'fusion'
            * "color"
            * "deliver"
    """
    current_page = resolve.GetCurrentPage()

    if current_page is None:
        msg = 'Failed to get current page. '
        msg += 'Please verify that the DaVinci Resolve\'s project is opend.'
        raise TyResolveModuleError(current_page, msg)

    return current_page

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
        raise TyResolveModuleError(ret_val, msg)

    return ret_val


@log_return_value
def refresh_lut_list():
    project = get_current_project()
    ret_val = project.RefreshLUTList()

    if ret_val is not True:
        msg = 'Failed to call "refresh_lut_list()"'
        raise TyResolveModuleError(ret_val, msg)

    return ret_val


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
        raise TyResolveModuleError(timeline, msg)

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
            print(f"    {name} = {value} was failed.")

    if is_success is False:
        msg = 'setup_project_settings() was failed. '
        msg += 'Please check your "params" parameters.'
        raise TyResolveModuleError(is_success, msg)

    return is_success


def get_project_resolution():
    width = get_project_setting(name="timelineResolutionWidth")
    height = get_project_setting(name="timelineResolutionHeight")

    return [int(width), int(height)]


def make_videoMonitorFormat_str(width, height, framerate):
    if str(width) == "1280":
        prefix = "HD"
        height = 720
    elif str(width) == "1920":
        prefix = "HD"
    elif str(width) == "2560":
        prefix = "HD"
        height = 1080
    elif str(width) == "2048":
        prefix = "2K"
    elif str(width) == "3840":
        prefix = "UHD"
    elif str(width) == "4096":
        prefix = "4K"
    else:
        raise ValueError("invalid width parameter")
    
    out_str = f"{prefix} {height}p {framerate}"

    return out_str


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
            if name == "timelineFrameRate" and value == "29.97":
                pass
            elif name == "timelineFrameRate" and value == "59.94":
                pass
            else:
                is_success = False

    if is_success is False:
        msg = 'set_timeline_settings() was failed. '
        msg += 'Please check your "params" parameters.'
        raise TyResolveModuleError(is_success, msg)

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
        clip, pos_frame_idx=None,
        start_frame=None, end_frame=None, media_type=None, track_index=1):
    """
    Parameters
    ----------
    clip : MediaPoolItem
        clip
    pos_frame_idx : float
        clip start position.
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

    if pos_frame_idx is not None:
        frame_idx = pos_frame_idx
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


def sec_to_frame_idx(sec: float) -> float:
    fps = int(round(get_project_setting(name="timelineFrameRate")))
    frame_idx = fps * sec

    return frame_idx


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


def timecode_to_frame_index(timecode: str, fps_float: float):
    fps = int(round(fps_float))

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


def is_rendering_in_progress():
    projectManager = resolve.GetProjectManager()
    project = projectManager.GetCurrentProject()
    if not project:
        return False

    return project.IsRenderingInProgress()


def run_rendering_and_wait_until_finish(project):
    project.AddRenderJob()
    project.StartRendering()
    project.DeleteAllRenderJobs()
    while is_rendering_in_progress():
        time.sleep(1)
    return


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
    fps_float = float(get_project_setting(name="timelineFrameRate"))
    fps = int(fps_float) if fps_float.is_integer() else fps_float
    width = int(get_project_setting(name="timelineResolutionWidth"))
    height = int(get_project_setting(name="timelineResolutionHeight"))
    directory = Path(__file__).resolve().parent
    dummy_video_path = directory / f"videos/dummy_video_{width}x{height}_{fps}P.mp4"

    return str(dummy_video_path)


@log_return_value
def append_fusion_composition_to_timeline(
        num_of_frame: float, pos_frame_idx: float | None = None):
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
        pos_frame_idx=pos_frame_idx
    )
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
    for value in comp.GetToolList().values():
        if value.Name == name:
            tool = value
            pass

    if tool is None:
        msg = 'Failed to get_comp_tool_by_name() '
        msg += f'"{name}" was not found'
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
def connect_mediaout(mediaout, source):
    """
    Connect source to MediaOut.
    """
    result = mediaout.ConnectInput("Input", source)

    if result is not True:
        msg = 'Failed to connect_mediaout() '
        raise TyResolveModuleError(result, msg)

    return result


@log_return_value
def connect_dctl(dctl, source):
    """
    Connect source to dctl's input.
    """
    result = dctl.ConnectInput("Source", source)

    if result is not True:
        msg = 'Failed to connect_dctl() '
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
    elif isinstance(value, str) or isinstance(value, int) or isinstance(value, dict):
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
    if "Font" in input_dict and "Style" in input_dict:
        is_font_available(
            family=input_dict["Font"],
            font_weight=input_dict["Style"]
        )

    for name, value in input_dict.items():
        set_tool_input(tool=tool, name=name, value=value)


def set_tool_topleft_color(tool, rgba=[0.18, 0.18, 0.18, 1.0]):
    channels = ["Red", "Green", "Blue", "Alpha"]
    for channel, value in zip(channels, rgba):
        set_tool_input(tool=tool, name=f"TopLeft{channel}", value=value)
    return True


@log_return_value
def set_tool_position(comp, tool, pos=(1, 1)):
    current_page = get_current_page()
    if current_page != "fusion":
        # activate `comp.CurrentFrame`
        open_page(page_name="fusion")

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


def add_dctl_comp(comp, dctl_path, option=None, base_pos=[0, 0]):
    """
    Parameters
    ----------
    comp : Fusion Compsition
        A composition instance
    dctl_path : str
        A relative dctl path from the DaVinci Resolve's LUT path.
    option : dect
        Options
    base_pos : list
        Tool position on the node editor.

    Examples
    --------
    >>> dcl.add_dctl_comp(
    >>>     comp=comp, dctl_path="TY_DCTL/draw_45deg_lines.dctl",
    >>>     base_pos=[x_pos+1, y_pos-1],
    >>>     option={
    >>>         "sliderFloatParam0": ppp.frame_marker_outline_width,
    >>>         "sliderFloatParam1": ppp.ramp_height * 0.93,
    >>>         "sliderIntParam0": 4
    >>>     }
    >>> )    
    """
    x_pos = base_pos[0]
    y_pos = base_pos[1]

    dctl = add_comp_tool(
        comp=comp,
        name="ofx.com.blackmagicdesign.resolvefx.DCTL",
        pos=(x_pos+0, y_pos-0)
    )

    # check if the dctl file exist
    target_file = Path(resolve_lut_path) / Path(dctl_path)
    if not target_file.is_file():
        msg = f"{target_file} does not exist."
        raise TyResolveModuleError(False, msg)

    # modify delimitter based on the platform (OS)
    dctl_os_path = str(Path(dctl_path))

    dctl_input = {
        "DCTLs": dctl_os_path,
        "reloadDCTLButton": 1.0,
    }
    if option is not None:
        dctl_input.update(option)

    set_multiple_tool_input(tool=dctl, input_dict=dctl_input)

    return dctl


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
        font_path = str((Path(__file__).parent / "fonts").resolve())
        msg = f'\n    Required font "{family} - {font_weight}" is not found.\n'
        msg += f'    Please install "{family}" font in the {font_path}\n'
        msg += '    And reboot DaVinci Resolve to refresh the font list.'
        raise TyResolveModuleError(is_available, msg)

    return is_available


def add_transparent_background(comp, pos):
    """
    Create transparent background node.
    """
    x_pos = pos[0]
    y_pos = pos[1]
    base_bg = add_comp_tool(
        comp=comp, name="Background", pos=(x_pos, y_pos)
    )
    base_bg_input = {
        "TopLeftRed": 0.0,
        "TopLeftGreen": 0.0,
        "TopLeftBlue": 0.0,
        "TopLeftAlpha": 0.0,
    }
    set_multiple_tool_input(
        tool=base_bg, input_dict=base_bg_input
    )

    return base_bg


def add_line_comp(
        comp, rgba, width, height, angle=0, pos=[0, 0], connect_fg=True,
        center={ 1: 0.5, 2: 0.5, 3: 0.0 }):
    x_pos = pos[0]
    y_pos = pos[1]

    line = add_comp_tool(comp=comp, name="RectangleMask", pos=[x_pos, y_pos-2])
    line_fg = add_comp_tool(comp=comp, name="Background", pos=[x_pos, y_pos-1])
    line_merge = add_comp_tool(comp=comp, name="Merge", pos=[x_pos, y_pos+0])

    line_input = {
        "Width": width,
        "Height": height,
        "Angle": angle,
        "Center": center,
    }
    set_multiple_tool_input(tool=line, input_dict=line_input)
    line_fg_input = {
        "TopLeftRed": rgba[0],
        "TopLeftGreen": rgba[1],
        "TopLeftBlue": rgba[2],
        "TopLeftAlpha": rgba[3],
        "EffectMask": line,
    }
    set_multiple_tool_input(tool=line_fg, input_dict=line_fg_input)

    if connect_fg:
        connect_merge_tool(
            merge_tool=line_merge,
            bg_tool=None, fg_tool=line_fg
        )
    else:
        connect_merge_tool(
            merge_tool=line_merge,
            bg_tool=line_fg, fg_tool=None
        )

    return line_merge


def force_rcm_update_via_page_switch():
    """
    Force a refresh of DaVinci Resolve's Color Management (RCM) settings
    for Fusion compositions.

    Background:
        Under certain conditions, DaVinci Resolve fails to correctly apply RCM settings
        to Fusion compositions.
        This function acts as a workaround by performing a specific page transition
        to force the RCM settings to update.

    Process:
        1. Open the Fusion page (FUSION_PAGE_STR) where the RCM is properly applied.
        2. Wait for 0.1 seconds to allow the page transition to complete.
        3. Return to the Edit page (EDIT_PAGE_STR).

    Note:
        This workaround is implemented to address a potential bug in DaVinci Resolve
        and may become unnecessary if the underlying issue is resolved in future updates.
    """
    open_page(page_name=FUSION_PAGE_STR)
    time.sleep(0.1)
    open_page(page_name=EDIT_PAGE_STR)



if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
