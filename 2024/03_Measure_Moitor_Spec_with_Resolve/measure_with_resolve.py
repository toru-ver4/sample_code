# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
import re
import sys
import time
from pathlib import Path
from pathlib import Path
from datetime import datetime

# import third-party libraries
from ty_display_pro_hl import read_xyz, save_measure_result, \
    calculate_elapsed_seconds

# import my libraries
import ty_davinci_control_lib as dcl
from create_tp_for_measure import create_tp_base_name, create_cv_list, \
    create_cc_tp_fname

"""
import sys
import importlib
folder_path = 'C:/Users/toruv/OneDrive/work/sample_code/2024/03_Measure_Moitor_Spec_with_Resolve'
sys.path.append(folder_path)
import measure_with_resolve as mwr
importlib.reload(mwr)
mwr.duration_measure()
"""


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


CCSS_RGBLED = "C:/Users/toruv/OneDrive/work/sample_code/2024/03_Measure_Moitor_Spec_with_Resolve/ccss/RGBLEDFamily_07Feb11.ccss"


def create_project(project_name="Dummy Project"):
    project_manager = dcl.get_project_manager()
    project = dcl.initialize_project(
        project_manager=project_manager, project_name=project_name)
    return project, project_manager


def remove_all_timeline(project):
    num_of_timeline = project.GetTimelineCount()
    print(f"num_of_timeline = {num_of_timeline}")
    media_pool = project.GetMediaPool()

    timelines = []

    for idx in range(num_of_timeline):
        timeline = project.GetTimelineByIndex(idx + 1)
        timelines.append(timeline)

    if timelines != []:
        media_pool.DeleteTimelines(timelines)


def remove_csv(file_path):
    # Pathオブジェクトを作成
    path = Path(file_path)

    if path.is_file() and path.suffix == '.csv':
        path.unlink()
        print(f"{file_path} has been deleted.")
    elif not path.is_file():
        print(f"{file_path} does not exist.")
    else:
        print(f"{file_path} is not a CSV file.")


def frame_number_to_timecode(frame_number, fps, timecode_offset="01:00:00:00"):
    # timecode_offsetを時間、分、秒、フレームに分解
    hours, minutes, seconds, frames = map(int, timecode_offset.split(':'))

    # 総フレーム数を追加
    total_frames = frames + frame_number

    # 総フレーム数を秒、分、時間に変換
    added_seconds = total_frames // fps
    frames = total_frames % fps
    seconds += added_seconds
    minutes += seconds // 60
    seconds %= 60
    hours += minutes // 60
    minutes %= 60

    # 新しいTimecodeをフォーマット
    new_timecode = f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"
    return new_timecode


def set_project_settings_bt2100(project):
    project_params = dict(
        timelineResolutionWidth=dcl.PRJ_TIMELINE_RESOLUTION_1920,
        timelineResolutionHeight=dcl.PRJ_TIMELINE_RESOLUTION_1080,
        timelinePlaybackFrameRate=dcl.PRJ_TIMELINE_PLAYBACK_FRAMERATE_24,
        videoMonitorFormat=dcl.PRJ_VIDEO_MONITOR_FORMAT_HD_1080P24FPS,
        timelineFrameRate=dcl.PRJ_TIMELINE_FRAMERATE_24,
        videoMonitorUse444SDI=dcl.PRJ_PARAM_DISABLE,
        videoMonitorSDIConfiguration=dcl.PRJ_SDI_SINGLE_LINK,
        videoDataLevels=dcl.PRJ_VIDEO_DATA_LEVEL_LIMITED,
        videoMonitorUseHDROverHDMI=dcl.PRJ_PARAM_ENABLE,
        colorScienceMode=dcl.PRJ_COLOR_SCIENCE_MODE_RCM_ON,
        rcmPresetMode=dcl.PRJ_PRESET_MODE_CUSTOM,
        separateColorSpaceAndGamma=dcl.PRJ_PARAM_ENABLE,
        colorSpaceInput=dcl.PRJ_COLOR_SPACE_REC2020,
        colorSpaceInputGamma=dcl.PRJ_GAMMA_STR_ST2084,
        colorSpaceTimeline=dcl.PRJ_COLOR_SPACE_REC2020,
        colorSpaceTimelineGamma=dcl.PRJ_GAMMA_STR_ST2084,
        colorSpaceOutput=dcl.PRJ_COLOR_SPACE_REC2020,
        colorSpaceOutputGamma=dcl.PRJ_GAMMA_STR_ST2084,
        timelineWorkingLuminance=dcl.PRJ_WORKING_LUMINANCE_MAX,
        timelineWorkingLuminanceMode=dcl.PRJ_LUMINANCE_MODE_CUSTOM,
        inputDRT=dcl.PRJ_PARAM_NONE,
        outputDRT=dcl.PRJ_PARAM_NONE,
        hdrMasteringLuminanceMax="1000",
        hdrMasteringOn=dcl.PRJ_PARAM_ENABLE
    )
    dcl.set_project_settings_from_dict(project=project, params=project_params)


def duration_measure(
        csv_name="./measure_result.csv", ccss_file=None,
        color_mask=[1, 1, 1], patch_area_ratio=0.03):
    remove_csv(file_path=csv_name)

    project_name = "Measure_AW3225QF"
    dcl.close_and_remove_project(project_name=project_name)
    project, project_manager = create_project(project_name=project_name)
    dcl.open_page(dcl.EDIT_PAGE_STR)
    remove_all_timeline(project=project)
    set_project_settings_bt2100(project=project)

    media_path = Path('C:/Users/toruv/OneDrive/work/sample_code/2024/03_Measure_Moitor_Spec_with_Resolve/tp_img')

    # add test pattern for measure
    cv_list = create_cv_list(num_of_block=64)
    fname_list = [
        create_tp_base_name(
            color_mask=color_mask, cv=cv/1023,
            patch_area_ratio=patch_area_ratio) + ".png"
        for cv in cv_list]
    fname_list = [str(media_path / Path(fname)) for fname in fname_list]

    clip_list = dcl.add_files_to_media_pool(media_path=fname_list)
    tp_timeline = dcl.create_timeline_from_clip(
        clip_list=clip_list, timeline_name="TP_Timeline")

    # add all black pattern
    fname_black = "C:/Users/toruv/OneDrive/work/sample_code/2024/03_Measure_Moitor_Spec_with_Resolve/tp_img/black.png"
    clip_list = dcl.add_files_to_media_pool(media_path=fname_black)
    black_timeline = dcl.create_timeline_from_clip(
        clip_list=clip_list, timeline_name="Black_Timeline")

    # initialize with black
    project.SetCurrentTimeline(black_timeline)
    time.sleep(1)

    # set 1000 nits
    project.SetCurrentTimeline(tp_timeline)
    frame_number = 48
    timecode_str = frame_number_to_timecode(frame_number=frame_number, fps=24)
    tp_timeline.SetCurrentTimecode(timecode_str)

    measure_period_second = 60
    measure_step_second = 5
    num_of_measure = measure_period_second // measure_step_second
    for _ in range(num_of_measure):
        large_xyz, Yxy = read_xyz(flush=False, ccss_file=ccss_file)
        ccss_name = Path(ccss_file).stem if ccss_file else "-"
        save_measure_result(
            large_xyz=large_xyz, Yxy=Yxy,
            csv_name=csv_name, ccss_name=ccss_name)
        time.sleep(measure_step_second)

    project.SetCurrentTimeline(black_timeline)
    time.sleep(10)

    dcl.save_project(project_manager=project_manager)
    # dcl.close_current_project()
    # patch_area_str = f"{int(patch_area_ratio * 100):03d}"
    # archive_path = Path(
    #     f"./project_archive/duration_measure_{patch_area_str}-window.dra").resolve()
    # dcl.archive_project(
    #     project_name=project_name, archive_path=str(archive_path))


def increase_cv_measure(
        csv_name="./measure_result.csv", ccss_file=None,
        color_mask=[1, 1, 1], patch_area_ratio=0.03):
    remove_csv(file_path=csv_name)

    project_name = "Measure_AW3225QF"
    dcl.close_and_remove_project(project_name=project_name)
    project, project_manager = create_project(project_name=project_name)
    dcl.open_page(dcl.EDIT_PAGE_STR)
    remove_all_timeline(project=project)
    set_project_settings_bt2100(project=project)

    media_path = Path('C:/Users/toruv/OneDrive/work/sample_code/2024/03_Measure_Moitor_Spec_with_Resolve/tp_img')

    # add test pattern for measure
    cv_list = create_cv_list(num_of_block=64)
    fname_list = [
        create_tp_base_name(
            color_mask=color_mask, cv=cv/1023,
            patch_area_ratio=patch_area_ratio) + ".png"
        for cv in cv_list]
    fname_list = [str(media_path / Path(fname)) for fname in fname_list]

    clip_list = dcl.add_files_to_media_pool(media_path=fname_list)
    tp_timeline = dcl.create_timeline_from_clip(
        clip_list=clip_list, timeline_name="TP_Timeline")

    # add all black pattern
    fname_black = "C:/Users/toruv/OneDrive/work/sample_code/2024/03_Measure_Moitor_Spec_with_Resolve/tp_img/black.png"
    clip_list = dcl.add_files_to_media_pool(media_path=fname_black)
    black_timeline = dcl.create_timeline_from_clip(
        clip_list=clip_list, timeline_name="Black_Timeline")

    # initialize with black
    project.SetCurrentTimeline(black_timeline)
    time.sleep(1)

    cv_list = create_cv_list(num_of_block=64)
    for frame_idx in range(len(cv_list)):
        project.SetCurrentTimeline(tp_timeline)
        timecode_str = frame_number_to_timecode(frame_number=frame_idx, fps=24)
        tp_timeline.SetCurrentTimecode(timecode_str)
        time.sleep(0.5)
        large_xyz, Yxy = read_xyz(flush=False, ccss_file=ccss_file)
        ccss_name = Path(ccss_file).stem if ccss_file else "-"
        save_measure_result(
            large_xyz=large_xyz, Yxy=Yxy,
            csv_name=csv_name, ccss_name=ccss_name)
        project.SetCurrentTimeline(black_timeline)
        time.sleep(1)

    dcl.save_project(project_manager=project_manager)
    dcl.close_current_project()
    # patch_area_str = f"{int(patch_area_ratio * 100):03d}"
    # archive_path = Path(
    #     f"./project_archive/increment_measure_{patch_area_str}-window.dra").resolve()
    # dcl.archive_project(
    #     project_name=project_name, archive_path=str(archive_path))


def create_cc_patch_measure_result_fname(luminance, window_size):
    window_size_int = int(window_size * 100)
    fname = f"./AW3225QF/cc_measure_lumi-{luminance:04d}_"
    fname += f"win-{window_size_int:03d}.csv"

    return fname


def cc_patch_measure(luminance, window_size, ccss_file):
    csv_name = create_cc_patch_measure_result_fname(
        luminance=luminance, window_size=window_size)
    remove_csv(file_path=csv_name)

    project_name = "Measure_AW3225QF"
    dcl.close_and_remove_project(project_name=project_name)
    project, project_manager = create_project(project_name=project_name)
    dcl.open_page(dcl.EDIT_PAGE_STR)
    remove_all_timeline(project=project)
    set_project_settings_bt2100(project=project)

    media_path = Path('C:/Users/toruv/OneDrive/work/sample_code/2024/03_Measure_Moitor_Spec_with_Resolve/img_cctp')

    # add test pattern for measure
    num_of_cc_patch = 18
    fname_list = [
        str(Path(create_cc_tp_fname(
            cc_idx=idx, luminance=luminance, window_size=window_size)
        ).resolve())
        for idx in range(num_of_cc_patch)]
    clip_list = dcl.add_files_to_media_pool(media_path=fname_list)
    tp_timeline = dcl.create_timeline_from_clip(
        clip_list=clip_list, timeline_name="TP_Timeline")

    # add all black pattern
    fname_black = "C:/Users/toruv/OneDrive/work/sample_code/2024/03_Measure_Moitor_Spec_with_Resolve/tp_img/black.png"
    clip_list = dcl.add_files_to_media_pool(media_path=fname_black)
    black_timeline = dcl.create_timeline_from_clip(
        clip_list=clip_list, timeline_name="Black_Timeline")

    # initialize with black
    project.SetCurrentTimeline(black_timeline)
    time.sleep(1)

    #################
    # Measure
    #################
    for frame_idx in range(num_of_cc_patch):
        print(f"Measure sequence [{frame_idx + 1} / {num_of_cc_patch}]")
        project.SetCurrentTimeline(tp_timeline)
        timecode_str = frame_number_to_timecode(frame_number=frame_idx, fps=24)
        tp_timeline.SetCurrentTimecode(timecode_str)
        time.sleep(0.5)
        large_xyz, Yxy = read_xyz(flush=False, ccss_file=ccss_file)
        ccss_name = Path(ccss_file).stem if ccss_file else "-"
        save_measure_result(
            large_xyz=large_xyz, Yxy=Yxy,
            csv_name=csv_name, ccss_name=ccss_name)
        project.SetCurrentTimeline(black_timeline)
        time.sleep(1)

    dcl.save_project(project_manager=project_manager)
    # window_size_str = f"{int(window_size * 100):03d}"
    # archive_path = Path(
    #     f"./project_archive/colorchecker_measure_{luminance}-nits_{window_size_str}-window.dra").resolve()
    # dcl.archive_project(
    #     project_name=project_name, archive_path=str(archive_path))


def demo_measure_for_blog():
    csv_name = "./demo_measure_result.csv"
    ccss_file = "./ccss/RGBLEDFamily_07Feb11.ccss"
    color_mask = [1, 1, 1]

    remove_csv(file_path=csv_name)

    project_name = "Measure_AW3225QF"
    dcl.close_and_remove_project(project_name=project_name)
    project, project_manager = create_project(project_name=project_name)
    dcl.open_page(dcl.EDIT_PAGE_STR)
    remove_all_timeline(project=project)
    set_project_settings_bt2100(project=project)

    media_path = Path('C:/Users/toruv/OneDrive/work/sample_code/2024/03_Measure_Moitor_Spec_with_Resolve/tp_img')

    # add test pattern for measure
    window_size_list = [
        0.03, 0.05, 0.10, 0.20, 0.30, 0.40,
        0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    fname_list = [
        create_tp_base_name(
            color_mask=color_mask, cv=1,
            patch_area_ratio=window_size_ratio) + ".png"
        for window_size_ratio in window_size_list]
    fname_list = [str(media_path / Path(fname)) for fname in fname_list]

    clip_list = dcl.add_files_to_media_pool(media_path=fname_list)
    tp_timeline = dcl.create_timeline_from_clip(
        clip_list=clip_list, timeline_name="TP_Timeline")

    # add all black pattern
    fname_black = "C:/Users/toruv/OneDrive/work/sample_code/2024/03_Measure_Moitor_Spec_with_Resolve/tp_img/black.png"
    clip_list = dcl.add_files_to_media_pool(media_path=fname_black)
    black_timeline = dcl.create_timeline_from_clip(
        clip_list=clip_list, timeline_name="Black_Timeline")

    # initialize with black
    project.SetCurrentTimeline(black_timeline)
    time.sleep(1)

    for frame_idx in range(len(window_size_list)):
        project.SetCurrentTimeline(tp_timeline)
        timecode_str = frame_number_to_timecode(frame_number=frame_idx, fps=24)
        tp_timeline.SetCurrentTimecode(timecode_str)
        time.sleep(0.5)
        large_xyz, Yxy = read_xyz(flush=False, ccss_file=ccss_file)
        ccss_name = Path(ccss_file).stem if ccss_file else "-"
        save_measure_result(
            large_xyz=large_xyz, Yxy=Yxy,
            csv_name=csv_name, ccss_name=ccss_name)
        project.SetCurrentTimeline(black_timeline)
        time.sleep(1)

    dcl.save_project(project_manager=project_manager)
    dcl.close_current_project()
    archive_path = Path("./project_archive/hoge.dra").resolve()
    dcl.archive_project(
        project_name=project_name, archive_path=str(archive_path))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # csv_name = "./measure_result/aw3225qf_1000nits_60s.csv"

    ###########################
    # Duration measure
    ###########################
    # color_mask = [1, 1, 1]
    # percent_list = [
    #     0.03, 0.05, 0.10, 0.20, 0.30, 0.40,
    #     0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    # for percent in percent_list:
    #     percent_str = f"{int(percent * 100):03d}"
    #     duration_measure(
    #         csv_name=f"./AW3225QF/measure_duration_{percent_str}.csv",
    #         ccss_file=CCSS_RGBLED, color_mask=color_mask,
    #         patch_area_ratio=percent)

    ###########################
    # White patch measure
    ###########################
    # condition = "Desktop"
    # condition = "Movie_HDR"
    # condition = "Game_HDR"
    # condition = "Custom_Color_HDR"
    # condition = "DisplayHDR_True_Black"
    # condition = "HDR_Peak_1000"
    # percent_list = [0.03, 0.10, 0.20, 0.50, 1.00]

    # for percent in percent_list:
    #     percent_str = f"{int(percent * 100):03d}"
    #     csv_name = f"./AW3225QF/measure_{condition}_{percent_str}_patch.csv"
    #     # csv_name = f"./AW3225QF/measure_{condition}_{percent_str}_patch2.csv"
    #     print(csv_name)
    #     increase_cv_measure(
    #         csv_name=csv_name, ccss_file=CCSS_RGBLED,
    #         color_mask=[1, 1, 1], patch_area_ratio=percent)

    ###########################
    # Color Checker measure
    ###########################
    # lumiannce_list = [100, 200, 400, 600, 1000]
    # window_size_list = [0.03, 0.10, 0.20, 0.50, 1.00]

    # for luminance in lumiannce_list:
    #     for window_size in window_size_list:
    #         print("="*80)
    #         print(f"luminance = {luminance}, window_size = {window_size}")
    #         print("="*80)
    #         cc_patch_measure(
    #             luminance=luminance, window_size=window_size,
    #             ccss_file=CCSS_RGBLED)

    #############################
    # Demo for Blog
    #############################
    # demo_measure_for_blog()
