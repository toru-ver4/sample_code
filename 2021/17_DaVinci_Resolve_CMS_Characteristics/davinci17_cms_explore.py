# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
from pathlib import Path

# import third-party libraries

# import my libraries
import ty_davinci_control_lib as dcl

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


TP_SRC_PATH = Path(
    'C:/Users/toruv/OneDrive/work/sample_code/2021/17_DaVinci_Resolve_CMS_Characteristics/img')
MEDIA_SRC_PATH = Path(
    'D:/abuse/2021/17_DaVinci_Resolve_CMS_Characteristics/src')
MEDIA_DST_PATH = Path(
    'D:/abuse/2021/17_DaVinci_Resolve_CMS_Characteristics/dst')
EXR_MIN_EXPOSURE = -6
EXR_MAX_EXPOSURE = 3


"""
import sys
sys.path.append("C:/Users/toruv/OneDrive/work/sample_code/temporary")
import resolve_17_4 as r174
import imp
imp.reload(r174)
r174.start_pos_test()
"""

def get_media_src_fname_sdr(idx=0):
    fname = MEDIA_SRC_PATH / f"src_sdr_{idx:04d}.png"
    return str(fname)


def get_media_src_fname_hdr(idx=0):
    fname = MEDIA_SRC_PATH / f"src_hdr_{idx:04d}.png"
    return str(fname)


def get_media_src_fname_exr(idx=0):
    fname = MEDIA_SRC_PATH / f"src_exr_{idx:04d}.exr"
    return str(fname)


def start_pos_test():
    media_video_path = Path('D:/abuse/2020/005_make_countdown_movie/movie_seq')
    out_path = Path(
        'D:/Resolve/render_out/resolve_17_4')
    format_str = None
    codec = None
    # preset_name = H265_CQP0_PRESET_NAME

    # main process
    print("script start")
    resolve, project_manager = dcl.init_davinci17(
        close_current_project=True)
    project = dcl.prepare_project(
        project_manager=project_manager,
        project_name="davinci_17_4_test")

    print("add media to pool")
    dcl.add_clips_to_media_pool(resolve, project, media_video_path)
    clip_list, clip_name_list\
        = dcl.get_media_pool_clip_list_and_clip_name_list(project)

    # # add video
    # selected_video_clip_list = []
    # for clip_obj, clip_name in zip(clip_list, clip_name_list):
    #     print(f"clip_name = {clip_name}")
    #     if clip_name_base in clip_name:
    #         print(f"{clip_name_base} is found!")
    #         selected_video_clip_list.append(clip_obj)
    #         break


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
