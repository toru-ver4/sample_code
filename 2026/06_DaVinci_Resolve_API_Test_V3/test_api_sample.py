"""Run from C:/Users/toruv/OneDrive/work/sample_code.

Unit checks:
  ty_lib/TY_DaVinci_Resolve_Control_Lib/.venv313/Scripts/python.exe -m pytest 2026/06_DaVinci_Resolve_API_Test_V3/test_api_sample.py -q -x -p no:cacheprovider
Live checks (Resolve running, PowerShell):
  $env:RUN_RESOLVE_SAMPLE_TESTS='1'; run the same command.
Live checks create UUID projects, render only to pytest's temporary directory,
and delete their projects after each case. No fixed sample project is deleted.
"""

import os
from types import SimpleNamespace
from uuid import uuid4

import pytest

import api_sapmle as sample


def test_snapshot_prefers_new_method():
    target = SimpleNamespace(
        GetSettings=lambda: {"timelineFrameRate": "24"},
        GetSetting=lambda: pytest.fail("Legacy method must not be used"),
    )
    assert sample.get_all_settings(target) == {"timelineFrameRate": "24"}


def test_snapshot_supports_absent_new_method():
    target = SimpleNamespace(GetSetting=lambda: {"timelineFrameRate": "24"})
    assert sample.get_all_settings(target) == {"timelineFrameRate": "24"}


def test_snapshot_failure_does_not_retry():
    target = SimpleNamespace(
        GetSettings=lambda: None,
        GetSetting=lambda: pytest.fail("Failed calls must not be retried"),
    )
    with pytest.raises(RuntimeError, match="Failed to read settings"):
        sample.get_all_settings(target)


@pytest.mark.skipif(
    os.environ.get("RUN_RESOLVE_SAMPLE_TESTS") != "1",
    reason="Set RUN_RESOLVE_SAMPLE_TESTS=1 for live Resolve checks",
)
@pytest.mark.parametrize("function_name", [
    "create_project_sample", "get_project_settings_sample",
    "project_settings_sample", "timeline_settings_sample", "encode_test",
    "fusion_key_frame_test", "draw_sharp_edge_rectangle_using_fusion_sample",
])
def test_live_sample(function_name, tmp_path):
    tdr = sample.tdr
    session = tdr.ResolveSession.connect()
    original = session.project_manager.GetCurrentProject()
    original_name = original.GetName() if original is not None else None
    original_page = session.resolve.GetCurrentPage()
    name = f"sample_21_1_check_{uuid4().hex}"
    kwargs = {"project_name": name}
    if function_name == "encode_test":
        kwargs["output_dir"] = tmp_path
    if function_name == "draw_sharp_edge_rectangle_using_fusion_sample":
        kwargs.update(center=(105.5 / 1920, 1 - 105.5 / 1080),
                      size=(9 / 1920, 11 / 1080))
    try:
        getattr(sample, function_name)(**kwargs)
        if function_name == "create_project_sample":
            project = tdr.load_project(session, name=name)
        else:
            project = tdr.get_current_project(session)
        assert project.GetName() == name
        project_before = sample.get_all_settings(project)
        timeline = project.GetCurrentTimeline()
        timeline_before = sample.get_all_settings(timeline) if timeline else None
        if timeline:
            sample.get_current_timeline_settings_sample()
        if function_name in {"timeline_settings_sample", "encode_test",
                             "fusion_key_frame_test"}:
            # Full snapshots retain the native numeric frame-rate value.
            assert float(timeline_before["timelineFrameRate"]) == 59.94
            assert timeline_before["useCustomSettings"] == "1"
            assert timeline_before["timelineResolutionWidth"] == "1920"
            assert timeline_before["colorSpaceOutput"] == "Rec.709"
        if function_name == "draw_sharp_edge_rectangle_using_fusion_sample":
            comp = timeline.GetItemListInTrack("video", 1)[0].GetFusionCompByIndex(1)
            mask = next(tool for tool in comp.GetToolList(False).values()
                        if tool.GetAttrs()["TOOLS_RegID"] == "RectangleMask")
            center = mask.GetInput("Center")
            assert center[1] == pytest.approx(105.5 / 1920)
            assert center[2] == pytest.approx(1 - 105.5 / 1080)
        if function_name == "encode_test":
            movies = list(tmp_path.glob("*.mov"))
            assert len(movies) == 1 and movies[0].stat().st_size > 0
        tdr.save_project(session)
        tdr.close_project(session, project=project)
        project = tdr.load_project(session, name=name)
        after = sample.get_all_settings(project)
        for key in ("timelineFrameRate", "timelineResolutionWidth",
                    "rcmPresetMode", "colorSpaceTimeline", "colorSpaceOutput"):
            assert after[key] == project_before[key], key
        if timeline_before:
            after = sample.get_all_settings(project.GetCurrentTimeline())
            for key in ("timelineFrameRate", "useCustomSettings",
                        "timelineResolutionWidth", "colorSpaceOutput"):
                # Inherited timelines omit useCustomSettings in snapshots.
                # Custom timelines must retain the explicit value checked above.
                assert after.get(key) == timeline_before.get(key), key
    finally:
        sample.close_and_delete_project_if_exists(session, name)
        if original_name in tdr.list_projects(session):
            tdr.load_project(session, name=original_name)
        if original_page:
            tdr.open_page(session, page=original_page)
