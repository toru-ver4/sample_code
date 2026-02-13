from pathlib import Path
import os
import sys
import unittest
from itertools import product
import json

import numpy as np

THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from create_hdr_media import (
    MDCV_PRIMARIES_LIST,
    MDCV_LUMINANCE_LIST,
    CLLI_LUMINANCE_LIST,
    KIND_AV1,
    KIND_AVIF,
    KIND_HEVC,
    KIND_PNG,
    make_media_file_name_without_ext
)
from parse_png_chunk_dump import parse_png_chunk_dump

import color_space as cs

from typing import Any


def find_values_by_key(
    obj: Any, target_key: str, parent_key: str | None = None
) -> list[Any]:
    """Collect values whose key matches ``target_key`` from a nested JSON-like object.

    The function traverses dictionaries and lists iteratively (depth-first) and
    returns every value found for the specified key, regardless of nesting depth.
    If ``parent_key`` is given, only values whose immediate parent dictionary key
    matches ``parent_key`` are collected.

    Args:
        obj: A JSON-like structure (typically composed of dict, list, and scalar values).
        target_key: The dictionary key to search for.
        parent_key: Optional parent dictionary key to limit matches.

    Returns:
        A list of matched values. Returns an empty list if the key is not found.
    """
    values: list[Any] = []
    stack: list[tuple[Any, str | None]] = [(obj, None)]

    while stack:
        current, current_parent_key = stack.pop()

        if isinstance(current, dict):
            if target_key in current:
                if (parent_key is None) or (current_parent_key == parent_key):
                    values.append(current[target_key])
            for key, value in current.items():
                stack.append((value, key))
        elif isinstance(current, list):
            for value in current:
                stack.append((value, current_parent_key))

    return values


def calc_expected_hevc_primaries_int(
        primary_str: str, unit: float = 0.00002, roundup: bool = True) -> list[list[int | None]]:
    primaries_int = [[None, None], [None, None], [None, None]]
    if primary_str is not None:
        primaries = cs.get_primaries(primary_str)
        if roundup:
            primaries_int = np.round(primaries / unit).astype(np.uint16).tolist()
        else:
            primaries_int = (primaries / unit).astype(np.uint16).tolist()

    return primaries_int


def calc_expected_hevc_primaries_float(primary_str: str) -> list[list[float | None]]:
    primaries = [[None, None], [None, None], [None, None]]
    if primary_str is not None:
        primaries = cs.get_primaries(primary_str).tolist()

    return primaries


class TestHevcMetadata(unittest.TestCase):

    def test_bitstream_metadata(self):
        for mdcv_primaries, mdcv_luminance, clli_luminance\
            in product(MDCV_PRIMARIES_LIST, MDCV_LUMINANCE_LIST, CLLI_LUMINANCE_LIST):
            # print(f"[TEST Condition] mdcv_primaries={mdcv_primaries}, mdcv_luminance={mdcv_luminance}, clli_lumiannce={clli_luminance}")
            file_name_without_ext = make_media_file_name_without_ext(
                kind=KIND_HEVC,
                suffix=None,
                mdcv_primaries=mdcv_primaries,
                mdcv_luminance=mdcv_luminance,
                clli_luminance=clli_luminance
            )
            file_name = file_name_without_ext + ".h265"
            if os.path.exists(file_name) is False:
                continue
            json_file = f"./data/{Path(file_name).name}.json"
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            cicp_expected_dict = {
                "-colour_primaries": 9,
                "-transfer_characteristic": 16,
                "-matrix_coeffs": 9,
                "-video_full_range_flag": 0,
            }
            for key, expected_value in cicp_expected_dict.items():
                actual_values = find_values_by_key(data, key)
                self.assertNotEqual(actual_values, [])
                for actual_value in actual_values:
                    # print(f"[ASSERT] {key} actual={actual_value} expected={expected_value}")
                    self.assertEqual(int(actual_value), expected_value)

            expected_primaries = calc_expected_hevc_primaries_float(primary_str=mdcv_primaries)
            mdcv_primaries_dict = {
                "-display_primaries_x": [expected_primaries[ii][0] for ii in [1, 2, 0]],
                "-display_primaries_y": [expected_primaries[ii][1] for ii in [1, 2, 0]],
            }
            for key, expected_value in mdcv_primaries_dict.items():
                actual_values = find_values_by_key(data, key)
                if None in expected_value:
                    # print(f"[ASSERT] {key} actual={actual_values} expected={expected_value}")
                    self.assertEqual(actual_values, [])
                else:
                    self.assertNotEqual(actual_values, [])
                    for actual_value in actual_values:
                        # print(f"[ASSERT] {key} actual={actual_value} expected={expected_value}")
                        arr = np.fromstring(actual_value, sep=" ")
                        np.testing.assert_almost_equal(arr, np.array(expected_value), decimal=4)

            parent_str = "SEIMessage"
            mdcv_clli_expected_dict = {
                "-white_point_x": (parent_str, 0.3127 if mdcv_primaries else None),
                "-white_point_y": (parent_str, 0.3290 if mdcv_primaries else None),
                "-max_display_mastering_luminance": (parent_str, mdcv_luminance if mdcv_luminance else None),
                "-min_display_mastering_luminance": (parent_str, 0 if mdcv_luminance else None),
                "-max_content_light_level": (parent_str, clli_luminance),
                "-max_pic_average_light_level": (parent_str, clli_luminance),
            }

            for key, (parent_key, expected_value) in mdcv_clli_expected_dict.items():
                actual_values = find_values_by_key(data, key, parent_key=parent_key)
                if expected_value is None:
                    # print(f"[ASSERT] {key} actual={actual_values} expected={expected_value}")
                    self.assertEqual(actual_values, [])
                else:
                    self.assertNotEqual(actual_values, [])
                    for actual_value in actual_values:
                        # print(f"[ASSERT] {key} actual={actual_value} expected={expected_value}")
                        self.assertAlmostEqual(float(actual_value), expected_value, places=4)

    def test_mp4_container_metadata(self):
        for mdcv_primaries, mdcv_luminance, clli_luminance, ext_str,\
            in product(MDCV_PRIMARIES_LIST, MDCV_LUMINANCE_LIST, CLLI_LUMINANCE_LIST, [".mp4", ".mov"]):
            # print(f"[TEST Condition] mdcv_primaries={mdcv_primaries}, mdcv_luminance={mdcv_luminance}, clli_lumiannce={clli_luminance}, ext_str={ext_str}")
            file_name_without_ext = make_media_file_name_without_ext(
                kind=KIND_HEVC,
                suffix=None,
                mdcv_primaries=mdcv_primaries,
                mdcv_luminance=mdcv_luminance,
                clli_luminance=clli_luminance
            )
            file_name = file_name_without_ext + ext_str
            if os.path.exists(file_name) is False:
                continue
            json_file = f"./data/{Path(file_name).name}.json"
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            cicp_expected_dict = {
                "-colour_primaries": 9,
                "-transfer_characteristic": 16,
                "-matrix_coeffs": 9,
                "-video_full_range_flag": 0,
            }
            for key, expected_value in cicp_expected_dict.items():
                actual_values = find_values_by_key(data, key)
                self.assertNotEqual(actual_values, [])
                for actual_value in actual_values:
                    # print(f"[ASSERT] {key} actual={actual_value} expected={expected_value}")
                    self.assertEqual(int(actual_value), expected_value)

            expected_primaries = calc_expected_hevc_primaries_int(primary_str=mdcv_primaries)
            mdcv_box_name = "MasteringDisplayColourVolumeBox"
            clli_box_name = "ContentLightLevelBox"
            mdcv_dict = {
                "-display_primaries_0_x": (mdcv_box_name, expected_primaries[1][0]),
                "-display_primaries_0_y": (mdcv_box_name, expected_primaries[1][1]),
                "-display_primaries_1_x": (mdcv_box_name, expected_primaries[2][0]),
                "-display_primaries_1_y": (mdcv_box_name, expected_primaries[2][1]),
                "-display_primaries_2_x": (mdcv_box_name, expected_primaries[0][0]),
                "-display_primaries_2_y": (mdcv_box_name, expected_primaries[0][1]),
                '-white_point_x': (mdcv_box_name, None if mdcv_primaries is None else int(round(0.3127 / 0.00002))),
                '-white_point_y': (mdcv_box_name, None if mdcv_primaries is None else int(round(0.3290 / 0.00002))),
                '-max_display_mastering_luminance': (mdcv_box_name, None if mdcv_luminance is None else int(round((mdcv_luminance) / 0.0001))),
                '-min_display_mastering_luminance': (mdcv_box_name, None if mdcv_luminance is None else int(round(0 / 0.0001))),
                '-max_content_light_level': (clli_box_name, clli_luminance),
                '-max_pic_average_light_level': (clli_box_name, clli_luminance),
            }
            for key, (parent_key, expected_value) in mdcv_dict.items():
                actual_values = find_values_by_key(data, key, parent_key=parent_key)
                if expected_value is None:
                    # print(f"[ASSERT] {key} actual={actual_values} expected={expected_value}")
                    self.assertEqual(actual_values, [])
                else:
                    self.assertNotEqual(actual_values, [])
                    for actual_value in actual_values:
                        # print(f"[ASSERT] {key} actual={actual_value} expected={expected_value}")
                        self.assertAlmostEqual(int(actual_value), expected_value, delta=1)


class TestAv1Metadata(unittest.TestCase):
    def test_bitstream_metadata(self):
        for mdcv_primaries, mdcv_luminance, clli_luminance\
            in product(MDCV_PRIMARIES_LIST, MDCV_LUMINANCE_LIST, CLLI_LUMINANCE_LIST):
            # print(f"[TEST Condition] mdcv_primaries={mdcv_primaries}, mdcv_luminance={mdcv_luminance}, clli_lumiannce={clli_luminance}")
            file_name_without_ext = make_media_file_name_without_ext(
                kind=KIND_AV1,
                suffix=None,
                mdcv_primaries=mdcv_primaries,
                mdcv_luminance=mdcv_luminance,
                clli_luminance=clli_luminance
            )
            file_name = file_name_without_ext + ".obu"
            if os.path.exists(file_name) is False:
                continue
            json_file = f"./data/{Path(file_name).name}.json"
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            cicp_expected_dict = {
                "-color_primaries": 9,
                "-transfer_characteristics": 16,
                "-matrix_coefficients": 9,
                "-color_range": 0,
            }
            for key, expected_value in cicp_expected_dict.items():
                actual_values = find_values_by_key(data, key)
                self.assertNotEqual(actual_values, [])
                for actual_value in actual_values:
                    # print(f"[ASSERT] {key} actual={actual_value} expected={expected_value}")
                    self.assertEqual(int(actual_value), expected_value)

            expected_primaries = calc_expected_hevc_primaries_float(primary_str=mdcv_primaries)
            mdcv_primaries_dict = {
                "-display_primaries_x": ("OBU", [expected_primaries[ii][0] for ii in [0, 1, 2]]),
                "-display_primaries_y": ("OBU", [expected_primaries[ii][1] for ii in [0, 1, 2]]),
            }
            for key, (parent_key, expected_value) in mdcv_primaries_dict.items():
                actual_values = find_values_by_key(data, key, parent_key=parent_key)
                if None in expected_value:
                    # print(f"[ASSERT] {key} actual={actual_values} expected={expected_value}")
                    self.assertEqual(actual_values, [])
                else:
                    self.assertNotEqual(actual_values, [])
                    for actual_value in actual_values:
                        # print(f"[ASSERT] {key} actual={actual_value} expected={expected_value}")
                        arr = np.fromstring(actual_value, sep=" ")
                        np.testing.assert_almost_equal(arr, np.array(expected_value), decimal=4)

            mdcv_clli_other_dict = {
                "-white_point_x": ("OBU", 0.3127 if mdcv_primaries else None),
                "-white_point_y": ("OBU", 0.3290 if mdcv_primaries else None),
                '-max_display_mastering_luminance': ("OBU", mdcv_luminance if mdcv_luminance else None),
                '-min_display_mastering_luminance': ("OBU", 0 if mdcv_luminance else None),
                '-max_content_light_level': ("OBU", clli_luminance),
                '-max_pic_average_light_level': ("OBU", clli_luminance),
            }
            for key, (parent_key, expected_value) in mdcv_clli_other_dict.items():
                actual_values = find_values_by_key(data, key, parent_key=parent_key)
                if expected_value is None:
                    # print(f"[ASSERT] {key} actual={actual_values} expected={expected_value}")
                    self.assertEqual(actual_values, [])
                else:
                    self.assertNotEqual(actual_values, [])
                    for actual_value in actual_values:
                        # print(f"[ASSERT] {key} actual={actual_value} expected={expected_value}")
                        self.assertAlmostEqual(float(actual_value), expected_value, places=4)

    def test_container_metadata(self):
        for mdcv_primaries, mdcv_luminance, clli_luminance, container_ext\
            in product(MDCV_PRIMARIES_LIST, MDCV_LUMINANCE_LIST, CLLI_LUMINANCE_LIST, [".mp4", ".mov"]):
            # print(f"[TEST Condition] mdcv_primaries={mdcv_primaries}, mdcv_luminance={mdcv_luminance}, clli_lumiannce={clli_luminance}")
            file_name_without_ext = make_media_file_name_without_ext(
                kind=KIND_AV1,
                suffix=None,
                mdcv_primaries=mdcv_primaries,
                mdcv_luminance=mdcv_luminance,
                clli_luminance=clli_luminance
            )
            file_name = file_name_without_ext + container_ext
            if os.path.exists(file_name) is False:
                continue
            json_file = f"./data/{Path(file_name).name}.json"
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            cicp_expected_dict = {
                "-color_primaries": 9,
                "-transfer_characteristics": 16,
                "-matrix_coefficients": 9,
                "-color_range": 0,
            }
            for key, expected_value in cicp_expected_dict.items():
                actual_values = find_values_by_key(data, key)
                self.assertNotEqual(actual_values, [])
                for actual_value in actual_values:
                    # print(f"[ASSERT] {key} actual={actual_value} expected={expected_value}")
                    self.assertEqual(int(actual_value), expected_value)

            expected_primaries = calc_expected_hevc_primaries_int(primary_str=mdcv_primaries)
            mdcv_box_name = "MasteringDisplayColourVolumeBox"
            clli_box_name = "ContentLightLevelBox"
            mdcv_dict = {
                "-display_primaries_0_x": (mdcv_box_name, expected_primaries[1][0]),
                "-display_primaries_0_y": (mdcv_box_name, expected_primaries[1][1]),
                "-display_primaries_1_x": (mdcv_box_name, expected_primaries[2][0]),
                "-display_primaries_1_y": (mdcv_box_name, expected_primaries[2][1]),
                "-display_primaries_2_x": (mdcv_box_name, expected_primaries[0][0]),
                "-display_primaries_2_y": (mdcv_box_name, expected_primaries[0][1]),
                '-white_point_x': (mdcv_box_name, None if mdcv_primaries is None else int(round(0.3127 / 0.00002))),
                '-white_point_y': (mdcv_box_name, None if mdcv_primaries is None else int(round(0.3290 / 0.00002))),
                '-max_display_mastering_luminance': (mdcv_box_name, None if mdcv_luminance is None else int(round((mdcv_luminance) / 0.0001))),
                '-min_display_mastering_luminance': (mdcv_box_name, None if mdcv_luminance is None else int(round(0 / 0.0001))),
                '-max_content_light_level': (clli_box_name, clli_luminance),
                '-max_pic_average_light_level': (clli_box_name, clli_luminance),
            }
            for key, (parent_key, expected_value) in mdcv_dict.items():
                actual_values = find_values_by_key(data, key, parent_key=parent_key)
                if expected_value is None:
                    # print(f"[ASSERT] {key} actual={actual_values} expected={expected_value}")
                    self.assertEqual(actual_values, [])
                else:
                    self.assertNotEqual(actual_values, [])
                    for actual_value in actual_values:
                        # print(f"[ASSERT] {key} actual={actual_value} expected={expected_value}")
                        self.assertAlmostEqual(int(actual_value), expected_value, delta=1)


class TestAvifMetadata(unittest.TestCase):
    def test_bitstream_metadata(self):
        for mdcv_primaries, mdcv_luminance, clli_luminance\
            in product(MDCV_PRIMARIES_LIST, MDCV_LUMINANCE_LIST, CLLI_LUMINANCE_LIST):
            # print(f"[TEST Condition] mdcv_primaries={mdcv_primaries}, mdcv_luminance={mdcv_luminance}, clli_lumiannce={clli_luminance}")
            file_name_without_ext = make_media_file_name_without_ext(
                kind=KIND_AVIF,
                suffix=None,
                mdcv_primaries=mdcv_primaries,
                mdcv_luminance=mdcv_luminance,
                clli_luminance=clli_luminance
            )
            file_name = file_name_without_ext + ".obu"
            if os.path.exists(file_name) is False:
                continue
            json_file = f"./data/{Path(file_name).name}.json"
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            cicp_expected_dict = {
                "-color_primaries": 2,
                "-transfer_characteristics": 2,
                "-matrix_coefficients": 2,
                "-color_range": 1,
            }
            for key, expected_value in cicp_expected_dict.items():
                actual_values = find_values_by_key(data, key)
                self.assertNotEqual(actual_values, [])
                for actual_value in actual_values:
                    # print(f"[ASSERT] {key} actual={actual_value} expected={expected_value}")
                    self.assertEqual(int(actual_value), expected_value)

            mdcv_primaries_dict = {
                "-display_primaries_x": ("OBU", None),
                "-display_primaries_y": ("OBU", None),
                "-white_point_x": ("OBU", None),
                "-white_point_y": ("OBU", None),
                '-max_display_mastering_luminance': ("OBU", None),
                '-min_display_mastering_luminance': ("OBU", None),
                '-max_content_light_level': ("OBU", None),
                '-max_pic_average_light_level': ("OBU", None),
            }
            for key, (parent_key, expected_value) in mdcv_primaries_dict.items():
                actual_values = find_values_by_key(data, key, parent_key=parent_key)
                if expected_value is None:
                    # print(f"[ASSERT] {key} actual={actual_values} expected={expected_value}")
                    self.assertEqual(actual_values, [])
                else:
                    self.assertNotEqual(actual_values, [])
                    for actual_value in actual_values:
                        # print(f"[ASSERT] {key} actual={actual_value} expected={expected_value}")
                        arr = np.fromstring(actual_value, sep=" ")
                        np.testing.assert_almost_equal(arr, np.array(expected_value), decimal=4)

    def test_container_metadata(self):
        for mdcv_primaries, mdcv_luminance, clli_luminance, container_ext\
            in product(MDCV_PRIMARIES_LIST, MDCV_LUMINANCE_LIST, CLLI_LUMINANCE_LIST, [".avif"]):
            # print(f"[TEST Condition] mdcv_primaries={mdcv_primaries}, mdcv_luminance={mdcv_luminance}, clli_lumiannce={clli_luminance}")
            file_name_without_ext = make_media_file_name_without_ext(
                kind=KIND_AVIF,
                suffix=None,
                mdcv_primaries=mdcv_primaries,
                mdcv_luminance=mdcv_luminance,
                clli_luminance=clli_luminance
            )
            file_name = file_name_without_ext + container_ext
            if os.path.exists(file_name) is False:
                continue
            json_file = f"./data/{Path(file_name).name}.json"
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            cicp_expected_dict = {
                "-colour_primaries": 9,
                "-transfer_characteristics": 16,
                "-matrix_coefficients": 0,
                "-full_range_flag": 1,
            }
            for key, expected_value in cicp_expected_dict.items():
                actual_values = find_values_by_key(data, key)
                self.assertNotEqual(actual_values, [])
                for actual_value in actual_values:
                    # print(f"[ASSERT] {key} actual={actual_value} expected={expected_value}")
                    self.assertEqual(int(actual_value), expected_value)


            clli_box_name = "ContentLightLevelBox"
            mdcv_dict = {
                '-max_content_light_level': (clli_box_name, clli_luminance),
                '-max_pic_average_light_level': (clli_box_name, clli_luminance),
            }
            for key, (parent_key, expected_value) in mdcv_dict.items():
                actual_values = find_values_by_key(data, key, parent_key=parent_key)
                if expected_value is None:
                    # print(f"[ASSERT] {key} actual={actual_values} expected={expected_value}")
                    self.assertEqual(actual_values, [])
                else:
                    self.assertNotEqual(actual_values, [])
                    for actual_value in actual_values:
                        # print(f"[ASSERT] {key} actual={actual_value} expected={expected_value}")
                        self.assertEqual(int(actual_value), expected_value)


class TestPngMetadata(unittest.TestCase):
    def test_bitstream_metadata(self):
        for mdcv_primaries, mdcv_luminance, clli_luminance\
            in product(MDCV_PRIMARIES_LIST, MDCV_LUMINANCE_LIST, CLLI_LUMINANCE_LIST):
            # print(f"[TEST Condition] mdcv_primaries={mdcv_primaries}, mdcv_luminance={mdcv_luminance}, clli_lumiannce={clli_luminance}")
            file_name_without_ext = make_media_file_name_without_ext(
                kind=KIND_PNG,
                suffix=None,
                mdcv_primaries=mdcv_primaries,
                mdcv_luminance=mdcv_luminance,
                clli_luminance=clli_luminance
            )
            file_name = file_name_without_ext + ".h265"
            if os.path.exists(file_name) is False:
                continue
            json_file = f"./data/{Path(file_name).name}.json"
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            cicp_expected_dict = {
                "-colour_primaries": 9,
                "-transfer_characteristic": 16,
                "-matrix_coeffs": 9,
                "-video_full_range_flag": 0,
            }
            for key, expected_value in cicp_expected_dict.items():
                actual_values = find_values_by_key(data, key)
                self.assertNotEqual(actual_values, [])
                for actual_value in actual_values:
                    # print(f"[ASSERT] {key} actual={actual_value} expected={expected_value}")
                    self.assertEqual(int(actual_value), expected_value)

            expected_primaries = calc_expected_hevc_primaries_float(primary_str=mdcv_primaries)
            mdcv_primaries_dict = {
                "-display_primaries_x": [expected_primaries[ii][0] for ii in [1, 2, 0]],
                "-display_primaries_y": [expected_primaries[ii][1] for ii in [1, 2, 0]],
            }
            for key, expected_value in mdcv_primaries_dict.items():
                actual_values = find_values_by_key(data, key)
                # print(f"[ASSERT] {key} actual={actual_values} expected={expected_value}")
                if None in expected_value:
                    self.assertEqual(actual_values, [])
                else:
                    self.assertNotEqual(actual_values, [])
                    for actual_value in actual_values:
                        arr = np.fromstring(actual_value, sep=" ")
                        np.testing.assert_almost_equal(arr, np.array(expected_value), decimal=4)

            parent_str = "SEIMessage"
            mdcv_clli_expected_dict = {
                "-white_point_x": (parent_str, 0.3127 if mdcv_primaries else None),
                "-white_point_y": (parent_str, 0.3290 if mdcv_primaries else None),
                "-max_display_mastering_luminance": (parent_str, mdcv_luminance if mdcv_luminance else None),
                "-min_display_mastering_luminance": (parent_str, 0 if mdcv_luminance else None),
                "-max_content_light_level": (parent_str, clli_luminance),
                "-max_pic_average_light_level": (parent_str, clli_luminance),
            }

            for key, (parent_key, expected_value) in mdcv_clli_expected_dict.items():
                actual_values = find_values_by_key(data, key, parent_key=parent_key)
                # print(f"[ASSERT] {key} actual={actual_values} expected={expected_value}")
                if expected_value is None:
                    self.assertEqual(actual_values, [])
                else:
                    self.assertNotEqual(actual_values, [])
                    for actual_value in actual_values:
                        self.assertAlmostEqual(float(actual_value), expected_value, places=4)

    def test_chunk_data(self):
        for mdcv_primaries, mdcv_luminance, clli_luminance\
            in product(MDCV_PRIMARIES_LIST, MDCV_LUMINANCE_LIST, CLLI_LUMINANCE_LIST):
            # print(f"[TEST Condition] mdcv_primaries={mdcv_primaries}, mdcv_luminance={mdcv_luminance}, clli_lumiannce={clli_luminance}")
            file_name_without_ext = make_media_file_name_without_ext(
                kind=KIND_PNG,
                suffix=None,
                mdcv_primaries=mdcv_primaries,
                mdcv_luminance=mdcv_luminance,
                clli_luminance=clli_luminance
            )
            file_name = file_name_without_ext + ".png"
            if os.path.exists(file_name) is False:
                continue
            parse_file = f"./data/{Path(file_name).name}.txt"
            parse_data = parse_png_chunk_dump(parse_file)

            cicp_expected_dict = {
                "chromaticity": np.concatenate([[cs.D65], cs.get_primaries(cs.BT2020)]),
                "range": "Full range",
            }
            for key, expected_value in cicp_expected_dict.items():
                if isinstance(expected_value, np.ndarray):
                    np.testing.assert_almost_equal(parse_data['cicp'][key], expected_value, decimal=4)
                elif isinstance(expected_value, str):
                    self.assertEqual(parse_data['cicp'][key], expected_value)
                else:
                    pass

            mdcv_expected_dict = {
                "chromaticity": None if mdcv_primaries is None else np.concatenate([[cs.D65], cs.get_primaries(mdcv_primaries)]),
                "luminance_cd_m2": None if mdcv_luminance is None else np.array([mdcv_luminance, 0.0]),
            }
            for key, expected_value in mdcv_expected_dict.items():
                if expected_value is None:
                    self.assertEqual(parse_data['mdcv'], expected_value)
                else:
                    np.testing.assert_almost_equal(parse_data['mdcv'][key], expected_value, decimal=4)

            clli_expected_dict = {
                "light_level_cd_m2": None if clli_luminance is None else np.array([clli_luminance, clli_luminance]),
            }
            for key, expected_value in clli_expected_dict.items():
                if expected_value is None:
                    self.assertEqual(parse_data['clli'], expected_value)
                else:
                    np.testing.assert_almost_equal(parse_data['clli'][key], expected_value, decimal=4)

            """
                'cicp': 
                    {'chromaticity': 
                        array([[ 0.3127,  0.329 ],
                            [ 0.708 ,  0.292 ],
                            [ 0.17  ,  0.797 ],
                            [ 0.131 ,  0.046 ]]),
                    'range': 'Full range'
                    },
                'mdcv':
                    {'chromaticity':
                        array([[ 0.3127,  0.329 ],
                            [ 0.64  ,  0.33  ],
                            [ 0.3   ,  0.6   ],
                            [ 0.15  ,  0.06  ]]),
                    'luminance_cd_m2': array([ 10000.,      0.])
                    },
                'clli':
                    {'light_level_cd_m2': array([ 100.,  100.])}
                    }
            """


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # unittest.main()
    # unittest.main(defaultTest="TestHevcMetadata.test_bitstream_metadata")
    # unittest.main(defaultTest="TestAvifMetadata.test_container_metadata")
    unittest.main(defaultTest="TestPngMetadata.test_chunk_data")
