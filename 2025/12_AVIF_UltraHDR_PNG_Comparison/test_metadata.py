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

import color_space as cs

from typing import Any


def find_values_by_key(obj: Any, target_key: str) -> list[Any]:
    """Collect all values whose key matches ``target_key`` from a nested JSON-like object.

    The function traverses dictionaries and lists iteratively (depth-first) and
    returns every value found for the specified key, regardless of nesting depth.

    Args:
        obj: A JSON-like structure (typically composed of dict, list, and scalar values).
        target_key: The dictionary key to search for.

    Returns:
        A list of matched values. Returns an empty list if the key is not found.
    """
    values: list[Any] = []
    stack: list[Any] = [obj]

    while stack:
        current = stack.pop()

        if isinstance(current, dict):
            if target_key in current:
                values.append(current[target_key])
                break
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)

    return values


def calc_expected_hevc_primaries_int(primary_str: str) -> list[list[int | None]]:
    primaries_int = [[None, None], [None, None], [None, None]]
    if primary_str is not None:
        unit = 0.00002
        primaries = cs.get_primaries(primary_str)
        primaries_int = np.round(primaries / unit).astype(np.uint16).tolist()

    return primaries_int


def calc_expected_hevc_primaries_float(primary_str: str) -> list[list[float | None]]:
    primaries = [[None, None], [None, None], [None, None]]
    if primary_str is not None:
        primaries = cs.get_primaries(primary_str).tolist()

    return primaries


class TestHevcMetadata(unittest.TestCase):

    # def test_bitstream_metadata(self):
    #     for mdcv_primaries, mdcv_luminance, clli_luminance\
    #         in product(MDCV_PRIMARIES_LIST, MDCV_LUMINANCE_LIST, CLLI_LUMINANCE_LIST):
    #         file_name_without_ext = make_media_file_name_without_ext(
    #             kind=KIND_HEVC,
    #             suffix=None,
    #             mdcv_primaries=mdcv_primaries,
    #             mdcv_luminance=mdcv_luminance,
    #             clli_luminance=clli_luminance
    #         )
    #         file_name = file_name_without_ext + ".h265"
    #         if os.path.exists(file_name) is False:
    #             continue
    #         json_file = f"./data/{Path(file_name).name}.json"
    #         with open(json_file, "r", encoding="utf-8") as f:
    #             data = json.load(f)

    #         cicp_expected_dict = {
    #             "-colour_primaries": 9,
    #             "-transfer_characteristic": 16,
    #             "-matrix_coeffs": 9,
    #             "-video_full_range_flag": 0,
    #         }
    #         for key, expected_value in cicp_expected_dict.items():
    #             actual_values = find_values_by_key(data, key)
    #             self.assertNotEqual(actual_values, [])
    #             for actual_value in actual_values:
    #                 print(f"[ASSERT] {key} actual={actual_value} expected={expected_value}")
    #                 self.assertEqual(int(actual_value), expected_value)

    #         expected_primaries = calc_expected_hevc_primaries_float(primary_str=mdcv_primaries)
    #         mdcv_primaries_dict = {
    #             "-display_primaries_x": [expected_primaries[ii][0] for ii in [1, 2, 0]],
    #             "-display_primaries_y": [expected_primaries[ii][1] for ii in [1, 2, 0]],
    #         }
    #         for key, expected_value in mdcv_primaries_dict.items():
    #             actual_values = find_values_by_key(data, key)
    #             if None in expected_value:
    #                 print(f"[ASSERT] {key} actual={actual_values} expected={expected_value}")
    #                 self.assertEqual(actual_values, [])
    #             else:
    #                 for actual_value in actual_values:
    #                     print(f"[ASSERT] {key} actual={actual_value} expected={expected_value}")
    #                     arr = np.fromstring(actual_value, sep=" ")
    #                     np.testing.assert_almost_equal(arr, np.array(expected_value), decimal=4)

    #         mdcv_clli_expected_dict = {
    #             "-white_point_x": 0.3127,
    #             "-white_point_y": 0.3290,
    #             "-max_display_mastering_luminance": mdcv_luminance,
    #             "-min_display_mastering_luminance": 0,
    #             "max_content_light_level": clli_luminance,
    #             "max_pic_average_light_level": clli_luminance,
    #         }

    #         for key, expected_value in mdcv_clli_expected_dict.items():
    #             actual_values = find_values_by_key(data, key)
    #             if expected_value is None:
    #                 print(f"[ASSERT] {key} actual={actual_values} expected={expected_value}")
    #                 self.assertEqual(actual_values, [])
    #             else:
    #                 for actual_value in actual_values:
    #                     print(f"[ASSERT] {key} actual={actual_value} expected={expected_value}")
    #                     self.assertAlmostEqual(float(actual_value), expected_value, places=4)

    def test_mp4_container_metadata(self):
        for mdcv_primaries, mdcv_luminance, clli_luminance\
            in product(MDCV_PRIMARIES_LIST, MDCV_LUMINANCE_LIST, CLLI_LUMINANCE_LIST):
            file_name_without_ext = make_media_file_name_without_ext(
                kind=KIND_HEVC,
                suffix=None,
                mdcv_primaries=mdcv_primaries,
                mdcv_luminance=mdcv_luminance,
                clli_luminance=clli_luminance
            )
            file_name = file_name_without_ext + ".mp4"
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
                    print(f"[ASSERT] {key} actual={actual_value} expected={expected_value}")
                    self.assertEqual(int(actual_value), expected_value)

            expected_primaries = calc_expected_hevc_primaries_int(primary_str=mdcv_primaries)
            mdcv_dict = {
                "-display_primaries_0_x": expected_primaries[0][0],
                "-display_primaries_0_x": expected_primaries[0][1],
                "-display_primaries_1_x": expected_primaries[1][0],
                "-display_primaries_1_x": expected_primaries[1][1],
                "-display_primaries_2_x": expected_primaries[2][0],
                "-display_primaries_2_x": expected_primaries[2][1],
                '-white_point_x': int(0.3127 * 0.00002),
                '-white_point_y': int(0.3290 * 0.00002),
                '-max_display_mastering_luminance': int(mdcv_luminance * 0.0001),
                '-min_display_mastering_luminance': 0,
                '-max_content_light_level': clli_luminance,
                '-max_pic_average_light_level': clli_luminance,
            }
            for key, expected_value in mdcv_dict.items():
                actual_values = find_values_by_key(data, key)
                if None in expected_value:
                    print(f"[ASSERT] {key} actual={actual_values} expected={expected_value}")
                    self.assertEqual(actual_values, [])
                else:
                    for actual_value in actual_values:
                        print(f"[ASSERT] {key} actual={actual_value} expected={expected_value}")
                        arr = np.fromstring(actual_value, sep=" ")
                        np.testing.assert_almost_equal(arr, np.array(expected_value), decimal=4)

            break


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    unittest.main()
