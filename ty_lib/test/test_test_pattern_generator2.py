"""Tests for :mod:`test_pattern_generator2`.

Usage
-----
Run the following command from the ``ty_lib`` directory::

    python -m pytest -q test/test_test_pattern_generator2.py

The ICC integration test requires ExifTool, avifenc/avifdec, cjxl/djxl, and
heif-enc/heif-dec. It is skipped when any of these commands is unavailable.
"""

import subprocess
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import test_pattern_generator2 as tpg


def test_add_hdr_info_to_png_preserves_rgb_values(tmp_path):
    """Verify HDR metadata insertion without changing RGB samples.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory supplied by pytest.

    Returns
    -------
    None

    Examples
    --------
    Run this test with pytest:

    ``pytest test/test_test_pattern_generator2.py``
    """
    repository_root = Path(__file__).resolve().parents[2]
    source_png = repository_root / (
        "2019/012_colour_v0.3.14_check/img/"
        "Gamma 2.4_ITU-R BT.709_D65_1920x1080_rev07_type1.png")
    destination_png = tmp_path / "hdr_metadata.png"
    rgb_before = tpg.img_read_as_float(str(source_png))

    tpg.add_hdr_info_to_png(
        src_png=str(source_png),
        dst_png=str(destination_png),
        clli=(1000.0, 400.0),
        cicp=[9, 16, 9, 1])

    rgb_after = tpg.img_read_as_float(str(destination_png))
    output_data = destination_png.read_bytes()
    np.testing.assert_array_equal(rgb_after, rgb_before)
    assert b"cICP" in output_data
    assert b"cLLI" in output_data
    first_image_chunk = min(
        output_data.index(b"PLTE")
        if b"PLTE" in output_data else len(output_data),
        output_data.index(b"IDAT"))
    assert output_data.index(b"cICP") < first_image_chunk
    assert output_data.index(b"cLLI") < first_image_chunk


@pytest.fixture
def input_files(tmp_path):
    input_path = tmp_path / "input.png"
    profile_path = tmp_path / "profile.icc"
    input_path.write_bytes(b"image data")
    profile_path.write_bytes(b"profile data")
    return input_path, profile_path


def test_add_icc_profile_using_exiftool(input_files, tmp_path):
    input_path, profile_path = input_files
    output_path = tmp_path / "output.PNG"

    def run_exiftool(command, **kwargs):
        Path(command[2]).write_bytes(input_path.read_bytes() + b" with ICC")
        return subprocess.CompletedProcess(command, 0, "", "")

    with patch("test_pattern_generator2.shutil.which",
               return_value="/usr/bin/exiftool"), \
            patch("test_pattern_generator2.subprocess.run",
                  side_effect=run_exiftool) as run:
        result = tpg.add_icc_profile_using_exiftool(
            str(input_path), str(output_path), str(profile_path))

    assert result is None
    assert output_path.read_bytes() == b"image data with ICC"
    assert run.call_args.args[0][1] == "-o"
    assert "-EXIF:All=" in run.call_args.args[0]
    assert f"-ICC_Profile<={profile_path}" in run.call_args.args[0]


@pytest.mark.parametrize(
    "extension", [".avif", ".png", ".jxl", ".heif", ".heic"])
def test_add_icc_profile_accepts_supported_extensions(
        extension, input_files, tmp_path):
    original_input, profile_path = input_files
    input_path = original_input.with_suffix(extension.upper())
    original_input.rename(input_path)
    output_path = tmp_path / f"output{extension}"

    def run_exiftool(command, **kwargs):
        Path(command[2]).write_bytes(b"output")
        return subprocess.CompletedProcess(command, 0, "", "")

    with patch("test_pattern_generator2.shutil.which",
               return_value="exiftool"), \
            patch("test_pattern_generator2.subprocess.run",
                  side_effect=run_exiftool):
        tpg.add_icc_profile_using_exiftool(
            str(input_path), str(output_path), str(profile_path))

    assert output_path.read_bytes() == b"output"


@pytest.mark.parametrize(
    "input_name,output_name",
    [("input.tiff", "output.tiff"), ("input.png", "output.avif")])
def test_add_icc_profile_rejects_invalid_extensions(
        input_name, output_name, tmp_path):
    with patch("test_pattern_generator2.subprocess.run") as run:
        with pytest.raises(ValueError):
            tpg.add_icc_profile_using_exiftool(
                str(tmp_path / input_name), str(tmp_path / output_name),
                str(tmp_path / "profile.icc"))
    run.assert_not_called()


def test_add_icc_profile_rejects_same_file(input_files):
    input_path, profile_path = input_files
    with pytest.raises(ValueError, match="different files"):
        tpg.add_icc_profile_using_exiftool(
            str(input_path), str(input_path), str(profile_path))


def test_add_icc_profile_requires_input_and_profile(tmp_path):
    with pytest.raises(FileNotFoundError, match="Input image"):
        tpg.add_icc_profile_using_exiftool(
            str(tmp_path / "missing.png"), str(tmp_path / "output.png"),
            str(tmp_path / "missing.icc"))

    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"image")
    with pytest.raises(FileNotFoundError, match="ICC profile"):
        tpg.add_icc_profile_using_exiftool(
            str(input_path), str(tmp_path / "output.png"),
            str(tmp_path / "missing.icc"))


def test_add_icc_profile_requires_exiftool(input_files, tmp_path):
    input_path, profile_path = input_files
    with patch("test_pattern_generator2.shutil.which", return_value=None), \
            pytest.raises(FileNotFoundError, match="ExifTool"):
        tpg.add_icc_profile_using_exiftool(
            str(input_path), str(tmp_path / "output.png"), str(profile_path))


def test_add_icc_profile_preserves_existing_output_on_failure(
        input_files, tmp_path):
    input_path, profile_path = input_files
    output_path = tmp_path / "output.png"
    output_path.write_bytes(b"existing output")

    with patch("test_pattern_generator2.shutil.which",
               return_value="exiftool"), \
            patch("test_pattern_generator2.subprocess.run",
                  return_value=subprocess.CompletedProcess(
                      [], 1, "", "invalid image")):
        with pytest.raises(RuntimeError, match="invalid image"):
            tpg.add_icc_profile_using_exiftool(
                str(input_path), str(output_path), str(profile_path))

    assert input_path.read_bytes() == b"image data"
    assert output_path.read_bytes() == b"existing output"


def test_add_icc_profile_preserves_pixel_values_for_supported_formats(
        tmp_path):
    required_commands = {
        "exiftool", "avifenc", "avifdec", "pngcheck", "cjxl", "djxl",
        "jxlinfo", "heif-enc", "heif-dec", "heif-info"}
    missing_commands = sorted(
        command for command in required_commands if shutil.which(command) is None)
    if missing_commands:
        pytest.skip(f"Required commands not found: {', '.join(missing_commands)}")

    repository_root = Path(__file__).resolve().parents[2]
    source_png = repository_root / (
        "2019/012_colour_v0.3.14_check/img/"
        "Gamma 2.4_ITU-R BT.709_D65_1920x1080_rev07_type1.png")
    profile_path = repository_root / (
        "2026/04_ICC_Profile_for_HDR_Media/icc/"
        "bt2020_PQ_with_CICP.icc")
    if not source_png.is_file() or not profile_path.is_file():
        pytest.skip("The source image or ICC profile is not available")

    input_paths = {".png": tmp_path / "input.png"}
    shutil.copyfile(source_png, input_paths[".png"])
    encode_commands = {
        ".avif": [
            "avifenc", "-l", "-d", "10", "-r", "full",
            str(source_png), str(tmp_path / "input.avif")],
        ".jxl": [
            "cjxl", str(source_png), str(tmp_path / "input.jxl"),
            "-q", "100"],
        ".heif": [
            "heif-enc", "--lossless", "-b", "10", "-o",
            str(tmp_path / "input.heif"), str(source_png)],
        ".heic": [
            "heif-enc", "--lossless", "-b", "10", "-o",
            str(tmp_path / "input.heic"), str(source_png)],
    }
    for extension, command in encode_commands.items():
        subprocess.run(command, check=True, capture_output=True, text=True)
        input_paths[extension] = tmp_path / f"input{extension}"

    for extension, input_path in input_paths.items():
        output_path = tmp_path / f"output{extension}"
        tpg.add_icc_profile_using_exiftool(
            str(input_path), str(output_path), str(profile_path))

        info_command = {
            ".avif": ["avifdec", "--info", str(output_path)],
            ".png": ["pngcheck", str(output_path)],
            ".jxl": ["jxlinfo", str(output_path)],
            ".heif": ["heif-info", str(output_path)],
            ".heic": ["heif-info", str(output_path)],
        }[extension]
        info_result = subprocess.run(
            info_command, check=True, capture_output=True, text=True)
        info_text = f"{info_result.stdout}\n{info_result.stderr}"
        assert "error" not in info_text.lower(), info_text

        if extension == ".png":
            before_png = input_path
            after_png = output_path
        else:
            before_png = tmp_path / f"before_{extension[1:]}.png"
            after_png = tmp_path / f"after_{extension[1:]}.png"
            decoder = {
                ".avif": "avifdec",
                ".jxl": "djxl",
                ".heif": "heif-dec",
                ".heic": "heif-dec",
            }[extension]
            subprocess.run(
                [decoder, str(input_path), str(before_png)], check=True,
                capture_output=True, text=True)
            subprocess.run(
                [decoder, str(output_path), str(after_png)], check=True,
                capture_output=True, text=True)

        before = tpg.img_read(str(before_png))
        after = tpg.img_read(str(after_png))
        assert np.array_equal(before, after), extension
