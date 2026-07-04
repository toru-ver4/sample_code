"""Tests for :mod:`make_test_images`."""

# Run from the repository root (/mnt/c/Users/toruv/OneDrive/work/sample_code):
# python -m pytest -q 2026/04_ICC_Profile_for_HDR_Media/test/test_make_test_images.py

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "make_test_images.py"
SPEC = importlib.util.spec_from_file_location("make_test_images", MODULE_PATH)
make_test_images = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(make_test_images)


def test_write_clean_png_removes_comparison_metadata(tmp_path):
    """Verify that source metadata is removed without changing IDAT data.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory supplied by pytest.

    Returns
    -------
    None

    Examples
    --------
    Run with ``pytest test/test_make_test_images.py``.
    """
    source = tmp_path / "source.png"
    clean = tmp_path / "clean.png"
    idat = b"coded pixel data"
    chunks = [
        (b"IHDR", b"header"),
        (b"eXIf", b"source metadata"),
        (b"cICP", make_test_images.CICP),
        (b"cLLI", b"light levels"),
        (b"iCCP", b"profile"),
        (b"IDAT", idat),
        (b"IEND", b""),
    ]
    source.write_bytes(
        make_test_images.PNG_SIGNATURE + b"".join(
            make_test_images.make_png_chunk(kind, payload)
            for kind, payload in chunks))

    make_test_images.write_clean_png(source, clean)

    clean_chunks = make_test_images.png_chunks(clean.read_bytes())
    assert clean_chunks == [
        (b"IHDR", b"header"), (b"IDAT", idat), (b"IEND", b"")]


@pytest.mark.parametrize(
    "cicp,clli,expected_cicp,expected_clli",
    [
        (False, False, None, None),
        (False, True, None, make_test_images.CLLI),
        (True, False, list(make_test_images.CICP), None),
        (True, True, list(make_test_images.CICP), make_test_images.CLLI),
    ])
def test_make_png_uses_test_pattern_generator_for_hdr_metadata(
        tmp_path, cicp, clli, expected_cicp, expected_clli):
    """Verify that PNG HDR metadata is delegated to the shared library.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory supplied by pytest.
    cicp : bool
        Whether CICP metadata is requested.
    clli : bool
        Whether CLLI metadata is requested.
    expected_cicp : list of int or None
        Expected shared-library CICP argument.
    expected_clli : tuple of int or None
        Expected shared-library CLLI argument.

    Returns
    -------
    None

    Examples
    --------
    Run with ``pytest test/test_make_test_images.py``.
    """
    clean = tmp_path / "clean.png"
    output = tmp_path / "output.png"

    with patch.object(make_test_images.tpg, "add_hdr_info_to_png") as add_hdr:
        make_test_images.make_png(clean, output, cicp, clli, False)

    add_hdr.assert_called_once_with(
        src_png=str(clean), dst_png=str(output),
        clli=expected_clli, cicp=expected_cicp)


def test_make_png_uses_test_pattern_generator_for_icc(tmp_path):
    """Verify that PNG ICC embedding is delegated to the shared library.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory supplied by pytest.

    Returns
    -------
    None

    Examples
    --------
    Run with ``pytest test/test_make_test_images.py``.
    """
    clean = tmp_path / "clean.png"
    output = tmp_path / "output.png"
    temporary = output.with_suffix(".icc-temp.png")

    def add_icc(**kwargs):
        Path(kwargs["output_img_fname"]).write_bytes(b"image with ICC")

    with patch.object(make_test_images.tpg, "add_hdr_info_to_png"), \
            patch.object(
                make_test_images.tpg, "add_icc_profile_using_exiftool",
                side_effect=add_icc) as embed_icc:
        make_test_images.make_png(clean, output, False, False, True)

    embed_icc.assert_called_once_with(
        input_img_fname=str(output),
        output_img_fname=str(temporary),
        icc_profile_fname=str(make_test_images.ICC_PATH))
    assert output.read_bytes() == b"image with ICC"
