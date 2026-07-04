#!/usr/bin/env python3
"""Generate HDR metadata test images and the browser test page."""

from __future__ import annotations

import itertools
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[1]
sys.path.append(str(REPOSITORY_ROOT / "ty_lib"))

import test_pattern_generator2 as tpg


SOURCE = REPOSITORY_ROOT / (
    "2019/012_colour_v0.3.14_check/img/"
    "SMPTE ST2084_ITU-R BT.2020_D65_1920x1080_rev08_type1.png")
OUTPUT_DIR = ROOT / "test_img"
ICC_PATH = ROOT / "icc/bt2020_PQ_with_CICP.icc"
HTML_PATH = ROOT / "hdr_judge_check.html"
CICP = bytes((9, 16, 0, 1))
CLLI = (203, 203)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def run(command: list[str], capture: bool = False) -> str:
    """Run a command and stop immediately when it fails.

    Parameters
    ----------
    command : list of str
        Command and arguments to execute.
    capture : bool, optional
        Return standard output when true.

    Returns
    -------
    str
        Captured standard output, or an empty string.

    Examples
    --------
    >>> run(["true"])
    ''
    """
    result = subprocess.run(
        command, check=False, text=True, capture_output=capture)
    if result.returncode:
        detail = result.stderr if capture else ""
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{detail}")
    return result.stdout if capture else ""


def png_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    """Parse complete PNG chunks.

    Parameters
    ----------
    data : bytes
        PNG file data.

    Returns
    -------
    list of tuple of bytes
        Chunk type and payload pairs.

    Examples
    --------
    >>> png_chunks(PNG_SIGNATURE + struct.pack(">I", 0) + b"IEND" + b"\xaeB`\x82")
    [(b'IEND', b'')]
    """
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("Invalid PNG signature")
    result = []
    offset = len(PNG_SIGNATURE)
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError("Truncated PNG chunk")
        size = struct.unpack_from(">I", data, offset)[0]
        end = offset + 12 + size
        if end > len(data):
            raise ValueError("Invalid PNG chunk size")
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + size]
        result.append((kind, payload))
        offset = end
        if kind == b"IEND":
            break
    return result


def make_png_chunk(kind: bytes, payload: bytes) -> bytes:
    """Build one PNG chunk.

    Parameters
    ----------
    kind : bytes
        Four-byte chunk type.
    payload : bytes
        Chunk payload.

    Returns
    -------
    bytes
        Encoded PNG chunk.

    Examples
    --------
    >>> make_png_chunk(b"IEND", b"").hex()
    '0000000049454e44ae426082'
    """
    return (struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))


def write_clean_png(source: Path, destination: Path) -> None:
    """Remove color and HDR metadata without changing PNG image data.

    Parameters
    ----------
    source : Path
        Source PNG path.
    destination : Path
        Clean PNG path.

    Returns
    -------
    None

    Examples
    --------
    >>> write_clean_png(Path("source.png"), Path("clean.png"))
    """
    removed = {
        b"cICP", b"cLLI", b"iCCP", b"sRGB", b"gAMA", b"cHRM", b"eXIf"}
    chunks = png_chunks(source.read_bytes())
    destination.write_bytes(PNG_SIGNATURE + b"".join(
        make_png_chunk(kind, payload)
        for kind, payload in chunks if kind not in removed))


def make_png(clean: Path, output: Path, cicp: bool, clli: bool,
             icc: bool) -> None:
    """Create one PNG metadata combination.

    Parameters
    ----------
    clean : Path
        Metadata-free source PNG.
    output : Path
        Output PNG path.
    cicp : bool
        Add the PNG cICP chunk.
    clli : bool
        Add the PNG cLLI chunk.
    icc : bool
        Add the ICC profile.

    Returns
    -------
    None

    Examples
    --------
    >>> make_png(Path("clean.png"), Path("out.png"), True, False, False)
    """
    tpg.add_hdr_info_to_png(
        src_png=str(clean),
        dst_png=str(output),
        clli=CLLI if clli else None,
        cicp=list(CICP) if cicp else None)
    if icc:
        temporary = output.with_suffix(".icc-temp.png")
        tpg.add_icc_profile_using_exiftool(
            input_img_fname=str(output),
            output_img_fname=str(temporary),
            icc_profile_fname=str(ICC_PATH))
        temporary.replace(output)


def disable_bmff_nclx(path: Path) -> None:
    """Disable BMFF colr properties without changing coded image bytes.

    Parameters
    ----------
    path : Path
        AVIF or HEIC file to update.

    Returns
    -------
    None

    Notes
    -----
    libavif and libheif always emit an nclx property. Replacing the property
    box type with a free-space box retains all offsets and coded bytes while
    making the color profile absent to conforming BMFF readers.

    Examples
    --------
    >>> disable_bmff_nclx(Path("image.avif"))
    """
    data = path.read_bytes()
    positions = [index for index in range(len(data) - 7)
                 if data[index:index + 8] == b"colrnclx"]
    if not positions:
        if b"colrprof" in data or b"colrrICC" in data:
            return
        raise ValueError(f"Expected an nclx or ICC colr property in {path}")
    if len(positions) != 1:
        raise ValueError(f"Expected exactly one nclx colr box in {path}")
    position = positions[0]
    path.write_bytes(data[:position] + b"free" + data[position + 4:])


def encode_avif(clean: Path, output: Path, cicp: bool, clli: bool,
                icc: bool) -> None:
    """Encode one 10-bit lossless RGB AVIF combination.

    Parameters
    ----------
    clean : Path
        Metadata-free source PNG.
    output : Path
        Output AVIF path.
    cicp : bool
        Add CICP 9-16-0-1.
    clli : bool
        Add CLLI 203/203.
    icc : bool
        Add the generated ICC profile.

    Returns
    -------
    None

    Examples
    --------
    >>> encode_avif(Path("clean.png"), Path("out.avif"), True, True, False)
    """
    command = ["avifenc", "--lossless", "--depth", "10", "--yuv", "444",
               "--range", "full", "--ignore-icc", "--ignore-exif"]
    if cicp:
        command += ["--cicp", "9/16/0"]
    if clli:
        command += ["--clli", "203,203"]
    if icc:
        command += ["--icc", str(ICC_PATH)]
    command += [str(clean), str(output)]
    run(command)
    if not cicp:
        disable_bmff_nclx(output)


def encode_heic(clean: Path, output: Path, cicp: bool, clli: bool) -> None:
    """Encode one 10-bit lossless RGB HEIC combination.

    Parameters
    ----------
    clean : Path
        Metadata-free source PNG.
    output : Path
        Output HEIC path.
    cicp : bool
        Add CICP 9-16-0-1.
    clli : bool
        Add CLLI 203/203.

    Returns
    -------
    None

    Examples
    --------
    >>> encode_heic(Path("clean.png"), Path("out.heic"), True, True)
    """
    command = ["heif-enc", "--lossless", "--bit-depth", "10"]
    if cicp:
        command += ["--matrix_coefficients", "0", "--colour_primaries", "9",
                    "--transfer_characteristic", "16",
                    "--full_range_flag", "1"]
    if clli:
        command += ["--clli", "203,203"]
    command += [str(clean), "-o", str(output)]
    run(command)
    if not cicp:
        disable_bmff_nclx(output)


def encode_jxl(clean: Path, output: Path, cicp: bool, icc: bool) -> None:
    """Encode one 16-bit lossless JPEG XL combination.

    Parameters
    ----------
    clean : Path
        Metadata-free source PNG.
    output : Path
        Output JPEG XL path.
    cicp : bool
        Use the BT.2020/PQ structured color encoding.
    icc : bool
        Add the generated ICC profile.

    Returns
    -------
    None

    Examples
    --------
    >>> encode_jxl(Path("clean.png"), Path("out.jxl"), True, False)
    """
    command = ["cjxl", str(clean), str(output), "-q", "100"]
    if cicp:
        command += ["-x", "color_space=RGB_D65_202_Rel_PeQ"]
    if icc:
        command += ["-x", f"icc_pathname={ICC_PATH}"]
    run(command)


def state(value: bool) -> str:
    """Convert a boolean to a filename state.

    Parameters
    ----------
    value : bool
        State to convert.

    Returns
    -------
    str
        ``on`` or ``off``.

    Examples
    --------
    >>> state(True)
    'on'
    """
    return "on" if value else "off"


def output_name(fmt: str, cicp: bool, clli: bool, icc: bool) -> str:
    """Create a descriptive output filename.

    Parameters
    ----------
    fmt : str
        Image format and extension.
    cicp : bool
        CICP state.
    clli : bool
        CLLI state.
    icc : bool
        ICC state.

    Returns
    -------
    str
        Output filename.

    Examples
    --------
    >>> output_name("png", True, False, True)
    'png_cicp-on_clli-off_icc-on.png'
    """
    return (f"{fmt}_cicp-{state(cicp)}_clli-{state(clli)}_"
            f"icc-{state(icc)}.{fmt}")


def create_html(records: list[tuple[str, bool, bool, bool, str]]) -> None:
    """Write the standalone browser test page.

    Parameters
    ----------
    records : list of tuple
        Format, metadata states, and filename for every image.

    Returns
    -------
    None

    Examples
    --------
    >>> create_html([])
    """
    sections = []
    for fmt in ("png", "avif", "heic", "jxl"):
        rows = []
        for _, cicp, clli, icc, filename in [r for r in records if r[0] == fmt]:
            label = (f"CICP: {state(cicp)}, CLLI: {state(clli)}, "
                     f"ICC: {state(icc)}")
            path = f"test_img/{filename}"
            rows.append(f"""<tr><td>{state(cicp)}</td><td>{state(clli)}</td>
<td>{state(icc)}</td><td><a href="{path}" target="_blank" rel="noopener">{label}</a></td>
<td><img src="{path}" alt="{label}" loading="lazy"></td></tr>""")
        sections.append(f"""<h2>{fmt.upper()}</h2><table><thead><tr><th>CICP</th>
<th>CLLI</th><th>ICC Profile</th><th>Open image</th><th>Inline preview</th></tr>
</thead><tbody>{''.join(rows)}</tbody></table>""")
    HTML_PATH.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>HDR metadata test</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;background:#181818;color:#eee}}
table{{border-collapse:collapse;width:100%;margin-bottom:32px}}th,td{{border:1px solid #666;padding:8px}}
th{{background:#333}}a{{color:#8cf}}img{{display:block;max-width:320px;max-height:180px;background:#222}}</style>
</head><body><h1>HDR metadata test images</h1><p>CICP target: 9-16-0-1; MaxCLL/MaxFALL: 203/203 nits.</p>
<p><strong>JPEG XL limitation:</strong> cjxl 0.12.0 replaces the structured BT.2020/PQ
color encoding when an ICC profile is supplied. Therefore the JXL CICP-on/ICC-on file
contains CICP 9-16-0-1 in its ICC profile, but not as a second structured color encoding.</p>
{''.join(sections)}</body></html>\n""", encoding="utf-8")


def extract_icc_cicp(profile: bytes) -> bytes | None:
    """Read a CICP tag from an ICC profile.

    Parameters
    ----------
    profile : bytes
        Complete ICC profile data.

    Returns
    -------
    bytes or None
        Four CICP bytes when present.

    Examples
    --------
    >>> extract_icc_cicp(b"") is None
    True
    """
    if len(profile) < 132:
        return None
    count = struct.unpack_from(">I", profile, 128)[0]
    for index in range(count):
        entry = 132 + index * 12
        signature, offset, size = struct.unpack_from(">4sII", profile, entry)
        if signature == b"cicp":
            if offset + size > len(profile) or size < 12:
                raise ValueError("Invalid ICC CICP tag")
            return profile[offset + 8:offset + 12]
    return None


def verify(records: list[tuple[str, bool, bool, bool, str]], clean: Path) -> None:
    """Verify generated metadata, pixels, file count, and HTML links.

    Parameters
    ----------
    records : list of tuple
        Expected output records.
    clean : Path
        Metadata-free reference PNG.

    Returns
    -------
    None

    Examples
    --------
    >>> verify([], Path("clean.png"))
    """
    if len(records) != 24 or len(list(OUTPUT_DIR.glob("*"))) != 24:
        raise ValueError("Exactly 24 output images were not generated")
    html = HTML_PATH.read_text(encoding="utf-8")
    if any(f"test_img/{record[4]}" not in html for record in records):
        raise ValueError("HTML does not link all 24 images")
    if extract_icc_cicp(ICC_PATH.read_bytes()) != CICP:
        raise ValueError("Generated ICC profile does not contain CICP 9-16-0-1")

    reference_idat = [payload for kind, payload in png_chunks(clean.read_bytes())
                      if kind == b"IDAT"]
    with tempfile.TemporaryDirectory() as temp_name:
        temp = Path(temp_name)
        decoded_reference: dict[tuple[str, bool], bytes] = {}
        for fmt, cicp, clli, icc, filename in records:
            path = OUTPUT_DIR / filename
            info = ""
            has_icc = False
            if fmt == "png":
                run(["pngcheck", str(path)], capture=True)
                chunks = png_chunks(path.read_bytes())
                kinds = [kind for kind, _ in chunks]
                if (b"cICP" in kinds) != cicp or (b"cLLI" in kinds) != clli:
                    raise ValueError(f"Unexpected PNG HDR metadata: {filename}")
                if [p for k, p in chunks if k == b"IDAT"] != reference_idat:
                    raise ValueError(f"PNG pixels changed: {filename}")
                profiles = [payload for kind, payload in chunks if kind == b"iCCP"]
                has_icc = bool(profiles)
                if profiles:
                    nul = profiles[0].index(b"\0")
                    embedded = zlib.decompress(profiles[0][nul + 2:])
                    if extract_icc_cicp(embedded) != CICP:
                        raise ValueError(f"Invalid embedded ICC CICP: {filename}")
            elif fmt in {"avif", "heic"}:
                info = run(["heif-info", str(path)], capture=True)
                expected_profile = "colour primaries: 9" in info
                if expected_profile != cicp:
                    raise ValueError(f"Unexpected BMFF CICP: {filename}")
                expected_clli = "MaxCLL=203 MaxFALL=203" in info
                if expected_clli != clli:
                    raise ValueError(f"Unexpected BMFF CLLI: {filename}")
                has_icc = "color profile: prof" in info
                decoder = "avifdec" if fmt == "avif" else "heif-dec"
                decoded = temp / f"{filename}.png"
                run([decoder, str(path), str(decoded)])
                key = (fmt, cicp)
                pixels = b"".join(p for k, p in png_chunks(decoded.read_bytes())
                                  if k == b"IDAT")
                if key in decoded_reference and decoded_reference[key] != pixels:
                    raise ValueError(f"Decoded pixels vary by metadata: {filename}")
                decoded_reference[key] = pixels
            else:
                info = run(["jxlinfo", str(path)], capture=True)
                has_pq = "Transfer function: PQ" in info
                if has_pq != cicp and not (cicp and icc):
                    raise ValueError(f"Unexpected JPEG XL CICP: {filename}")
                has_icc = "ICC" in info
                decoded = temp / f"{filename}.png"
                run(["djxl", str(path), str(decoded)])
                key = (fmt, cicp)
                pixels = b"".join(p for k, p in png_chunks(decoded.read_bytes())
                                  if k == b"IDAT")
                if key in decoded_reference and decoded_reference[key] != pixels:
                    raise ValueError(f"Decoded pixels vary by metadata: {filename}")
                decoded_reference[key] = pixels
            if has_icc != icc:
                raise ValueError(f"Unexpected ICC profile state: {filename}")


def main() -> None:
    """Generate and verify all requested artifacts.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Examples
    --------
    >>> main()
    """
    required = [
        "exiftool", "avifenc", "avifdec", "pngcheck", "heif-enc",
        "heif-dec", "cjxl", "djxl", "jxlinfo", "heif-info", "iccFromXml"]
    missing = [command for command in required if shutil.which(command) is None]
    if missing:
        raise FileNotFoundError(f"Required commands not found: {', '.join(missing)}")
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Source image not found: {SOURCE}")

    run(["python3", str(ROOT / "create_icc_profile.py")])
    OUTPUT_DIR.mkdir(exist_ok=True)
    for existing in OUTPUT_DIR.iterdir():
        if not existing.is_file():
            raise ValueError(f"Unexpected non-file output entry: {existing}")
        existing.unlink()
    records = []
    with tempfile.TemporaryDirectory() as temp_name:
        clean = Path(temp_name) / "clean.png"
        write_clean_png(SOURCE, clean)
        for cicp, clli, icc in itertools.product((False, True), repeat=3):
            filename = output_name("png", cicp, clli, icc)
            make_png(clean, OUTPUT_DIR / filename, cicp, clli, icc)
            records.append(("png", cicp, clli, icc, filename))
            filename = output_name("avif", cicp, clli, icc)
            encode_avif(clean, OUTPUT_DIR / filename, cicp, clli, icc)
            records.append(("avif", cicp, clli, icc, filename))
        for cicp, clli in itertools.product((False, True), repeat=2):
            filename = output_name("heic", cicp, clli, False)
            encode_heic(clean, OUTPUT_DIR / filename, cicp, clli)
            records.append(("heic", cicp, clli, False, filename))
        for cicp, icc in itertools.product((False, True), repeat=2):
            filename = output_name("jxl", cicp, False, icc)
            encode_jxl(clean, OUTPUT_DIR / filename, cicp, icc)
            records.append(("jxl", cicp, False, icc, filename))
        create_html(records)
        verify(records, clean)
    print("Generated and verified 24 images and hdr_judge_check.html")
    print("WARNING: JPEG XL cannot retain structured CICP and ICC simultaneously")


if __name__ == "__main__":
    main()
