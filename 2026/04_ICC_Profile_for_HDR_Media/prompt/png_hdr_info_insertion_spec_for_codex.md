# PNG HDR Metadata Insertion Specification for Codex

This document describes the expected implementation for a Python function that inserts HDR-related PNG chunks into an existing PNG file.

Target function:

```python
def add_hdr_info_to_png(
    src_png: str,
    dst_png: str,
    clli: tuple[float, float] | None = None,
    cicp: list[int] | None = None,
) -> None:
    ...
```

The function must support adding the following PNG chunks:

- `cLLI`: Content Light Level Information
- `cICP`: Coding-independent code points

If `clli` is `None`, the `cLLI` chunk must not be added.

If `cicp` is `None`, the `cICP` chunk must not be added.

If both `clli` and `cicp` are `None`, the function should simply copy `src_png` to `dst_png` without modifying the PNG structure.

Please add this function to `ty_lib\test_pattern_generator2.py`

---

## Reference specifications

Primary references:

- PNG Third Edition, W3C: <https://www.w3.org/TR/png-3/>
- PNG chunk structure: <https://www.w3.org/TR/png-3/#5Chunk-layout>
- `cICP` chunk: <https://www.w3.org/TR/png-3/#11cICP>
- `cLLI` chunk: <https://www.w3.org/TR/png-3/#11cLLI>

---

## PNG file structure overview

A PNG file consists of:

```text
[8-byte PNG signature]
[chunk 0]
[chunk 1]
[chunk 2]
...
[IEND chunk]
```

The PNG signature is always:

```text
89 50 4E 47 0D 0A 1A 0A
```

After the signature, the file is a sequence of chunks. PNG does not have a global table of chunk addresses. Chunk locations are determined by sequentially parsing the file from the beginning.

---

## PNG chunk layout

Each PNG chunk has this binary layout:

```text
Length      4 bytes  big-endian unsigned integer
Chunk Type  4 bytes  ASCII chunk name, e.g. "IHDR", "IDAT", "cICP"
Chunk Data  N bytes  where N == Length
CRC         4 bytes  CRC-32 of Chunk Type + Chunk Data
```

Therefore, the total size of a chunk is:

```text
Length + 12 bytes
```

because:

```text
4 bytes Length + 4 bytes Chunk Type + Length bytes Data + 4 bytes CRC
```

Important notes:

- `Length` describes only the size of `Chunk Data`.
- `Length` does not include the `Length` field itself.
- `Length` does not include the `Chunk Type` field.
- `Length` does not include the `CRC` field.
- The CRC is calculated over `Chunk Type + Chunk Data`.
- The CRC does not include the `Length` field.
- PNG does not have a file-level CRC.
- Each chunk has its own CRC.
- Multi-byte integers in PNG chunks are stored in big-endian byte order.

---

## Required parsing behavior

The function must parse the PNG file as follows:

1. Read the entire input file as bytes.
2. Verify the 8-byte PNG signature.
3. Start parsing chunks at byte offset `8`.
4. For each chunk:
   1. Read 4 bytes as `Length` using big-endian byte order.
   2. Read 4 bytes as `Chunk Type`.
   3. Skip `Length` bytes of `Chunk Data`.
   4. Skip 4 bytes of `CRC`.
   5. Move to the next chunk by adding `Length + 12` to the current offset.
5. Continue until the `IEND` chunk is found.

The function should validate at least the following:

- The file begins with the PNG signature.
- The first chunk after the signature is `IHDR`.
- The file contains an `IEND` chunk.
- Chunk lengths do not run past the end of the file.

---

## Insertion location

Both `cICP` and `cLLI` must be inserted before `PLTE` and before `IDAT`.

Recommended insertion rule:

```text
Insert new HDR metadata chunks immediately before the first PLTE or IDAT chunk, whichever appears first.
```

This rule works for both palette-based PNGs and truecolor PNGs:

- Indexed-color PNGs usually contain `PLTE` before `IDAT`.
- Truecolor PNGs may not contain `PLTE`, so insertion before the first `IDAT` is appropriate.

The function must not insert `cICP` or `cLLI` after `IDAT`.

---

## Existing chunk behavior

The implementation should avoid creating duplicate `cICP` or `cLLI` chunks.

Recommended behavior:

- If an existing `cICP` chunk is found and `cicp is not None`, replace the existing `cICP` chunk.
- If an existing `cLLI` chunk is found and `clli is not None`, replace the existing `cLLI` chunk.
- If an existing `cICP` chunk is found and `cicp is None`, preserve it unchanged.
- If an existing `cLLI` chunk is found and `clli is None`, preserve it unchanged.

This means `None` means “do not add or modify that chunk type”.

---

## Output chunk ordering

When both `cICP` and `cLLI` are newly added or replaced, write them in this order:

```text
cICP
cLLI
```

Recommended output structure:

```text
[PNG signature]
[IHDR]
[other chunks before PLTE/IDAT, excluding replaced cICP/cLLI]
[cICP, if cicp is not None]
[cLLI, if clli is not None]
[PLTE or IDAT and the remaining chunks]
```

Do not modify `IDAT` chunk contents.

Do not decompress or recompress image data.

Do not change RGB, YCbCr, palette, or alpha values.

---

## cICP chunk specification

The `cICP` chunk stores four 1-byte fields.

Chunk type:

```text
cICP
```

Chunk data length:

```text
4 bytes
```

Chunk data layout:

```text
color_primaries           1 byte
transfer_function         1 byte
matrix_coefficients       1 byte
video_full_range_flag     1 byte
```

Function argument:

```python
cicp: list[int] | None
```

Expected value when not `None`:

```python
[color_primaries, transfer_function, matrix_coefficients, video_full_range_flag]
```

Validation:

- `cicp` must be `None` or a list of exactly 4 integers.
- Each element must be in the range `0 <= value <= 255`.
- If `cicp is None`, do not add or replace the `cICP` chunk.

Example:

```python
cicp = [9, 16, 9, 1]
```

This corresponds to:

```text
color_primaries       = 9
transfer_function     = 16
matrix_coefficients   = 9
full_range_flag       = 1
```

The exact semantic meaning of the values is defined by the CICP code point registry/specification. The function only needs to encode the provided integer values into the PNG `cICP` chunk.

Binary chunk construction:

```text
Length      00 00 00 04
Chunk Type  63 49 43 50    ASCII "cICP"
Data        4 bytes from cicp list
CRC         CRC32("cICP" + Data)
```

Total chunk size:

```text
4 + 4 + 4 + 4 = 16 bytes
```

---

## cLLI chunk specification

The `cLLI` chunk stores content light level information.

Chunk type:

```text
cLLI
```

Chunk data length:

```text
8 bytes
```

Chunk data layout:

```text
MaxCLL   4 bytes, unsigned integer, big-endian
MaxFALL  4 bytes, unsigned integer, big-endian
```

The PNG `cLLI` chunk stores these values in units of `0.0001 cd/m^2`.

Function argument:

```python
clli: tuple[float, float] | None
```

Expected value when not `None`:

```python
(max_cll_nits, max_fall_nits)
```

where:

- `max_cll_nits` is MaxCLL in `cd/m^2`, commonly called nits.
- `max_fall_nits` is MaxFALL in `cd/m^2`, commonly called nits.

Validation:

- `clli` must be `None` or a tuple/list of exactly 2 numeric values.
- Both values must be finite numbers.
- Both values must be greater than or equal to `0`.
- After conversion to PNG units, each value must fit in an unsigned 32-bit integer.
- If `clli is None`, do not add or replace the `cLLI` chunk.

Conversion from luminance value to PNG integer value:

```python
png_value = round(luminance_nits / 0.0001)
```

Equivalent:

```python
png_value = round(luminance_nits * 10000)
```

Examples:

```text
1000.0 cd/m^2 -> 10000000 -> 0x00989680
400.0 cd/m^2  -> 4000000  -> 0x003D0900
```

Binary chunk construction:

```text
Length      00 00 00 08
Chunk Type  63 4C 4C 49    ASCII "cLLI"
Data        MaxCLL  4 bytes, big-endian
            MaxFALL 4 bytes, big-endian
CRC         CRC32("cLLI" + Data)
```

Total chunk size:

```text
4 + 4 + 8 + 4 = 20 bytes
```

---

## CRC calculation

Use standard CRC-32 as used by PNG.

In Python, use:

```python
import zlib

crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
crc_bytes = crc.to_bytes(4, "big")
```

The CRC input must be:

```text
Chunk Type + Chunk Data
```

Do not include the `Length` field in the CRC calculation.

---

## Recommended helper functions

Codex should implement small helper functions rather than putting all binary operations in one large function.

Recommended helpers:

```python
def _read_u32be(data: bytes, offset: int) -> int:
    ...


def _make_png_chunk(chunk_type: bytes, chunk_data: bytes) -> bytes:
    ...


def _make_cicp_chunk(cicp: list[int]) -> bytes:
    ...


def _make_clli_chunk(clli: tuple[float, float]) -> bytes:
    ...


def _iter_png_chunks(data: bytes):
    ...
```

`_make_png_chunk()` should create any PNG chunk from a 4-byte chunk type and arbitrary chunk data:

```python
def _make_png_chunk(chunk_type: bytes, chunk_data: bytes) -> bytes:
    length = len(chunk_data).to_bytes(4, "big")
    crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
    return length + chunk_type + chunk_data + crc.to_bytes(4, "big")
```

---

## Recommended algorithm

High-level algorithm:

```text
1. Read src_png as bytes.
2. Validate PNG signature.
3. Parse all chunks into a list of chunk records:
   - start offset
   - end offset
   - chunk type
   - chunk data
   - original raw bytes
4. Validate that the first chunk is IHDR.
5. Validate that IEND exists.
6. Build new cICP chunk if cicp is not None.
7. Build new cLLI chunk if clli is not None.
8. Rebuild the PNG byte stream:
   - Start with PNG signature.
   - Iterate through existing chunks in order.
   - Skip existing cICP if cicp is not None.
   - Skip existing cLLI if clli is not None.
   - Immediately before the first PLTE or IDAT, insert the newly requested chunks.
   - Write all other existing chunks unchanged.
9. If no PLTE or IDAT is found before IEND, raise ValueError because the PNG is malformed.
10. Write the result to dst_png.
```

Pseudo-code:

```python
def add_hdr_info_to_png(src_png, dst_png, clli=None, cicp=None):
    data = Path(src_png).read_bytes()
    validate_png_signature(data)
    chunks = parse_chunks(data)

    if clli is None and cicp is None:
        Path(dst_png).write_bytes(data)
        return

    new_chunks = []
    if cicp is not None:
        new_chunks.append(make_cicp_chunk(cicp))
    if clli is not None:
        new_chunks.append(make_clli_chunk(clli))

    output = bytearray()
    output += PNG_SIGNATURE

    inserted = False

    for chunk in chunks:
        chunk_type = chunk.type

        if cicp is not None and chunk_type == b"cICP":
            continue

        if clli is not None and chunk_type == b"cLLI":
            continue

        if not inserted and chunk_type in (b"PLTE", b"IDAT"):
            for new_chunk in new_chunks:
                output += new_chunk
            inserted = True

        output += chunk.raw_bytes

    if not inserted:
        raise ValueError("Could not find PLTE or IDAT insertion point")

    Path(dst_png).write_bytes(bytes(output))
```

---

## Error handling requirements

Raise `ValueError` for invalid input, including:

- Input file is not a PNG.
- PNG signature is invalid.
- First chunk is not `IHDR`.
- `IEND` chunk is missing.
- A chunk length points beyond the end of the file.
- `cicp` is not `None` and is not exactly 4 integers.
- `cicp` contains values outside `0..255`.
- `clli` is not `None` and is not exactly 2 numeric values.
- `clli` contains negative, infinite, or NaN values.
- Converted `cLLI` integer values do not fit in unsigned 32-bit integers.

---

## Non-goals

The function must not:

- Decode image pixels.
- Modify RGB values.
- Modify YCbCr values.
- Modify alpha values.
- Modify palette entries.
- Decompress `IDAT`.
- Recompress `IDAT`.
- Rewrite unrelated chunks.
- Add an ICC profile.
- Convert color spaces.
- Infer CICP values automatically.
- Infer CLLI values automatically from pixels.

The function only inserts or replaces PNG metadata chunks.

---

## Example usage

Add both `cICP` and `cLLI`:

```python
add_hdr_info_to_png(
    src_png="input.png",
    dst_png="output.png",
    cicp=[9, 16, 9, 1],
    clli=(1000.0, 400.0),
)
```

Add only `cICP`:

```python
add_hdr_info_to_png(
    src_png="input.png",
    dst_png="output.png",
    cicp=[9, 16, 9, 1],
    clli=None,
)
```

Add only `cLLI`:

```python
add_hdr_info_to_png(
    src_png="input.png",
    dst_png="output.png",
    cicp=None,
    clli=(1000.0, 400.0),
)
```

Add neither; copy input to output:

```python
add_hdr_info_to_png(
    src_png="input.png",
    dst_png="output.png",
    cicp=None,
    clli=None,
)
```

---

## Suggested implementation style

Use only Python standard library modules if possible:

```python
from pathlib import Path
import math
import zlib
```

The implementation should be deterministic and should preserve all unrelated PNG chunks byte-for-byte.

Only the following changes should occur:

- requested `cICP` chunk is inserted or replaced;
- requested `cLLI` chunk is inserted or replaced;
- if both arguments are `None`, the file is copied unchanged.

## テストについて

2026/04_ICC_Profile_for_HDR_Media/test/test_add_hdr_info_to_png.py にて実施すること。
内容は、cICP, cLLI の追加前、追加語で RGB値が変わっていないこと。
テスト画像には `2019\012_colour_v0.3.14_check\img\Gamma 2.4_ITU-R BT.709_D65_1920x1080_rev07_type1.png` を使うこと。
RGB値の読み込みには `ty_lib\test_pattern_generator2.py` の `img_read_as_float` を使うこと
