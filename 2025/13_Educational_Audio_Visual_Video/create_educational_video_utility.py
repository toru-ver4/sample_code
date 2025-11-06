"""
細切れの .srt を結合して 1 本の .srt にするスクリプト。
- 入力: ./asset/{project_title}/{03d}-{project_title}.srt (例: 000-reirei.srt)
- 先頭行の "1" を (03dの数値 + 1) に置換してから結合
- 出力: ./asset/{project_title}/_{project_title}.srt
"""

import argparse
import glob
import os
import re
from pathlib import Path

SRT_GLOB_TEMPLATE = "./asset/{proj}/[0-9][0-9][0-9]-{proj}.srt"
OUT_TEMPLATE = "./asset/{proj}/_{proj}.srt"

def extract_03d_num(filename: str) -> int:
    """
    ファイル名から {03d} の数値を抽出 (例: 001-reirei.srt -> 1)
    """
    base = os.path.basename(filename)
    m = re.match(r"^(\d{3})-", base)
    if not m:
        raise ValueError(f"Unexpected filename format: {filename}")
    return int(m.group(1))

def rewrite_first_index(lines, new_index: int):
    """
    .srt ファイルの 1 行目 (通常は "1") を new_index に置換する。
    1 行目が数値なら上書きする。改行は元のまま維持。
    """
    if not lines:
        return lines

    first = lines[0]
    # 改行を保持
    newline = "\n"
    if first.endswith("\r\n"):
        newline = "\r\n"
    elif first.endswith("\n"):
        newline = "\n"
    else:
        # 行末に改行が無い場合は後で付ける
        newline = ""

    # 数値のみの行なら置換（空白は許容）
    if re.fullmatch(r"\s*\d+\s*", first.rstrip("\r\n")):
        lines[0] = f"{new_index}{newline}"
    # それ以外は触らない
    return lines

def main(project_title: str):
    in_glob = SRT_GLOB_TEMPLATE.format(proj=project_title)
    files = sorted(glob.glob(in_glob), key=extract_03d_num)

    if not files:
        raise SystemExit(f"No SRT files found for project '{project_title}' under ./asset/{project_title}/")

    out_path = OUT_TEMPLATE.format(proj=project_title)
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)

    chunks = []
    for fp in files:
        part_no = extract_03d_num(fp)          # 0,1,2,...
        new_index = part_no + 1                # 指定仕様: {03d} + 1

        # UTF-8(BOMあり/なし)の双方に耐性
        with open(fp, "r", encoding="utf-8-sig") as f:
            lines = f.read().splitlines(keepends=True)

        lines = rewrite_first_index(lines, new_index)

        # 各ファイル間の区切りとして、最後が改行で終わっていなければ 1 行足す
        if lines and not (lines[-1].endswith("\n") or lines[-1].endswith("\r\n")):
            lines[-1] = lines[-1] + "\n"

        chunks.append("".join(lines))

    # 連結して出力
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("".join(chunks))

    print(f"Written: {out_path}  (merged {len(files)} files)")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Merge fragmented SRTs into one.")
    parser.add_argument("project_title", help="プロジェクト名 (例: reirei)")
    args = parser.parse_args()
    main(args.project_title)
