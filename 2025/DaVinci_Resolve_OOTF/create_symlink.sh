#!/bin/bash

# パラメータ
SRC="./DCTL"
DST="/Library/Application Support/Blackmagic Design/DaVinci Resolve/LUT/DCTL_OOTF"

# 絶対パスに変換
SRC_ABS=$(realpath "$SRC")

# ソースディレクトリの存在確認
if [ ! -d "$SRC_ABS" ]; then
  echo "Error: Source directory does not exist: $SRC_ABS"
  exit 1
fi

# 既存リンクやディレクトリの確認
if [ -e "$DST" ]; then
  echo "Error: Destination already exists: $DST"
  exit 1
fi

# シンボリックリンク作成（管理者権限が必要）
sudo ln -s "$SRC_ABS" "$DST"
if [ $? -eq 0 ]; then
  echo "Symbolic link created successfully: $SRC_ABS -> $DST"
else
  echo "Error: Failed to create symbolic link."
  exit 1
fi
