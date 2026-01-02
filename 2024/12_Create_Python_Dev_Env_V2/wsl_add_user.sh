#!/usr/bin/env bash
set -euo pipefail

# ===== 設定値 =====
USERNAME="toru"
UID_NUM=1000
GID_NUM=1001
PASSWORD_HASH='$1$jQHQ4kCa$qx6X2s6ee1GqpGszqZNtV0'   # password: "dummy"
# ==================

# グループ作成（既に存在する場合はスキップ）
if ! getent group "${GID_NUM}" >/dev/null; then
    groupadd -g "${GID_NUM}" "${USERNAME}"
fi

# ユーザー作成（既に存在する場合はスキップ）
if ! id -u "${USERNAME}" >/dev/null 2>&1; then
    useradd -m -s /bin/bash -u "${UID_NUM}" -g "${GID_NUM}" -p "${PASSWORD_HASH}" "${USERNAME}"
fi

# sudo グループへ追加
usermod -aG sudo "${USERNAME}"

# 初回ログイン時にパスワード変更を強制
chage -d 0 "${USERNAME}"

echo "User '${USERNAME}' has been configured."
