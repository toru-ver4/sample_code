#!/usr/bin/env bash
set -euo pipefail

# ===== Settings =====
USERNAME="toru"
UID_NUM=1000
GID_NUM=1001
PASSWORD_HASH='$1$jQHQ4kCa$qx6X2s6ee1GqpGszqZNtV0'   # password: "dummy"
# =====================

# Create group (skip if it already exists)
if ! getent group "${GID_NUM}" >/dev/null; then
    groupadd -g "${GID_NUM}" "${USERNAME}"
fi

# Create user (skip if it already exists)
if ! id -u "${USERNAME}" >/dev/null 2>&1; then
    useradd -m -s /bin/bash -u "${UID_NUM}" -g "${GID_NUM}" -p "${PASSWORD_HASH}" "${USERNAME}"
fi

# Add to sudo group
usermod -aG sudo "${USERNAME}"

# Add to docker group
sudo usermod -aG docker "${USERNAME}"

# Force password change on first login
chage -d 0 "${USERNAME}"

echo "User '${USERNAME}' has been configured."
