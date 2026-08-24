#!/usr/bin/env bash
# Push the Space files in this directory to the Hugging Face Space.
#
# The Space repo only holds the container definition; the model code is
# installed from GitHub by the Dockerfile. So the deploy order matters:
#
#   1. push miburi changes to github.com/m-hamza-mughal/miburi
#   2. run this script
#
# Step 1 is not optional -- the image builds `pip install miburi @ git+...`,
# so anything still only on your disk will not be in the image.
#
# Usage:
#   HF_TOKEN=hf_...  ./sync.sh                 # push to the default Space
#   MIBURI_SPACE_ID=user/other ./sync.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPACE_ID="${MIBURI_SPACE_ID:-m-hamza-mughal/miburi}"
WORK="${MIBURI_SPACE_CLONE:-${TMPDIR:-/tmp}/miburi-space-clone}"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "sync.sh: HF_TOKEN is not set (needs write access to ${SPACE_ID})" >&2
    exit 1
fi

REMOTE="https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE_ID}"

if [[ ! -d "${WORK}/.git" ]]; then
    echo "sync.sh: cloning ${SPACE_ID} -> ${WORK}"
    rm -rf "${WORK}"
    git clone "${REMOTE}" "${WORK}"
else
    git -C "${WORK}" remote set-url origin "${REMOTE}"
    git -C "${WORK}" fetch origin
    git -C "${WORK}" reset --hard origin/main
fi

# Everything here except this script and any local scratch.
rsync -av --delete \
    --exclude '.git' \
    --exclude 'sync.sh' \
    --exclude '__pycache__' \
    "${HERE}/" "${WORK}/"

cd "${WORK}"
git lfs install --local >/dev/null 2>&1 || true
git add -A
if git diff --cached --quiet; then
    echo "sync.sh: nothing to push"
    exit 0
fi
git -c user.name="miburi-sync" -c user.email="noreply@huggingface.co" \
    commit -m "sync from miburi repo ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
git push origin main
echo "sync.sh: pushed. Build logs: hf spaces logs ${SPACE_ID} --build"
