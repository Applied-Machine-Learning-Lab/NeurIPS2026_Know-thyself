#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DEPS="$ROOT/third_party"
KT="$ROOT"

clone_at() {
    local url="$1" dir="$2" commit="$3"
    if [[ ! -d "$dir/.git" ]]; then
        git clone "$url" "$dir"
    fi
    git -C "$dir" checkout --quiet "$commit"
    echo "[deps] $(basename "$dir") @ $commit"
}

clone_at https://github.com/hiyouga/LlamaFactory.git "$DEPS/LlamaFactory" \
    "$(cat "$KT/llamafactory_patches/BASE_COMMIT")"
clone_at https://github.com/hiyouga/EasyR1.git "$DEPS/EasyR1" \
    "$(cat "$KT/easyr1_patches/BASE_COMMIT")"
clone_at https://github.com/EleutherAI/lm-evaluation-harness.git "$DEPS/lm-evaluation-harness" \
    69145a03d3e6ad0e552f729b7fe57b2762e7e4ca
clone_at https://github.com/Trustworthy-Information-Access/Annotation-Efficient-Universal-Honesty-Alignment.git "$DEPS/EliCal" \
    1b566b65886f83fc9ef287e9927e86bb2bd95071

if git -C "$DEPS/LlamaFactory" apply --check "$KT/llamafactory_patches/llamafactory.patch" 2>/dev/null; then
    git -C "$DEPS/LlamaFactory" apply "$KT/llamafactory_patches/llamafactory.patch"
    echo "[deps] applied llamafactory.patch"
else
    echo "[deps] llamafactory.patch already applied or does not match; skipped"
fi

EASYR1_ROOT="$DEPS/EasyR1" bash "$KT/easyr1_patches/install.sh"
