#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
EASYR1_ROOT="${EASYR1_ROOT:-$HERE/../third_party/EasyR1}"

if [[ ! -d "$EASYR1_ROOT" ]]; then
    echo "[error] EasyR1 root not found at $EASYR1_ROOT"
    echo "        set EASYR1_ROOT=/path/to/EasyR1"
    exit 1
fi

if git -C "$EASYR1_ROOT" apply --check "$HERE/verl.patch" 2>/dev/null; then
    git -C "$EASYR1_ROOT" apply "$HERE/verl.patch"
    echo "[install] applied verl.patch"
else
    echo "[install] verl.patch already applied or does not match; skipped"
fi

mkdir -p "$EASYR1_ROOT/examples/reward_function" \
         "$EASYR1_ROOT/examples/format_prompt"

cp "$HERE"/reward_function/*.py "$EASYR1_ROOT/examples/reward_function/"
cp "$HERE"/format_prompt/*.jinja "$EASYR1_ROOT/examples/format_prompt/"
cp "$HERE/config.yaml" "$EASYR1_ROOT/examples/config.yaml"

echo "[install] copied reward functions, format prompts and config.yaml into $EASYR1_ROOT/examples"
