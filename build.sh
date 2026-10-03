#!/usr/bin/env sh
# TPT Local Chat - build pipeline (macOS/Linux). Windows canonical: build.ps1.
#
# Free edition only - this app is intentionally not a revenue line (apps.md
# §T2-19): no Pro feature flag, no Gumroad listing, no desktop build.
#   cargo build --release --target wasm32-unknown-unknown, then
#   wasm-bindgen --target web -> $TPT_WEB_REPO/public/apps/local-chat/ (glue
#   named local-chat.js, wasm keeps its _bg.wasm name), otherwise dist/web.
#
# Env:
#   TPT_WEB_REPO  path to the web repo; the bundle is copied to
#                 $TPT_WEB_REPO/public/apps/local-chat/, otherwise dist/web.

set -eu
cd "$(dirname "$0")"

CRATE=tpt-app-local-chat
TARGET=wasm32-unknown-unknown
OUT_NAME=local-chat

if [ -n "${TPT_WEB_REPO:-}" ]; then
  DEST="$TPT_WEB_REPO/public/apps/local-chat"
else
  DEST="dist/web"
fi

echo "== TPT Local Chat build =="
echo " edition: free (the only tier - local model, in-browser)"
echo " dest:    $DEST"

cargo build --release --target "$TARGET" -p "$CRATE"

WASM="target/$TARGET/release/tpt_app_local_chat.wasm"
[ -f "$WASM" ] || { echo "missing $WASM" >&2; exit 1; }

mkdir -p "$DEST"
wasm-bindgen --target web --out-name "$OUT_NAME" --out-dir "$DEST" "$WASM"

GLUE="$DEST/$OUT_NAME.js"
BG_WASM="$DEST/${OUT_NAME}_bg.wasm"
[ -f "$GLUE" ] || { echo "wasm-bindgen did not produce $GLUE" >&2; exit 1; }
[ -f "$BG_WASM" ] || { echo "wasm-bindgen did not produce $BG_WASM" >&2; exit 1; }

# Payload size report (local-model weights are tracked separately from this
# code-size line - they are the dominant size cost for this app).
RAW=$(wc -c < "$BG_WASM" | tr -d ' ')
if command -v gzip >/dev/null 2>&1; then
  GZ=$(gzip -9 -c "$BG_WASM" | wc -c | tr -d ' ')
  echo "-- $(basename "$BG_WASM"): $RAW bytes raw, $GZ bytes gzipped"
fi

echo "-- done"
[ -n "${TPT_WEB_REPO:-}" ] || echo "   (set TPT_WEB_REPO to copy the bundle into the hub web repo)"
