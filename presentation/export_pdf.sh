#!/usr/bin/env bash
# Exports the deck to PDF (posters instead of videos, all fragments shown).
# Run from anywhere while present.cmd (or `python -m http.server 8765`) is serving this folder.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
chrome="/c/Program Files/Google/Chrome/Application/chrome.exe"
timeout 180 "$chrome" --headless=new --disable-gpu --no-pdf-header-footer --virtual-time-budget=8000 \
  --print-to-pdf="$(cygpath -wa "$here")\deck.pdf" "http://127.0.0.1:8765/index.html?print-pdf"
echo "wrote $here/deck.pdf"
