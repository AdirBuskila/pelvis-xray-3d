#!/usr/bin/env bash
# Render videos locally one by one, verify each, copy to Google Drive. Usage: scripts/local_render.sh V09 V10 ...
set -u
cd "$(dirname "$0")/.."
DRIVE="G:/My Drive/x-ray-videos"
for id in "$@"; do
  entry=$(node scripts/entry.mjs "$id")
  out="out/final_$id.mp4"
  echo "== $id: rendering $(date +%H:%M)"
  if ! npx remotion render "$entry" "$id" "$out" --codec=h264 --crf=16 --concurrency=4 --log=error; then
    echo "!! $id render FAILED"; continue
  fi
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$out")
  streams=$(ffprobe -v error -show_entries stream=codec_type -of csv=p=0 "$out" | tr '\n' ',')
  vol=$(ffmpeg -hide_banner -i "$out" -af volumedetect -vn -f null - 2>&1 | grep -o "mean_volume: [-0-9.]* dB")
  echo "== $id: done $(date +%H:%M) duration=${dur}s streams=${streams} ${vol}"
  title=$(node -e "console.log(require('./scripts/titles.json')['$id'])")
  cp "$out" "$DRIVE/$title.mp4" && echo "== $id: copied to Drive as '$title.mp4'"
done
