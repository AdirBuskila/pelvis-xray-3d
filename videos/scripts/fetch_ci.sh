#!/usr/bin/env bash
# Wait for a CI render run, then assemble its videos LOCALLY from the chunk artifacts:
# join the video chunks (lossless concat), render the audio track locally (fast), mux, verify, copy to Drive.
# Works whether or not the run's own assemble job succeeded. Usage: scripts/fetch_ci.sh <run_id>
set -u
cd "$(dirname "$0")/.."
run=$1; DRIVE="G:/My Drive/x-ray-videos"; dir="out/ci_$run"
while [ "$(gh run view "$run" --json status --jq .status)" != "completed" ]; do sleep 120; done
bad=$(gh run view "$run" --json jobs --jq '[.jobs[] | select(.name|startswith("render")) | select(.conclusion!="success")] | length')
if [ "$bad" != "0" ]; then echo "!! run $run: $bad render chunks did not succeed"; exit 1; fi
rm -rf "$dir"; mkdir -p "$dir"
gh run download "$run" -p "chunk_*" -D "$dir/chunks" || { echo "!! chunk download failed"; exit 1; }
ids=$(find "$dir/chunks" -name "*.mp4" -printf "%f\n" | sed 's/_[0-9]*\.mp4$//' | sort -u)
for id in $ids; do
  find "$dir/chunks" -name "${id}_*.mp4" | sort -t_ -k2 | while read -r f; do echo "file '$(cygpath -w "$f" 2>/dev/null || echo "$f")'"; done > "$dir/list_$id.txt"
  n=$(wc -l < "$dir/list_$id.txt")
  ffmpeg -v error -y -f concat -safe 0 -i "$dir/list_$id.txt" -c copy "$dir/video_$id.mp4" || { echo "!! $id concat failed"; continue; }
  entry=$(node scripts/entry.mjs "$id")
  npx remotion render "$entry" "$id" "$dir/audio_$id.aac" --codec=aac --log=error || { echo "!! $id audio failed"; continue; }
  ffmpeg -v error -y -i "$dir/video_$id.mp4" -i "$dir/audio_$id.aac" -c:v copy -c:a copy -shortest -movflags +faststart "$dir/$id.mp4"
  f="$dir/$id.mp4"
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")
  streams=$(ffprobe -v error -show_entries stream=codec_type -of csv=p=0 "$f" | tr '\n' ',')
  vol=$(ffmpeg -hide_banner -i "$f" -af volumedetect -vn -f null - 2>&1 | grep -o "mean_volume: [-0-9.]* dB")
  title=$(node -e "console.log(require('./scripts/titles.json')['$id'])")
  echo "== $id chunks=$n duration=${dur}s streams=${streams} ${vol}"
  cp "$f" "$DRIVE/$title.mp4" && echo "== $id copied to Drive as '$title.mp4'"
done
