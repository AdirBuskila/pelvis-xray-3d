"""Regenerate only some narration segments (with retries), updating durations.json in place.
Usage: python -I v12_tts_one.py <script.json> <out_dir> <seg_id> [<seg_id> ...]   (same voice/rate as tts.py)"""
import asyncio, json, sys, subprocess, time
from pathlib import Path
import edge_tts

script, out, ids = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
cfg = json.load(open(script, encoding="utf-8"))
dur_path = out / "durations.json"
durations = json.load(open(dur_path)) if dur_path.exists() else {}
for seg in cfg["segments"]:
    if seg["id"] not in ids: continue
    mp3 = out / f"{seg['id']}.mp3"
    for attempt in range(8):
        try:
            asyncio.run(edge_tts.Communicate(seg["text"], cfg["voice"], rate=cfg.get("rate", "+0%")).save(str(mp3)))
            break
        except Exception as e:
            print("retry", seg["id"], attempt, type(e).__name__); time.sleep(5 + 5 * attempt)
    else:
        sys.exit(f"failed {seg['id']}")
    d = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(mp3)]).decode())
    durations[seg["id"]] = round(d, 3); print(seg["id"], d)
json.dump(durations, open(dur_path, "w"), indent=1)
