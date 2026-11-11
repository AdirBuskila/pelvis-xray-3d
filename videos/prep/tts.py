"""Generate narration segments with edge-tts (Microsoft neural voices). Usage: tts.py <script.json> <out_dir>"""
import asyncio, json, sys, subprocess
from pathlib import Path
import edge_tts

async def main(script, out):
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(script, encoding="utf-8"))
    durations = {}
    for seg in cfg["segments"]:
        mp3 = out / f"{seg['id']}.mp3"
        await edge_tts.Communicate(seg["text"], cfg["voice"], rate=cfg.get("rate", "+0%")).save(str(mp3))
        d = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                           "-of", "csv=p=0", str(mp3)]).decode())
        durations[seg["id"]] = round(d, 3)
        print(seg["id"], d)
    json.dump(durations, open(out / "durations.json", "w"), indent=1)

asyncio.run(main(Path(sys.argv[1]), Path(sys.argv[2])))
