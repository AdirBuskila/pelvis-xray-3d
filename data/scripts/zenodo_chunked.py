"""Robust Zenodo download: fixed-size Range chunks, each retried independently, then MD5-verified.
Usage: python zenodo_chunked.py <record_id> <filename> <out_dir>
Chunks already on disk with the right size are kept, so re-running continues where it stopped."""
import hashlib, json, os, sys, time, urllib.request, concurrent.futures as cf

rec, name, out = sys.argv[1], sys.argv[2], sys.argv[3]
meta = json.load(urllib.request.urlopen(f"https://zenodo.org/api/records/{rec}"))
f = next(x for x in meta["files"] if x["key"] == name)
size, md5 = f["size"], f["checksum"].split(":")[1]
url = f["links"]["self"]
CH = 64 * 2**20
parts_dir = os.path.join(out, name + ".parts"); os.makedirs(parts_dir, exist_ok=True)
ranges = [(i, s, min(s + CH, size) - 1) for i, s in enumerate(range(0, size, CH))]

def get(r):
    i, a, b = r
    p = os.path.join(parts_dir, f"{i:05d}")
    if os.path.exists(p) and os.path.getsize(p) == b - a + 1: return 0
    for attempt in range(20):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
            data = urllib.request.urlopen(req, timeout=60).read()
            if len(data) != b - a + 1: raise IOError(f"short chunk {len(data)}")
            open(p + ".tmp", "wb").write(data); os.replace(p + ".tmp", p); return 1
        except Exception as e:
            time.sleep(min(60, 3 * (attempt + 1)))
    raise RuntimeError(f"chunk {i} failed")

with cf.ThreadPoolExecutor(4) as ex:
    done = 0
    for n in ex.map(get, ranges):
        done += 1
        if done % 10 == 0: print(f"{done}/{len(ranges)} chunks", flush=True)

dst = os.path.join(out, name); h = hashlib.md5()
with open(dst, "wb") as fo:
    for i, _, _ in ranges:
        d = open(os.path.join(parts_dir, f"{i:05d}"), "rb").read(); h.update(d); fo.write(d)
assert os.path.getsize(dst) == size, "size mismatch"
assert h.hexdigest() == md5, f"md5 mismatch {h.hexdigest()} != {md5}"
for i, _, _ in ranges: os.remove(os.path.join(parts_dir, f"{i:05d}"))
os.rmdir(parts_dir)
print("OK", dst, size, "bytes, md5", md5)
