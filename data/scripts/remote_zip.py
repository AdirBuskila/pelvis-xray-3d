"""Read members of a remote zip via HTTP Range requests, without downloading the whole archive.
Usage: python remote_zip.py <url> list [pattern]
       python remote_zip.py <url> get <member_prefix> <out_dir>"""
import io, os, sys, time, urllib.request, zipfile, fnmatch

class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        r = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        self.size = int(r.headers["Content-Length"]); self.url = r.geturl()
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos
    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0: return 0
        for attempt in range(10):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos+n-1}"})
                data = urllib.request.urlopen(req, timeout=60).read(); break
            except Exception:
                time.sleep(3 * (attempt + 1))
        b[:len(data)] = data; self.pos += len(data); return len(data)

url, cmd = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(io.BufferedReader(HttpFile(url), buffer_size=4 * 2**20))
if cmd == "list":
    pat = sys.argv[3] if len(sys.argv) > 3 else "*"
    for i in z.infolist():
        if fnmatch.fnmatch(i.filename, pat): print(i.filename, i.file_size)
else:
    prefix, out = sys.argv[3], sys.argv[4]
    for i in z.infolist():
        if i.filename.startswith(prefix) and not i.is_dir():
            dst = os.path.join(out, i.filename); os.makedirs(os.path.dirname(dst), exist_ok=True)
            with z.open(i) as src, open(dst, "wb") as fo: fo.write(src.read())
            print("got", i.filename, i.file_size)
