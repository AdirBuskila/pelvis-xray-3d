"""Read single files out of a zip that lives on a web server, without downloading the whole zip.

How: a zip keeps its table of contents at the END of the file. Python's zipfile only needs
`seek` and `read`, so we give it a file-like object whose reads become HTTP Range requests
("send me bytes 1000-2000"). Opening the zip reads the table of contents; opening a member
reads only that member's bytes. The 3.2 GB small subset gives one subject in ~30-60 MB.

Same idea as data/scripts/remote_zip.py, packaged so notebooks can call it.
"""
from __future__ import annotations

import fnmatch
import io
import time
import urllib.request
import zipfile
from pathlib import Path


class HttpFile(io.RawIOBase):
    """A read-only, seekable file whose bytes come from HTTP Range requests."""

    def __init__(self, url: str):
        self.pos = 0
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
        self.size = int(head.headers["Content-Length"])
        self.url = head.geturl()  # follow redirects once, then reuse the final URL

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        last_error = None
        for attempt in range(8):  # servers drop connections now and then: retry with backoff
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
                data = urllib.request.urlopen(req, timeout=60).read()
                break
            except Exception as e:  # noqa: BLE001 (network errors come in many types)
                last_error = e
                time.sleep(2 * (attempt + 1))
        else:
            raise IOError(f"range request failed 8 times: {last_error}")
        buf[: len(data)] = data
        self.pos += len(data)
        return len(data)


def open_remote_zip(url: str) -> zipfile.ZipFile:
    # a 4 MB read buffer turns many tiny reads into a few big requests
    return zipfile.ZipFile(io.BufferedReader(HttpFile(url), buffer_size=4 * 2**20))


def list_members(url: str, pattern: str = "*") -> list[tuple[str, int]]:
    """(name, size in bytes) of every member matching a glob pattern."""
    z = open_remote_zip(url)
    return [(i.filename, i.file_size) for i in z.infolist() if fnmatch.fnmatch(i.filename, pattern)]


def extract(url: str, prefix: str, out_dir: str | Path, skip_existing: bool = True) -> list[Path]:
    """Extract every member whose name starts with `prefix` into out_dir (keeping its path)."""
    z = open_remote_zip(url)
    out_dir = Path(out_dir)
    written = []
    for info in z.infolist():
        if info.is_dir() or not info.filename.startswith(prefix):
            continue
        dst = out_dir / info.filename
        if skip_existing and dst.exists() and dst.stat().st_size == info.file_size:
            written.append(dst)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with z.open(info) as src, open(dst, "wb") as fo:
            fo.write(src.read())
        written.append(dst)
    return written
