from __future__ import annotations

import hashlib
import http.client
import io
import logging
import random
import socket
import subprocess
import time
from typing import List, Tuple
import urllib
import urllib.error
from urllib.request import Request, urlopen


def run(args: List[str], show_output: bool = True) -> subprocess.CompletedProcess:
    """Run a command line.

    Parameters
    ----------
    args: List[str]
        A list of argument
    show_output: bool (default: True)
        Output command line

    Return
    ------
    subprocess.CompletedProcess
        Return result obtained with subprocess
    """
    ret = subprocess.run(args, capture_output=True, encoding="utf8")
    if show_output and ret.stdout is not None:
        logging.info(ret.stdout)
    if show_output and ret.stderr is not None:
        logging.warning(ret.stderr)
    return ret


def md5(path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def url_download(url: str, path: str) -> None:
    try:
        with urlopen(Request(url)) as fod:
            with open(path, "wb") as dst:
                while True:
                    chunk = fod.read(2**10)
                    if chunk:
                        dst.write(chunk)
                    else:
                        break
    except Exception as e:
        print(str(e))


def url_download_to_memory(url: str, *, retries: int = 5, timeout: float = 30.0) -> Tuple[io.BytesIO | None, int]:
    """
    Download URL into memory.
    Returns (BytesIO, 0) on success, else (None, error_code).

    error_code conventions:
      - HTTP status code (e.g., 404, 503) when available
      - -1 for generic network/URL issues
      - -2 for remote disconnected / connection dropped
      - -3 for timeout
    """
    error_code = 0

    headers = {
        # A more common UA than python-urllib; many servers block "botty" UAs.
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "close",
    }

    for attempt in range(retries):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout) as resp:
                data = resp.read()  # urllib auto-decompresses gzip if Accept-Encoding set
            buf = io.BytesIO(data)
            buf.seek(0)
            return buf, 0

        except urllib.error.HTTPError as e:
            # Retry only on typical transient codes
            if e.code in (408, 425, 429) or (500 <= e.code <= 599):
                error_code = e.code
            else:
                return None, e.code

        except (socket.timeout, TimeoutError):
            error_code = -3

        except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError):
            error_code = -2

        except urllib.error.URLError:
            # DNS, connection refused, etc.
            error_code = -1

        # Backoff before retry (except after last attempt)
        if attempt < retries - 1:
            backoff = min(8.0, 0.5 * (2 ** attempt))  # 0.5,1,2,4,8...
            time.sleep(backoff + random.uniform(0, 0.2))

    return None, error_code