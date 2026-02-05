import hashlib
import io
import logging
import subprocess
from typing import List, Tuple
import urllib
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


def url_download_to_memory(url: str) -> Tuple[io.BytesIO | None, int]:
    retries = 3
    error_code = 0
    for _ in range(retries):
        try:
            # User-Agent improves compatibility with some servers
            req = Request(url, headers={"User-Agent": "python-urllib/3"})
            with urlopen(req) as resp:
                memory_buffer = io.BytesIO(resp.read())
            memory_buffer.seek(0)
            return memory_buffer, 0
        except urllib.error.HTTPError as e:
            # Retry only on 5xx; otherwise return immediately
            if 500 <= e.code <= 599:
                error_code = e.code
                continue
            return None, e.code
        except urllib.error.URLError:
            # Network/DNS issues—treat like retryable
            error_code = -1
            continue
    return None, error_code