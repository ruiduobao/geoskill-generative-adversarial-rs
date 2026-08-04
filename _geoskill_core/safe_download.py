"""geoskill_core.safe_download — 安全下载器（.part / 原子替换 / 重试 / resume / 校验）

参考实现：从 `landsat-download/landsat-download.py` 抽取的 `download_asset`
精简为可复用的独立函数。

设计要点：
- 先写 `*.part`，验证后 `os.replace()` 原子替换
- 网络错误自动重试（指数退避）
- HTTP 416/206 支持 Range resume
- 默认不覆盖（--overwrite 才会）
- 可选 SHA256 校验
- 上报进度（默认 silent，可选 callback）
- 上限超时 + 大小限制（防 OOM / 防巨文件）
- 退出码契约：4=网络, 6=校验失败

vendoring：本文件会 copy 到各 skill 内部。修改时请同步更新源仓库。
"""

from __future__ import annotations
import hashlib
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    import urllib.request
    import urllib.error
    _HAS_REQUESTS = False

from .errors import NetworkError, ValidationError


DEFAULT_USER_AGENT = "geoskill-core/0.1.0 (+https://clawhub.ai)"

DEFAULT_CHUNK_SIZE = 64 * 1024  # 64 KB
DEFAULT_TIMEOUT = 600
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 2.0  # 秒，指数退避基数

ProgressCallback = Callable[[int, int, float], None]  # (downloaded, total, speed_bps)


def _http_get(url: str, headers: Dict, timeout: int, range_from: Optional[int] = None):
    """跨 requests/urllib 的 HTTP GET，返回 (stream_iter, total_size, error)。

    - requests 模式：返回 r.iter_content() + Content-Length
    - urllib 模式：返回 chunk reader
    """
    if _HAS_REQUESTS:
        h = dict(headers)
        if range_from is not None:
            h["Range"] = f"bytes={range_from}-"
        try:
            r = requests.get(url, stream=True, timeout=timeout, headers=h)
        except Exception as e:
            return None, None, e
        try:
            r.raise_for_status()
        except Exception as e:
            return None, None, e
        total = int(r.headers.get("Content-Length", 0)) or None
        if range_from is not None:
            # Range request returns 206
            total_full = None
        return r.iter_content(chunk_size=DEFAULT_CHUNK_SIZE), total, None
    # urllib fallback
    try:
        req = urllib.request.Request(url, headers=headers)
        if range_from is not None:
            req.add_header("Range", f"bytes={range_from}-")
        r = urllib.request.urlopen(req, timeout=timeout)
        total = r.headers.get("Content-Length")
        total = int(total) if total else None
        return _UrllibChunkReader(r), total, None
    except urllib.error.HTTPError as e:
        return None, None, e
    except Exception as e:
        return None, None, e


class _UrllibChunkReader:
    def __init__(self, r):
        self.r = r
    def __iter__(self):
        return self
    def __next__(self):
        chunk = self.r.read(DEFAULT_CHUNK_SIZE)
        if not chunk:
            raise StopIteration
        return chunk


def _file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_download(
    url: str,
    dest_path: str,
    *,
    expected_sha256: Optional[str] = None,
    overwrite: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    user_agent: str = DEFAULT_USER_AGENT,
    progress_cb: Optional[ProgressCallback] = None,
    max_size_bytes: Optional[int] = None,
    resume: bool = True,
) -> Dict:
    """安全下载一个 URL 到本地文件。

    Args:
        url: 下载 URL
        dest_path: 目标文件路径
        expected_sha256: 期望的 SHA256（hex），None 跳过校验
        overwrite: 允许覆盖（默认 False，已存在则跳过）
        timeout: 单次请求超时（秒）
        max_retries: 最大重试次数（网络错误）
        user_agent: HTTP User-Agent
        progress_cb: 进度回调 (downloaded, total, speed_bps)
        max_size_bytes: 最大允许字节（None=无限），超过则失败
        resume: 允许 Range resume（.part 文件存在时）

    Returns:
        dict: {ok, path, sha256, size, bytes_downloaded, retries, message}

    Raises:
        NetworkError: 网络错误（已用尽重试）
        ValidationError: 校验失败 / 大小超限
    """
    dest_path = os.path.abspath(dest_path)
    tmp_path = dest_path + ".part"

    # 1. 已存在检查
    if os.path.exists(dest_path) and not os.path.exists(tmp_path):
        if not overwrite:
            sz = os.path.getsize(dest_path)
            sha = _file_sha256(dest_path)
            return {
                "ok": True, "path": dest_path, "sha256": sha, "size": sz,
                "bytes_downloaded": 0, "retries": 0,
                "message": "already exists, skipped (pass overwrite=True to redownload)",
            }
    # 2. 准备 .part（可能从已有部分 resume）
    resume_from = 0
    if os.path.exists(tmp_path) and resume:
        resume_from = os.path.getsize(tmp_path)

    headers = {"User-Agent": user_agent}
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            stream, total, err = _http_get(
                url, headers, timeout,
                range_from=resume_from if resume_from > 0 else None,
            )
            if err is not None:
                raise err
            # 写文件
            downloaded = resume_from
            t0 = time.time()
            mode = "ab" if resume_from > 0 else "wb"
            with open(tmp_path, mode) as f:
                for chunk in stream:
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if max_size_bytes is not None and downloaded > max_size_bytes:
                        raise ValidationError(
                            f"Download exceeded max_size_bytes={max_size_bytes}",
                            url=url, size=downloaded,
                        )
                    if progress_cb is not None:
                        elapsed = max(0.001, time.time() - t0)
                        speed = (downloaded - resume_from) / elapsed
                        progress_cb(downloaded, total, speed)
            # 3. 校验 SHA256（如果指定）
            if expected_sha256:
                actual = _file_sha256(tmp_path)
                if actual.lower() != expected_sha256.lower():
                    os.remove(tmp_path)
                    raise ValidationError(
                        f"SHA256 mismatch: expected={expected_sha256[:12]}..., got={actual[:12]}...",
                        url=url, expected=expected_sha256, actual=actual,
                    )
            # 4. 原子替换
            os.replace(tmp_path, dest_path)
            sz = os.path.getsize(dest_path)
            sha = _file_sha256(dest_path) if expected_sha256 else _file_sha256(dest_path)
            return {
                "ok": True, "path": dest_path, "sha256": sha, "size": sz,
                "bytes_downloaded": downloaded - resume_from,
                "retries": attempt,
                "message": "ok",
            }
        except Exception as e:
            last_err = e
            # 不可恢复错误：直接抛
            if isinstance(e, ValidationError):
                raise
            # 网络错误：清理 .part 重试
            if os.path.exists(tmp_path) and not resume:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            elif os.path.exists(tmp_path) and resume and attempt == max_retries:
                # 最后一次重试也失败：清理
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if attempt < max_retries:
                wait = DEFAULT_RETRY_BACKOFF ** attempt
                time.sleep(wait)
                continue
    # 重试耗尽
    raise NetworkError(
        f"Download failed after {max_retries+1} attempts: {last_err}",
        url=url, attempts=max_retries + 1, last_error=str(last_err)[:200],
    )


def safe_download_many(
    items: List[Dict],
    output_dir: str,
    **kwargs,
) -> List[Dict]:
    """批量下载。每个 item 是 {url, filename, expected_sha256?, ...}。

    Args:
        items: 列表，每项 {url, filename, expected_sha256?}
        output_dir: 目标目录
        **kwargs: 透传给 safe_download

    Returns:
        每项的 safe_download 结果（按输入顺序）
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for it in items:
        url = it["url"]
        filename = it.get("filename") or os.path.basename(url.split("?")[0]) or "file.bin"
        dest = os.path.join(output_dir, filename)
        expected = it.get("expected_sha256")
        try:
            r = safe_download(url, dest, expected_sha256=expected, **kwargs)
        except Exception as e:
            r = {
                "ok": False, "path": dest, "message": str(e)[:200],
                "code": getattr(e, "code", 4),
            }
        results.append(r)
    return results


__all__ = [
    "DEFAULT_USER_AGENT", "DEFAULT_CHUNK_SIZE", "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_RETRIES", "DEFAULT_RETRY_BACKOFF",
    "safe_download", "safe_download_many",
    "_file_sha256",
]
