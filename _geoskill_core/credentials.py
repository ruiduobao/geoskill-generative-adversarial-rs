"""geoskill_core.credentials — 统一凭证管理 (Phase 7, 2026-07-27).

设计原则：
- **不向 skill 源码/日志/文档硬编码密码**。所有默认凭证仅作为"环境变量
  未设置时的 fallback"，集中放在本模块内（geoskill-core 源仓库）。
- **可被环境变量覆盖**：`EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`
  / `EARTHDATA_TOKEN` / `FIRMS_MAP_KEY` / `OPENAI_API_KEY` /
  `CMA_API_KEY` / `EOG_USERNAME` / `EOG_PASSWORD` 任何一项显式设置
  都优先于默认值。
- **支持 .netrc**：若 ~/.netrc 中存在 `machine urs.earthdata.nasa.gov`
  行，优先取 .netrc 凭证。
- **支持用户级 secrets 文件** ``~/.geoskill/secrets.json``：Phase 7
  (2026-07-27) 新增。本文件在用户 home，**不** vendor 到任何 skill，
  **不** push 到 GitHub；用于把个人真实凭证（NASA Earthdata bearer
  token 等）放在 skill 之外。
- **不缓存密码**：每次调用读环境或 .netrc（避免长寿命进程泄露）。
- **统一接口**：`get_earthdata_creds()` / `get_earthdata_token()` /
  `get_firms_key()` / `get_cma_key()` / `get_openai_key()` /
  `get_eog_creds()` 六个 helper。
- **可在测试中被 monkeypatch**：通过 `set_default(name, value)` 在
  测试里临时改默认。

NASA Earthdata Login 是 EOSDIS 单一登录，覆盖以下数据源：
- LAADS DAAC (ladsweb.modaps.eosdis.nasa.gov) — MODIS L1 / 大气产品
- GES DISC (data.gesdisc.earthdata.nasa.gov) — GPM IMERG / MERRA-2
- ASF / NSIDC — Sentinel-1 / SMAP / 冰冻圈产品
- LP DAAC — MODIS Land / SRTM / ASTER GDEM
- CMR (cmr.earthdata.nasa.gov) — 元数据 + 下载 URL
- Worldview snapshot (worldview.earthdata.nasa.gov) — 影像快照
- AppEEARS (appeears.earthdata.nasa.gov) — 区域产品
- EARTHDATA 上的所有 DAAC
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Optional, Tuple

__all__ = [
    "get_earthdata_creds",
    "get_earthdata_token",
    "get_firms_key",
    "get_cma_key",
    "get_openai_key",
    "get_eog_creds",
    "set_default",
    "clear_default",
    "clear_all_defaults",
    "load_user_secrets",
    "describe_credentials",
    "USER_SECRETS_PATH",
]

# --- 默认值 (仅作为 fallback) -----------------------------------------
# 用户在 2026-07-26 指定的 NASA Earthdata 默认账号 (走 basic auth fallback).
# 这些值通过 helper 间接使用，**不应**出现在任何 skill 源码 / 日志 /
# 文档 / 报告里。skill 只能通过本模块拿凭证，禁止直接读这两个常量。
_DEFAULTS: dict[str, str] = {
    "EARTHDATA_USERNAME": "ruiduobao",
    "EARTHDATA_PASSWORD": "Ruiduobao123",
    "EARTHDATA_TOKEN": "",  # 用户级 secrets.json 提供（不走默认值以免推到 GitHub）
    "FIRMS_MAP_KEY": "",
    "CMA_API_KEY": "",
    "OPENAI_API_KEY": "",
    "EOG_USERNAME": "",
    "EOG_PASSWORD": "",
}

# .netrc 解析（仅在 UNIX-like / WSL 下 ~/.netrc 可用；Windows 下
# 通常用 %USERPROFILE%\_netrc，但 .netrc 本身仍是约定俗成的名称）。
_NETRC_HOSTS = {
    "urs.earthdata.nasa.gov": ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"),
    "firms.modaps.eosdis.nasa.gov": ("FIRMS_MAP_KEY",),
    "eogdata.mines.edu": ("EOG_USERNAME", "EOG_PASSWORD"),
}

# 用户级 secrets 文件位置（在用户 home，**不** vendor 到 skill 内部）。
# Phase 7 (2026-07-27): 包含 NASA Earthdata bearer token 等真实凭证。
USER_SECRETS_PATH = Path.home() / ".geoskill" / "secrets.json"

# 是否已加载过用户级 secrets（避免每次调用都重读）
_user_secrets_loaded = False


def set_default(name: str, value: str) -> None:
    """在运行时改默认值（仅用于测试 monkeypatch）。"""
    if name not in _DEFAULTS:
        raise KeyError(f"unknown credential name: {name}")
    _DEFAULTS[name] = value


def clear_default(name: str) -> None:
    """清掉默认值。env var 仍可能提供。"""
    if name not in _DEFAULTS:
        raise KeyError(f"unknown credential name: {name}")
    _DEFAULTS[name] = ""


def clear_all_defaults() -> None:
    for k in _DEFAULTS:
        _DEFAULTS[k] = ""


def load_user_secrets(path: Optional[Path] = None, *, force: bool = False) -> bool:
    """从 ``~/.geoskill/secrets.json`` 加载用户级凭证到 _DEFAULTS.

    Phase 7 (2026-07-27): 第一次调用自动加载（lazy）。之后每个 helper
    调用也会 lazy 加载，除非显式 ``force=True`` 强制重读。返回 True
    表示文件存在且读到了非空内容。

    文件格式（JSON）::

        {
          "EARTHDATA_USERNAME": "...",
          "EARTHDATA_PASSWORD": "...",
          "EARTHDATA_TOKEN": "...",
          ...
        }

    任何 ``_DEFAULTS`` 中已知的 key 都会被文件中的非空值覆盖。未知
    key 被忽略。下划线开头的 key（如 ``_profile``）永远被忽略。

    此函数是**幂等**的：多次调用只产生一次文件 IO。
    """
    global _user_secrets_loaded
    if _user_secrets_loaded and not force:
        return False
    _user_secrets_loaded = True
    target = path or USER_SECRETS_PATH
    if not target.is_file():
        return False
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    loaded_any = False
    for k, v in data.items():
        if not isinstance(k, str) or k.startswith("_"):
            continue
        if k in _DEFAULTS and isinstance(v, str) and v.strip():
            _DEFAULTS[k] = v.strip()
            loaded_any = True
    return loaded_any


def _ensure_user_secrets_loaded() -> None:
    """lazy 加载 — 每次 helper 调用都确保读到用户 secrets."""
    global _user_secrets_loaded
    if not _user_secrets_loaded:
        load_user_secrets()


def _read_netrc(host: str) -> Optional[Tuple[str, ...]]:
    """从 ~/.netrc 读指定 host 的凭证（无 token 格式）。"""
    for path in (Path.home() / ".netrc", Path.home() / "_netrc"):
        if not path.is_file():
            continue
        try:
            mode = path.stat().st_mode
            if mode & (stat.S_IRWXG | stat.S_IRWXO):
                pass
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        blocks = text.split("machine ")
        for block in blocks:
            if not block:
                continue
            head = block.splitlines()
            if not head:
                continue
            first = head[0].strip()
            if first != host:
                continue
            creds = {}
            for line in head[1:]:
                line = line.strip()
                if line.startswith("login "):
                    creds["login"] = line[6:].strip()
                elif line.startswith("password "):
                    creds["password"] = line[9:].strip()
                elif line.startswith("account "):
                    creds["account"] = line[8:].strip()
            if "login" in creds and "password" in creds:
                return (creds["login"], creds["password"])
            if "login" in creds:
                return (creds["login"],)
    return None


def _resolve(name: str) -> str:
    """env > 用户 secrets > .netrc > 默认. 空字符串视为未设."""
    env_val = os.environ.get(name, "").strip()
    if env_val:
        return env_val
    _ensure_user_secrets_loaded()
    default = _DEFAULTS.get(name, "")
    return default if default else ""


def _resolve_with_netrc(env_name: str, netrc_host: str, field_index: int) -> str:
    """env > 用户 secrets > .netrc > 默认."""
    env_val = os.environ.get(env_name, "").strip()
    if env_val:
        return env_val
    _ensure_user_secrets_loaded()
    netrc = _read_netrc(netrc_host)
    if netrc and len(netrc) > field_index and netrc[field_index]:
        return netrc[field_index]
    default = _DEFAULTS.get(env_name, "")
    return default if default else ""


def get_earthdata_creds() -> Tuple[str, str]:
    """返回 (username, password) — 任一为空则返回 ("", "").

    解析顺序：
    1. env: EARTHDATA_USERNAME / EARTHDATA_PASSWORD
    2. ~/.geoskill/secrets.json
    3. .netrc: machine urs.earthdata.nasa.gov
    4. _DEFAULTS 兜底
    """
    u = _resolve_with_netrc("EARTHDATA_USERNAME", "urs.earthdata.nasa.gov", 0)
    p = _resolve_with_netrc("EARTHDATA_PASSWORD", "urs.earthdata.nasa.gov", 1)
    if not p:
        netrc = _read_netrc("urs.earthdata.nasa.gov")
        if netrc and len(netrc) > 1 and netrc[1]:
            p = netrc[1]
    return (u, p)


def get_earthdata_token() -> str:
    """返回 bearer token — 用于 CMR / LAADS / GES DISC 等需要 token 的端点.

    解析顺序：
    1. env: EARTHDATA_TOKEN
    2. ~/.geoskill/secrets.json
    3. .netrc: machine urs.earthdata.nasa.gov account <TOKEN>
    4. _DEFAULTS（通常为空）
    """
    return _resolve("EARTHDATA_TOKEN")


def get_firms_key() -> str:
    return _resolve("FIRMS_MAP_KEY")


def get_cma_key() -> str:
    return _resolve("CMA_API_KEY")


def get_openai_key() -> str:
    return _resolve("OPENAI_API_KEY")


def get_eog_creds() -> Tuple[str, str]:
    u = _resolve_with_netrc("EOG_USERNAME", "eogdata.mines.edu", 0)
    p = _resolve_with_netrc("EOG_PASSWORD", "eogdata.mines.edu", 1)
    return (u, p)


def describe_credentials() -> dict:
    """诊断用 — 返回每个凭证的来源（不暴露明文密码）.

    source 字段:
    - ``"env"``         — 来自环境变量
    - ``"user_secrets"`` — 来自 ~/.geoskill/secrets.json
    - ``"netrc"``        — 来自 ~/.netrc
    - ``"default"``      — 来自 _DEFAULTS（geoskill-core 硬编码兜底）
    - ``"none"``         — 全部未配
    """
    def _info(env_name: str, *, netrc_host: str = None) -> dict:
        env_val = os.environ.get(env_name, "").strip()
        if env_val:
            return {"available": True, "source": "env"}
        # 用户 secrets
        if USER_SECRETS_PATH.is_file():
            try:
                data = json.loads(USER_SECRETS_PATH.read_text(encoding="utf-8"))
                if isinstance(data, dict) and not env_name.startswith("_"):
                    v = data.get(env_name, "")
                    if isinstance(v, str) and v.strip():
                        return {"available": True, "source": "user_secrets"}
            except (OSError, json.JSONDecodeError):
                pass
        if netrc_host:
            netrc = _read_netrc(netrc_host)
            if netrc and any(netrc):
                return {"available": True, "source": "netrc"}
        _ensure_user_secrets_loaded()
        default = _DEFAULTS.get(env_name, "")
        if default:
            return {"available": True, "source": "default"}
        return {"available": False, "source": "none"}

    return {
        "EARTHDATA_USERNAME": {**_info("EARTHDATA_USERNAME", netrc_host="urs.earthdata.nasa.gov"),
                              "hint": "Register at https://urs.earthdata.nasa.gov/users/new/"},
        "EARTHDATA_PASSWORD": {**_info("EARTHDATA_PASSWORD", netrc_host="urs.earthdata.nasa.gov"),
                              "hint": "Same as Earthdata Login password"},
        "EARTHDATA_TOKEN": {**_info("EARTHDATA_TOKEN"),
                            "hint": "Earthdata Login profile -> Generate Token (https://urs.earthdata.nasa.gov/profile)"},
        "FIRMS_MAP_KEY": {**_info("FIRMS_MAP_KEY"),
                          "hint": "Request at https://firms.modaps.eosdis.nasa.gov/api/map_key/"},
        "CMA_API_KEY": {**_info("CMA_API_KEY"),
                        "hint": "Register at http://data.cma.cn/ (free, Chinese phone)"},
        "OPENAI_API_KEY": {**_info("OPENAI_API_KEY"),
                           "hint": "Any OpenAI-compatible endpoint"},
        "EOG_USERNAME": {**_info("EOG_USERNAME", netrc_host="eogdata.mines.edu"),
                         "hint": "Free account at https://eogdata.mines.edu/register/ (since 2025)"},
        "EOG_PASSWORD": {**_info("EOG_PASSWORD", netrc_host="eogdata.mines.edu"),
                         "hint": "Same as EOG account password"},
    }
