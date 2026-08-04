"""Adapter shim — 让老 _place.py 调用统一委托给 _geoskill_core.aoi

Phase 1 迁移：保持 _place.py 公开 API 不变（返回 dict 格式），
但内部委托给 geoskill_core.aoi.resolve_place() → AOIManifest → dict adapter。

这样 12 份 _place.py 还在，但实现收口到 geoskill_core。

如果 _geoskill_core 不可用，回退到老实现（_legacy_* 函数保留为兜底）。
"""
from __future__ import annotations
import os
import sys
import warnings
from typing import Dict, List, Optional

# 优先用 _geoskill_core（统一实现）
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _geoskill_core import aoi as _aoi
    _USE_CORE = True
except Exception as _e:  # noqa: BLE001
    _USE_CORE = False
    warnings.warn(f"_geoskill_core.aoi not available, falling back to legacy: {_e}")


def _adapt_manifest_to_dict(m) -> Dict:
    """把 AOIManifest 转为老 dict 格式（向后兼容 place_info["lon"] 等访问）"""
    if m is None:
        return {}
    centroid = m.centroid_wgs84 or [None, None]
    return {
        "query": m.query,
        "normalised_query": m.query,
        "display_name": (m.notes or m.query or ""),
        "name": m.query,
        "lat": centroid[1] if len(centroid) > 1 else None,
        "lon": centroid[0] if len(centroid) > 0 else None,
        "bbox": m.bbox_wgs84,
        "osm_id": None,
        "osm_type": None,
        "country_code": "",
        "source": m.resolver,
        "candidates": m.ambiguity or [],
        "confidence": m.confidence,
        "resolver": m.resolver,
    }


def resolve_place(
    place: str,
    timeout: int = 15,
    user_agent: str = "geoskill-core/0.1.0 (+https://clawhub.ai)",
    prefer_country: str = "cn",
    buffer_deg: float = 0.05,
    allow_nominatim: bool = True,
    use_cache: bool = True,
) -> Dict:
    """Resolve a place name to a bbox (WGS84) plus display metadata.

    委托给 _geoskill_core.aoi.resolve_place()。返回 dict 与老 _place.py 完全一致。
    """
    if _USE_CORE:
        try:
            m = _aoi.resolve_place(
                place,
                timeout=timeout,
                user_agent=user_agent,
                prefer_country=prefer_country,
                buffer_deg=buffer_deg,
                allow_nominatim=allow_nominatim,
                use_cache=use_cache,
            )
            d = _adapt_manifest_to_dict(m)
            # buffer_deg_used（老接口）注入
            d["buffer_deg_used"] = buffer_deg
            return d
        except Exception as _e:
            # 网络/解析失败：尝试老实现兜底
            warnings.warn(f"_geoskill_core.aoi failed: {_e}, falling back to legacy")
    # ---- 兜底：老 _legacy 实现（从原 _place.py 复制保留）----
    return _legacy_resolve_place(
        place, timeout=timeout, user_agent=user_agent,
        prefer_country=prefer_country, buffer_deg=buffer_deg,
        allow_nominatim=allow_nominatim,
    )


# ---- 兜底实现（从原 _place.py 抽取，保留网络解析能力）----
import json
import re
import time

import requests

OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
DEFAULT_USER_AGENT = "codex-skill/1.0 (+https://clawhub.ai)"
DEFAULT_TIMEOUT = 15
DEFAULT_BUFFER_DEG = 0.05
_CHINESE_ADMIN_MARKERS = ("市", "省", "自治区", "区", "县", "旗")


def _chinese_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    chinese = sum(1 for ch in text if "一" <= ch <= "鿿")
    return chinese / max(1, len(text))


def _strip_chinese_place_hierarchy(place: str) -> List[str]:
    out: List[str] = []
    if not place:
        return out
    for i, ch in enumerate(place):
        if ch in _CHINESE_ADMIN_MARKERS:
            sub = place[: i + 1]
            if sub not in out and sub != place:
                out.append(sub)
            tail = place[i + 1 :]
            if tail and tail not in out and tail != place:
                out.append(tail)
    return out


def _open_meteo_search(query: str, language: str, timeout: int, user_agent: str):
    try:
        r = requests.get(OPEN_METEO_GEOCODE,
                          params={"name": query, "count": 10, "language": language, "format": "json"},
                          headers={"User-Agent": user_agent}, timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        return [{"_error": f"open-meteo: {e}"}]
    return [{
        "name": x.get("name"),
        "lat": float(x.get("latitude", 0.0)),
        "lon": float(x.get("longitude", 0.0)),
        "display_name": ", ".join(p for p in (x.get("name"), x.get("admin1"), x.get("admin2"), x.get("admin3"), x.get("country")) if p),
        "country_code": (x.get("country_code") or "").lower(),
        "population": x.get("population"),
        "source": "open-meteo",
    } for x in (r.json() or {}).get("results", []) or []]


def _nominatim_search(query: str, timeout: int, user_agent: str):
    try:
        r = requests.get(NOMINATIM_URL,
                          params={"q": query, "format": "jsonv2", "limit": 5, "addressdetails": 1},
                          headers={"User-Agent": user_agent, "Accept-Language": "zh-CN,zh;q=0.9"},
                          timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        return [{"_error": f"nominatim: {e}"}]
    out = []
    for x in r.json() or []:
        bbox = x.get("boundingbox") or []
        if len(bbox) != 4:
            continue
        try:
            out.append({
                "name": x.get("name") or x.get("display_name", ""),
                "lat": float(x.get("lat", 0.0)),
                "lon": float(x.get("lon", 0.0)),
                "display_name": x.get("display_name", ""),
                "country_code": ((x.get("address") or {}).get("country_code") or "").lower(),
                "bbox": [float(bbox[2]), float(bbox[0]), float(bbox[3]), float(bbox[1])],
                "osm_id": x.get("osm_id"),
                "osm_type": x.get("osm_type"),
                "source": "nominatim",
            })
        except (TypeError, ValueError):
            continue
    return out


def _bbox_from_point(lat: float, lon: float, buffer_deg: float) -> List[float]:
    return [max(-180.0, lon - buffer_deg), max(-90.0, lat - buffer_deg),
            min(180.0, lon + buffer_deg), min(90.0, lat + buffer_deg)]


def _score_open_meteo(candidate: Dict, query: str) -> int:
    name = (candidate.get("name") or "").replace(" ", "")
    q = (query or "").replace(" ", "")
    score = 0
    if name == q:
        score += 100
    if q in name or name in q:
        score += 30
    pop = candidate.get("population") or 0
    if pop > 1_000_000:
        score += 10
    elif pop > 100_000:
        score += 5
    if candidate.get("country_code", "") == "cn":
        score += 20
    return score


def _legacy_resolve_place(place: str, timeout: int, user_agent: str,
                          prefer_country: str, buffer_deg: float,
                          allow_nominatim: bool) -> Dict:
    if not place or not place.strip():
        raise ValueError("--place must not be empty")
    normalised = re.sub(r"\s+", "", place.strip())
    is_chinese = _chinese_char_ratio(normalised) > 0.4
    language = "zh" if is_chinese else "en"
    queries = [normalised]
    if is_chinese:
        if normalised and normalised[-1] in "市省区县旗":
            stripped = normalised[:-1]
            if stripped and stripped not in queries:
                queries.append(stripped)
        for c in _strip_chinese_place_hierarchy(normalised):
            if c and c not in queries:
                queries.append(c)
    all_candidates: List[Dict] = []
    for q in queries:
        primary = [c for c in _open_meteo_search(q, language, timeout, user_agent) if "_error" not in c]
        if prefer_country:
            primary = [c for c in primary if c.get("country_code") == prefer_country] or primary
        for c in primary[:5]:
            c = dict(c); c["source_query"] = q
            all_candidates.append(c)
        if all_candidates:
            break
    if not all_candidates:
        raise ValueError(f"Could not resolve place {place!r}. Try a more general name.")
    all_candidates.sort(key=lambda c: _score_open_meteo(c, normalised), reverse=True)
    chosen = all_candidates[0]
    bbox = _bbox_from_point(chosen["lat"], chosen["lon"], buffer_deg)
    if allow_nominatim:
        nom = [c for c in _nominatim_search(normalised, timeout, user_agent) if "_error" not in c]
        if prefer_country:
            nom = [c for c in nom if c.get("country_code") == prefer_country] or nom
        if nom:
            nom_chosen = nom[0]
            return {
                "query": place, "normalised_query": normalised,
                "display_name": nom_chosen.get("display_name", ""),
                "name": nom_chosen.get("name", place),
                "lat": nom_chosen["lat"], "lon": nom_chosen["lon"],
                "bbox": nom_chosen["bbox"],
                "osm_id": nom_chosen.get("osm_id"), "osm_type": nom_chosen.get("osm_type"),
                "country_code": nom_chosen.get("country_code", ""),
                "source": "nominatim", "candidates": all_candidates[:5],
                "buffer_deg_used": buffer_deg,
            }
    return {
        "query": place, "normalised_query": normalised,
        "display_name": chosen.get("display_name", ""),
        "name": chosen.get("name", place),
        "lat": chosen["lat"], "lon": chosen["lon"],
        "bbox": bbox, "osm_id": None, "osm_type": None,
        "country_code": chosen.get("country_code", ""),
        "source": "open-meteo", "candidates": all_candidates[:5],
        "buffer_deg_used": buffer_deg,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python _place.py <place name> [--no-nominatim]", file=sys.stderr)
        sys.exit(2)
    use_nominatim = "--no-nominatim" not in sys.argv
    place = next(a for a in sys.argv[1:] if not a.startswith("--"))
    res = resolve_place(place, allow_nominatim=use_nominatim)
    print(json.dumps(res, ensure_ascii=False, indent=2))