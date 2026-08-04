"""geoskill_core.aoi — 地名/坐标 → AOI manifest

设计要点：
- 离线优先：HARDCODED_BBOXES（28+ 个中国 + 全球主要国家）
- 在线 fallback：Open-Meteo geocoding → Nominatim
- 统一返回 AOIManifest（geoskill_core.manifest）
- 缓存：内存 + 可选文件 cache
- 歧义处理：返回 candidates 列表，不静默取第一个

vendoring：本文件会 copy 到各 skill 内部。修改时请同步更新源仓库。
"""

from __future__ import annotations
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    import urllib.request
    import urllib.parse
    _HAS_REQUESTS = False

from .manifest import AOIManifest
from .errors import NoMatchError, NetworkError, UsageError


# ---- 离线硬编码 bbox（28+ 中国 + 全球）----

# WGS84 [west, south, east, north]
HARDCODED_BBOXES: Dict[str, Tuple[float, float, float, float]] = {
    # ---- China 国家 + 一级 ----
    "中国": (73.0, 18.0, 135.0, 54.0),
    "china": (73.0, 18.0, 135.0, 54.0),
    "cn": (73.0, 18.0, 135.0, 54.0),
    # 直辖市
    "北京": (115.7, 39.4, 116.7, 40.2),
    "北京市": (115.7, 39.4, 116.7, 40.2),
    "beijing": (115.7, 39.4, 116.7, 40.2),
    "上海": (120.8, 30.7, 122.2, 31.6),
    "上海市": (120.8, 30.7, 122.2, 31.6),
    "shanghai": (120.8, 30.7, 122.2, 31.6),
    "天津": (116.7, 38.5, 118.1, 40.2),
    "天津市": (116.7, 38.5, 118.1, 40.2),
    "重庆": (105.5, 28.5, 110.5, 32.0),
    "重庆市": (105.5, 28.5, 110.5, 32.0),
    "chongqing": (105.5, 28.5, 110.5, 32.0),
    # 省会（精选 15 个）
    "成都": (102.9, 30.1, 104.9, 31.5),
    "成都市": (102.9, 30.1, 104.9, 31.5),
    "chengdu": (102.9, 30.1, 104.9, 31.5),
    "广州": (112.9, 22.4, 114.0, 23.9),
    "广州市": (112.9, 22.4, 114.0, 23.9),
    "guangzhou": (112.9, 22.4, 114.0, 23.9),
    "深圳": (113.7, 22.4, 114.6, 22.9),
    "深圳市": (113.7, 22.4, 114.6, 22.9),
    "shenzhen": (113.7, 22.4, 114.6, 22.9),
    "杭州": (118.3, 29.8, 120.5, 30.6),
    "杭州市": (118.3, 29.8, 120.5, 30.6),
    "hangzhou": (118.3, 29.8, 120.5, 30.6),
    "南京": (118.3, 31.2, 119.2, 32.6),
    "南京市": (118.3, 31.2, 119.2, 32.6),
    "nanjing": (118.3, 31.2, 119.2, 32.6),
    "武汉": (113.7, 29.9, 115.1, 31.4),
    "武汉市": (113.7, 29.9, 115.1, 31.4),
    "wuhan": (113.7, 29.9, 115.1, 31.4),
    "西安": (107.9, 33.7, 109.8, 34.7),
    "西安市": (107.9, 33.7, 109.8, 34.7),
    "xian": (107.9, 33.7, 109.8, 34.7),
    "苏州": (119.9, 30.8, 121.1, 32.1),
    "苏州市": (119.9, 30.8, 121.1, 32.1),
    "郑州": (112.7, 34.2, 114.0, 35.0),
    "郑州市": (112.7, 34.2, 114.0, 35.0),
    "青岛": (119.7, 35.6, 121.0, 36.5),
    "青岛市": (119.7, 35.6, 121.0, 36.5),
    "济南": (116.1, 36.0, 117.5, 37.0),
    "济南市": (116.1, 36.0, 117.5, 37.0),
    # ---- 全球 ----
    "world": (-180.0, -90.0, 180.0, 90.0),
    "global": (-180.0, -90.0, 180.0, 90.0),
    "usa": (-125.0, 24.0, -66.0, 50.0),
    "united states": (-125.0, 24.0, -66.0, 50.0),
    "japan": (130.0, 30.0, 146.0, 46.0),
    "日本": (130.0, 30.0, 146.0, 46.0),
    "india": (68.0, 6.0, 98.0, 36.0),
    "印度": (68.0, 6.0, 98.0, 36.0),
    "europe": (-10.0, 35.0, 40.0, 70.0),
    "欧洲": (-10.0, 35.0, 40.0, 70.0),
}

# 行政区 marker
_CHINESE_ADMIN_MARKERS = ("市", "省", "自治区", "区", "县", "旗")

# 默认 User-Agent
DEFAULT_USER_AGENT = "geoskill-core/0.1.0 (+https://clawhub.ai)"

# 默认 timeout
DEFAULT_TIMEOUT = 15

# 默认 buffer
DEFAULT_BUFFER_DEG = 0.05  # ~5 km

# 缓存目录（None = 不缓存文件）
DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".geoskill_core_cache")


# ---- 内部工具 ----


def _chinese_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    chinese = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return chinese / max(1, len(text))


def _strip_chinese_place_hierarchy(place: str) -> List[str]:
    """北京市朝阳区 → ['朝阳区', '北京市朝阳区', '北京市', '北京']"""
    out: List[str] = []
    if not place:
        return out
    markers = _CHINESE_ADMIN_MARKERS
    for i, ch in enumerate(place):
        if ch in markers:
            sub_with = place[: i + 1]
            if sub_with not in out and sub_with != place:
                out.append(sub_with)
            tail = place[i + 1 :]
            if tail and tail not in out and tail != place:
                out.append(tail)
    return out


def _bbox_from_point(lat: float, lon: float, buffer_deg: float) -> List[float]:
    return [
        max(-180.0, lon - buffer_deg),
        max(-90.0, lat - buffer_deg),
        min(180.0, lon + buffer_deg),
        min(90.0, lat + buffer_deg),
    ]


def _normalize_query(place: str) -> str:
    return re.sub(r"\s+", "", place.strip())


# ---- 缓存 ----


_memory_cache: Dict[str, AOIManifest] = {}


def _cache_get(query: str) -> Optional[AOIManifest]:
    if query in _memory_cache:
        return _memory_cache[query]
    if DEFAULT_CACHE_DIR and os.path.isdir(DEFAULT_CACHE_DIR):
        path = os.path.join(DEFAULT_CACHE_DIR, _cache_key(query) + ".json")
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                m = AOIManifest.from_dict(data)
                _memory_cache[query] = m
                return m
            except Exception:
                return None
    return None


def _cache_put(query: str, manifest: AOIManifest) -> None:
    _memory_cache[query] = manifest
    if DEFAULT_CACHE_DIR:
        try:
            os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
            path = os.path.join(DEFAULT_CACHE_DIR, _cache_key(query) + ".json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(manifest.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass


def _cache_key(query: str) -> str:
    import hashlib
    return hashlib.sha1(query.encode("utf-8")).hexdigest()[:16]


# ---- 离线硬编码 lookup ----


def _hardcoded_lookup(normalised: str) -> Optional[AOIManifest]:
    """精确匹配 HARDCODED_BBOXES"""
    bbox = HARDCODED_BBOXES.get(normalised)
    if bbox is None:
        return None
    w, s, e, n = bbox
    return AOIManifest(
        query=normalised,
        bbox_wgs84=[w, s, e, n],
        centroid_wgs84=[(w + e) / 2.0, (s + n) / 2.0],
        resolver="hardcoded",
        confidence=0.98,
    )


# ---- HTTP helper（requests / urllib 兼容）----


def _http_get_json(url: str, params: Dict, headers: Dict, timeout: int) -> Optional[Dict]:
    if _HAS_REQUESTS:
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None
    else:
        try:
            q = urllib.parse.urlencode(params)
            req = urllib.request.Request(f"{url}?{q}", headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            return None


# ---- Open-Meteo ----


def _open_meteo_search(query: str, language: str, timeout: int, user_agent: str) -> List[Dict]:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    data = _http_get_json(
        url,
        {"name": query, "count": 10, "language": language, "format": "json"},
        {"User-Agent": user_agent},
        timeout,
    )
    if data is None:
        return []
    out: List[Dict] = []
    for r in data.get("results", []) or []:
        out.append({
            "name": r.get("name"),
            "lat": float(r.get("latitude", 0.0)),
            "lon": float(r.get("longitude", 0.0)),
            "display_name": ", ".join(
                p for p in (
                    r.get("name"),
                    r.get("admin1"),
                    r.get("admin2"),
                    r.get("admin3"),
                    r.get("country"),
                ) if p
            ),
            "country_code": (r.get("country_code") or "").lower(),
            "feature_code": r.get("feature_code"),
            "admin1": r.get("admin1"),
            "admin2": r.get("admin2"),
            "population": r.get("population"),
            "source": "open-meteo",
        })
    return out


# ---- Nominatim ----


def _nominatim_search(query: str, timeout: int, user_agent: str) -> List[Dict]:
    url = "https://nominatim.openstreetmap.org/search"
    data = _http_get_json(
        url,
        {
            "q": query,
            "format": "jsonv2",
            "limit": 5,
            "addressdetails": 1,
        },
        {"User-Agent": user_agent, "Accept-Language": "zh-CN,zh;q=0.9"},
        timeout,
    )
    if data is None:
        return []
    out: List[Dict] = []
    for r in data:
        bbox = r.get("boundingbox") or []
        if len(bbox) != 4:
            continue
        try:
            out.append({
                "name": r.get("name") or r.get("display_name", ""),
                "lat": float(r.get("lat", 0.0)),
                "lon": float(r.get("lon", 0.0)),
                "display_name": r.get("display_name", ""),
                "country_code": ((r.get("address") or {}).get("country_code") or "").lower(),
                "bbox": [
                    float(bbox[2]), float(bbox[0]),
                    float(bbox[3]), float(bbox[1]),
                ],
                "osm_id": r.get("osm_id"),
                "osm_type": r.get("osm_type"),
                "source": "nominatim",
            })
        except (TypeError, ValueError):
            continue
    return out


# ---- 评分 ----


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
    cc = candidate.get("country_code", "")
    if cc == "cn":
        score += 20
    return score


# ---- 公共 API ----


def resolve_place(
    place: str,
    timeout: int = DEFAULT_TIMEOUT,
    user_agent: str = DEFAULT_USER_AGENT,
    prefer_country: str = "cn",
    buffer_deg: float = DEFAULT_BUFFER_DEG,
    allow_nominatim: bool = True,
    use_cache: bool = True,
) -> AOIManifest:
    """把 place name 解析为 AOI manifest。

    解析顺序：
    1. 内存 / 文件缓存
    2. HARDCODED_BBOXES 离线匹配
    3. Open-Meteo geocoding（限流友好）
    4. Nominatim（默认 fallback，可关闭）

    Raises:
        UsageError: place 为空
        NoMatchError: 所有来源都查不到
        NetworkError: 网络问题（保留为可识别错误）
    """
    if not place or not place.strip():
        raise UsageError("place must not be empty", place=place)
    normalised = _normalize_query(place)
    if use_cache:
        cached = _cache_get(normalised)
        if cached is not None:
            return cached
    # 1. 离线
    hc = _hardcoded_lookup(normalised)
    if hc is not None:
        if use_cache:
            _cache_put(normalised, hc)
        return hc
    # 2. Open-Meteo
    is_chinese = _chinese_char_ratio(normalised) > 0.4
    language = "zh" if is_chinese else "en"
    queries_to_try: List[str] = [normalised]
    if is_chinese:
        if normalised and normalised[-1] in "市省区县旗":
            stripped = normalised[:-1]
        else:
            stripped = ""
        if stripped and stripped != normalised and stripped not in queries_to_try:
            queries_to_try.append(stripped)
        for c in _strip_chinese_place_hierarchy(normalised):
            if c and c not in queries_to_try:
                queries_to_try.append(c)
            if c and c[-1] in "市省区县旗":
                cs = c[:-1]
                if cs and cs != c and cs not in queries_to_try:
                    queries_to_try.append(cs)
    all_candidates: List[Dict] = []
    for q in queries_to_try:
        results = _open_meteo_search(q, language, timeout, user_agent)
        if prefer_country:
            results = [r for r in results if r.get("country_code") == prefer_country] or results
        for c in results[:5]:
            c = dict(c)
            c["source_query"] = q
            all_candidates.append(c)
        if all_candidates:
            break  # 找到就停
    if not all_candidates:
        raise NoMatchError(
            f"Could not resolve place {place!r} via Open-Meteo. "
            "Try a more general name or pass --bbox explicitly.",
            place=place,
        )
    # 评分 + 选最佳
    all_candidates.sort(key=lambda c: _score_open_meteo(c, normalised), reverse=True)
    chosen = all_candidates[0]
    bbox = _bbox_from_point(chosen["lat"], chosen["lon"], buffer_deg)
    # 3. Nominatim 富化（可选）
    if allow_nominatim:
        nom = _nominatim_search(normalised, timeout, user_agent)
        if prefer_country:
            nom = [c for c in nom if c.get("country_code") == prefer_country] or nom
        if nom:
            nom_chosen = nom[0]
            bbox = nom_chosen["bbox"]
            return AOIManifest(
                query=place,
                bbox_wgs84=bbox,
                centroid_wgs84=[nom_chosen["lon"], nom_chosen["lat"]],
                resolver="nominatim",
                confidence=0.85,
                ambiguity=[
                    {"name": c.get("name"), "country": c.get("country_code"), "lat": c.get("lat"), "lon": c.get("lon")}
                    for c in all_candidates[:5]
                ],
                notes=f"resolved via Nominatim; {len(nom)} candidates",
            )
    # 默认返回 Open-Meteo 选出的
    return AOIManifest(
        query=place,
        bbox_wgs84=bbox,
        centroid_wgs84=[chosen["lon"], chosen["lat"]],
        resolver="open-meteo",
        confidence=0.75,
        ambiguity=[
            {"name": c.get("name"), "country": c.get("country_code"), "lat": c.get("lat"), "lon": c.get("lon")}
            for c in all_candidates[:5]
        ],
        notes=f"resolved via Open-Meteo; {len(all_candidates)} candidates",
    )


def parse_bbox_arg(bbox_str: str) -> List[float]:
    """解析 --bbox "W,S,E,N" 字符串。

    Raises:
        UsageError: 格式错或值不在合法范围
    """
    if not bbox_str:
        raise UsageError("--bbox must not be empty", bbox=bbox_str)
    parts = re.split(r"[,;\s]+", bbox_str.strip())
    if len(parts) != 4:
        raise UsageError(
            f"--bbox must be 'W,S,E,N' (got {len(parts)} numbers)",
            bbox=bbox_str,
        )
    try:
        nums = [float(p) for p in parts]
    except ValueError as e:
        raise UsageError(f"--bbox contains non-numeric value: {e}", bbox=bbox_str)
    w, s, e, n = nums
    if not (-180.0 <= w < e <= 180.0):
        raise UsageError(f"--bbox west/east out of range or inverted: W={w} E={e}", bbox=nums)
    if not (-90.0 <= s < n <= 90.0):
        raise UsageError(f"--bbox south/north out of range or inverted: S={s} N={n}", bbox=nums)
    return [w, s, e, n]


def resolve_or_parse(place: Optional[str], bbox: Optional[str], **kwargs) -> AOIManifest:
    """根据参数选择 place 解析或 bbox 解析。两者都给时优先 place。

    Raises:
        UsageError: 两个都给或两个都不给
    """
    if place and bbox:
        raise UsageError("pass either --place or --bbox, not both", place=place, bbox=bbox)
    if not place and not bbox:
        raise UsageError("must pass --place or --bbox", place=place, bbox=bbox)
    if bbox:
        nums = parse_bbox_arg(bbox)
        w, s, e, n = nums
        return AOIManifest(
            query=f"bbox:{bbox}",
            bbox_wgs84=nums,
            centroid_wgs84=[(w + e) / 2.0, (s + n) / 2.0],
            resolver="bbox",
            confidence=1.0,
        )
    return resolve_place(place, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "HARDCODED_BBOXES",
    "DEFAULT_USER_AGENT",
    "DEFAULT_TIMEOUT",
    "DEFAULT_BUFFER_DEG",
    "DEFAULT_CACHE_DIR",
    "resolve_place",
    "parse_bbox_arg",
    "resolve_or_parse",
    "_chinese_char_ratio",
    "_strip_chinese_place_hierarchy",
    "_bbox_from_point",
    "_hardcoded_lookup",
]
