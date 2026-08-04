"""geoskill_core.manifest — 四个 manifest 数据结构 + JSON Schema

四个契约（规划文档 §2.2）：
- AOI manifest     : query / bbox_wgs84 / geometry_wgs84 / centroid / resolver / confidence / ambiguity
- Dataset manifest : provider / collection / item_id / asset / time / unit / scale / license / url
- Output manifest  : 所有产物 / CRS / bbox / resolution / nodata / statistics / 软件版本
- Error manifest   : 错误类型 / 退出码 / 消息 / 详情（stderr 输出，--json-errors 时为 JSON）

约束：
- 字段名 snake_case
- bbox_wgs84 顺序固定 [west, south, east, north]
- 所有时间用 ISO8601 UTC
- CRS 用 EPSG 整数
- 数值字段单位必须写在字段名或 metadata
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Union
import json
import datetime as _dt


# ---- AOI manifest ----


@dataclass
class AOIManifest:
    """Area of Interest 解析结果。

    resolver: 解析来源（china-admin-divisions / nominatim / open-meteo / fixed 等）
    confidence: 0.0-1.0 置信度（行政匹配=0.98+，地理编码=0.5-0.9）
    ambiguity: 候选列表（多个同名地点时）
    """
    query: str
    bbox_wgs84: Optional[List[float]] = None  # [W, S, E, N]
    geometry_wgs84: Optional[Dict[str, Any]] = None  # GeoJSON geometry 或 None
    centroid_wgs84: Optional[List[float]] = None  # [lon, lat]
    resolver: str = "unknown"
    confidence: float = 0.0
    ambiguity: List[Dict[str, Any]] = field(default_factory=list)
    notes: Optional[str] = None

    def is_valid(self) -> bool:
        return (
            self.bbox_wgs84 is not None
            and len(self.bbox_wgs84) == 4
            and self.bbox_wgs84[0] < self.bbox_wgs84[2]
            and self.bbox_wgs84[1] < self.bbox_wgs84[3]
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AOIManifest":
        return cls(**d)


# ---- Dataset manifest ----


@dataclass
class AssetEntry:
    """单个 asset（文件 / 流 / band）"""
    key: str
    href: str
    media_type: str = ""  # e.g. image/tiff; application/json
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None  # sha256:...
    role: Optional[str] = None  # data / metadata / thumbnail / index


@dataclass
class DatasetManifest:
    """某个具体数据条目（如一个 Landsat 9 场景 / 一个 ERA5 月值）"""
    provider: str  # e.g. microsoft-pc, nasa-power, fao-soilgrids
    collection: str  # e.g. landsat-c2-l2, era5-single-levels
    item_id: str  # provider-specific 唯一 ID
    datetime: str  # ISO8601 UTC
    bbox_wgs84: Optional[List[float]] = None  # 数据本身的覆盖范围（可能 > AOI）
    assets: List[AssetEntry] = field(default_factory=list)
    license: Optional[str] = None  # e.g. CC-BY-4.0, public-domain
    version: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["assets"] = [asdict(a) for a in self.assets]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetManifest":
        d = dict(d)
        assets = d.pop("assets", []) or []
        d["assets"] = [AssetEntry(**a) for a in assets]
        return cls(**d)


# ---- Output manifest ----


@dataclass
class OutputFile:
    path: str  # 绝对或相对路径
    kind: str  # raster / vector / table / json / text
    size_bytes: Optional[int] = None
    crs_epsg: Optional[int] = None
    bbox_wgs84: Optional[List[float]] = None
    resolution_m: Optional[float] = None
    nodata: Optional[float] = None
    band_count: Optional[int] = None
    feature_count: Optional[int] = None  # vector
    row_count: Optional[int] = None  # table


@dataclass
class OutputManifest:
    """一次 skill 调用的产物清单 + QA 摘要"""
    skill: str
    skill_version: str
    command: str
    started_at: str  # ISO8601 UTC
    finished_at: str  # ISO8601 UTC
    exit_code: int
    inputs: Dict[str, Any] = field(default_factory=dict)  # place / bbox / time / ...
    aoi: Optional[AOIManifest] = None
    datasets: List[DatasetManifest] = field(default_factory=list)
    outputs: List[OutputFile] = field(default_factory=list)
    qa: Dict[str, Any] = field(default_factory=dict)  # {feature_count, cloud_pct, ...}
    software: Dict[str, str] = field(default_factory=dict)  # {python: 3.12, gdal: 3.8, ...}
    error: Optional[Dict[str, Any]] = None  # 失败时填充

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.aoi is not None:
            d["aoi"] = self.aoi.to_dict()
        d["datasets"] = [ds.to_dict() for ds in self.datasets]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OutputManifest":
        aoi_d = d.pop("aoi", None)
        aoi = AOIManifest.from_dict(aoi_d) if aoi_d else None
        ds_d = d.pop("datasets", []) or []
        datasets = [DatasetManifest.from_dict(x) for x in ds_d]
        d["aoi"] = aoi
        d["datasets"] = datasets
        return cls(**d)


# ---- Error manifest ----


@dataclass
class ErrorManifest:
    """统一错误格式（stderr / --json-errors）"""
    kind: str  # EUsage / EDepend / ENetwork / ENoMatch / EValidate / EProcess
    code: int  # 退出码
    message: str
    skill: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


# ---- JSON Schema（轻量版，不引 jsonschema 依赖）----


AOI_SCHEMA = {
    "type": "object",
    "required": ["query", "resolver", "confidence"],
    "properties": {
        "query": {"type": "string"},
        "bbox_wgs84": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
            "description": "[west, south, east, north]",
        },
        "geometry_wgs84": {"type": ["object", "null"]},
        "centroid_wgs84": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2,
            "description": "[lon, lat]",
        },
        "resolver": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "ambiguity": {"type": "array"},
        "notes": {"type": ["string", "null"]},
    },
}

DATASET_SCHEMA = {
    "type": "object",
    "required": ["provider", "collection", "item_id", "datetime"],
    "properties": {
        "provider": {"type": "string"},
        "collection": {"type": "string"},
        "item_id": {"type": "string"},
        "datetime": {"type": "string"},
        "bbox_wgs84": {"type": "array"},
        "assets": {"type": "array"},
        "license": {"type": ["string", "null"]},
        "version": {"type": ["string", "null"]},
        "extra": {"type": "object"},
    },
}

OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["skill", "command", "exit_code"],
    "properties": {
        "skill": {"type": "string"},
        "skill_version": {"type": "string"},
        "command": {"type": "string"},
        "started_at": {"type": "string"},
        "finished_at": {"type": "string"},
        "exit_code": {"type": "integer"},
        "inputs": {"type": "object"},
        "aoi": AOI_SCHEMA,
        "datasets": {"type": "array", "items": DATASET_SCHEMA},
        "outputs": {"type": "array"},
        "qa": {"type": "object"},
        "software": {"type": "object"},
        "error": {"type": ["object", "null"]},
    },
}

ERROR_SCHEMA = {
    "type": "object",
    "required": ["kind", "code", "message", "skill"],
    "properties": {
        "kind": {"type": "string"},
        "code": {"type": "integer"},
        "message": {"type": "string"},
        "skill": {"type": "string"},
        "details": {"type": "object"},
        "timestamp": {"type": "string"},
    },
}


def validate_dict(d: Dict[str, Any], schema: Dict[str, Any], path: str = "") -> List[str]:
    """轻量 schema 验证：只检查 required / type / array length。
    返回错误消息列表（空列表 = 通过）。
    """
    errs: List[str] = []
    for req in schema.get("required", []):
        if req not in d:
            errs.append(f"{path}.{req}: missing required")
    for k, v in d.items():
        sub = schema.get("properties", {}).get(k)
        if sub is None:
            continue
        sub_type = sub.get("type")
        if sub_type and not _type_ok(v, sub_type):
            errs.append(f"{path}.{k}: type mismatch (got {type(v).__name__}, expected {sub_type})")
        if sub_type == "array" and isinstance(v, list):
            mn = sub.get("minItems"); mx = sub.get("maxItems")
            if mn is not None and len(v) < mn:
                errs.append(f"{path}.{k}: too few items ({len(v)} < {mn})")
            if mx is not None and len(v) > mx:
                errs.append(f"{path}.{k}: too many items ({len(v)} > {mx})")
    return errs


def _type_ok(v: Any, t: Union[str, List[str]]) -> bool:
    if isinstance(t, list):
        return any(_type_ok(v, x) for x in t)
    if t == "string":
        return isinstance(v, str)
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "boolean":
        return isinstance(v, bool)
    if t == "object":
        return isinstance(v, dict)
    if t == "array":
        return isinstance(v, list)
    if t == "null":
        return v is None
    return True


__all__ = [
    "AOIManifest", "DatasetManifest", "AssetEntry",
    "OutputManifest", "OutputFile", "ErrorManifest",
    "AOI_SCHEMA", "DATASET_SCHEMA", "OUTPUT_SCHEMA", "ERROR_SCHEMA",
    "validate_dict",
]
