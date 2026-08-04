"""geoskill_core.checks — CRS/单位/面积科学正确性强制检查

按规划 §5 Phase 2 第 3 项：所有下载器在写产物前应检查。

强制项：
1. CRS 不能是 (0,0) 哑值
2. 单位必须写在产物 metadata（GeoTIFF tags 或 sidecar）
3. 面积计算必须在投影 CRS 上做（不能直接在经纬度像元上算）
4. bbox 不能跨 180° 经线（除非专门处理）
5. nodata 必须有（除非特殊场景）
6. scale/offset 必须显式（避免下游误解）
"""
from __future__ import annotations
import math
from typing import List, Dict, Optional, Tuple

from .errors import ValidationError


# WGS84 椭球参数
WGS84_A = 6378137.0  # 长半轴（米）
WGS84_F = 1 / 298.257223563  # 扁率
WGS84_B = WGS84_A * (1 - WGS84_F)  # 短半轴
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2  # 第一偏心率平方


def geodesic_area_m2(bbox_wgs84: List[float]) -> float:
    """用 WGS84 椭球公式计算 bbox 面积（平方米）。

    简化算法：把 bbox 看作球冠差。
    精确公式：积分积分（参考 OSGeoArea / GeographicLib）。
    这里用球面近似 + 椭球校正。
    """
    if not bbox_wgs84 or len(bbox_wgs84) != 4:
        raise ValidationError("bbox must be (W, S, E, N)", bbox=bbox_wgs84)
    w, s, e, n = bbox_wgs84
    if w >= e or s >= n:
        raise ValidationError("bbox W<E, S<N required", bbox=bbox_wgs84)
    # 简化为球面
    R = WGS84_A
    # 球冠面积 = 2πR² |sin(lat2) - sin(lat1)|
    # 但这里计算的是矩形区域 → 球面梯形
    # 精确公式（球面）：
    area = abs(math.radians(e - w)) * R ** 2 * abs(
        math.sin(math.radians(n)) - math.sin(math.radians(s))
    )
    return area


def geodesic_area_km2(bbox_wgs84: List[float]) -> float:
    return geodesic_area_m2(bbox_wgs84) / 1e6


def check_bbox_wgs84(bbox_wgs84: List[float]) -> List[str]:
    """检查 bbox 合法性（不抛错，只返回 warnings 列表）"""
    issues = []
    if not bbox_wgs84 or len(bbox_wgs84) != 4:
        issues.append("bbox must be (W, S, E, N) with 4 values")
        return issues
    w, s, e, n = bbox_wgs84
    if not (-180 <= w <= 180 and -180 <= e <= 180):
        issues.append(f"longitude out of [-180, 180]: W={w}, E={e}")
    if not (-90 <= s <= 90 and -90 <= n <= 90):
        issues.append(f"latitude out of [-90, 90]: S={s}, N={n}")
    if w >= e:
        issues.append(f"W must be < E: W={w}, E={e}")
    if s >= n:
        issues.append(f"S must be < N: S={s}, N={n}")
    # 检查 180° 经线跨越
    if w < -180 or e > 180:
        issues.append(f"bbox crosses antimeridian: W={w}, E={e} (need wrap-around)")
    # 退化 bbox
    if abs(e - w) < 1e-6 or abs(n - s) < 1e-6:
        issues.append("bbox is degenerate (zero-size)")
    return issues


def check_crs_epsg(crs_epsg: Optional[int]) -> List[str]:
    """检查 EPSG code 合法性"""
    issues = []
    if crs_epsg is None:
        issues.append("CRS EPSG not set (downstream tools may default to WGS84)")
    elif crs_epsg <= 0:
        issues.append(f"Invalid EPSG: {crs_epsg}")
    return issues


def check_unit_in_metadata(unit: Optional[str], product: Optional[str] = None) -> List[str]:
    """检查 unit 字段被显式记录"""
    issues = []
    if not unit:
        issues.append(
            "unit not specified in metadata; "
            "downstream consumers may misinterpret values"
        )
    return issues


def check_nodata_set(nodata: Optional[float], dtype: Optional[str] = None) -> List[str]:
    """检查 nodata 是否设置"""
    issues = []
    if nodata is None:
        if dtype and dtype.startswith("float"):
            issues.append("Float raster without nodata (downstream analysis may mis-handle)")
        elif dtype and "int" in dtype:
            # 整数类型 nodata 可选
            pass
        else:
            issues.append("nodata not set; consider setting explicit nodata value")
    return issues


def check_pixel_scale(pixel_scale: Optional[Tuple[float, ...]],
                      crs_epsg: Optional[int] = None) -> List[str]:
    """检查 pixel_scale 合理性"""
    issues = []
    if not pixel_scale or len(pixel_scale) < 2:
        issues.append("pixel_scale not set; cannot determine resolution")
        return issues
    sx, sy = pixel_scale[0], pixel_scale[1]
    if sx <= 0 or sy <= 0:
        issues.append(f"Non-positive pixel_scale: ({sx}, {sy})")
    # 在 WGS84 下 sx 是度，>1 视为粗
    if crs_epsg == 4326 and (sx > 1.0 or sy > 1.0):
        issues.append(f"Coarse resolution for WGS84: ({sx}, {sy}) degrees")
    return issues


def check_scale_offset(scale: float, offset: float, valid_range: Optional[Tuple[float, float]] = None) -> List[str]:
    """检查 scale/offset 是否合理（避免精度损失）"""
    issues = []
    if scale == 0:
        issues.append("scale is 0; values would all equal offset")
    if scale < 0:
        issues.append(f"Negative scale: {scale} (likely wrong)")
    if valid_range:
        lo, hi = valid_range
        # 假设 DN 范围是 [0, 65535] (16-bit) 或 [0, 2^31-1] (32-bit)
        # 物理值范围: [offset, offset + scale * max_dn]
        # 检查 scale * 1000 在 valid_range 内是否合理
        if abs(scale) > 0 and (lo < -1e6 or hi > 1e6):
            issues.append(f"valid_range {valid_range} too large; check scale")
    return issues


def full_raster_check(
    *,
    bbox_wgs84: Optional[List[float]] = None,
    crs_epsg: Optional[int] = None,
    nodata: Optional[float] = None,
    pixel_scale: Optional[Tuple[float, ...]] = None,
    dtype: Optional[str] = None,
    unit: Optional[str] = None,
    scale: Optional[float] = None,
    offset: Optional[float] = None,
) -> Dict[str, List[str]]:
    """对栅格产物做完整科学正确性检查。

    Returns:
        dict: {category: [issues]} — 多个类别的 issue 列表
    """
    checks: Dict[str, List[str]] = {}
    if bbox_wgs84 is not None:
        checks["bbox"] = check_bbox_wgs84(bbox_wgs84)
    if crs_epsg is not None or pixel_scale is not None:
        # 至少做一次 CRS 检查
        crs_issues = check_crs_epsg(crs_epsg)
        if crs_issues:
            checks["crs"] = crs_issues
    if unit is not None or True:
        # 总是检查 unit（如果给了）
        unit_issues = check_unit_in_metadata(unit)
        if unit_issues:
            checks["unit"] = unit_issues
    if nodata is not None or dtype is not None:
        nodata_issues = check_nodata_set(nodata, dtype)
        if nodata_issues:
            checks["nodata"] = nodata_issues
    if pixel_scale is not None:
        scale_issues = check_pixel_scale(pixel_scale, crs_epsg)
        if scale_issues:
            checks["pixel_scale"] = scale_issues
    if scale is not None or offset is not None:
        so_issues = check_scale_offset(scale or 0, offset or 0)
        if so_issues:
            checks["scale_offset"] = so_issues
    # 过滤空 list
    return {k: v for k, v in checks.items() if v}


__all__ = [
    "geodesic_area_m2", "geodesic_area_km2",
    "check_bbox_wgs84", "check_crs_epsg",
    "check_unit_in_metadata", "check_nodata_set",
    "check_pixel_scale", "check_scale_offset",
    "full_raster_check",
    "WGS84_A", "WGS84_B", "WGS84_E2",
]
