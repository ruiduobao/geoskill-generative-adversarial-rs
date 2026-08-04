"""geoskill_core.sensors — 传感器/产品 registry

每个 entry 描述一个遥感产品的关键元数据：
- bands: 各波段的物理含义 + 数值含义（reflectance, radiance, temperature, etc.）
- scale: 缩放因子（DN → 物理值 = DN * scale + offset）
- offset: 偏移（默认 0）
- unit: 物理单位
- nodata: 无效值（可能在 DN 或物理值空间）
- qa: QA bit 定义（pixel_qa, qa_pixel, etc.）
- crs_default: 默认投影（UTM / Sinusoidal / WGS84）
- resolution_m: 空间分辨率
- license: 数据许可
- source: 数据源（planetary-computer, laads, nasa-power, fao-soilgrids, etc.）
- version: 产品版本

vendoring：本文件会 copy 到各 skill 内部。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class BandSpec:
    """单个波段的元数据"""
    key: str
    number: Optional[int]  # band number in collection
    physical: str  # 'reflectance', 'radiance', 'temperature_K', 'temperature_C', 'ndvi', ...
    description: str
    scale: float = 1.0
    offset: float = 0.0
    unit: str = ""  # 物理单位
    valid_range: Optional[Tuple[float, float]] = None  # 物理值空间


@dataclass
class QABitSpec:
    """QA bit 定义"""
    bit: int
    name: str
    meaning: str


@dataclass
class ProductSpec:
    """一个遥感产品"""
    collection: str  # e.g. "landsat-c2-l2", "modis-11A1-061"
    name: str  # 人类可读
    provider: str  # e.g. "planetary-computer", "laads", "nasa-power"
    bands: List[BandSpec] = field(default_factory=list)
    scale: float = 1.0
    offset: float = 0.0
    unit: str = ""
    nodata_dn: Optional[float] = None
    nodata_physical: Optional[float] = None
    crs_default: str = "EPSG:4326"
    resolution_m: Optional[float] = None
    license: str = "public-domain"
    version: str = "latest"
    qa_bits: List[QABitSpec] = field(default_factory=list)
    notes: str = ""


# ---- 传感器 / 产品 registry ----

REGISTRY: Dict[str, ProductSpec] = {}


def register(spec: ProductSpec) -> ProductSpec:
    REGISTRY[spec.collection] = spec
    return spec


# ---- Landsat 8/9 Collection 2 Level 2 ----

LANDSAT_C2_L2_BANDS = [
    BandSpec("SR_B1", 1, "reflectance", "Coastal Aerosol", 2.75e-5, -0.2, "", (0.0, 1.0)),
    BandSpec("SR_B2", 2, "reflectance", "Blue", 2.75e-5, -0.2, "", (0.0, 1.0)),
    BandSpec("SR_B3", 3, "reflectance", "Green", 2.75e-5, -0.2, "", (0.0, 1.0)),
    BandSpec("SR_B4", 4, "reflectance", "Red", 2.75e-5, -0.2, "", (0.0, 1.0)),
    BandSpec("SR_B5", 5, "reflectance", "NIR", 2.75e-5, -0.2, "", (0.0, 1.0)),
    BandSpec("SR_B6", 6, "reflectance", "SWIR1", 2.75e-5, -0.2, "", (0.0, 1.0)),
    BandSpec("SR_B7", 7, "reflectance", "SWIR2", 2.75e-5, -0.2, "", (0.0, 1.0)),
    BandSpec("ST_B10", 10, "radiance", "Thermal Infrared", 3.3420e-4, 0.1, "W/(m²·sr·µm)", None),
    BandSpec("QA_PIXEL", None, "qa", "Pixel QA bit array", 1.0, 0.0, "bitmask", None),
    BandSpec("QA_RADSAT", None, "qa", "Radiometric Saturation QA", 1.0, 0.0, "bitmask", None),
]

LANDSAT_C2_L2_QA_BITS = [
    QABitSpec(0, "Fill", "No data (0=valid)"),
    QABitSpec(1, "Dilated Cloud", "Cloud dilation (1=dilated)"),
    QABitSpec(2, "Cirrus", "Cirrus cloud (1=cirrus)"),
    QABitSpec(3, "Cloud", "Cloud (1=cloud)"),
    QABitSpec(4, "Cloud Shadow", "Cloud shadow (1=shadow)"),
    QABitSpec(5, "Snow", "Snow (1=snow)"),
    QABitSpec(6, "Clear", "Clear sky (1=clear)"),
    QABitSpec(7, "Water", "Water (1=water)"),
]

register(ProductSpec(
    collection="landsat-c2-l2",
    name="Landsat 8/9 Collection 2 Level 2 Science Products",
    provider="planetary-computer",
    bands=LANDSAT_C2_L2_BANDS,
    crs_default="EPSG:32633",  # UTM
    resolution_m=30.0,
    license="public-domain",
    version="2",
    qa_bits=LANDSAT_C2_L2_QA_BITS,
    notes="Surface Reflectance: 0.0001 * DN - 0.2 (unitless reflectance)",
))


# ---- Sentinel-2 L2A ----

SENTINEL2_L2A_BANDS = [
    BandSpec("B01", 1, "reflectance", "Coastal Aerosol", 1.0e-4, 0.0, "", (0.0, 1.0)),
    BandSpec("B02", 2, "reflectance", "Blue", 1.0e-4, 0.0, "", (0.0, 1.0)),
    BandSpec("B03", 3, "reflectance", "Green", 1.0e-4, 0.0, "", (0.0, 1.0)),
    BandSpec("B04", 4, "reflectance", "Red", 1.0e-4, 0.0, "", (0.0, 1.0)),
    BandSpec("B05", 5, "reflectance", "Red Edge 1", 1.0e-4, 0.0, "", (0.0, 1.0)),
    BandSpec("B06", 6, "reflectance", "Red Edge 2", 1.0e-4, 0.0, "", (0.0, 1.0)),
    BandSpec("B07", 7, "reflectance", "Red Edge 3", 1.0e-4, 0.0, "", (0.0, 1.0)),
    BandSpec("B08", 8, "reflectance", "NIR", 1.0e-4, 0.0, "", (0.0, 1.0)),
    BandSpec("B8A", 9, "reflectance", "Narrow NIR", 1.0e-4, 0.0, "", (0.0, 1.0)),
    BandSpec("B09", 10, "reflectance", "Water Vapour", 1.0e-4, 0.0, "", (0.0, 1.0)),
    BandSpec("B11", 11, "reflectance", "SWIR1", 1.0e-4, 0.0, "", (0.0, 1.0)),
    BandSpec("B12", 12, "reflectance", "SWIR2", 1.0e-4, 0.0, "", (0.0, 1.0)),
    BandSpec("SCL", None, "classification", "Scene Classification", 1.0, 0.0, "", (0.0, 11.0)),
]

register(ProductSpec(
    collection="sentinel-2-l2a",
    name="Sentinel-2 Level 2A Surface Reflectance",
    provider="planetary-computer",
    bands=SENTINEL2_L2A_BANDS,
    crs_default="EPSG:32633",
    resolution_m=10.0,
    license="public-domain",
    version="2A",
    notes="Reflectance: 0.0001 * DN (unitless)",
))


# ---- MODIS LST (MOD11A1 / MYD11A1 daily 1km) ----

MODIS_LST_DAILY_BANDS = [
    BandSpec("LST_Day_1km", None, "temperature_K", "Day Land Surface Temperature", 0.02, 0.0, "K", (200.0, 340.0)),
    BandSpec("LST_Night_1km", None, "temperature_K", "Night Land Surface Temperature", 0.02, 0.0, "K", (200.0, 340.0)),
    BandSpec("QC_Day", None, "qa", "Day QA", 1.0, 0.0, "bitmask", None),
    BandSpec("QC_Night", None, "qa", "Night QA", 1.0, 0.0, "bitmask", None),
    BandSpec("Emis_31", None, "emissivity", "Band 31 Emissivity", 0.002, 0.49, "", (0.49, 1.0)),
    BandSpec("Emis_32", None, "emissivity", "Band 32 Emissivity", 0.002, 0.49, "", (0.49, 1.0)),
]

register(ProductSpec(
    collection="modis-11A1-061",
    name="MODIS/Terra Land Surface Temperature Daily 1km",
    provider="laads",
    bands=MODIS_LST_DAILY_BANDS,
    crs_default="EPSG:4326",  # Sinusoidal in raw, WGS84 in reprojected
    resolution_m=1000.0,
    license="public-domain",
    version="6.1",
    notes="LST: 0.02 * DN (Kelvin). 转换为 °C: LST_K - 273.15.",
))


# ---- GPM IMERG ----

GPM_IMERG_BANDS = [
    BandSpec("precipitation", None, "precipitation", "Precipitation rate", 0.1, 0.0, "mm/hr", (0.0, 300.0)),
    BandSpec("precipitationCal", None, "precipitation", "Calibrated precipitation", 0.1, 0.0, "mm/hr", (0.0, 300.0)),
    BandSpec("randomError", None, "uncertainty", "Random error", 0.1, 0.0, "mm/hr", None),
    BandSpec("IRprecipitation", None, "precipitation", "IR-derived precipitation", 0.1, 0.0, "mm/hr", None),
]

register(ProductSpec(
    collection="gpm-imerg",
    name="GPM IMERG Final Precipitation",
    provider="ges-disc",
    bands=GPM_IMERG_BANDS,
    crs_default="EPSG:4326",
    resolution_m=10000.0,  # 0.1° ~ 10 km
    license="public-domain",
    version="07",
    notes="Precipitation: 0.1 * DN (mm/hr). 半小时/日/月聚合可选.",
))


# ---- ERA5 Single Levels ----

ERA5_BANDS = [
    BandSpec("t2m", None, "temperature_K", "2 metre temperature", 1.0, 0.0, "K", (180.0, 340.0)),
    BandSpec("sp", None, "pressure", "Surface pressure", 1.0, 0.0, "Pa", (87000.0, 110000.0)),
    BandSpec("tp", None, "precipitation_m", "Total precipitation (m)", 1000.0, 0.0, "mm", (0.0, 1000.0)),
    BandSpec("u10", None, "wind_speed", "10 metre U wind", 1.0, 0.0, "m/s", (-100.0, 100.0)),
    BandSpec("v10", None, "wind_speed", "10 metre V wind", 1.0, 0.0, "m/s", (-100.0, 100.0)),
]

register(ProductSpec(
    collection="era5-single-levels",
    name="ERA5 Single Levels Monthly Means",
    provider="planetary-computer",
    bands=ERA5_BANDS,
    crs_default="EPSG:4326",
    resolution_m=27830.0,  # 0.25°
    license="Copernicus",
    version="5",
    notes="2m 温度: K; 降水: m (单位转换 × 1000 → mm).",
))


# ---- NASA POWER ----

NASA_POWER_BANDS = [
    BandSpec("ALLSKY_SFC_SW_DWN", None, "irradiance", "All-sky surface SW downward", 0.01, 0.0, "MJ/m²/day", (0.0, 50.0)),
    BandSpec("T2M", None, "temperature_C", "2 metre temperature", 1.0, 0.0, "°C", (-90.0, 60.0)),
    BandSpec("PRECTOTCORR", None, "precipitation", "Precipitation corrected", 1.0, 0.0, "mm/day", (0.0, 500.0)),
    BandSpec("WS10M", None, "wind_speed", "10 metre wind speed", 0.1, 0.0, "m/s", (0.0, 50.0)),
]

register(ProductSpec(
    collection="nasa-power",
    name="NASA POWER Agroclimatology",
    provider="nasa-power",
    bands=NASA_POWER_BANDS,
    crs_default="EPSG:4326",
    resolution_m=55660.0,  # 0.5° × 0.5°
    license="public-domain",
    version="latest",
    notes="SW radiation: 0.01 * MJ/m²/day. 风速: 0.1 * m/s.",
))


# ---- ISRIC SoilGrids ----

SOILGRIDS_BANDS = [
    BandSpec("phh2o", None, "pH", "pH in H2O", 0.1, 0.0, "pH", (0.0, 14.0)),
    BandSpec("ocd", None, "carbon_density", "Organic carbon density", 1.0, 0.0, "g/dm³", (0.0, 500.0)),
    BandSpec("clay", None, "fraction", "Clay fraction", 1.0, 0.0, "%", (0.0, 100.0)),
    BandSpec("sand", None, "fraction", "Sand fraction", 1.0, 0.0, "%", (0.0, 100.0)),
    BandSpec("silt", None, "fraction", "Silt fraction", 1.0, 0.0, "%", (0.0, 100.0)),
    BandSpec("bdod", None, "bulk_density", "Bulk density", 1.0, 0.0, "kg/dm³", (0.0, 2.0)),
    BandSpec("cec", None, "exchange", "Cation exchange capacity", 1.0, 0.0, "mmol(c)/kg", (0.0, 1000.0)),
]

register(ProductSpec(
    collection="soilgrids",
    name="ISRIC SoilGrids 250m v2.0",
    provider="isric",
    bands=SOILGRIDS_BANDS,
    crs_default="EPSG:4326",
    resolution_m=250.0,
    license="CC-BY-4.0",
    version="2.0",
    notes="6 standard depths: 0-5, 5-15, 15-30, 30-60, 60-100, 100-200 cm.",
))


def get_product(collection: str) -> Optional[ProductSpec]:
    """查 product spec by collection id"""
    return REGISTRY.get(collection)


def apply_scale(dn, product: ProductSpec, band: str):
    """把 DN 转为物理值: physical = dn * scale + offset
    若 band 有自己的 scale/offset，优先用。
    支持 float / int / numpy 1D 数组输入。
    """
    b = next((x for x in product.bands if x.key == band), None)
    if b is not None:
        scale, offset = b.scale, b.offset
    else:
        scale, offset = product.scale, product.offset
    try:
        # numpy 数组优先
        import numpy as np
        if isinstance(dn, np.ndarray):
            return dn * scale + offset
    except ImportError:
        pass
    return float(dn) * scale + offset


def to_celsius(kelvin):
    """Kelvin → Celsius。支持 float / int / numpy 1D 数组。"""
    try:
        import numpy as np
        if isinstance(kelvin, np.ndarray):
            return kelvin - 273.15
    except ImportError:
        pass
    return float(kelvin) - 273.15


__all__ = [
    "BandSpec", "QABitSpec", "ProductSpec", "REGISTRY",
    "register", "get_product", "apply_scale", "to_celsius",
]
