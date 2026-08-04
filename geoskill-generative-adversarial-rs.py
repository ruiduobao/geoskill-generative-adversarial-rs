#!/usr/bin/env python3
"""generative-adversarial-rs — 生成对抗遥感应用

用"生成式"思想修复与增强遥感影像：
- **云去除**：检测云掩膜并用 GAN 生成器（U-Net）重建被云污染的像元；
- **影像增强**：用同一个生成器作去噪自编码器，对低对比度/含噪影像做增强。

**核心模型（pix2pix 简化，Isola et al. 2017）**：
- Generator：U-Net 编码器-解码器，含 3 级下采样与对应上采样 + skip connection；
- Discriminator：PatchGAN（70×70 receptive field 的小型 CNN），像元块级真/伪判别；
- 损失：BCE（对抗）+ L1（重建，权重 100）；BCEWithLogitsLoss 数值稳定；
- 训练/推理均在 CUDA GPU（torch >= 2.x，cuDNN 9.x）。

随 skill 附带预训练权重 ``gan_cloudremoval_weights.pt``
（合成云去除对，~4.5MB）；若权重缺失则在首次运行时自动用 GPU 训练并落盘。
原始 numpy 算子（detect_cloud_mask / inpaint_masked / histogram_match /
contrast_stretch / remove_clouds）保留为**对照基线与单元测试入口**。

数据源：本地单波段 GeoTIFF，或 ``--synthetic`` 生成"真值 -> 加云"的自洽实验。

隐私声明 / Privacy：
- 默认离线运行，``--synthetic`` 模式完全无网络。
- 所有处理在本地完成，不上传任何用户数据。

Usage:
    python generative-adversarial-rs.py --input cloudy.tif --mode cloud-removal --output-dir ./out
    python generative-adversarial-rs.py --bbox 116 39 117 40 --synthetic --output-dir ./out

License: MIT
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

VERSION = "1.0.0"
SKILL_NAME = "generative-adversarial-rs"

try:
    from _geoskill_core.errors import (
        GeoSkillError, UsageError, DependencyError, ValidationError, ProcessError,
        to_exit_code,
    )
    from _geoskill_core.manifest import OutputManifest, OutputFile
except ImportError:  # pragma: no cover
    class GeoSkillError(Exception):
        def __init__(self, message: str, code: int = 7, kind: str = "EGeo", **kw):
            super().__init__(message)
            self.message, self.code, self.kind = message, code, kind

    class UsageError(GeoSkillError):
        def __init__(self, m, **k): super().__init__(m, code=2, kind="EUsage", **k)

    class DependencyError(GeoSkillError):
        def __init__(self, m, **k): super().__init__(m, code=3, kind="EDep", **k)

    class ValidationError(GeoSkillError):
        def __init__(self, m, **k): super().__init__(m, code=6, kind="EValidate", **k)

    class ProcessError(GeoSkillError):
        def __init__(self, m, **k): super().__init__(m, code=7, kind="EProcess", **k)

    def to_exit_code(exc):
        return getattr(exc, "code", 7)

    OutputManifest = None
    OutputFile = None


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


# ===========================================================================
# 经典算子（保留：单元测试覆盖 + 对照基线）
# ===========================================================================
def detect_cloud_mask(image: np.ndarray, percentile: float = 90.0,
                      threshold: Optional[float] = None) -> np.ndarray:
    """检测云掩膜：亮度高于阈值（或分位数）的像元视为云。"""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValidationError("detect_cloud_mask expects a 2D image", shape=list(img.shape))
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros(img.shape, dtype=bool)
    if threshold is None:
        threshold = float(np.percentile(finite, percentile))
    return img > threshold


def inpaint_masked(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """用有效像元插值重建掩膜像元（numpy 对照基线）。

    掩膜内用线性插值（Delaunay/griddata），凸包外用最近邻兜底。
    """
    from scipy.interpolate import griddata
    img = np.asarray(image, dtype=np.float64)
    mask = np.asarray(mask).astype(bool)
    if img.shape != mask.shape:
        raise ValidationError("image/mask shape mismatch",
                              image=list(img.shape), mask=list(mask.shape))
    if not mask.any():
        return img.copy()
    if not (~mask).any():
        return img.copy()

    ys, xs = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    valid = ~mask
    points = np.column_stack([ys[valid], xs[valid]])
    values = img[valid]
    targets = np.column_stack([ys[mask], xs[mask]])

    filled_linear = griddata(points, values, targets, method="linear")
    nan_idx = np.isnan(filled_linear)
    if nan_idx.any():
        filled_nearest = griddata(points, values, targets[nan_idx], method="nearest")
        filled_linear[nan_idx] = filled_nearest

    out = img.copy()
    out[mask] = filled_linear
    return out


def histogram_match(source: np.ndarray, reference: np.ndarray,
                    n_bins: int = 256) -> np.ndarray:
    """直方图匹配（numpy 对照基线）。"""
    src = np.asarray(source, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    src_valid = src[np.isfinite(src)]
    ref_valid = ref[np.isfinite(ref)]
    if src_valid.size == 0 or ref_valid.size == 0:
        raise ValidationError("source/reference is empty")
    lo = float(min(src_valid.min(), ref_valid.min()))
    hi = float(max(src_valid.max(), ref_valid.max()))
    if hi <= lo:
        return src.copy()
    edges = np.linspace(lo, hi, n_bins + 1)
    src_hist, _ = np.histogram(src_valid, bins=edges)
    ref_hist, _ = np.histogram(ref_valid, bins=edges)
    src_cdf = np.cumsum(src_hist).astype(np.float64)
    ref_cdf = np.cumsum(ref_hist).astype(np.float64)
    src_cdf /= max(src_cdf[-1], 1.0)
    ref_cdf /= max(ref_cdf[-1], 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = np.interp(src, centers, np.interp(src_cdf, ref_cdf, centers))
    return out


def contrast_stretch(image: np.ndarray, plow: float = 2.0, phigh: float = 98.0) -> np.ndarray:
    """百分位拉伸到 [0, 1]（numpy 对照基线）。"""
    img = np.asarray(image, dtype=np.float64)
    if not 0.0 <= plow < phigh <= 100.0:
        raise UsageError("need 0 <= plow < phigh <= 100", plow=plow, phigh=phigh)
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        raise ValidationError("image is empty")
    lo = float(np.percentile(finite, plow))
    hi = float(np.percentile(finite, phigh))
    if hi <= lo:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def psnr(reference: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if ref.shape != tgt.shape:
        raise ValidationError("psnr shape mismatch")
    mse = float(np.mean((ref - tgt) ** 2))
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * np.log10((data_range ** 2) / mse))


def remove_clouds(scene: np.ndarray, percentile: float = 90.0,
                  threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """云去除流程（numpy 对照基线）：检测云 -> 插值重建。"""
    mask = detect_cloud_mask(scene, percentile=percentile, threshold=threshold)
    result = inpaint_masked(scene, mask)
    info = {
        "cloud_fraction": float(np.mean(mask)),
        "n_cloud_pixels": int(mask.sum()),
    }
    return result, mask, info


# ===========================================================================
# 合成数据：真值场景 + 云污染（自洽实验）
# ===========================================================================
def generate_synthetic(
    bbox: List[float],
    width: int = 64,
    height: int = 64,
    cloud_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """返回 (cloudy_scene, cloud_mask_truth, clean_truth, info)。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    base = 0.25 + 0.4 * (xx / max(width - 1, 1)) + 0.1 * (yy / max(height - 1, 1))
    truth = base.copy()
    truth[height // 4:height // 4 + 10, width // 4:width // 4 + 10] = 0.9
    truth[2 * height // 3:, 2 * width // 3:] = 0.1
    truth += rng.normal(0, 0.01, truth.shape)
    truth = np.clip(truth, 0.0, 1.0)

    mask = np.zeros((height, width), dtype=bool)
    target_px = int(cloud_fraction * height * width)
    covered = 0
    while covered < target_px:
        cy = rng.integers(0, height)
        cx = rng.integers(0, width)
        sigma = float(rng.uniform(3.0, 7.0))
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma * sigma)) > 0.3
        mask |= blob
        covered = int(mask.sum())

    cloudy = truth.copy()
    cloudy[mask] = np.clip(rng.normal(0.95, 0.03, size=int(mask.sum())), 0.0, 1.0)

    info = {"bbox": bbox, "width": width, "height": height,
            "cloud_fraction": float(np.mean(mask))}
    return cloudy.astype(np.float32), mask, truth.astype(np.float32), info


# ===========================================================================
# 深度学习模块：U-Net Generator + PatchGAN Discriminator
# ===========================================================================
WEIGHTS_FILENAME = "gan_cloudremoval_weights.pt"
_MODEL_CACHE: Dict[str, Any] = {}


def _prepare_dll_paths() -> None:
    """Windows：把 conda env 的 Library\\bin 注册到 DLL 搜索路径。

    torch 的 cuDNN 依赖（cudnn64_9.dll / cudnn_graph64_9.dll / cudart64_12.dll
    等）装在 <env>\\Library\\bin；不注册则 torch 加载 cuDNN 失败，更严重时
    ``torch.backends.cudnn.is_available()`` 探测调用会直接崩进程
    （exit 0xC0000409，try/except 不可捕获）。详见 SHARED_ISSUES SI-004。
    """
    if os.name != "nt":
        return
    lib_bin = os.path.join(sys.prefix, "Library", "bin")
    if not os.path.isdir(lib_bin):
        return
    if lib_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = lib_bin + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(lib_bin)
    except (OSError, AttributeError):
        pass


def _require_torch():
    _prepare_dll_paths()
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise DependencyError(
            "torch is required for generative-adversarial-rs "
            "(pip install torch or use the bundled env)") from exc
    import torch
    return torch


def _cuda_device(torch_mod):
    if not torch_mod.cuda.is_available():
        raise DependencyError(
            "generative-adversarial-rs requires a CUDA GPU; "
            "torch.cuda.is_available() is False in this environment")
    # cuDNN 探测：成功则启用加速；探测崩进程时退回 CUDA 原生卷积（仍在 GPU）。
    try:
        torch_mod.backends.cudnn.enabled = bool(torch_mod.backends.cudnn.is_available())
    except Exception:  # pragma: no cover
        torch_mod.backends.cudnn.enabled = False
    return torch_mod.device("cuda")


def _conv_block(torch_mod, cin: int, cout: int, act: str = "leaky"):
    nn = torch_mod.nn
    layers = [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout)]
    if act == "leaky":
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    else:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def _build_generator(torch_mod, in_ch: int = 1, out_ch: int = 1, base: int = 16):
    """U-Net Generator：3 级下采样 + 对应上采样 + skip connection。"""
    nn = torch_mod.nn

    class GeneratorUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = _conv_block(torch_mod, in_ch, base, "leaky")
            self.enc2 = _conv_block(torch_mod, base, base * 2, "leaky")
            self.enc3 = _conv_block(torch_mod, base * 2, base * 4, "leaky")
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = _conv_block(torch_mod, base * 4, base * 8, "leaky")
            self.up3 = nn.ConvTranspose2d(base * 8, base * 8, 2, stride=2)
            self.dec3 = _conv_block(torch_mod, base * 8 + base * 4, base * 4, "relu")
            self.up2 = nn.ConvTranspose2d(base * 4, base * 4, 2, stride=2)
            self.dec2 = _conv_block(torch_mod, base * 4 + base * 2, base * 2, "relu")
            self.up1 = nn.ConvTranspose2d(base * 2, base * 2, 2, stride=2)
            self.dec1 = _conv_block(torch_mod, base * 2 + base, base, "relu")
            self.head = nn.Conv2d(base, out_ch, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))
            d3 = self.dec3(torch_mod.cat([self.up3(b), e3], 1))
            d2 = self.dec2(torch_mod.cat([self.up2(d3), e2], 1))
            d1 = self.dec1(torch_mod.cat([self.up1(d2), e1], 1))
            return torch_mod.sigmoid(self.head(d1))

    return GeneratorUNet()


def _build_discriminator(torch_mod, in_ch: int = 1, base: int = 16):
    """PatchGAN Discriminator：70×70 receptive field 的小型 CNN。"""
    nn = torch_mod.nn

    class DiscriminatorPatch(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, base, 4, 2, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(base, base * 2, 4, 2, 1),
                nn.BatchNorm2d(base * 2),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(base * 2, base * 4, 4, 2, 1),
                nn.BatchNorm2d(base * 4),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(base * 4, 1, 4, 1, 1),  # patch-wise logits
            )

        def forward(self, x):
            return self.net(x)

    return DiscriminatorPatch()


def weights_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), WEIGHTS_FILENAME)


def _synthesize_training_pair(rng: np.random.Generator, h: int = 64, w: int = 64):
    """生成 (cloudy, clean) 训练对。"""
    cloudy, _, truth, _ = generate_synthetic(
        [116, 39, 117, 40], width=w, height=h, seed=int(rng.integers(1, 1 << 30)))
    return cloudy.astype(np.float32), truth.astype(np.float32)


def train_gan(G, D, device, torch_mod,
              epochs: int = 20, n_pairs: int = 200, batch: int = 16,
              lr: float = 2e-4, seed: int = 42) -> Dict[str, Any]:
    """在合成云去除对上训练 GAN（GPU）。返回训练元信息。"""
    torch_mod.manual_seed(seed)
    G, D = G.to(device), D.to(device)
    rng = np.random.default_rng(seed)
    pairs = [_synthesize_training_pair(rng) for _ in range(n_pairs)]
    rng_h = np.random.default_rng(seed + 1)
    holdout = [_synthesize_training_pair(rng_h) for _ in range(32)]

    opt_g = torch_mod.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch_mod.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = torch_mod.nn.BCEWithLogitsLoss()
    l1 = torch_mod.nn.L1Loss()

    step = 0
    for _ in range(epochs):
        idx = rng.permutation(len(pairs))
        for s in range(0, len(idx), batch):
            chunk = [pairs[i] for i in idx[s:s + batch]]
            cloudy = torch_mod.from_numpy(np.stack([p[0] for p in chunk]))[:, None].to(device).float()
            clean = torch_mod.from_numpy(np.stack([p[1] for p in chunk]))[:, None].to(device).float()
            # Discriminator step
            D.zero_grad()
            pred_real = D(clean)
            fake = G(cloudy).detach()
            pred_fake = D(fake)
            loss_d = bce(pred_real, torch_mod.ones_like(pred_real)) + \
                     bce(pred_fake, torch_mod.zeros_like(pred_fake))
            loss_d.backward()
            opt_d.step()
            # Generator step (adversarial + L1 reconstruction)
            G.zero_grad()
            fake_g = G(cloudy)
            loss_g_adv = bce(D(fake_g), torch_mod.ones_like(pred_real))
            loss_g_l1 = l1(fake_g, clean) * 100.0
            (loss_g_adv + loss_g_l1).backward()
            opt_g.step()
            step += 1

    # holdout 评估
    G.eval()
    psnr_sum = 0.0
    with torch_mod.no_grad():
        for cloudy, clean in holdout:
            x = torch_mod.from_numpy(cloudy[None, None]).to(device).float()
            rec = G(x)[0, 0].cpu().numpy()
            psnr_sum += psnr(clean, rec)
    meta = {
        "arch": "unet-patchgan-lite",
        "base": 16,
        "epochs": epochs,
        "n_pairs": n_pairs,
        "batch": batch,
        "steps": step,
        "n_params_G": int(sum(p.numel() for p in G.parameters())),
        "n_params_D": int(sum(p.numel() for p in D.parameters())),
        "holdout_psnr": round(psnr_sum / len(holdout), 4),
        "trained_at": _utc_now(),
        "framework": f"torch {torch_mod.__version__} / {torch_mod.cuda.get_device_name(0)}",
    }
    return meta


def _ensure_gan() -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """加载随附权重；缺失时在 GPU 上训练并落盘。返回 (G, D, device, meta)。"""
    if _MODEL_CACHE:
        return (_MODEL_CACHE["G"], _MODEL_CACHE["D"],
                _MODEL_CACHE["device"], _MODEL_CACHE["meta"])
    torch_mod = _require_torch()
    device = _cuda_device(torch_mod)
    G = _build_generator(torch_mod)
    D = _build_discriminator(torch_mod)
    path = weights_path()
    if os.path.exists(path):
        ck = torch_mod.load(path, map_location="cpu")
        G.load_state_dict(ck["G_state_dict"])
        D.load_state_dict(ck["D_state_dict"])
        meta = dict(ck.get("meta", {}))
        meta["weights"] = os.path.basename(path)
    else:
        meta = train_gan(G, D, device, torch_mod)
        meta["weights"] = "trained-on-the-fly (weights file missing)"
        torch_mod.save({"G_state_dict": G.state_dict(),
                        "D_state_dict": D.state_dict(),
                        "meta": meta}, path)
    G = G.to(device).eval()
    D = D.to(device).eval()
    _MODEL_CACHE.update(G=G, D=D, device=device, meta=meta)
    return G, D, device, meta


def _pad_to_multiple(arr2d: np.ndarray, multiple: int = 4) -> Tuple[np.ndarray, int, int]:
    """把 2D 数组 pad 到 multiple 的倍数，返回 (padded, h, w)。"""
    h, w = arr2d.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        arr2d = np.pad(arr2d, ((0, ph), (0, pw)), mode="edge")
    return arr2d, h, w


def gan_inference(G, device, torch_mod, image: np.ndarray) -> np.ndarray:
    """Generator 前向推理：input → reconstruction，自动 pad 到 4 的倍数再裁回。"""
    if image.ndim != 2:
        raise ValidationError("gan_inference expects 2D image", shape=list(image.shape))
    arr, h, w = _pad_to_multiple(image.astype(np.float32), multiple=4)
    t = torch_mod.from_numpy(arr[None, None]).to(device).float()
    with torch_mod.no_grad():
        out = G(t)[0, 0]
    return out[:h, :w].cpu().numpy()


def gan_cloud_removal(image: np.ndarray, percentile: float = 90.0,
                      threshold: Optional[float] = None
                      ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """GAN 路径的云去除：分位数检测 + U-Net 生成器重建。"""
    mask = detect_cloud_mask(image, percentile=percentile, threshold=threshold)
    info: Dict[str, Any] = {
        "cloud_fraction": float(np.mean(mask)),
        "n_cloud_pixels": int(mask.sum()),
    }
    if not mask.any():
        info["backend"] = "torch gan (no cloud; generator skipped)"
        return image.astype(np.float32), mask, info

    torch_mod = _require_torch()
    G, _, device, meta = _ensure_gan()
    reconstruction = gan_inference(G, device, torch_mod, image.astype(np.float32))

    out = image.astype(np.float32).copy()
    out[mask] = reconstruction[mask]
    info["backend"] = f"torch gan ({meta.get('arch', 'unet-patchgan-lite')})"
    info["device"] = str(device)
    info["weights"] = str(meta.get("weights", "n/a"))
    info["holdout_psnr"] = meta.get("holdout_psnr")
    return out, mask, info


def gan_enhance(image: np.ndarray, plow: float = 2.0, phigh: float = 98.0,
                ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """GAN 路径的增强：Generator 用作 denoising/autoencoder。

    1) 把影像归一到 [0,1]；
    2) Generator 前向；
    3) 反归一到原始值域。
    """
    if not 0.0 <= plow < phigh <= 100.0:
        # 与 contrast_stretch 保持一致：值域/顺序违反 → UsageError (exit 2)
        raise UsageError("need 0 <= plow < phigh <= 100", plow=plow, phigh=phigh)
    torch_mod = _require_torch()
    G, _, device, meta = _ensure_gan()
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValidationError("input is empty")
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        raise ValidationError("input has zero range; cannot enhance")
    normed = np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    rec = gan_inference(G, device, torch_mod, normed)
    out = (rec * (hi - lo) + lo).astype(np.float32)
    info = {
        "std_before": float(np.std(image)),
        "std_after": float(np.std(out)),
        "backend": f"torch gan ({meta.get('arch', 'unet-patchgan-lite')})",
        "device": str(device),
        "weights": str(meta.get("weights", "n/a")),
        "holdout_psnr": meta.get("holdout_psnr"),
    }
    return out, info


# ===========================================================================
# 校验
# ===========================================================================
def validate_bbox(bbox: Optional[List[float]]) -> List[float]:
    """校验 bbox [W, S, E, N]；不合法抛 ValidationError（exit 6）。"""
    if bbox is None:
        raise UsageError("provide --bbox (synthetic mode) or --input <raster>")
    try:
        w, s, e, n = (float(v) for v in bbox)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"bbox values must be numeric: {bbox}") from exc
    if not (-180.0 <= w <= 180.0 and -180.0 <= e <= 180.0):
        raise ValidationError(f"bbox longitude out of range [-180, 180]: W={w}, E={e}")
    if not (-90.0 <= s <= 90.0 and -90.0 <= n <= 90.0):
        raise ValidationError(f"bbox latitude out of range [-90, 90]: S={s}, N={n}")
    if w > e:
        raise ValidationError(
            f"bbox W ({w}) > E ({e}); antimeridian-crossing bbox is not supported — "
            "split the request into two bboxes on either side of +/-180")
    if s > n:
        raise ValidationError(f"bbox S ({s}) > N ({n})")
    return [w, s, e, n]


# ===========================================================================
# GeoTIFF I/O
# ===========================================================================
def write_geotiff(path: str, array: np.ndarray, bbox: List[float], nodata: float = -9999.0) -> None:
    import rasterio
    from rasterio.transform import from_bounds

    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    nb, h, w = arr.shape
    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], w, h)
    profile = {
        "driver": "GTiff", "height": h, "width": w, "count": nb,
        "dtype": "float32", "crs": "EPSG:4326", "transform": transform,
        "nodata": nodata, "compress": "deflate",
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        for b in range(nb):
            dst.write(arr[b].astype("float32"), b + 1)


def read_geotiff(path: str) -> Tuple[np.ndarray, List[float]]:
    """读取栅格，返回 (cube, bbox)。NoData 像元一并读回（保留测试签名）。"""
    import rasterio
    if not os.path.exists(path):
        raise UsageError(f"input raster not found: {path}", path=path)
    with rasterio.open(path) as src:
        cube = src.read().astype(np.float32)
        b = src.bounds
        bbox = [b.left, b.bottom, b.right, b.top]
    return cube, bbox


def read_geotiff_masked(path: str) -> Tuple[np.ndarray, List[float], np.ndarray]:
    """读取栅格并返回 (cube, bbox, valid_mask)。NoData 像元 mask=False。"""
    import rasterio
    if not os.path.exists(path):
        raise UsageError(f"input raster not found: {path}", path=path)
    with rasterio.open(path) as src:
        cube = src.read().astype(np.float32)
        b = src.bounds
        bbox = [b.left, b.bottom, b.right, b.top]
        try:
            mask = src.read_masks(1) > 0
        except Exception:
            mask = np.ones(cube.shape[1:], dtype=bool)
        nd = src.nodata
    if nd is not None:
        for k in range(cube.shape[0]):
            mask &= ~np.isclose(cube[k], nd)
    return cube, bbox, mask


# ===========================================================================
# Manifest
# ===========================================================================
def write_manifest(output_dir, args, outputs, qa, started_at, exit_code, bbox):
    if OutputManifest is None:
        return None
    cmd = " ".join([SKILL_NAME] + sys.argv[1:])
    man = OutputManifest(
        skill=SKILL_NAME, skill_version=VERSION, command=cmd,
        started_at=started_at, finished_at=_utc_now(), exit_code=exit_code,
        inputs={
            "input": getattr(args, "input", None),
            "mode": getattr(args, "mode", None),
            "synthetic": bool(getattr(args, "synthetic", False)),
        },
        outputs=[OutputFile(**o) for o in outputs],
        qa=qa,
        software={"python": sys.version.split()[0], "skill": SKILL_NAME},
    )
    path = os.path.join(output_dir, "output-manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(man.to_dict(), f, ensure_ascii=False, indent=2, default=str)
    return path


# ===========================================================================
# 主流程
# ===========================================================================
def process(args: argparse.Namespace) -> int:
    started_at = _utc_now()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    bbox = list(args.bbox) if args.bbox else None
    truth: Optional[np.ndarray] = None
    valid: Optional[np.ndarray] = None

    if args.input and not args.synthetic:
        cube, file_bbox, valid = read_geotiff_masked(args.input)
        bbox = validate_bbox(bbox if bbox is not None else file_bbox)
        if cube.size == 0:
            raise ValidationError("input raster is empty")
        if valid is not None and not bool(valid.any()):
            raise ValidationError(
                "input raster contains no valid (non-NoData) pixels")
        image = cube[0] if cube.ndim == 3 else cube
        # 用 NaN 标记无效像元（避免 GAN 看到 nodata 值）
        if valid is not None:
            image = np.where(valid, image, np.nan).astype(np.float32)
        source_note = args.input
    else:
        bbox = validate_bbox(bbox)
        image, _, truth, _ = generate_synthetic(bbox, seed=args.seed)
        source_note = "synthetic"

    if image.size == 0:
        raise ValidationError("input raster is empty")

    outputs: List[Dict[str, Any]] = []
    qa: Dict[str, Any] = {"source": source_note, "mode": args.mode}

    if args.mode == "cloud-removal":
        result, mask, info = gan_cloud_removal(image, percentile=args.percentile)
        out_tif = os.path.join(output_dir, "cloud_removed.tif")
        write_geotiff(out_tif, result.astype(np.float32), bbox)
        mask_tif = os.path.join(output_dir, "cloud_mask.tif")
        write_geotiff(mask_tif, mask.astype(np.float32), bbox)
        outputs += [
            {"path": out_tif, "kind": "raster", "crs_epsg": 4326,
             "bbox_wgs84": bbox, "band_count": 1},
            {"path": mask_tif, "kind": "raster", "crs_epsg": 4326,
             "bbox_wgs84": bbox, "band_count": 1},
        ]
        qa["cloud_fraction"] = info["cloud_fraction"]
        qa["backend"] = info["backend"]
        qa["device"] = info["device"]
        qa["weights"] = info["weights"]
        if "holdout_psnr" in info:
            qa["model_holdout_psnr"] = info["holdout_psnr"]
        if truth is not None:
            qa["psnr_before"] = psnr(truth, image)
            qa["psnr_after"] = psnr(truth, result)
    else:  # enhance
        result, info = gan_enhance(image, plow=args.plow, phigh=args.phigh)
        out_tif = os.path.join(output_dir, "enhanced.tif")
        write_geotiff(out_tif, result.astype(np.float32), bbox)
        outputs.append({"path": out_tif, "kind": "raster", "crs_epsg": 4326,
                        "bbox_wgs84": bbox, "band_count": 1})
        qa["std_before"] = info["std_before"]
        qa["std_after"] = info["std_after"]
        qa["backend"] = info["backend"]
        qa["device"] = info["device"]
        qa["weights"] = info["weights"]
        if "holdout_psnr" in info:
            qa["model_holdout_psnr"] = info["holdout_psnr"]

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(qa, f, ensure_ascii=False, indent=2)
    outputs.append({"path": metrics_path, "kind": "json"})

    man_path = write_manifest(output_dir, args, outputs, qa, started_at, 0, bbox)

    if not args.quiet:
        print(f"[{SKILL_NAME}] source: {source_note}  mode: {args.mode}")
        print(f"[{SKILL_NAME}] backend: {qa.get('backend','?')} on {qa.get('device','?')}")
        if args.mode == "cloud-removal":
            print(f"[{SKILL_NAME}] cloud fraction: {qa['cloud_fraction']:.3f}")
            if "psnr_after" in qa:
                print(f"[{SKILL_NAME}] PSNR before/after: "
                      f"{qa['psnr_before']:.2f} / {qa['psnr_after']:.2f} dB")
        else:
            print(f"[{SKILL_NAME}] std before/after: "
                  f"{qa['std_before']:.4f} / {qa['std_after']:.4f}")
        for o in outputs:
            print(f"[{SKILL_NAME}] output: {o['path']}")
        if man_path:
            print(f"[{SKILL_NAME}] manifest: {man_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=SKILL_NAME,
        description="Generative cloud removal (U-Net GAN) and image enhancement (torch/CUDA).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                   help="geographic extent [minLon minLat maxLon maxLat]")
    p.add_argument("--input", help="input GeoTIFF (first band used)")
    p.add_argument("--mode", default="cloud-removal", choices=["cloud-removal", "enhance"],
                   help="generative task (default: cloud-removal)")
    p.add_argument("--percentile", type=float, default=88.0,
                   help="cloud brightness percentile threshold")
    p.add_argument("--plow", type=float, default=2.0, help="stretch low percentile (enhance)")
    p.add_argument("--phigh", type=float, default=98.0, help="stretch high percentile (enhance)")
    p.add_argument("--seed", type=int, default=42, help="seed for synthetic data")
    p.add_argument("--synthetic", action="store_true", help="generate synthetic data (offline)")
    p.add_argument("--output-dir", default="./output", help="output directory")
    p.add_argument("--quiet", action="store_true", help="suppress console output")
    p.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return process(args)
    except GeoSkillError as exc:
        print(f"[{SKILL_NAME}] ERROR [{exc.kind}] {exc.message}", file=sys.stderr)
        return exc.code
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"[{SKILL_NAME}] ERROR {exc}", file=sys.stderr)
        return to_exit_code(exc)


if __name__ == "__main__":
    sys.exit(main())
