# Generative Adversarial Remote Sensing (geoskill-generative-adversarial-rs)

> GAN (U-Net + PatchGAN) cloud removal / image enhancement, with torch+CUDA GPU training and inference

---

## 1. Overview

Uses a "generative" approach to repair and enhance imagery: cloud removal mode detects the cloud mask and reconstructs contaminated pixels with the **U-Net Generator**; enhancement mode uses the same Generator as a denoising autoencoder for contrast/quality improvement. This skill is a true DL implementation of **GAN-based cloud inpainting / image enhancement (pix2pix style, Isola et al. 2017)**: - **Generator**: U-Net encoder-decoder with 3 downsampling levels, corresponding upsampling and skip connections; - **Discriminator**: PatchGAN (a small CNN with a 70×70 receptive field) that discriminates real/fake at the patch level; - **Loss**: BCE (adversarial) + L1 (reconstruction, weight 100); BCEWithLogitsLoss for numerical stability; - **Training and inference both run on CUDA GPU** (torch >= 2.x, cuDNN 9.x). Pretrained weights `gan_cloudremoval_weights.pt` ship with the skill (synthetic cloud-removal pairs, 1.5 MB, Generator 329K + Discriminator 42K parameters); if the weights are missing, training is automatically run on the GPU at first use and the weights are saved to disk. The original numpy operators (detect_cloud_mask / inpaint_masked / histogram_match / contrast_stretch / remove_clouds) are kept as a **baseline for comparison and as unit test entry points**.

## 2. Features

Uses a "generative" approach to repair and enhance imagery: cloud removal mode detects the cloud mask and reconstructs contaminated pixels with the **U-Net Generator**; enhancement mode uses the same Generator as a denoising autoencoder for contrast/quality improvement. This skill is a true DL implementation of **GAN-based cloud inpainting / image enhancement (pix2pix style, Isola et al. 2017)**: - **Generator**: U-Net encoder-decoder with 3 downsampling levels, corresponding upsampling and skip connections; - **Discriminator**: PatchGAN (a small CNN with a 70×70 receptive field) that discriminates real/fake at the patch level; - **Loss**: BCE (adversarial) + L1 (reconstruction, weight 100); BCEWithLogitsLoss for numerical stability; - **Training and inference both run on CUDA GPU** (torch >= 2.x, cuDNN 9.x). Pretrained weights `gan_cloudremoval_weights.pt` ship with the skill (synthetic cloud-removal pairs, 1.5 MB, Generator 329K + Discriminator 42K parameters); if the weights are missing, training is automatically run on the GPU at first use and the weights are saved to disk. The original numpy operators (detect_cloud_mask / inpaint_masked / histogram_match / contrast_stretch / remove_clouds) are kept as a **baseline for comparison and as unit test entry points**.

## 3. Quick Start

```bash
pip install -r requirements.txt
python geoskill-generative-adversarial-rs.py --bbox 116 39 117 40 --synthetic --output-dir ./out
```

## 4. CLI Parameters

Run `python <skill>.py --help` for the full list. Common parameters:

| Parameter | Type | Description |
|---|---|---|
| `--bbox` | `float[4]` | WGS84 bounding box `min_lon min_lat max_lon max_lat` |
| `--input` | `path` | Local input file (GeoJSON/GeoTIFF/etc.) |
| `--output-dir` | `path` | Output directory (default `./output`) |
| `--synthetic` | `flag` | Use synthetic data instead of real input |
| `--quiet` | `flag` | Suppress non-essential stdout |

## 5. Input / Output

| File | Format | Description |
|---|---|---|
| `cloud_removed.tif / enhanced.tif` | GeoTIFF | Reconstructed or enhanced image (per mode) |
| `cloud_mask.tif` | GeoTIFF | Cloud mask (cloud removal mode) |
| `metrics.json` | JSON | Metrics: cloud fraction, PSNR before/after, std before/after, backend/device/weights, etc. |
| `output-manifest.json` | JSON | Run manifest (inputs/outputs/QA/exit code) |

## 6. Technical Principle

(see SKILL.md for details)

## 7. Methodology

This skill has been methodologically reviewed. See [`REVIEW.md`](./REVIEW.md) for:

- P0/P1/P2 issue counts and verdicts
- Reproduction commands
- Known limitations and edge cases

## 8. License

MIT License. See [`LICENSE`](./LICENSE) for full text.

---

# 生成对抗遥感应用（中文版）

> 本部分为完整中文文档，与上方英文部分对应。

---
name: geoskill-generative-adversarial-rs
description: 'GAN (U-Net + PatchGAN) 云去除 / 影像增强，torch+CUDA GPU 训练与推理'
---

# 生成对抗遥感应用 | Generative Adversarial Remote Sensing

用"生成式"思想修复与增强影像：云去除模式检测云掩膜并用 **U-Net Generator** 重建被污染像元；增强模式用同一个 Generator 作 denoising autoencoder 做对比度/质量提升。

本 skill 是 **GAN 云修复/影像增强（pix2pix 风格，Isola et al. 2017）** 的真 DL 实现：

- **Generator**：U-Net 编码器-解码器，3 级下采样 + 对应上采样 + skip connection；
- **Discriminator**：PatchGAN（70×70 receptive field 的小型 CNN），像元块级真/伪判别；
- **损失**：BCE（对抗）+ L1（重建，权重 100）；BCEWithLogitsLoss 数值稳定；
- **训练/推理均在 CUDA GPU**（torch >= 2.x，cuDNN 9.x）。

随 skill 附带预训练权重 `gan_cloudremoval_weights.pt`（合成云去除对，1.5MB，Generator 329K + Discriminator 42K 参数）；若权重缺失则在首次运行时自动用 GPU 训练并落盘。原始 numpy 算子（detect_cloud_mask / inpaint_masked / histogram_match / contrast_stretch / remove_clouds）保留为**对照基线与单元测试入口**。

## 依赖

```bash
pip install numpy rasterio scipy torch
```

## 使用方法

### 示例 1（合成数据，离线）

```bash
python geoskill-generative-adversarial-rs.py --bbox 116.0 39.0 117.0 40.0 --synthetic --output-dir ./out
```

### 示例 2：合成云去除（含 PSNR 对比）

```bash
python geoskill-generative-adversarial-rs.py --bbox 116.0 39.0 117.0 40.0 --synthetic --mode cloud-removal --output-dir ./out
```

### 示例 3：真实影像云去除

```bash
python geoskill-generative-adversarial-rs.py --input cloudy.tif --percentile 88 --output-dir ./out
```

### 示例 4：影像增强

```bash
python geoskill-generative-adversarial-rs.py --input dim.tif --mode enhance --plow 2 --phigh 98 --output-dir ./out
```

## 输出

| 文件 | 格式 | 说明 |
|---|---|---|
| `cloud_removed.tif / enhanced.tif` | GeoTIFF | 重建或增强后的影像（按模式） |
| `cloud_mask.tif` | GeoTIFF | 云掩膜（云去除模式） |
| `metrics.json` | JSON | 云占比、PSNR 前后、std 前后、backend/device/weights 等指标 |
| `output-manifest.json` | JSON | 运行清单（输入/输出/QA/退出码） |

## 数据源 / Source

本地单波段 GeoTIFF，或 `--synthetic`（真值场景 + 高斯云斑块，真值留存评估）。

## 局限

- 模型权重仅在合成光谱对（渐变 + 4 类地物 + 高斯云斑块）上训练，真实卫星影像上未做外场精度标定；用于筛查级而非定量化。
- 跨 180° 经线 bbox 不支持（明确报错：拆分到两侧）。
- NoData 影像自动 NaN 化后再走 GAN，但极小/极窄有效像元场景未充分测试。

## 隐私声明 / Privacy

- 默认离线运行，`--synthetic` 模式完全无网络。
- 所有处理在本地完成，不上传任何用户数据。

## License

MIT
