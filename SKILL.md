---
name: geoskill-generative-adversarial-rs
description: 'GAN (U-Net + PatchGAN) 云去除 / 影像增强，torch+CUDA GPU 训练与推理'
---

# 生成对抗遥感应用 | Generative Adversarial Remote Sensing

Repairs and enhances imagery with a "generative" approach: in cloud removal mode, a cloud mask is detected and the contaminated pixels are reconstructed with a **U-Net Generator**; in enhancement mode, the same generator acts as a denoising autoencoder to improve contrast/quality.

This skill is a true deep-learning implementation of **GAN-based cloud inpainting / image enhancement (pix2pix style, Isola et al. 2017)**:

- **Generator**: U-Net encoder-decoder with 3 levels of downsampling + corresponding upsampling + skip connections;
- **Discriminator**: PatchGAN (a small CNN with a 70×70 receptive field) judging real/fake at the pixel-patch level;
- **Loss**: BCE (adversarial) + L1 (reconstruction, weight 100); BCEWithLogitsLoss for numerical stability;
- **Training and inference both run on CUDA GPUs** (torch >= 2.x, cuDNN 9.x).

Pretrained weights `gan_cloudremoval_weights.pt` ship with the skill (synthetic cloud-removal pairs, 1.5 MB, Generator 329K + Discriminator 42K parameters); if the weights are missing, they are automatically trained on the GPU and saved at first run. The original numpy operators (detect_cloud_mask / inpaint_masked / histogram_match / contrast_stretch / remove_clouds) are kept as a baseline reference and unit-test entry points.

## Dependencies / 依赖

```bash
pip install numpy rasterio scipy torch
```

## Usage / 使用方法

### Example 1 (synthetic data, offline)

```bash
python geoskill-generative-adversarial-rs.py --bbox 116.0 39.0 117.0 40.0 --synthetic --output-dir ./out
```

### Example 2: synthetic cloud removal (with PSNR comparison)

```bash
python geoskill-generative-adversarial-rs.py --bbox 116.0 39.0 117.0 40.0 --synthetic --mode cloud-removal --output-dir ./out
```

### Example 3: cloud removal on real imagery

```bash
python geoskill-generative-adversarial-rs.py --input cloudy.tif --percentile 88 --output-dir ./out
```

### Example 4: image enhancement

```bash
python geoskill-generative-adversarial-rs.py --input dim.tif --mode enhance --plow 2 --phigh 98 --output-dir ./out
```

## Output / 输出

| File | Format | Description |
|---|---|---|
| `cloud_removed.tif / enhanced.tif` | GeoTIFF | Reconstructed or enhanced image (depending on mode) |
| `cloud_mask.tif` | GeoTIFF | Cloud mask (cloud-removal mode) |
| `metrics.json` | JSON | Metrics: cloud fraction, PSNR before/after, std before/after, backend/device/weights, etc. |
| `output-manifest.json` | JSON | Run manifest (inputs/outputs/QA/exit code) |

## Data Source / 数据源 / Source

Local single-band GeoTIFF, or `--synthetic` (ground-truth scene + Gaussian cloud patches, with ground truth retained for evaluation).

## Limitations / 局限

- The model weights are trained only on synthetic spectral pairs (gradient + 4 land-cover classes + Gaussian cloud patches) and have not been field-accuracy calibrated on real satellite imagery; intended for screening-level rather than quantitative use.
- Bboxes crossing the 180° meridian are not supported (explicit error: split across the two sides).
- NoData imagery is automatically converted to NaN before the GAN, but scenarios with extremely small/narrow valid pixels are not fully tested.

## Privacy / 隐私声明 / Privacy

- Runs offline by default; `--synthetic` mode requires no network at all.
- All processing is performed locally; no user data is uploaded.

## License / License

MIT

---

<!-- ===== 中文原文 (Chinese Original) ===== -->

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
