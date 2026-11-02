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

> 中文完整版：需要时再读 [references/SKILL.zh-CN.md](references/SKILL.zh-CN.md)，不要默认全量加载。
