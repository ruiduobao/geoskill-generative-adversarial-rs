# Generative Adversarial Remote Sensing (geoskill-generative-adversarial-rs)

> GAN (U-Net + PatchGAN) cloud removal / image enhancement, with torch+CUDA GPU training and inference

---

## 1. Overview

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

## 8. License

MIT License. See [`LICENSE`](./LICENSE) for full text.

---

> 中文完整版：[references/SKILL.zh-CN.md](references/SKILL.zh-CN.md)
