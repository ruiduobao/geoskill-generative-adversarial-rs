"""Core algorithm tests for generative-adversarial-rs."""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from conftest import mod


class TestDetectCloudMask:
    def test_bright_blob_detected(self):
        img = np.full((32, 32), 0.2)
        img[10:16, 10:16] = 0.95
        mask = mod.detect_cloud_mask(img, percentile=90.0)
        assert mask[12, 12] == True
        assert mask[0, 0] == False

    def test_explicit_threshold(self):
        img = np.array([[0.1, 0.5], [0.9, 0.2]])
        mask = mod.detect_cloud_mask(img, threshold=0.4)
        np.testing.assert_array_equal(mask, [[False, True], [True, False]])

    def test_empty_image(self):
        mask = mod.detect_cloud_mask(np.full((4, 4), np.nan))
        assert not mask.any()

    def test_rejects_3d(self):
        with pytest.raises(mod.ValidationError):
            mod.detect_cloud_mask(np.zeros((2, 4, 4)))


class TestInpaintMasked:
    def test_recovers_gradient(self):
        """线性渐变上挖洞，插值应还原渐变值。"""
        img = np.tile(np.linspace(0, 1, 32), (32, 1))  # 按列线性
        mask = np.zeros_like(img, dtype=bool)
        mask[:, 14:18] = True
        out = mod.inpaint_masked(img, mask)
        np.testing.assert_allclose(out[mask], img[mask], atol=1e-6)

    def test_no_mask_identity(self):
        img = np.random.uniform(0, 1, (8, 8))
        out = mod.inpaint_masked(img, np.zeros((8, 8), dtype=bool))
        np.testing.assert_array_equal(out, img)

    def test_all_masked_returns_copy(self):
        img = np.random.uniform(0, 1, (8, 8))
        out = mod.inpaint_masked(img, np.ones((8, 8), dtype=bool))
        np.testing.assert_array_equal(out, img)

    def test_shape_mismatch_raises(self):
        with pytest.raises(mod.ValidationError):
            mod.inpaint_masked(np.zeros((8, 8)), np.zeros((4, 4), dtype=bool))


class TestHistogramMatch:
    def test_matches_reference_distribution(self):
        """匹配后的分位数应接近参考的分位数。"""
        rng = np.random.default_rng(0)
        source = rng.uniform(0, 1, (64, 64))
        reference = rng.uniform(10, 20, (64, 64))
        matched = mod.histogram_match(source, reference)
        assert np.percentile(matched, 50) == pytest.approx(np.percentile(reference, 50), abs=1.0)
        assert np.percentile(matched, 90) == pytest.approx(np.percentile(reference, 90), abs=1.5)

    def test_identity_when_same_distribution(self):
        rng = np.random.default_rng(1)
        img = rng.uniform(0, 1, (32, 32))
        matched = mod.histogram_match(img, img)
        np.testing.assert_allclose(matched, img, atol=0.02)

    def test_empty_raises(self):
        with pytest.raises(mod.ValidationError):
            mod.histogram_match(np.full((4, 4), np.nan), np.ones((4, 4)))


class TestContrastStretch:
    def test_output_range(self):
        img = np.random.uniform(100, 500, (32, 32))
        out = mod.contrast_stretch(img, plow=2, phigh=98)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_monotonic(self):
        img = np.array([[1.0, 2.0, 3.0, 4.0, 100.0]])
        out = mod.contrast_stretch(img, plow=0, phigh=100)
        assert np.all(np.diff(out) >= -1e-12)

    def test_bad_percentiles(self):
        with pytest.raises(mod.UsageError):
            mod.contrast_stretch(np.ones((4, 4)), plow=98, phigh=2)


class TestPSNR:
    def test_identical(self):
        img = np.random.uniform(0, 1, (8, 8))
        assert mod.psnr(img, img) >= 99.0

    def test_shape_mismatch(self):
        with pytest.raises(mod.ValidationError):
            mod.psnr(np.zeros((4, 4)), np.zeros((5, 5)))


class TestRemoveClouds:
    def test_inpaint_improves_psnr(self):
        """云去除后 PSNR 应高于带云影像。"""
        cloudy, mask, truth, _ = mod.generate_synthetic([116, 39, 117, 40], seed=3)
        result, mask_det, info = mod.remove_clouds(cloudy.astype(np.float64),
                                                   percentile=85.0)
        assert info["cloud_fraction"] > 0.05
        psnr_before = mod.psnr(truth, cloudy)
        psnr_after = mod.psnr(truth, result)
        assert psnr_after > psnr_before

    def test_unchanged_pixels_preserved(self):
        cloudy, _, _, _ = mod.generate_synthetic([116, 39, 117, 40], seed=4)
        result, mask, _ = mod.remove_clouds(cloudy.astype(np.float64), percentile=85.0)
        np.testing.assert_array_equal(result[~mask], cloudy[~mask])


class TestSynthetic:
    def test_shapes_and_cloud_stats(self):
        cloudy, mask, truth, info = mod.generate_synthetic(
            [116, 39, 117, 40], cloud_fraction=0.2, seed=5)
        assert cloudy.shape == mask.shape == truth.shape == (64, 64)
        assert abs(info["cloud_fraction"] - 0.2) < 0.1
        # 云区被显著增亮
        assert cloudy[mask].mean() > truth[mask].mean() + 0.3


class TestGeoTiffIO:
    def test_roundtrip(self, tmp_path):
        arr = np.random.uniform(0, 1, (16, 16)).astype(np.float32)
        bbox = [116.0, 39.0, 117.0, 40.0]
        path = str(tmp_path / "x.tif")
        mod.write_geotiff(path, arr, bbox)
        back, rb = mod.read_geotiff(path)
        np.testing.assert_allclose(back[0], arr, atol=1e-5)
        np.testing.assert_allclose(rb, bbox, atol=1e-6)

    def test_read_missing_raises(self):
        with pytest.raises(mod.UsageError):
            mod.read_geotiff("/nonexistent/x.tif")
