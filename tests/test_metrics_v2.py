"""v2 metrics on synthetic inputs where the answer is known (WP0 item 3)."""
import numpy as np
import pytest
from scipy import ndimage

from sonospadepp.v2 import metrics as M

DR = 60.0


def log_compress(A, dr=DR):
    """Inverse of metrics.envelope: x = 1 + 20 log10(A / max A) / DR, clipped to [0, 1]."""
    return np.clip(1.0 + 20.0 * np.log10(A / A.max()) / dr, 0.0, 1.0)


def test_envelope_inverts_log_compression():
    A = np.random.default_rng(0).uniform(0.01, 1.0, (32, 32))
    np.testing.assert_allclose(M.envelope(log_compress(A)), A / A.max(), rtol=1e-9)


def test_rayleigh_speckle_gives_nakagami_m_near_one():
    rng = np.random.default_rng(0)
    A = rng.rayleigh(1.0, (256, 256))
    m, log_om = M.nakagami_windows(log_compress(A), np.ones_like(A, bool), win=15, stride=8)
    assert len(m) > 500
    assert abs(np.median(m) - 1.0) < 0.12 and abs(np.mean(m) - 1.0) < 0.15


def test_nakagami_m_two_is_recovered():
    rng = np.random.default_rng(1)
    I = rng.gamma(shape=2.0, scale=0.5, size=(256, 256))          # A^2 ~ Gamma(m=2, Omega/m), Omega = 1
    m, _ = M.nakagami_windows(log_compress(np.sqrt(I)), np.ones_like(I, bool), win=15, stride=8)
    assert abs(np.median(m) - 2.0) < 0.35


def test_pre_rayleigh_below_one_and_mask_respected():
    rng = np.random.default_rng(2)
    I = rng.gamma(shape=0.6, scale=1 / 0.6, size=(128, 128))
    x = log_compress(np.sqrt(I))
    mask = np.zeros_like(x, bool)
    mask[20:100, 20:100] = True
    m, _ = M.nakagami_windows(x, mask, win=9, stride=4, erode=2)
    assert np.median(m) < 0.9
    assert len(M.nakagami_windows(x, np.zeros_like(mask), win=9)[0]) == 0


def _correlated_field(sig_ax, sig_lat, shape=(256, 256), seed=0):
    g = ndimage.gaussian_filter(np.random.default_rng(seed).normal(size=shape), (sig_ax, sig_lat))
    return 0.5 + 0.02 * g / g.std()                    # small amplitude: the envelope map is ~linear


def test_speckle_size_recovers_gaussian_correlation_length():
    x = _correlated_field(2.0, 0.0)
    acc, cnt = M.speckle_acov_sum(x, np.ones_like(x, bool), patch=16, stride=8, depth_bins=1, erode=0)
    ax, lat = M.fwhm_from_acov(acc[0])
    expected_ax = 2.0 * np.sqrt(2) * 2 * np.sqrt(2 * np.log(2))      # Gaussian acov: sigma * sqrt(2)
    assert cnt[0] > 100
    assert abs(ax - expected_ax) / expected_ax < 0.15
    assert lat < 1.5                                    # white along the lateral axis


def test_speckle_size_depth_bins_and_orientation():
    x = _correlated_field(0.0, 2.5)
    acc, cnt = M.speckle_acov_sum(x, np.ones_like(x, bool), patch=16, stride=8, depth_bins=3, erode=0)
    assert (cnt > 0).all()
    for b in range(3):
        ax, lat = M.fwhm_from_acov(acc[b])
        assert lat > 2 * ax


def test_w1_1d_shift_and_identity():
    a = np.random.default_rng(0).normal(size=5000)
    assert M.w1_1d(a, a) < 1e-12
    assert abs(M.w1_1d(a, a + 0.7) - 0.7) < 1e-6
    assert np.isnan(M.w1_1d(a, []))


def test_kid_and_fd_zero_for_same_distribution_positive_for_shift():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(400, 16)), rng.normal(size=(400, 16))
    c = rng.normal(size=(400, 16)) + 1.0
    k_same, _ = M.kid(a, b)
    k_diff, _ = M.kid(a, c)
    assert abs(k_same) < 0.02 and k_diff > 0.2
    assert M.fd(a, b) < 1.0 and M.fd(a, c) > 10.0


def test_tissue_crops_lie_in_mask():
    img = np.arange(128 * 128, dtype=np.float32).reshape(128, 128) / (128 * 128)
    mask = np.zeros((128, 128), bool)
    mask[40:90, 30:100] = True
    cr = M.tissue_crops(img, mask, size=32, max_crops=8, min_cover=0.9, rng=np.random.default_rng(0))
    assert cr.shape == (8, 32, 32)
    assert len(M.tissue_crops(img, np.zeros_like(mask), size=32)) == 0


def _local_translator(img, lab):
    out = np.asarray(img, np.float32).copy()
    out += 0.05 * np.asarray(lab, np.float32)            # strictly per-pixel: label changes act locally
    return out


def _blurry_translator(img, lab):
    return np.asarray(img, np.float32) + 0.05 * ndimage.uniform_filter(np.asarray(lab, np.float32), 5)


def _global_translator(img, lab):
    return np.asarray(img, np.float32) + 0.05 * float(np.mean(lab))


def _toy():
    lab = np.full((64, 64), 12, np.int64)
    lab[10:40, 10:40] = 4
    lab[45:60, 5:60] = 7
    return np.random.default_rng(0).uniform(0, 1, (64, 64)).astype(np.float32), lab


def test_locality_band0_equals_v1_definition():
    from usbg import texture_eval as TE
    img, lab = _toy()
    for f in (_local_translator, _blurry_translator, _global_translator):
        v1 = TE.label_swap_sensitivity(f, img, lab, min_px=100)
        v2 = M.label_swap_locality(f, img, lab, min_px=100, band=0)
        assert v1.keys() == v2.keys()
        for c in v1:
            assert v1[c]["locality"] == pytest.approx(v2[c]["locality"], rel=1e-6)


def test_locality_band_separates_local_from_global():
    img, lab = _toy()
    loc = M.label_swap_locality(_blurry_translator, img, lab, min_px=100, band=8)
    glob = M.label_swap_locality(_global_translator, img, lab, min_px=100, band=8)
    assert min(v["locality"] for v in loc.values()) > 100       # outside the band: no change at all
    assert max(v["locality"] for v in glob.values()) < 1.5


def test_style_swap_locality_on_toy_styled_translator():
    img, lab = _toy()

    def styled(im, lb, style):
        out = np.asarray(im, np.float32).copy()
        for t, s in style.items():
            out[lb == t] += s
        return out

    r = M.style_swap_locality(styled, img, lab, {4: 0.0, 7: 0.0}, {4: 0.3, 7: 0.0}, tissue=4, min_px=100)
    assert r["locality"] > 1e3


def test_bootstrap_ci_brackets_mean():
    x = np.random.default_rng(0).normal(2.0, 1.0, 400)
    lo, hi = M.bootstrap_ci(lambda i: x[i].mean(), len(x), n_boot=500)
    assert lo < 2.0 < hi and hi - lo < 0.3
