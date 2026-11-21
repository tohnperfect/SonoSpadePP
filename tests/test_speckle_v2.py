"""A5 pieces: the torch speckle statistics agree with the evaluation metrics, the loss is differentiable, and
a noise-injecting generator starts as an exact copy of v1."""
import numpy as np
import torch

from sonospadepp.v2 import metrics as M
from sonospadepp.v2 import speckle as S
from sonospadepp.v2.generator import build_generator


def _rayleigh_frame(seed=0, hw=96):
    A = np.random.default_rng(seed).rayleigh(1.0, (hw, hw))
    return np.clip(1 + 20 * np.log10(A / A.max()) / 60.0, 0, 1)


def test_torch_window_stats_match_metrics():
    x = _rayleigh_frame()
    mask = np.zeros_like(x, bool)
    mask[10:80, 12:90] = True
    m_np, lo_np = M.nakagami_windows(x, mask, win=9, stride=4, erode=2)
    lm_t, lo_t = S.window_stats(torch.tensor(x, dtype=torch.float64)[None, None],
                                torch.tensor(mask, dtype=torch.float64)[None, None], win=9, stride=4, erode=2)
    # same windows (avg_pool2d grid starts at 0; the numpy grid is centred at win//2: identical positions)
    np.testing.assert_allclose(np.sort(np.exp(lm_t.numpy())), np.sort(m_np), rtol=1e-6)
    np.testing.assert_allclose(np.sort(lo_t.numpy()), np.sort(lo_np), rtol=1e-6)


def test_speckle_loss_is_differentiable_and_zero_against_itself():
    torch.manual_seed(0)
    x = torch.tensor(_rayleigh_frame(1), dtype=torch.float32)[None, None]
    seg = torch.ones(1, 96, 96, dtype=torch.long)
    bank = S.SpeckleBank()
    lm, lo = S.window_stats(x, torch.ones_like(x))
    prof, pairs = S.acov_profiles(x, torch.ones_like(x))
    bank.update(1, lm, lo, prof, pairs)
    loss, n = S.speckle_loss(x, seg, bank, n_classes=2)
    assert n == 1 and float(loss) < 1e-4
    y = (x * 0.5 + 0.4).clone().requires_grad_(True)            # smoother, brighter: a worse match
    loss2, _ = S.speckle_loss(y, seg, bank, n_classes=2)
    loss2.backward()
    assert float(loss2) > 0.05 and torch.isfinite(y.grad).all() and y.grad.abs().sum() > 0


def test_noise_generator_starts_as_v1():
    from usbg import texture_gan as TG
    torch.manual_seed(0)
    v1 = TG._spade_generator(13, ngf=8, n_blocks=1, spade_hidden=8)
    v2 = build_generator(13, ngf=8, n_blocks=1, spade_hidden=8, noise_inject=True)
    missing, unexpected = v2.load_state_dict(v1.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("noise_scale") for k in missing)
    phys = torch.rand(2, 1, 32, 32) * 2 - 1
    seg = TG.onehot_labels(np.random.default_rng(0).integers(0, 13, (2, 32, 32)), 13)
    with torch.no_grad():
        assert torch.equal(v1(phys, seg), v2(phys, seg))
        f1, f2 = v1.forward_features(phys, seg), v2.forward_features(phys, seg)
    assert all(torch.equal(a, b) for a, b in zip(f1, f2))
    with torch.no_grad():
        v2.noise_scale[1].fill_(0.5)
        assert not torch.equal(v1(phys, seg), v2(phys, seg))
