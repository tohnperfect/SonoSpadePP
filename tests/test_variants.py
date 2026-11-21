"""The M3 trainer reads its tissue targets from the one-hot block, never from the HU channel, and a
step runs end to end on CPU for every variant."""
import numpy as np
import pytest
import torch
from usbg import segmenter as SEG
from usbg import texture_gan as TG

from sonospadepp.labels import N_GROUPED, grouped_map
from sonospadepp.variants import HUSonoSPADE, Translator, cond_tensor


def _s_us(n_lab):
    return SEG.Segmenter(device="cpu", n_classes=N_GROUPED, base=8, depth=2, group_map=grouped_map(n_lab))


@pytest.mark.parametrize("n_lab,use_hu", [(13, False), (14, False), (13, True)])
def test_one_step_and_roundtrip(tmp_path, n_lab, use_hu):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    kw = dict(s_us=_s_us(n_lab), ngf=8, ndf=8, n_blocks=1, spade_hidden=8, d_depth=2,
              lambda_tc=1.0, lambda_lm=1.0, r1_gamma=1.0)
    gan = HUSonoSPADE("cpu", n_lab=n_lab, **kw) if use_hu else TG.SonoSPADE("cpu", n_classes=n_lab, **kw)
    lab = rng.integers(0, n_lab, (2, 32, 32))
    z = rng.normal(size=(2, 32, 32)).astype(np.float32) * 5
    phys = torch.rand(2, 1, 32, 32) * 2 - 1
    real = torch.rand(2, 1, 32, 32) * 2 - 1
    out = gan.step(phys, cond_tensor(lab, z, n_lab, use_hu, "cpu"), real)
    assert np.isfinite(out["g"]) and np.isfinite(out["d"])
    path = tmp_path / "g.pt"
    gan.save(str(path))
    tr = Translator(str(path), device="cpu")
    assert tr.use_hu == use_hu and tr.n_lab == n_lab
    img = tr(np.random.rand(32, 32).astype(np.float32), lab[0], z[0] if use_hu else None)
    assert img.shape == (32, 32) and 0 <= img.min() and img.max() <= 1


def test_hu_channel_does_not_change_targets():
    gan = HUSonoSPADE("cpu", n_lab=13, s_us=_s_us(13), ngf=8, ndf=8, n_blocks=1, spade_hidden=8, d_depth=2)
    lab = np.full((1, 16, 16), 1)
    big = cond_tensor(lab, np.full((1, 16, 16), 100.0, np.float32), 13, True, "cpu")  # channel ~ +1
    assert torch.equal(big[:, :13].argmax(1), torch.ones(1, 16, 16, dtype=torch.long))
    assert big.shape[1] == 14 and big[:, 13].max() > 0.99
