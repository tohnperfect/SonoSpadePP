"""A5 (WP5): per-tissue speckle statistics as a differentiable training loss.

Same quantities as the evaluation metrics (sonospadepp.v2.metrics), in torch:
  * envelope A = 10 ** ((x - 1) * DR / 20) from the log-compressed frame in [0, 1];
  * window-level Nakagami log m and log Omega (masked moments of I = A^2 on win x win windows, stride s, inside
    the tissue mask eroded by `erode` px, full coverage only);
  * axial and lateral autocovariance profiles of the high-passed envelope at lags 1..L, per depth bin (thirds of
    the frame). The FWHM used as the evaluation metric is a threshold crossing of this profile and is not
    smooth, so the loss matches the profile itself (an L1 term), which fixes the FWHM.
The loss per tissue: 1D W1 between fake and real-bank samples of log m and of log Omega, plus the L1 profile
term. Real statistics come from the real pool masked by S_US pseudo-labels, kept in a bank (a reservoir of
recent window values and an EMA of the profiles), as for the feature matching of WP1.
"""
from __future__ import annotations


def envelope_t(x01, dr_db):
    import torch
    return torch.pow(torch.tensor(10.0, device=x01.device), (x01 - 1.0) * (dr_db / 20.0))


def _erode(mask, it):
    import torch.nn.functional as F
    if it <= 0:
        return mask
    return 1.0 - F.max_pool2d(1.0 - mask, 2 * it + 1, stride=1, padding=it)


def window_stats(x01, mask, win=7, stride=3, dr_db=60.0, erode=1):
    """x01, mask: (B, 1, H, W). Returns (log_m, log_omega) 1D tensors over fully-covered windows."""
    import torch
    import torch.nn.functional as F
    A = envelope_t(x01.clamp(0, 1), dr_db)
    inten = A * A
    m = _erode(mask, erode)
    sm = F.avg_pool2d(m, win, stride)
    valid = sm >= 0.999
    if not bool(valid.any()):
        e = x01.new_zeros(0)
        return e, e
    si = F.avg_pool2d(inten * m, win, stride) / sm.clamp(min=1e-6)
    si2 = F.avg_pool2d(inten * inten * m, win, stride) / sm.clamp(min=1e-6)
    var = (si2 - si * si).clamp(min=1e-12)
    mu = si.clamp(min=1e-12)
    return (2 * torch.log(mu) - torch.log(var))[valid], torch.log(mu)[valid]


def acov_profiles(x01, mask, lags=4, depth_bins=3, dr_db=60.0, erode=1, hp=9):
    """Normalized autocovariance of the high-passed envelope at lags 1..L, axial (rows) and lateral (columns),
    per depth bin: returns (prof (depth_bins, 2, L), pairs (depth_bins,)) summed over the batch."""
    import torch
    import torch.nn.functional as F
    A = envelope_t(x01.clamp(0, 1), dr_db)
    a = A - F.avg_pool2d(A, hp, 1, hp // 2, count_include_pad=False)
    m = _erode(mask, erode)
    H = x01.shape[-2]
    rows = torch.arange(H, device=x01.device).view(1, 1, H, 1).float()
    out = x01.new_zeros(depth_bins, 2, lags)
    n0 = x01.new_zeros(depth_bins)
    for b in range(depth_bins):
        band = ((rows >= b * H / depth_bins) & (rows < (b + 1) * H / depth_bins)).float()
        mb = m * band
        c0 = (a * a * mb).sum()
        n = mb.sum()
        n0[b] = n
        if n < 16:
            continue
        var = c0 / n
        for d in range(1, lags + 1):
            for ax in (0, 1):
                if ax == 0:
                    pa, pb, ma, mc = a[..., :-d, :], a[..., d:, :], mb[..., :-d, :], m[..., d:, :]
                else:
                    pa, pb, ma, mc = a[..., :, :-d], a[..., :, d:], mb[..., :, :-d], m[..., :, d:]
                w = ma * mc
                out[b, ax, d - 1] = (pa * pb * w).sum() / w.sum().clamp(min=1.0) / var.clamp(min=1e-12)
    return out, n0


def w1_to_reference(x, ref, n_q=64):
    """Differentiable 1D W1 between sample x (grad) and a fixed reference sample, by matched quantiles."""
    import torch
    q = (torch.arange(n_q, device=x.device, dtype=x.dtype) + 0.5) / n_q
    return (torch.quantile(x, q) - torch.quantile(ref.to(x.dtype), q)).abs().mean()


class SpeckleBank:
    """Real statistics per tissue: reservoirs of recent window log m / log Omega values and an EMA of the
    autocovariance profiles."""

    def __init__(self, size=4096, momentum=0.99):
        self.size, self.m = size, momentum
        self.logm, self.logom, self.prof, self.prof_n = {}, {}, {}, {}

    def _push(self, store, k, v):
        import torch
        v = v.detach()
        cur = store.get(k)
        store[k] = (v if cur is None else torch.cat([cur, v]))[-self.size:]

    def update(self, k, logm, logom, prof, pairs):
        if logm.numel():
            self._push(self.logm, k, logm)
            self._push(self.logom, k, logom)
        ok = pairs >= 16
        if bool(ok.any()):
            p = prof.detach()
            if k not in self.prof:
                self.prof[k], self.prof_n[k] = p.clone(), ok.clone()
            else:
                upd = ok[:, None, None]
                self.prof[k] = self.prof[k].where(~upd, self.prof[k] * self.m + p * (1 - self.m))
                first = ok & ~self.prof_n[k]
                self.prof[k] = self.prof[k].where(~first[:, None, None], p)
                self.prof_n[k] = self.prof_n[k] | ok


def speckle_loss(fake01, seg_t, bank, n_classes, fan=None, min_windows=8, w_prof=1.0, **kw):
    """Mean over tissues present (with enough windows and a bank entry) of
    W1(log m) + W1(log Omega) + w_prof * L1(profiles). seg_t: (B, H, W) labels in S_US's scheme."""
    total, terms = 0.0, 0
    for k in range(1, n_classes):
        if k not in bank.logm or bank.logm[k].numel() < 64:
            continue
        mk = (seg_t == k).float()[:, None]
        if fan is not None:
            mk = mk * fan
        if mk.sum() < 64:
            continue
        lm, lo = window_stats(fake01, mk, **{a: kw[a] for a in ("win", "stride", "dr_db", "erode") if a in kw})
        if lm.numel() < min_windows:
            continue
        loss = w1_to_reference(lm, bank.logm[k]) + w1_to_reference(lo, bank.logom[k])
        if k in bank.prof and w_prof > 0:
            prof, pairs = acov_profiles(fake01, mk, dr_db=kw.get("dr_db", 60.0))
            ok = (pairs >= 16) & bank.prof_n[k]
            if bool(ok.any()):
                loss = loss + w_prof * (prof[ok] - bank.prof[k][ok]).abs().mean()
        total, terms = total + loss, terms + 1
    return (total / terms if terms else None), terms
