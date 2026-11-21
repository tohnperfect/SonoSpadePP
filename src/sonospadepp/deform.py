"""deform.py: deformable US-to-CT refinement of the rigid registration (register.py).

The rigid transform aligns the kidney as a whole; probe pressure and patient position bend it.
Here a smooth displacement field u, on a control grid (spacing h mm) spanning the US volume and
interpolated trilinearly, is added after the rigid map:

    q(p) = T_rigid(p) + u(p)        p: US world point (mm), q: CT world point (mm)

and fitted so that the signed distance to the CT kidney surface at q equals the signed distance to
the US kidney surface at p, over a band around the US surface (both clipped at +-clip mm):

    loss = mean (sdf_CT(q(p)) - sdf_US(p))^2 + lam * mean |grad u|^2 + mu * mean |u|^2

Only the two kidney masks drive the fit. The dataset's landmarks are not used, so their residual
before and after is an independent check of whether the kidney's interior moved closer.
"""
from __future__ import annotations

import numpy as np


def sdf_mm(mask: np.ndarray, spacing) -> np.ndarray:
    """Signed distance (mm) to the mask surface: negative inside, positive outside."""
    from scipy.ndimage import distance_transform_edt
    m = mask.astype(bool)
    if not m.any():
        raise ValueError("empty mask")
    out = distance_transform_edt(~m, sampling=spacing)
    inside = distance_transform_edt(m, sampling=spacing)
    return (out - inside).astype(np.float32)


def _corners(shape, aff):
    idx = np.array([[i, j, k] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1) for k in (0, shape[2] - 1)], float)
    return idx @ aff[:3, :3].T + aff[:3, 3]


class DeformField:
    """Displacement field (mm) on a world-aligned control grid; u has shape (3, nz, ny, nx)."""

    def __init__(self, lo, hi, h=8.0, u=None):
        self.lo, self.hi, self.h = np.asarray(lo, float), np.asarray(hi, float), float(h)
        self.n = np.maximum(np.ceil((self.hi - self.lo) / h).astype(int) + 1, 2)      # (nx, ny, nz)
        self.hi = self.lo + (self.n - 1) * h
        self.u = np.zeros((3, self.n[2], self.n[1], self.n[0]), np.float32) if u is None else np.asarray(u, np.float32)

    @classmethod
    def covering(cls, shape, aff, h=8.0, margin=10.0):
        c = _corners(shape, aff)
        return cls(c.min(0) - margin, c.max(0) + margin, h)

    def _norm(self, pts):
        return 2.0 * (pts - self.lo) / (self.hi - self.lo) - 1.0

    def disp_torch(self, u_t, pts_t):
        """u_t (1,3,nz,ny,nx) torch, pts_t (N,3) torch world mm -> (N,3) displacement."""
        import torch
        import torch.nn.functional as F
        lo = torch.as_tensor(self.lo, dtype=pts_t.dtype, device=pts_t.device)
        hi = torch.as_tensor(self.hi, dtype=pts_t.dtype, device=pts_t.device)
        g = (2.0 * (pts_t - lo) / (hi - lo) - 1.0).view(1, -1, 1, 1, 3)
        return F.grid_sample(u_t, g, mode="bilinear", padding_mode="border", align_corners=True).view(3, -1).t()

    def disp(self, pts):
        """(N,3) world mm -> (N,3) displacement mm (numpy, CPU)."""
        import torch
        u_t = torch.as_tensor(self.u)[None]
        out = []
        for i in range(0, len(pts), 2_000_000):
            out.append(self.disp_torch(u_t, torch.as_tensor(np.asarray(pts[i:i + 2_000_000], np.float32))).numpy())
        return np.concatenate(out) if out else np.zeros((0, 3), np.float32)

    def jacobian_stats(self):
        """min det(I + grad u) and max |grad u| over the control grid (u per mm)."""
        u = self.u
        g = np.stack([np.stack(np.gradient(u[c], self.h, self.h, self.h), 0) for c in range(3)], 0)  # (3c,3axis z,y,x)
        J = np.zeros(u.shape[1:] + (3, 3), np.float32)
        ax = {0: 2, 1: 1, 2: 0}          # world x <- array axis 2, y <- 1, z <- 0
        for c in range(3):
            for d in range(3):
                J[..., c, d] = g[c, ax[d]] + (1.0 if c == d else 0.0)
        det = np.linalg.det(J.reshape(-1, 3, 3))
        return float(det.min()), float(np.abs(g).max())

    def state(self):
        return {"lo": self.lo.tolist(), "hi": self.hi.tolist(), "h": self.h, "u": self.u}


def fit_deform(us_mask, us_aff, ct_mask, ct_aff, T_ct_from_us, *, h=8.0, lam=2000.0, mu=1e-3, clip=15.0,
               band=12.0, stride=4, iters=300, lr=0.2, batch=60000, seed=0, device="cpu",
               l_us=None, l_ct=None, w_ldk=0.0):
    """Fit the field. With l_us/l_ct and w_ldk > 0, landmark pairs (mm) also pull: + w_ldk * mean |q(l_us) - l_ct|^2.
    Returns (DeformField, info dict with the loss before and after)."""
    import torch
    import torch.nn.functional as F
    rng = np.random.default_rng(seed)
    # US side: downsampled mask, its SDF, the band points (world mm)
    m = us_mask[::stride, ::stride, ::stride]
    aff_s = us_aff.copy(); aff_s[:3, :3] = us_aff[:3, :3] * stride
    sp = np.sqrt((aff_s[:3, :3] ** 2).sum(0))
    s_us = sdf_mm(m, sp)
    sel = np.argwhere(np.abs(s_us) < band)
    pts = (sel @ aff_s[:3, :3].T + aff_s[:3, 3]).astype(np.float32)
    tgt = np.clip(s_us[tuple(sel.T)], -clip, clip).astype(np.float32)
    # CT side: SDF of the kidney mask in a crop around it
    nz = np.argwhere(ct_mask)
    sp_ct = np.sqrt((ct_aff[:3, :3] ** 2).sum(0))
    pad = np.ceil(40.0 / sp_ct).astype(int)
    a0 = np.maximum(nz.min(0) - pad, 0); a1 = np.minimum(nz.max(0) + pad + 1, ct_mask.shape)
    crop = ct_mask[a0[0]:a1[0], a0[1]:a1[1], a0[2]:a1[2]]
    s_ct = np.clip(sdf_mm(crop, sp_ct), -clip, clip)
    inv_ct = np.linalg.inv(ct_aff)
    field = DeformField.covering(us_mask.shape, us_aff, h=h)

    dev = torch.device(device)
    T = torch.as_tensor(T_ct_from_us, dtype=torch.float32, device=dev)
    Ainv = torch.as_tensor(inv_ct, dtype=torch.float32, device=dev)
    a0_t = torch.as_tensor(a0, dtype=torch.float32, device=dev)
    shp = torch.as_tensor(np.array(crop.shape) - 1, dtype=torch.float32, device=dev)
    sdf_t = torch.as_tensor(s_ct, device=dev)[None, None]                 # (1,1,I,J,K)
    pts_t = torch.as_tensor(pts, device=dev); tgt_t = torch.as_tensor(tgt, device=dev)
    u = torch.zeros((1,) + field.u.shape, device=dev, requires_grad=True)
    opt = torch.optim.Adam([u], lr=lr)

    def ct_sdf_at(q):
        vox = q @ Ainv[:3, :3].t() + Ainv[:3, 3] - a0_t                    # (N,3) i,j,k in the crop
        g = (2.0 * vox / shp - 1.0)[:, [2, 1, 0]].view(1, -1, 1, 1, 3)    # grid order: k (W), j (H), i (D)
        return F.grid_sample(sdf_t, g, mode="bilinear", padding_mode="border", align_corners=True).view(-1)

    def data_loss(idx, uu):
        p = pts_t[idx]
        q = p @ T[:3, :3].t() + T[:3, 3] + field.disp_torch(uu, p)
        return ((ct_sdf_at(q) - tgt_t[idx]) ** 2).mean()

    allidx = torch.arange(len(pts), device=dev)
    with torch.no_grad():
        before = float(data_loss(allidx, torch.zeros_like(u)))
    use_l = l_us is not None and w_ldk > 0 and len(l_us)
    if use_l:
        lu = torch.as_tensor(np.asarray(l_us, np.float32), device=dev); lc = torch.as_tensor(np.asarray(l_ct, np.float32), device=dev)
    for it in range(iters):
        idx = torch.as_tensor(rng.choice(len(pts), size=min(batch, len(pts)), replace=False), device=dev)
        reg = sum(((u.diff(dim=d) / field.h) ** 2).mean() for d in (2, 3, 4))
        loss = data_loss(idx, u) + lam * reg + mu * (u ** 2).mean()
        if use_l:
            ql = lu @ T[:3, :3].t() + T[:3, 3] + field.disp_torch(u, lu)
            loss = loss + w_ldk * ((ql - lc) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        after = float(data_loss(allidx, u))
    field.u = u.detach().cpu().numpy()[0]
    jmin, gmax = field.jacobian_stats()
    return field, {"sdf_mse_before": before, "sdf_mse_after": after, "jac_min": jmin, "grad_max": gmax,
                   "n_points": int(len(pts))}


def map_points(field: DeformField | None, T_ct_from_us, pts_us_mm):
    """US world mm -> CT world mm, rigid then (if given) the displacement."""
    q = pts_us_mm @ T_ct_from_us[:3, :3].T + T_ct_from_us[:3, 3]
    if field is not None:
        q = q + field.disp(pts_us_mm)
    return q


def dice_mapped(us_mask, us_aff, ct_mask, ct_aff, T_ct_from_us, field=None, step=3):
    """Kidney Dice on a strided US grid: US mask vs the CT mask pulled back through the mapping."""
    from scipy.ndimage import map_coordinates
    idx = np.stack(np.meshgrid(*[np.arange(0, s, step) for s in us_mask.shape], indexing="ij"), -1).reshape(-1, 3)
    m_us = us_mask[tuple(idx.T)]
    pts = idx @ us_aff[:3, :3].T + us_aff[:3, 3]
    q = map_points(field, T_ct_from_us, pts.astype(np.float32))
    vox = q @ np.linalg.inv(ct_aff)[:3, :3].T + np.linalg.inv(ct_aff)[:3, 3]
    m_ct = map_coordinates(ct_mask.astype(np.uint8), vox.T, order=0, mode="constant") > 0
    return float(2 * (m_us & m_ct).sum() / (m_us.sum() + m_ct.sum() + 1e-9))


__all__ = ["sdf_mm", "DeformField", "fit_deform", "map_points", "dice_mapped"]
