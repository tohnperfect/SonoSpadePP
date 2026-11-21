"""SonoSPADEv2: v1's SonoSPADE trainer (usbg.texture_gan.SonoSPADE, unchanged) plus the v2 additions behind
flags. With every v2 flag off, step() performs exactly v1's computation in v1's order (tests check bitwise
equality on CPU), so v2 results differ from v1 only by what the flags add.

v2 additions (config dict `v2`, see DEFAULT_V2):
  A1 (WP1)  fm_disc, lambda_fm_adv, lambda_fm_tissue: a frozen foundation-model encoder with trainable heads
            (D_fm, non-saturating loss, DiffAugment on its inputs) and per-tissue masked feature matching
            against an EMA bank of real statistics. The OASIS discriminator is kept exactly as in v1.
Both real and fake frames are multiplied by the real pool's fan mask before D_fm and the feature matching,
so the frame outline (linear render vs curvilinear sector) cannot be what D_fm learns.
"""
from __future__ import annotations

from usbg import texture_gan as TG

DEFAULT_V2 = {
    "fm_disc": "none",              # none | dinov2 | usfm | echocare
    "lambda_fm_adv": 0.0,
    "lambda_fm_tissue": 0.0,
    "fm_layers": None,              # encoder block indices; None = DEFAULT_LAYERS[fm_disc]
    "fm_bank_momentum": 0.99,
    "fm_min_cells": 4,
    "fm_min_updates": 10,
    "diffaug": "color,translation,cutout",
    "use_fan": True,
    # A2 (WP2)
    "style": "none",                # none | sean | inade (INADE: the WP2 fallback)
    "style_dim": 16,
    "lambda_style_rec": 1.0,        # latent regression: E_s(fake) must recover the code used
    "lambda_style_kl": 0.01,        # exemplar codes stay near the prior N(0, I)
    "style_prior_frac": 0.5,        # share of fakes whose codes are drawn from the prior
    "lambda_ms": 1.0,               # INADE: mode-seeking term (MSGAN), keeps the noise codes in use
    "inade_sigma0": 0.05,           # INADE: initial per-class modulation scale
    # A5 (WP5)
    "lambda_speckle": 0.0,          # > 0 switches the per-tissue speckle-statistics loss on
    "noise_inject": False,
    "gain_sigma": 0.0,              # > 0: gain-field generator (keeps the render's fine structure)
    "dyn_range_db": 60.0,
    "speckle_w_prof": 1.0,
}
DEFAULT_LAYERS = {"dinov2": [2, 5, 8, 11], "usfm": [2, 5, 8, 11], "echocare": [1, 2, 3, 4]}


class SonoSPADEv2(TG.SonoSPADE):
    def __init__(self, device, v2=None, fan=None, **kw):
        import torch
        super().__init__(device, **kw)
        self.v2 = dict(DEFAULT_V2, **(v2 or {}))
        c = self.v2
        lr, b1, b2 = kw.get("lr", 2e-4), kw.get("beta1", 0.5), kw.get("beta2", 0.999)
        self.fan = None
        if fan is not None and c["use_fan"]:
            self.fan = torch.as_tensor(fan, dtype=torch.float32, device=device)[None, None]
        self.fm = self.D_fm = self.bank = None
        use_adv = c["lambda_fm_adv"] > 0
        use_tis = c["lambda_fm_tissue"] > 0
        if c["fm_disc"] != "none" and (use_adv or use_tis):
            from .features import FMEncoder
            from .fm_disc import FMHeads, TissueBank
            self.fm = FMEncoder(c["fm_disc"], device)
            self.fm_layers = list(c["fm_layers"] or DEFAULT_LAYERS[c["fm_disc"]])
            if use_adv:
                self.D_fm = FMHeads(self.fm.layer_dims(self.fm_layers)).to(device)
                self.opt_dfm = torch.optim.Adam(self.D_fm.parameters(), lr=lr, betas=(b1, b2))
            if use_tis:
                self.bank = TissueBank(c["fm_bank_momentum"])
        # A2 / A5: a v2 generator (v1 modules and names, additions start with no effect)
        self.E_s = None
        if c["noise_inject"] or c["style"] != "none" or c["gain_sigma"] > 0:
            from .generator import build_generator
            from .style import StyleEncoder, convert_style

            def make():
                g = build_generator(self.n_classes, ngf=self._ngf, n_blocks=self._n_blocks,
                                    spade_hidden=self._spade_hidden, noise_inject=c["noise_inject"],
                                    gain_sigma=c["gain_sigma"])
                convert_style(g, c["style"], c["style_dim"], self.n_classes, c["inade_sigma0"])
                return g.to(device)
            g_new = make()
            g_new.load_state_dict(self.G.state_dict(), strict=False)      # v1's initialization carries over
            self.G = g_new
            if self.G_ema is not None:
                self.G_ema = make()
                self.G_ema.load_state_dict(self.G.state_dict())
                for p in self.G_ema.parameters():
                    p.requires_grad_(False)
            self.opt_g = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(b1, b2))
            if c["style"] == "sean":
                self.E_s = StyleEncoder(self.n_seg, c["style_dim"]).to(device)
                self.opt_es = torch.optim.Adam(self.E_s.parameters(), lr=lr, betas=(b1, b2))
                self.style_ema = torch.zeros(self.n_seg, c["style_dim"], device=device)
        self.sbank = None
        if c["lambda_speckle"] > 0:
            from .speckle import SpeckleBank
            self.sbank = SpeckleBank()

    # ---------------------------------------------------------------------------------------------
    def _fm_in(self, x, aug: bool):
        from .diffaug import diff_augment
        x01 = self._to01(x)
        if self.fan is not None:
            x01 = x01 * self.fan
        return diff_augment(x01, self.v2["diffaug"]) if aug else x01

    def _v2_d_step(self, real, fake, pseudo):
        """D_fm update and the real-statistics bank. Runs after v1's D step, before the G step."""
        import torch
        from .fm_disc import masked_stats, ns_d_loss, token_weights
        out = {}
        if self.D_fm is not None:
            with torch.no_grad():
                fr = self.fm.intermediates(self._fm_in(real, aug=True), self.fm_layers)
                ff = self.fm.intermediates(self._fm_in(fake.detach(), aug=True), self.fm_layers)
            self.opt_dfm.zero_grad()
            loss = ns_d_loss(self.D_fm(fr), self.D_fm(ff))
            loss.backward()
            self.opt_dfm.step()
            out["dfm"] = float(loss.detach())
        if self.sbank is not None:
            from .speckle import acov_profiles, window_stats
            with torch.no_grad():
                r01 = self._to01(real)
                for k in range(1, self.n_seg):
                    mk = (pseudo == k).float()[:, None]
                    if self.fan is not None:
                        mk = mk * self.fan
                    if mk.sum() < 64:
                        continue
                    lm, lo = window_stats(r01, mk, dr_db=self.v2["dyn_range_db"])
                    prof, pairs = acov_profiles(r01, mk, dr_db=self.v2["dyn_range_db"])
                    self.sbank.update(k, lm, lo, prof, pairs)
        if self.fm is None:
            return out
        if self.bank is not None:
            with torch.no_grad():
                feats = self.fm.intermediates(self._fm_in(real, aug=False), self.fm_layers)
                for li, f in enumerate(feats):
                    w = token_weights(pseudo, self.n_seg, f.shape[-2:], fan=self.fan)
                    for k in range(1, self.n_seg):
                        mu, var, n = masked_stats(f, w[:, k])
                        if mu is not None and n >= self.v2["fm_min_cells"]:
                            self.bank.update((li, k), mu, var)
        return out

    def _v2_g_terms(self, fake, target_lab):
        """Extra generator terms {name: (weight, tensor)}; empty when every v2 flag is off."""
        from .fm_disc import fm_tissue_loss, ns_g_loss, token_weights
        terms = {}
        if self.sbank is not None:
            from .speckle import speckle_loss
            loss, n = speckle_loss(self._to01(fake), self._seg_target(target_lab), self.sbank, self.n_seg,
                                   fan=self.fan, dr_db=self.v2["dyn_range_db"], w_prof=self.v2["speckle_w_prof"])
            if loss is not None:
                terms["speckle"] = (self.v2["lambda_speckle"], loss)
        if self.fm is None:
            return terms
        if self.D_fm is not None:
            ff = self.fm.intermediates(self._fm_in(fake, aug=True), self.fm_layers)
            terms["fm_adv"] = (self.v2["lambda_fm_adv"], ns_g_loss(self.D_fm(ff)))
        if self.bank is not None:
            fu = self.fm.intermediates(self._fm_in(fake, aug=False), self.fm_layers)
            seg_t = self._seg_target(target_lab)
            w = [token_weights(seg_t, self.n_seg, f.shape[-2:], fan=self.fan) for f in fu]
            loss, n = fm_tissue_loss(fu, w, self.bank, self.n_seg, min_cells=self.v2["fm_min_cells"],
                                     min_updates=self.v2["fm_min_updates"])
            if loss is not None:
                terms["fm_tissue"] = (self.v2["lambda_fm_tissue"], loss)
        return terms

    # ---------------------------------------------------------------------------------------------
    def generator_losses(self, phys, label_onehot, fake):
        """v1's generator_losses, except that the conditioning may carry style channels after the K label
        channels: the targets come from the label channels only. Identical to v1 for a K-channel input."""
        if label_onehot.shape[1] == self.n_classes:
            return super().generator_losses(phys, label_onehot, fake)
        from usbg.segmenter import seg_ce_dice_loss
        seg_target = self._seg_target(label_onehot[:, :self.n_classes].argmax(dim=1))
        loss_g_adv = self._balanced_ce(self.D(fake), seg_target)
        fk = self.G.forward_features(fake, label_onehot)
        fq = self.G.forward_features(phys, label_onehot)
        q, ids = self.H(fk); k, _ = self.H(fq, patch_ids=ids)
        loss_nce = TG._patchnce_loss(q, k, self.tau)
        seg_logits = self.S.net(self._to01(fake))
        loss_seg = seg_ce_dice_loss(seg_logits, seg_target, n_classes=self.n_seg, dice_weight=self.lambda_seg_dice)
        out = {"adv": loss_g_adv, "nce": loss_nce, "seg": loss_seg}
        if self.lambda_tc and self.lambda_tc > 0:
            shifted = self.G(self._shift(phys, 2, 2), self._shift(label_onehot, 2, 2))
            out["tc"] = (shifted - self._shift(fake, 2, 2)).abs().mean()
        return out

    def _style_codes(self, real, pseudo, target_lab):
        """Codes for this batch's fakes: exemplar codes from the real frames (reparameterized; tissues absent
        from the exemplar use the running bank mean) or prior draws. Returns codes, mu_r, logvar_r, present_r."""
        import torch
        c = self.v2
        mu, lv, pres = self.E_s(self._to01(real), pseudo)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        fallback = self.style_ema[None].expand_as(z)
        z = torch.where(pres[..., None], z, fallback)
        prior = torch.randn_like(z)
        use_prior = (torch.rand(z.shape[0], 1, 1, device=z.device) < c["style_prior_frac"])
        codes = torch.where(use_prior, prior, z)
        with torch.no_grad():
            for k in range(self.n_seg):
                sel = mu[pres[:, k], k]
                if len(sel):
                    self.style_ema[k].mul_(0.99).add_(sel.mean(0), alpha=0.01)
        return codes, mu, lv, pres

    def _ms_term(self, phys, label_onehot, seg_t, fake, smap):
        """INADE's mode-seeking term (Mao et al., 2019): a second fake from new codes for the same content;
        L = 1 / (mean |fake - fake2| / mean |S - S2| + eps), so the output must move when the codes move."""
        import torch
        from .style import style_map
        codes2 = torch.randn(fake.shape[0], self.n_seg, self.v2["style_dim"], device=fake.device)
        smap2 = style_map(codes2, seg_t, self.n_seg)
        fake2 = self.G(phys, torch.cat([label_onehot, smap2], 1))
        lz = (fake - fake2).abs().mean() / (smap - smap2).abs().mean().clamp(min=1e-6)
        return 1.0 / (lz + 1e-5), lz

    def _step_style(self, phys, label_onehot, real):
        """v1's step with style codes (A2): the generator's conditioning is [label one-hot; style map]; G and
        E_s also minimize latent regression and the KL term. D, OASIS and every v1 term are unchanged."""
        import torch
        from .style import style_map
        c = self.v2
        target_lab = label_onehot.argmax(dim=1)
        seg_t = self._seg_target(target_lab)
        with torch.no_grad():
            pseudo = self.S.net(self._to01(real)).argmax(dim=1)
        inade = c["style"] == "inade"
        if inade:                                    # no encoder: every tissue instance draws its code from N(0, I)
            codes = torch.randn(phys.shape[0], self.n_seg, c["style_dim"], device=phys.device)
        else:
            codes, mu_r, lv_r, pres_r = self._style_codes(real, pseudo, target_lab)
        smap = style_map(codes, seg_t, self.n_seg)
        cond = torch.cat([label_onehot, smap], 1)
        fake = self.G(phys, cond)
        B, _, H, W = fake.shape

        self.opt_d.zero_grad()
        use_r1 = self.r1_gamma and self.r1_gamma > 0
        real_in = real.detach().requires_grad_(True) if use_r1 else real
        d_real = self.D(real_in)
        loss_d = self._balanced_ce(d_real, pseudo) + self._balanced_ce(self.D(fake.detach()),
                                                                      torch.full_like(target_lab, self.n_seg))
        extra = {}
        if use_r1:
            grad = torch.autograd.grad(d_real.sum(), real_in, create_graph=True)[0]
            loss_r1 = grad.pow(2).flatten(1).sum(1).mean() / float(H * W)
            loss_d = loss_d + 0.5 * self.r1_gamma * loss_r1
            extra["r1"] = float(loss_r1.detach())
        if self.lambda_lm and self.lambda_lm > 0:
            d_fake = self.D(fake.detach())
            m = self._labelmix_mask(B, H, W)
            mixed = m * real.detach() + (1.0 - m) * fake.detach()
            target_mix = m * d_real.detach() + (1.0 - m) * d_fake.detach()
            loss_lm = (self.D(mixed) - target_mix).pow(2).mean()
            loss_d = loss_d + self.lambda_lm * loss_lm
            extra["lm"] = float(loss_lm.detach())
        loss_d.backward(); self.opt_d.step()
        extra.update(self._v2_d_step(real, fake, pseudo))

        self.opt_g.zero_grad(); self.opt_h.zero_grad()
        if not inade:
            self.opt_es.zero_grad()
        g = self.generator_losses(phys, cond, fake)
        loss_g = g["adv"] + self.lambda_nce * g["nce"] + self.lambda_seg * g["seg"]
        if "tc" in g:
            loss_g = loss_g + self.lambda_tc * g["tc"]
        for name, (wt, val) in self._v2_g_terms(fake, target_lab).items():
            loss_g = loss_g + wt * val
            extra[name] = float(val.detach())
        if inade:
            if c["lambda_ms"] > 0:
                l_ms, lz = self._ms_term(phys, label_onehot, seg_t, fake, smap)
                loss_g = loss_g + c["lambda_ms"] * l_ms
                extra["ms"], extra["ms_ratio"] = float(l_ms.detach()), float(lz.detach())
        else:
            mu_f, _, pres_f = self.E_s(self._to01(fake), seg_t)
            if bool(pres_f.any()):
                l_rec = (mu_f - codes.detach()).abs().mean(-1)[pres_f].mean()
                loss_g = loss_g + c["lambda_style_rec"] * l_rec
                extra["style_rec"] = float(l_rec.detach())
            if bool(pres_r.any()):
                kl = (-0.5 * (1 + lv_r - mu_r ** 2 - lv_r.exp())).mean(-1)[pres_r].mean()
                loss_g = loss_g + c["lambda_style_kl"] * kl
                extra["style_kl"] = float(kl.detach())
        loss_g.backward(); self.opt_g.step(); self.opt_h.step()
        if not inade:
            self.opt_es.step()
        self._update_ema()
        out = {"g": float(loss_g.detach()), "d": float(loss_d.detach()), "adv": float(g["adv"].detach()),
               "nce": float(g["nce"].detach()), "seg": float(g["seg"].detach())}
        if "tc" in g:
            out["tc"] = float(g["tc"].detach())
        out.update(extra)
        return out

    def step(self, phys, label_onehot, real):
        """v1's SonoSPADE.step verbatim (usbg 0.2.0), with two hooks: _v2_d_step after the D update and
        _v2_g_terms added to the generator loss. Both are no-ops when the v2 flags are off."""
        import torch
        if self.E_s is not None or self.v2["style"] == "inade":
            return self._step_style(phys, label_onehot, real)
        target_lab = label_onehot.argmax(dim=1)
        fake = self.G(phys, label_onehot)
        B, _, H, W = fake.shape

        self.opt_d.zero_grad()
        with torch.no_grad():
            pseudo = self.S.net(self._to01(real)).argmax(dim=1)
        use_r1 = self.r1_gamma and self.r1_gamma > 0
        real_in = real.detach().requires_grad_(True) if use_r1 else real
        d_real = self.D(real_in)
        loss_d_real = self._balanced_ce(d_real, pseudo)
        d_fake = self.D(fake.detach())
        fake_tgt = torch.full_like(target_lab, self.n_seg)
        loss_d_fake = self._balanced_ce(d_fake, fake_tgt)
        loss_d = loss_d_real + loss_d_fake
        extra = {}
        if use_r1:
            grad = torch.autograd.grad(d_real.sum(), real_in, create_graph=True)[0]
            loss_r1 = grad.pow(2).flatten(1).sum(1).mean() / float(H * W)
            loss_d = loss_d + 0.5 * self.r1_gamma * loss_r1
            extra["r1"] = float(loss_r1.detach())
        if self.lambda_lm and self.lambda_lm > 0:
            m = self._labelmix_mask(B, H, W)
            mixed = m * real.detach() + (1.0 - m) * fake.detach()
            target_mix = m * d_real.detach() + (1.0 - m) * d_fake.detach()
            loss_lm = (self.D(mixed) - target_mix).pow(2).mean()
            loss_d = loss_d + self.lambda_lm * loss_lm
            extra["lm"] = float(loss_lm.detach())
        loss_d.backward(); self.opt_d.step()

        extra.update(self._v2_d_step(real, fake, pseudo))                       # v2 hook 1

        self.opt_g.zero_grad(); self.opt_h.zero_grad()
        g = self.generator_losses(phys, label_onehot, fake)
        loss_g = g["adv"] + self.lambda_nce * g["nce"] + self.lambda_seg * g["seg"]
        if "tc" in g:
            loss_g = loss_g + self.lambda_tc * g["tc"]
        for name, (wt, val) in self._v2_g_terms(fake, target_lab).items():     # v2 hook 2
            loss_g = loss_g + wt * val
            extra[name] = float(val.detach())
        loss_g.backward(); self.opt_g.step(); self.opt_h.step()
        self._update_ema()
        out = {"g": float(loss_g.detach()), "d": float(loss_d.detach()),
               "adv": float(g["adv"].detach()), "nce": float(g["nce"].detach()),
               "seg": float(g["seg"].detach())}
        if "tc" in g:
            out["tc"] = float(g["tc"].detach())
        out.update(extra)
        return out

    def save(self, path, run_config=None):
        """v1's checkpoint keys unchanged (G_spade = EMA generator, G_raw, n_classes, ngf, n_blocks,
        spade_hidden), plus variant, v2_config and run_config (plain types, weights_only-loadable)."""
        import torch
        super().save(path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.v2.items()}
        state.update({"variant": "sonospade_v2", "v2_config": cfg, "run_config": run_config or {}})
        if self.v2["style"] == "inade":                 # codes are N(0, I) by design: the bank is the prior
            import torch as _t
            d = self.v2["style_dim"]
            self.style_bank = {"mean": _t.zeros(self.n_seg, d), "sd": _t.ones(self.n_seg, d),
                               "count": _t.zeros(self.n_seg)}
        if self.E_s is not None or self.v2["style"] == "inade":
            if self.E_s is not None:
                state["E_s"] = self.E_s.state_dict()
            state["group_map"] = [int(v) for v in self.gmap_t.cpu().tolist()] if self.gmap_t is not None else None
            state["n_seg"] = int(self.n_seg)
            if getattr(self, "style_bank", None) is not None:
                state["style_bank"] = self.style_bank
        torch.save(state, path)

    def build_style_bank(self, real_pool01, batch=16):
        """After training: encode every real-pool frame with E_s under its S_US pseudo-label and fit the
        per-tissue Gaussian used at inference (style=None -> mean, 'sample' -> draw)."""
        import numpy as np
        import torch
        from .style import StyleBank
        bank = StyleBank(self.n_seg, self.v2["style_dim"])
        self.E_s.eval()
        with torch.no_grad():
            for j in range(0, len(real_pool01), batch):
                x = torch.from_numpy(np.asarray(real_pool01[j:j + batch], np.float32))[:, None].to(self.device)
                pseudo = self.S.net(x).argmax(dim=1)
                mu, _, pres = self.E_s(x, pseudo)
                bank.add(mu, pres)
        self.E_s.train()
        self.style_bank = bank.state()
        return self.style_bank
