"""Inference for v2 checkpoints. The runtime API keeps v1's call, translate_aligned(phys, lab), and adds an
optional style argument for WP2. A v1 checkpoint loads too (it simply has no v2 keys), so this translator is a
drop-in for the USB Gym texture slot."""
from __future__ import annotations

import numpy as np


class SonoSPADEv2Translator:
    wants_label = True

    def __init__(self, ckpt_path, device=None):
        import torch
        from usbg import texture_gan as TG
        from usbg.render_lotus import get_device
        self.device = device or get_device()
        state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.n_classes = state.get("n_classes", 13)
        self.config = dict(state.get("v2_config", {}))
        self.variant = state.get("variant", "sonospade")
        from .generator import build_generator
        from .style import StyleEncoder, convert_style
        c = self.config
        self.G = build_generator(self.n_classes, ngf=state.get("ngf", 32), n_blocks=state.get("n_blocks", 6),
                                 spade_hidden=state.get("spade_hidden", 64), noise_inject=c.get("noise_inject", False),
                                 gain_sigma=c.get("gain_sigma", 0.0))
        self.style_kind = c.get("style", "none")
        self.has_style = self.style_kind in ("sean", "inade")
        self.E_s = None
        if self.has_style:
            convert_style(self.G, self.style_kind, c["style_dim"], self.n_classes, c.get("inade_sigma0", 0.05))
            self.n_seg = state["n_seg"]
            gm = state.get("group_map")
            self.group_map = np.asarray(gm if gm is not None else range(self.n_classes), np.int64)
            if self.style_kind == "sean":
                self.E_s = StyleEncoder(self.n_seg, c["style_dim"]).to(self.device)
                self.E_s.load_state_dict(state["E_s"])
                self.E_s.eval()
            sb = state["style_bank"]
            self.bank_mean, self.bank_sd = sb["mean"].float().cpu(), sb["sd"].float().cpu()
        self.G.to(self.device)
        self.G.load_state_dict(state["G_spade"])
        self.G.eval()
        self.noise_seed = None                   # set to an int to fix the injected noise (locality metrics)

    def sample_style(self, tissue_grouped: int, rng=None):
        """A code drawn from the style bank's Gaussian for one tissue (grouped id)."""
        rng = rng if rng is not None else np.random.default_rng()
        m, sd = self.bank_mean[tissue_grouped].numpy(), self.bank_sd[tissue_grouped].numpy()
        return (m + sd * rng.standard_normal(m.shape)).astype(np.float32)

    def encode_styles(self, img01, label, grouped=False):
        """Per-tissue codes (mean) of a real frame under a given label map (canonical ids unless grouped=True):
        {grouped tissue id: code}. The entry point SonoSpadePP uses with CT label slices."""
        import torch
        if self.E_s is None:
            raise ValueError("this checkpoint has no style encoder (INADE codes are noise; no encoding)")
        lab = np.asarray(label, np.int64)
        lab = lab if grouped else self.group_map[lab]
        x = torch.from_numpy(np.asarray(img01, np.float32))[None, None].to(self.device)
        with torch.no_grad():
            mu, _, pres = self.E_s(x, torch.from_numpy(lab)[None].to(self.device))
        return {k: mu[0, k].cpu().numpy() for k in range(self.n_seg) if bool(pres[0, k])}

    def _cond(self, seg, lab, style):
        import torch
        from .style import style_map
        if not self.has_style:
            if style:
                raise ValueError("this checkpoint has no style codes")
            return seg
        codes = self.bank_mean.clone()
        for t, v in (style or {}).items():
            codes[int(t)] = torch.as_tensor(self.sample_style(int(t)) if isinstance(v, str) else v)
        lg = torch.from_numpy(self.group_map[np.asarray(lab, np.int64)])[None].to(self.device)
        return torch.cat([seg, style_map(codes[None].to(self.device), lg, self.n_seg)], 1)

    def translate_aligned(self, img01, aligned_label, style=None, noise_seed=None):
        """style: None -> each tissue's bank mean (deterministic up to injected noise); {grouped tissue id: code
        array or 'sample'} sets or samples that tissue's appearance; other tissues use their bank mean."""
        import torch
        from usbg import texture_gan as TG
        img = np.asarray(img01, np.float32)
        h, w = img.shape
        lab = np.asarray(aligned_label)
        if lab.shape != (h, w):
            from PIL import Image
            lab = np.asarray(Image.fromarray(lab.astype(np.uint8)).resize((w, h), Image.NEAREST))
        seg = self._cond(TG.onehot_labels(lab, self.n_classes, self.device), lab, style)
        if hasattr(self.G, "noise_gen"):          # injected noise: fresh per call, or fixed for paired comparisons
            seed = noise_seed if noise_seed is not None else self.noise_seed
            self.G.noise_gen = None if seed is None else torch.Generator().manual_seed(int(seed))
        a = torch.from_numpy(img * 2 - 1)[None, None].to(self.device)
        with torch.no_grad():
            out = self.G(a, seg)[0, 0].cpu().numpy()
        return np.clip((out + 1) / 2, 0, 1).astype(np.float32)

    def translate(self, img01, label_slice):
        from usbg.segmenter import align_label_to_image
        img = np.asarray(img01, np.float32)
        return self.translate_aligned(img, align_label_to_image(label_slice, img.shape))

    __call__ = translate
