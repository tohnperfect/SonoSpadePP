#!/usr/bin/env python3
"""Run USB Gym's own texture trainer (its scripts/train_texture_gan.py, unchanged) in the v2
environment, so that its --s-us may also be a v2 foundation-model segmenter (sonospadepp.v2.fm_seg). Only the
segmenter loader is widened: a checkpoint of kind "fm_segmenter" loads as FMSegmenter, anything else exactly as
before. Run with the gym root as the working directory:

  cd $USBG_GYM_ROOT && /path/to/this/repo/.venv/bin/python -I \\
      /path/to/this/repo/scripts/v2/usbgym/train_texture_gym.py --variant sonospade --pairs ... --s-us ...
"""
import os
import runpy
import sys

GYM = os.environ.get("USBG_GYM_ROOT", "")          # a USB Gym checkout (https://github.com/tohnperfect/USBGym)
if not GYM:
    raise SystemExit("set USBG_GYM_ROOT to a USB Gym checkout")
sys.path.insert(0, os.path.join(GYM, "src"))          # the gym's usbg, ahead of the v2 environment's SonoSPADE usbg
for m in [m for m in sys.modules if m == "usbg" or m.startswith("usbg.")]:
    del sys.modules[m]

import torch  # noqa: E402
import usbg.segmenter as SEG  # noqa: E402

assert os.path.realpath(SEG.__file__).startswith(os.path.realpath(GYM)), SEG.__file__
_orig_load = SEG.Segmenter.load


def _load_any(path, device="cpu"):
    st = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(st, dict) and st.get("kind") == "fm_segmenter":
        from sonospadepp.v2.fm_seg import FMSegmenter
        return FMSegmenter.load(path, device=device)
    return _orig_load(path, device=device)


SEG.Segmenter.load = staticmethod(_load_any)
sys.argv = [os.path.join(GYM, "scripts", "train_texture_gan.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
