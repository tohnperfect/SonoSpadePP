# Trained weights

The checkpoints behind the paper's tables are release assets of this repository (never files in git).
This file lists every asset with the path the scripts expect, its size and its SHA-256.

## Terms

All checkpoints listed here are offered under **CC BY-NC 4.0** (https://creativecommons.org/licenses/by-nc/4.0/):
non-commercial use, with attribution. They are for research simulation only, not for clinical use.

When you use them, cite this work (`CITATION.cff`) and:

* SonoSPADE: T. Intharah, Z. Gao, H. Dong, "SonoSPADE: Real-Time, Per-Tissue Ultrasound Texture Synthesis for
  Closed-Loop Acquisition," MICCAI 2026 Workshops (SASHIMI).
* USFM, for every checkpoint whose name contains `usfm` or `tier0` (the segmenter is initialized from USFM
  weights, which are CC BY-NC 4.0): J. Jiao et al., "USFM: A Universal Ultrasound Foundation Model Generalized to
  Tasks and Organs towards Label Efficient Image Analysis," Med. Image Anal., 2024, doi:10.1016/j.media.2024.103202.
* TRUSTED, for every checkpoint marked "Trained on TRUSTED": W. Ndzimbong et al., "TRUSTED: The Paired 3D
  Transabdominal Ultrasound and CT Human Data for Kidney Segmentation and Registration Research," Sci. Data, 2025,
  doi:10.1038/s41597-025-04467-1. The dataset itself is available only from its authors under their data usage
  agreement and is not part of this release.
* The real ultrasound frames used in training: S. Vitale et al., "Improving Realism in Patient-Specific Abdominal
  Ultrasound Simulation Using CycleGANs," Int. J. Comput. Assist. Radiol. Surg., 2020, doi:10.1007/s11548-019-02046-5.

## Not released

* Models trained with EchoCare features (the EchoCare feature-matching component and the EchoCare-initialized
  segmenter with its generator): EchoCare's weights carry no license.
* Tier 2 has no checkpoint. It is a one-shot fine-tune of a tier-1 model on one image of the patient, run at
  evaluation time by `scripts/adapt_m9a.py`; only its predictions were stored, and those are TRUSTED-derived images.
* The label-free baselines of Table II (CUT, CycleGAN, S-CycleGAN) and the workshop paper's SonoSPADE generator are
  not products of this study. The generator is `sonospade_sashimi_linear.pt` in the SonoSPADE v0.2.0 release; the
  baselines retrain with `scripts/v1/train_texture_gan.py`.
* Models of the earlier CT-intensity study (M1 to M6), the CT-intensity variants of tiers 1 and 2, the
  registration-network variant, the retrieval hybrid and the selection runs.

## Use

Download an asset and place it at the path in the table (relative to this repository unless noted), or pass it
with `--ckpt`. A tier-1 or pix2pix file is given together with its base generator, as the chains do:

```bash
python -I scripts/v2/eval_v2.py --model tier1 --seed 0 --kind paired_lab \
    --ckpt "runs/b0_s0.pt::runs/m7/lab_s0_f0.pt" --skip-tstr
```

`sha256sum -c WEIGHTS_SHA256SUMS.txt` checks a folder of downloaded assets.

## Assets

### SonoSPADE retrained (linear)

SonoSPADE retrained in this study's environment, three seeds (Table II, row "SonoSPADE retrained"). Trained on simulator pairs from TotalSegmentator CTs and the unlabeled real pool.

| Asset | Path the scripts expect | Bytes | SHA-256 |
|---|---|---:|---|
| `sonospade_linear_s0.pt` | `runs/v2/v1_repro/sonospade_s0.pt` | 35,928,929 | `7205b697bf66631775d82a3b8a0855eb2b73f1b5d8096a86918b487624d21733` |
| `sonospade_linear_s1.pt` | `runs/v2/v1_repro/sonospade_s1.pt` | 35,928,929 | `2efa8774532435aa4a2a7746c593ddad9fb91a346f1d1c76e611c361f01c0d81` |
| `sonospade_linear_s2.pt` | `runs/v2/v1_repro/sonospade_s2.pt` | 35,928,929 | `81458dcdc7e735352628eaafe9fb4c134e0789b85ed776e45c4eba33585a50c1` |

### Tier 0: SonoSPADE with the USFM segmenter (linear)

Tier 0 as reported: SonoSPADE trained with a segmenter initialized from USFM, three seeds, and that segmenter (Tables II and III).

| Asset | Path the scripts expect | Bytes | SHA-256 |
|---|---|---:|---|
| `tier0_linear_s0.pt` | `runs/v2/wp3/gen_sususfm_s0.pt` | 35,937,157 | `4c35f279e65a4a6374c1419e76570b7c5f180a64a0734428f3958c5a0b24c7fd` |
| `tier0_linear_s1.pt` | `runs/v2/wp3/gen_sususfm_s1.pt` | 35,937,157 | `0a1d59659ced558054506484414d4608a3d11d1f9a5992c9216819467fb6fd48` |
| `tier0_linear_s2.pt` | `runs/v2/wp3/gen_sususfm_s2.pt` | 35,937,157 | `57ff6dfced194d8c2e1324462343f1bd8058bb7f4d20a8c242c6512a6c69a7e4` |
| `s_us_usfm_linear.pt` | `runs/v2/wp3/s_us_usfm.pt` | 345,503,755 | `fbb1e623cb535b13b352e4b9c88ee9c1decadbd0f22c5204d8a2c36e12803ff3` |

### Component: DINOv2-initialized segmenter (linear)

Tier-0 component, segmenter initialized from DINOv2 (one seed) and that segmenter.

| Asset | Path the scripts expect | Bytes | SHA-256 |
|---|---|---:|---|
| `component_susdinov2_linear_s0.pt` | `runs/v2/wp3/gen_susdinov2_s0.pt` | 35,937,769 | `2d2174b111cbe2ca6bb5b3ad6a0e4af99d7f1c5bc649440e85787a52bca7e145` |
| `s_us_dinov2_linear.pt` | `runs/v2/wp3/s_us_dinov2.pt` | 345,431,081 | `d15702fa1be3acc02a43f2a0c8ad8dbd241eec8293ad8af518636b2d275795f8` |

### Component: DINOv2 feature matching (linear)

Tier-0 component, per-tissue feature matching with DINOv2 features, three seeds (Table III).

| Asset | Path the scripts expect | Bytes | SHA-256 |
|---|---|---:|---|
| `component_fm_dinov2_linear_s0.pt` | `runs/v2/wp1_final/e1b_dinov2_tis0.1_full_s0.pt` | 35,941,035 | `e7010e10ee85787db2d0731345cdf78e3aa2e10dd77cb3109a870b06c6afbbc4` |
| `component_fm_dinov2_linear_s1.pt` | `runs/v2/wp1_final/e1b_dinov2_tis0.1_full_s1.pt` | 35,941,163 | `b38c21c4787bfe06639625669117de4b311c5180268b3b3a6584728ebc3b6e10` |
| `component_fm_dinov2_linear_s2.pt` | `runs/v2/wp1_final/e1b_dinov2_tis0.1_full_s2.pt` | 35,941,163 | `2c2237fc7c33f91b920aa717dc51b80d88f51cf56f992f4354712e12b74a51f3` |

### Component: per-region style codes (linear)

Tier-0 component, per-region style codes, three seeds (Table III; one checkpoint serves both evaluation modes).

| Asset | Path the scripts expect | Bytes | SHA-256 |
|---|---|---:|---|
| `component_style_linear_s0.pt` | `runs/v2/wp2/e2_sean_full_s0.pt` | 41,070,869 | `5642cfe9bf5a478b13bac12e98b1220bf808c5d0eab117c8af5b6dcf1706e99b` |
| `component_style_linear_s1.pt` | `runs/v2/wp2/e2_sean_full_s1.pt` | 41,070,869 | `d0ec3999fc0e13a90b6938de2b95d5de23d875c7e523aea58a96c764fbc96605` |
| `component_style_linear_s2.pt` | `runs/v2/wp2/e2_sean_full_s2.pt` | 41,070,869 | `23f6123d08b09285e03ac93ccd55b1d4aece783923ddebfa8adf65862719a1e7` |

### SonoSPADE retrained (fan)

SonoSPADE retrained on the sector-warped pairs of the same-patient study, three seeds, and its segmenter (Table I, row "SonoSPADE"). No TRUSTED data.

| Asset | Path the scripts expect | Bytes | SHA-256 |
|---|---|---:|---|
| `sonospade_fan_s0.pt` | `runs/b0_s0.pt` | 35,907,651 | `4565770a0b7598f8832c48e64ffcaa326fea6d85bb8005371d6b34579e58252c` |
| `sonospade_fan_s1.pt` | `runs/b0_s1.pt` | 35,907,651 | `09cf0269ddd0d7b2ad91a4419038c43f7abd32c2b4e47d4794b1a16c01b2d894` |
| `sonospade_fan_s2.pt` | `runs/b0_s2.pt` | 35,907,651 | `ddb9ccb76f875686aacfb3f59f77daf5a77e2c2067010cce0cf3898907a0d853` |
| `s_us_fan.pt` | `runs/s_us_phys_b0__lab_base.pt` | 34,613,057 | `f0cad749dcfa27d03c4f1937887a059bd78f2980a95840dff073a5bbcc993aee` |

### Tier 0 (fan)

Tier 0 in the fan geometry: the same recipe with the USFM-initialized segmenter, three seeds, and that segmenter. No TRUSTED data.

| Asset | Path the scripts expect | Bytes | SHA-256 |
|---|---|---:|---|
| `tier0_fan_s0.pt` | `runs/b0v2_s0.pt` | 35,926,759 | `304317ae0f0ea5bd869244f9080a8cc3159179f74dae579b4a0a5432bd3b087b` |
| `tier0_fan_s1.pt` | `runs/b0v2_s1.pt` | 35,926,759 | `692dc98456bab5e21176290039bb8a625a867482d251f26d8ca7f3920116673a` |
| `tier0_fan_s2.pt` | `runs/b0v2_s2.pt` | 35,926,759 | `87426a7ca91a4c6a4ee394ee9d01510e4e4048168f6d5a905e78c0295ac0b006` |
| `s_us_usfm_fan.pt` | `runs/s_us_usfm_phys_b0__lab_base.pt` | 345,508,355 | `126ebd52812138ab3cd4e458018dc9e5ee5f56b3ae3fb34d58d1ead56636ca80` |

### Unpaired control on TRUSTED planes (fan)

Unpaired control: SonoSPADE's unpaired recipe fine-tuned on TRUSTED planes, three seeds x five patient folds. **Trained on TRUSTED.**

| Asset | Path the scripts expect | Bytes | SHA-256 |
|---|---|---:|---|
| `unpaired_control_fan_s0_f0.pt` | `runs/b0t/b0t_s0_f0.pt` | 35,927,947 | `1384997709c86806ed10f93a5dccbe988f6ba01f565170ed328691c2e9fddba0` |
| `unpaired_control_fan_s0_f1.pt` | `runs/b0t/b0t_s0_f1.pt` | 35,927,947 | `f2f19fc238ab93ad782f1152eb479c238ef1a3587e60f87f277f4c9ba7b5a88a` |
| `unpaired_control_fan_s0_f2.pt` | `runs/b0t/b0t_s0_f2.pt` | 35,927,947 | `5f570f8204429dda5eab4a0e25f6be8842de91838385a04ac838f7d41f54d35d` |
| `unpaired_control_fan_s0_f3.pt` | `runs/b0t/b0t_s0_f3.pt` | 35,927,947 | `333865b3ada7b8f4b42611d40a7bea6fde9261435739801d9fa53c4f9a84520a` |
| `unpaired_control_fan_s0_f4.pt` | `runs/b0t/b0t_s0_f4.pt` | 35,927,947 | `1ac5b93fd732c11b431966c36847eaa24e44e890ffb660193c96b832f5c7c20d` |
| `unpaired_control_fan_s1_f0.pt` | `runs/b0t/b0t_s1_f0.pt` | 35,927,947 | `72810a338426b5c630e375d8f8a02e533361352cb21ed065ccf7d602d3cf89b3` |
| `unpaired_control_fan_s1_f1.pt` | `runs/b0t/b0t_s1_f1.pt` | 35,927,947 | `f14fb4548aac54bd5890270dc88de44d72dab0bc7c6d671e5724108ee3730557` |
| `unpaired_control_fan_s1_f2.pt` | `runs/b0t/b0t_s1_f2.pt` | 35,927,947 | `938df5ffe4e69237cc4837bbe55af118795ef53e9a49fcd8fb2f339a6add9db7` |
| `unpaired_control_fan_s1_f3.pt` | `runs/b0t/b0t_s1_f3.pt` | 35,927,947 | `3aff4025e85235eb065912e785b1348537e0cd27f43f5db7e3261b3a773a5fbc` |
| `unpaired_control_fan_s1_f4.pt` | `runs/b0t/b0t_s1_f4.pt` | 35,927,947 | `05992c2fdba81ed86e833f5c33587aec4ab9d3238a5dcf3e4a9a4c5eaed6f86c` |
| `unpaired_control_fan_s2_f0.pt` | `runs/b0t/b0t_s2_f0.pt` | 35,927,947 | `73cfba1a7f088a38496f10a41f05b12b09523df3aea6c23674c39caa9517e937` |
| `unpaired_control_fan_s2_f1.pt` | `runs/b0t/b0t_s2_f1.pt` | 35,927,947 | `360130fba37e7579ccf08ae11ca10a39d5577df83d4b72b86876c07f52c0735a` |
| `unpaired_control_fan_s2_f2.pt` | `runs/b0t/b0t_s2_f2.pt` | 35,927,947 | `621a66419ff7f620b01b4cf0c36f8bab973996f6e6e869de2ff624fb2d6ceecc` |
| `unpaired_control_fan_s2_f3.pt` | `runs/b0t/b0t_s2_f3.pt` | 35,927,947 | `d89e0b897040afe001c0c58a46202776aaa9e096f757cdf67b1da82c624b4fa3` |
| `unpaired_control_fan_s2_f4.pt` | `runs/b0t/b0t_s2_f4.pt` | 35,927,947 | `81b9a3c2464865eab893bf7bc67d16b08518e9c11218fe406f6fbaa0876953f9` |

### Tier 1: paired planes of other patients (fan)

Tier 1: fine-tuned on registered CT and ultrasound planes of the patients outside the test fold, three seeds x five folds. Each file holds the fine-tuned generator state and loads on top of the matching `sonospade_fan_s<seed>.pt`. **Trained on TRUSTED.**

| Asset | Path the scripts expect | Bytes | SHA-256 |
|---|---|---:|---|
| `tier1_fan_s0_f0.pt` | `runs/m7/lab_s0_f0.pt` | 18,052,363 | `004afd4ea8c6baa389c23c8d6d836822d0b492336ba0a9abc1738fdcb8795c86` |
| `tier1_fan_s0_f1.pt` | `runs/m7/lab_s0_f1.pt` | 18,052,363 | `f9b5c9b1e9309f1b91aae4d1482573b22db6b13e59535b0718c1c81d4c5a9214` |
| `tier1_fan_s0_f2.pt` | `runs/m7/lab_s0_f2.pt` | 18,052,363 | `d65353651fd14b3c641d8db0fe760599a382997f728ee0b3d95845199397d1bf` |
| `tier1_fan_s0_f3.pt` | `runs/m7/lab_s0_f3.pt` | 18,052,363 | `ff24caa0b1d722d9d69ecc958592d7b43d091fbbc48556091c311c2c24824e14` |
| `tier1_fan_s0_f4.pt` | `runs/m7/lab_s0_f4.pt` | 18,052,363 | `f2800f0dbfd15d9b65a7fe10ec63efdfa360f6f2d40fc15e78f054096c90ee0f` |
| `tier1_fan_s1_f0.pt` | `runs/m7/lab_s1_f0.pt` | 18,052,363 | `ae6294be9c0c864254c202e4fbf96edd7d876569054daf1703bc4dc861d06bdb` |
| `tier1_fan_s1_f1.pt` | `runs/m7/lab_s1_f1.pt` | 18,052,363 | `91a00b6c614c9b27e959970a668f4d917b0bbda0c33f1d2e03cebf931d1821c5` |
| `tier1_fan_s1_f2.pt` | `runs/m7/lab_s1_f2.pt` | 18,052,363 | `78efa9f92c8539142eaa707ed9923f5e4dd15f4ea4a39363e2be41d68a5cbf99` |
| `tier1_fan_s1_f3.pt` | `runs/m7/lab_s1_f3.pt` | 18,052,363 | `7d8d8de82cc39ac4f9afa1cdc7a12a7f80251254462bdb281a611c73e3567d35` |
| `tier1_fan_s1_f4.pt` | `runs/m7/lab_s1_f4.pt` | 18,052,363 | `4196377eeaa2e04466c17710cf04bcb541937fe9a5db009ab327891093fe9954` |
| `tier1_fan_s2_f0.pt` | `runs/m7/lab_s2_f0.pt` | 18,052,363 | `7c08ffb9988a6059721d86ca2a589c99c1fb97b8424089018936f7d48d9737a9` |
| `tier1_fan_s2_f1.pt` | `runs/m7/lab_s2_f1.pt` | 18,052,363 | `df17d8ea05ef5e6fd678014b472a499fbbd4de07d82fb68e972b6a99decde6ea` |
| `tier1_fan_s2_f2.pt` | `runs/m7/lab_s2_f2.pt` | 18,052,363 | `2fe9079d682c57db75cb0ebf0c7169c68107c846a7492a70fb5bf92fd354c859` |
| `tier1_fan_s2_f3.pt` | `runs/m7/lab_s2_f3.pt` | 18,052,363 | `48c9b57230fde45679ed4db6c7d422731376b446702c01b252e9638ccec7638d` |
| `tier1_fan_s2_f4.pt` | `runs/m7/lab_s2_f4.pt` | 18,052,363 | `89374f66a10b351ecf0e34a1491a64397b509ea51652338e23f7456829a8def8` |

### pix2pix baseline (fan)

pix2pix baseline with the same data and budget as tier 1, three seeds x five folds; loads like tier 1. **Trained on TRUSTED.**

| Asset | Path the scripts expect | Bytes | SHA-256 |
|---|---|---:|---|
| `pix2pix_fan_s0_f0.pt` | `runs/m7/p2p_s0_f0.pt` | 18,052,363 | `ef34e0d7066017f70f8a842ff46ee16f18330accd7976f102bc6f90d8b28347e` |
| `pix2pix_fan_s0_f1.pt` | `runs/m7/p2p_s0_f1.pt` | 18,052,363 | `7514ab98f1ff0732a1138c1a3c2beda73fc26a42e6fc78ff4c700ee7164f23db` |
| `pix2pix_fan_s0_f2.pt` | `runs/m7/p2p_s0_f2.pt` | 18,052,363 | `410d80c1b17e21b3f79855137b3a8fab769269bd881dcf6a8d63568ba590d943` |
| `pix2pix_fan_s0_f3.pt` | `runs/m7/p2p_s0_f3.pt` | 18,052,363 | `b41c860c4f324561fbe9cf8b2820a05f27f06ab1ba40a9d695344828f306b264` |
| `pix2pix_fan_s0_f4.pt` | `runs/m7/p2p_s0_f4.pt` | 18,052,363 | `36bbdc7da2656b76ff14ce97f253ce586ab1b60cac337577563e3be4cfbc0eb5` |
| `pix2pix_fan_s1_f0.pt` | `runs/m7/p2p_s1_f0.pt` | 18,052,363 | `42e1f9f69a76324580345a2cbe58d2d0c7e841b671fe69dd57e360d62ddaa381` |
| `pix2pix_fan_s1_f1.pt` | `runs/m7/p2p_s1_f1.pt` | 18,052,363 | `e13e8a4f7e125d573092d7313dadaf7f4eab3821553ab0eff5c7988603945076` |
| `pix2pix_fan_s1_f2.pt` | `runs/m7/p2p_s1_f2.pt` | 18,052,363 | `b2983a713467f5e3cc99412b571d13601000e10d70729de63aba6b1450681ea2` |
| `pix2pix_fan_s1_f3.pt` | `runs/m7/p2p_s1_f3.pt` | 18,052,363 | `c900fa25ce165fb3e68e9cd6a379b25d14d5674a487cb4ed8dbfd125fefbae67` |
| `pix2pix_fan_s1_f4.pt` | `runs/m7/p2p_s1_f4.pt` | 18,052,363 | `85911e4ad1b7d2465e4e38d59f4b41ce96b76ce53856bf9ac899b1f0e8ffa02a` |
| `pix2pix_fan_s2_f0.pt` | `runs/m7/p2p_s2_f0.pt` | 18,052,363 | `8f8ddc1764a5848232c8fca8ca1abd6ea6f5442d135a4afa1bd72f177bdd4ab3` |
| `pix2pix_fan_s2_f1.pt` | `runs/m7/p2p_s2_f1.pt` | 18,052,363 | `3b290e1ca2c50cee75a7630afc07fa26b88808daaf6e555dc39a8537434a8b2c` |
| `pix2pix_fan_s2_f2.pt` | `runs/m7/p2p_s2_f2.pt` | 18,052,363 | `626163ae13d8c0e8055d7e8e02f5c63f86c5e099695c88c77fb65561c015699f` |
| `pix2pix_fan_s2_f3.pt` | `runs/m7/p2p_s2_f3.pt` | 18,052,363 | `66727473e92d7d2962d4045bf68a7d1dfecd604dbe76065c3f36effa4346735f` |
| `pix2pix_fan_s2_f4.pt` | `runs/m7/p2p_s2_f4.pt` | 18,052,363 | `382c6ecca6d0242b69d6e29b8a31bc33c83a85df4e99710917ef6214f3f459fe` |

### Closed loop in USB Gym (curvilinear)

Texture stages of the closed-loop evaluation in USB Gym's SonoSPADE checkpoint layout (curvilinear probe): SonoSPADE retrained (v1r), tier 0 with its USFM-initialized segmenter, and tier 1 started from v1r (seed 0, fold 0; **trained on TRUSTED**). Paths are relative to the USB Gym checkout.

| Asset | Path the scripts expect | Bytes | SHA-256 |
|---|---|---:|---|
| `sonospade_v1r_curvi.pt` | `data/dataset/v2fm/sonospade_v1r_curvi.pt` | 35,931,071 | `9aa1fa8e4ab673f424251fd59421ce44c006bb55bc937dc28dd4e22865cf7cbd` |
| `tier0_curvi.pt` | `data/dataset/v2fm/sonospade_v2_curvi.pt` | 35,930,765 | `e4fcd3368bb63032948dbb93a9f6d8003dc124e121938b7ed8edf7a4160a1a89` |
| `s_us_usfm_curvi.pt` | `data/dataset/v2fm/s_us_usfm_curvi.pt` | 345,505,267 | `c6c245c4209a6bd2db100821e0efc437f7af8cb8a37265e0ad727b1e7986703c` |
| `tier1_curvi.pt` | `data/dataset/v2fm/sonospade_t1g_curvi.pt` | 35,912,255 | `c0ba1d4998498c33f06e505e012667525657188a56f3fffb4a61ffe31234f8f7` |

72 files, 3.30 GB in total.
