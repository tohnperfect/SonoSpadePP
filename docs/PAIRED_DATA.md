# Is there public CT + ultrasound from the same patient?

Why it matters: without the same patient in both modalities, a CT-informed texture can only be
shown to look more like real ultrasound in general (distribution metrics, a downstream segmenter).
Showing it reproduces *this patient's* texture needs a same-patient pair.

Checked in November 2026 against the dataset pages, papers and repositories.

## Short answer

S-CycleGAN and the Vitale / Kaggle set are **unpaired**. The only public, same-patient,
transabdominal CT + B-mode set found is **TRUSTED**, and it covers the **kidney only**.

| Dataset | Anatomy | Same patient? | What is released | Registration | Access / license |
|---|---|---|---|---|---|
| S-CycleGAN (Song & Chong, IEEE CBS 2024) | abdomen | **No**: AbdomenCT-1K CT vs Kaggle ussimandsegm US, described as unpaired | code only (github.com/yhsong98/ct-us-i2i-translation) | none | no license stated on GitHub |
| Kaggle ussimandsegm (Vitale et al. 2019) | abdomen | **No**: real US from 11 volunteers; simulated US ray-cast from 13 VISCERAL CTs of other people | 617 real (61 annotated) + 926 simulated images | none | "Data files © Original Authors", Kaggle login |
| **TRUSTED** (Ndzimbong et al., Sci Data 2025, doi 10.1038/s41597-025-04467-1) | **kidney** | **Yes**: 48 patients, US right after CT | CT (one NIfTI per patient, contrast phase not stated) + 59 single-kidney **3D** US volumes (Acuson S3000, 0.3 mm), no probe tracking | not co-registered; 7 landmarks per kidney in both, STAPLE masks; best benchmark error 4.47 mm | figshare record says CC BY 4.0, but the paper asks for a signed Data Usage Agreement; 16 GB |
| LiverRegNet (Xu et al., BSPC 2025) | liver | in vitro specimens (not patients; unverified) | 32 3D-US / 3D-CT pairs with rigid transforms | rigid labels | email request |
| Rabbani et al. 2021 | liver (laparoscopic) | yes, 4 patients | pre-op CT + intra-op laparoscopic US | via laparoscope | link in the paper returns 404 |
| TCIA overlap (CMB-*, CPTAC-*) | mixed | yes, about 28 patients have both | clinical CT and US from different dates | none | CC BY 4.0 |
| Balgrist lumbar spine (Sci Data 2025) | spine (bone) | yes, 63 volunteers, same day | ultra-low-dose CT + tracked 2D US | not confirmed | CC BY 4.0 |
| LOTUS (Velikova 2023) | aorta / abdomen | No: Synapse CT vs volunteer sweeps | code only | none | n/a |

The MR + ultrasound sets (RESECT, ReMIND, CuRIOUS, µ-RegPro, Prostate-MRI-US-Biopsy) are paired,
but they are brain or prostate MRI, not CT.

## What this means for SonoSpadePP

* The comparison in this repo is **unpaired**: CT content from TotalSegmentator, real reference
  from Kaggle ussimandsegm. It measures realism and downstream transfer, not patient fidelity.
* A per-patient check is possible for the **kidney only**, using TRUSTED: register each 3D US
  volume to its CT with the landmarks, cut matching 2D planes from both, render the CT plane
  through each variant, and compare to the US plane. Expect about 4.5 mm residual misalignment
  and a kidney-only conclusion. Read the Data Usage Agreement before downloading.
* For liver, bowel and aorta there is no public patient-paired set. That needs local collection
  (patients who already have an abdominal CT and get a tracked ultrasound), under ethics approval.
