# Report outline

The report is yours to write (in English; the lecturer asked for a short report). This file is only the skeleton:
headings, what each section must contain, and where its numbers come from. Under most headings a short Hebrew note
says what to think about while writing. Delete the guidance when you're done.

Every number comes from a file listed by unit 08's **reproduce-everything** cell. Never type in a number by hand
that no notebook produced.

Length: 6–10 pages with figures. Figures: the demo (two X-rays, predicted mesh vs CT), the story table's dot plot,
one best and one worst case, the sim-vs-real comparison.

---

## 1. Research question

- One paragraph: reconstruct the 3D pelvis (a 64³ occupancy, then a mesh) from two X-rays (AP + LAT) with a
  2D-encoder / 3D-decoder network trained on simulated X-rays, and measure how much is lost on real X-rays.
- The four sub-questions (from the proposal): does it beat non-neural baselines? Does the second view help? How much
  does the amount of training data matter? How large is the simulated-to-real gap?

<div dir="rtl">

> שאלה אחת שאפשר לענות עליה במספרים.‏ כתבו מה הייתה ההשערה **לפני** הניסויים, לא אחרי.

</div>

## 2. Data and licences

- **TotalSegmentator** (Wasserthal et al. 2023): CT with 117 labelled structures, **CC BY 4.0**. The subset (Zenodo
  10047263) locally, the full v2.0.1 (Zenodo 10047292) on Kaggle. Cite it in every figure that shows it.
- **DeepFluoro** (Grupp et al. 2020): real fluoroscopy of 6 cadavers with CT and labels, through the Hugging Face
  mirror `eigenvivek/xvr-data`, **CC BY-NC-ND 4.0**. Used for evaluation only. The repo holds **results only**
  (aggregate numbers): no derived data, no images, no figures made from it.
- Tools: DiffDRR (MIT), PyTorch, scikit-image (marching cubes).
- Screening: how many CTs had a complete, metal-free pelvis (unit 01: 27 of 102 in the subset; unit 02: __ of 1,228
  in the full set), and why the rest were rejected (`screening.json`).
- The patient-level split (`splits/split_v1.json`, 70/10/20, made once and never regenerated) and its sizes.
- The processed dataset: the 320 mm canonical cube, 64³ voxels of 5 mm, RAS axes; the DRR geometry (source–detector
  1,020 mm, source–isocentre 750 mm, 128² pixels of 4 mm).

<div dir="rtl">

> זה הסעיף שהמרצה ביקש במפורש.‏ לכל מאגר: מה הוא, הרישיון, ומה **מותר** לעשות איתו.‏ על DeepFluoro כתבו במפורש שה־repo מכיל רק מספרים מצטברים, ולמה (NC-ND: בלי שימוש מסחרי ובלי יצירות נגזרות).

</div>

## 3. Method

- From CT to training pairs: HU → μ, the perspective DRR (unit 01's parallel DRR as the idea, DiffDRR for the data),
  the oracle silhouettes. One figure: a CT slice, the AP and the LAT.
- The network: one 2D encoder per view, lifting each view's features along its rays (AP along A, LAT along R), fusion
  by concatenation, a 3D decoder with skip connections (unit 04 "as built": a 16² bottleneck, InstanceNorm, 679 k
  parameters for 2 views). One diagram.
- Training: BCE + soft Dice, Adam 1e-3, batch 8, 4,000 steps, intensity augmentation only (and why not geometric), the
  checkpoint with the best val Dice.
- The pipeline at test time (unit 08): standardise each image, sigmoid > 0.5, marching cubes → a mesh in mm.
- Metrics: Dice, IoU, HD95, ASSD, NSD at 1.5 mm and 5 mm (and why both, with 5 mm voxels), Chamfer on the meshes, the
  CE angle.

<div dir="rtl">

> כתבו כך שסטודנט מהקורס יוכל לשחזר את השיטה בלי ה־repo.‏ לכל בחירה שעשיתם (InstanceNorm, ‏16² ולא 8², סף 0.5) משפט אחד של "למה".

</div>

## 4. Baselines

- The visual hull from the oracle silhouettes (an upper bound for silhouette methods; why it fills the concavities).
- Nearest-neighbour retrieval (AP, AP + LAT), the mean shape, and the oracle retrieval (a **ceiling**, not a method:
  it uses the test GT). Tatarchenko et al. 2019 as the reason retrieval is the floor.
- Source: `results/baselines.csv` (unit 03).

<div dir="rtl">

> למה ה־mean shape כל כך חזק אצלנו, ומה זה אומר על הרשת אם היא רק קצת יותר טובה ממנו?

</div>

## 5. The controlled experiment

- Arm 1: 1 view vs 2 views, everything else fixed, 3 seeds each. Arm 2: 25 % vs 100 % of the training set, 2 views.
- The story table (unit 08, part B): mean ± sd **over test subjects**, n = subjects (seeds averaged per subject first).
- Unit 06's paired comparison (per-subject differences and their spread), not "do the error bars overlap".
- Source: `results/experiment_v1.csv` (unit 06); the runs' `log.csv` for the curves.

<div dir="rtl">

> מה ההבדל בין "התוצאה השתפרה" לבין "השיפור גדול מהרעש בין seeds ובין מטופלים"?‏ ‏n = 9 מטופלי test: כתבו את זה ליד כל מספר.

</div>

## 6. Simulated vs real

- Why there is no real lateral view (DeepFluoro's frames are all near-AP), so the gap is measured two ways: the 1-view
  model on a real frame vs on a DRR at the same pose, and the 2-view model on (real AP + simulated LAT) vs (simulated
  AP + simulated LAT).
- The gap in Dice and HD95, with n (specimens, frames), and the MC-dropout uncertainty.
- Source: `results/sim_vs_real_v1.csv` (unit 07), aggregates only.

<div dir="rtl">

> הפער הוא החידוש של הפרויקט (אין מספר כזה ב־X2CT-GAN ובעבודות הדומות לו).‏ אבל עם 6 גופות, כמה בטוח המספר?‏ כתבו גם מה **לא** נבדק.

</div>

## 7. Error analysis

- The worst test cases (unit 08's best/worst, unit 07's 10 worst): what goes wrong (size, position, shape, a missing
  part) and in which images.
- Dice vs the CE angle: do they agree? (The NeurIPS 2023 benchmark says not always.)
- The re-projection check (unit 07): does the predicted bone project onto the pelvis in the X-ray?

<div dir="rtl">

> מקרה רע אחד עם תמונה מלמד יותר מעוד שורה בטבלה.‏ לכל כשל נסו לנסח השערה אחת שאפשר לבדוק.

</div>

## 8. Limitations

- **Resolution:** 64³ voxels of 5 mm. Thin structures (the pubic rami, the acetabular rim) are one or two voxels thick,
  and NSD at 1.5 mm is below one voxel. A 1 mm error is invisible here; a 5 mm error is a whole voxel.
- **Small data:** the local subset had 27 usable subjects (14 / 4 / 9); the full set ~__. With 9 test subjects the
  intervals are wide.
- **Real X-rays:** DeepFluoro has near-AP frames only, from 6 specimens, all cadavers (no breathing, different soft
  tissue).
- **Geometry mismatch:** the network was trained with one fixed camera (750 / 1,020 mm, the pelvis centred at the
  isocentre). Real frames have other poses and other detectors, and the pelvis isn't centred. The cube is placed using
  the CT, which a real patient wouldn't have.
- Simulated X-rays are cleaner than real ones: no scatter, no beam hardening, no implants (CTs with metal were removed).

<div dir="rtl">

> מגבלה כתובה טוב אומרת **איך** היא משפיעה על המספרים (מגדילה אותם?‏ מקטינה?‏ לא ידוע?), ולא רק שהיא קיימת.

</div>

## 9. Personal contribution

- The table from unit 08's contribution cell, built from every unit's `LOG.md`: who drove each TODO, who explained it
  back, the hints asked, the reveals.
- A few sentences each: what each of you led (the proposal's leads) and what each of you can explain without notes.

<div dir="rtl">

> המרצה דורש תיעוד של התרומה האישית.‏ הטבלה היא הראיה, והמשפטים הם ההסבר.‏ אם היו רמזים מדרגה 3 או reveals, כתבו אותם בגלוי.

</div>

## 10. Reproducibility (appendix)

- The output of unit 08's reproduce-everything cell, with every line ✔: which notebook or run made each file, the
  commit, the dataset build, the library versions.
- How to rerun: the repo URL, the Kaggle dataset `pelvis-xray3d-v1`, the run names.

## References

- Wasserthal et al. 2023, TotalSegmentator (Radiology: AI). Grupp et al. 2020, DeepFluoro (IJCARS). Gopalakrishnan
  and Golland 2022, DiffDRR. Tatarchenko et al. 2019 (CVPR). Shakya and Khanal 2023, the X-ray-to-3D benchmark
  (NeurIPS Datasets and Benchmarks). Ronneberger et al. 2015, U-Net. Lorensen and Cline 1987, marching cubes.
