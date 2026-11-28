## 7. Limitations (D7)

**Question.** Where does seam carving fail, and what does a protect mask fix? Avidan and Shamir name two limits (§6): content too dense to leave low-energy paths, and a layout that does not let seams go around important parts. All ten `insan`, `mimari` and `yogun` photos were narrowed by 30% with seam carving and with scaling; the four `insan` photos were also carved with their protect mask (`apply_mask_weights`). Numbers come from `results/tables/d7_limits.csv`.

### 7.1 Layout: people and straight lines

![Original, seam carving, scaling and seam carving with protect mask, -30%](FIG/d7_limits_insan_01_emre.png)

Without a mask, the seams of every `insan` photo run through the person: they removed 30.4% (ahsen), 42.1% (berkay), 51.6% (emre) and 35.3% (mirza) of the pixels inside the protect mask, and in three of the four photos the face became visibly narrow (clearly for emre and berkay); in `insan_01_mirza` the seams went mostly through the body. In `mimari_01_furkan` many seams cross the monument, and its straight diagonal edge became stepped; in `mimari_01_ahsen` the library facade became noticeably narrower.

![Removed seams without and with the protect mask](FIG/d7_seams_insan_01_emre.png)

### 7.2 What the protect mask fixes, and what it does not

With the mask, no protected pixel was removed in any of the four photos, and people keep their proportions. The cost moves elsewhere: all seams must pass through the background, so in `insan_01_emre` the city behind the person is squeezed and the tall building at the right edge becomes thin and bent. A mask protects only what was painted; nothing protects straight lines in the `mimari` photos.

### 7.3 Dense content (`yogun`)

![Original, seam carving -30%, scaling -30%](FIG/d7_limits_yogun_01_furkan.png)

At −30%, our two `yogun` photos did not fail as clearly as expected: seams passed through the side windows and decorated side panels, and the results look close to scaling. With only two dense photos and a single reduction, this does not show where the method starts to break on dense content.
