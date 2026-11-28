## 6.2 Results: Shrinking

### D2: Seam carving against scaling and cropping

We reduced the width of all 13 `manzara`, `insan` and `mimari` photos by 10, 25 and 50 % with seam carving, scaling, centre crop and best crop, and the height of one landscape (`manzara_01_ahsen`) by 25 %. We judged the results by eye; this comparison has no numerical quality measure.

At −10 % all four methods look acceptable; the differences grow with the reduction. Seam carving is clearly best on the landscapes. In `manzara_01_mirza` at −50 % it keeps every windmill and house at its original proportions and removes sky, clouds and water; scaling halves the windmills, and both crops cut off part of the row. The harbour (`manzara_01_emre`), the mooring posts (`manzara_01_furkan`) and the height reduction behave the same way: the seams run through sky, water and lawn.

Seam carving fails on people. In `insan_01_emre` the smooth skin of the face has less energy than the textured city behind it, so many seams cross the face. It is already narrowed at −25 % and badly distorted at −50 %. All four `insan` photos show this to some degree, and here the crops are better because they keep the person intact. This is the case the paper handles with a protection mask (Section 2.5).

Architecture lies in between. Up to −25 % the buildings stay recognisable, but at −50 % straight lines break: the library facade in `mimari_01_ahsen` bends and the edge of the monument in `mimari_01_furkan` becomes jagged. Scaling keeps the lines straight but makes every shape thinner.

On an Intel Core Ultra 7 256V laptop the median time per seam was 86–111 ms, depending on the reduction, so −50 % took 25–46 s per photo. This is four to five times the D6 time, which was measured on a different machine.

![Width −50 % on a landscape: original, seam carving, scaling, centre crop, best crop (D2)](FIG/d2_shrink_manzara_01_mirza_50.png)

![Width −50 % on a portrait: the face is distorted by seam carving (D2)](FIG/d2_shrink_insan_01_emre_50.png)

![The 200 seams removed at −25 %, drawn on the original: many cross the low-energy face (D2)](FIG/d2_seams_insan_01_emre.png)

### D3: Energy preservation

Following paper Section 3.1, we measure the mean of the original energy map over the pixels that remain. We use the first `manzara` and the first `mimari` photo (`manzara_01_ahsen`, `mimari_01_ahsen`) and remove 0–50 % of the width in 5 % steps. The energy is not recomputed after removal, so random removal leaves the mean unchanged on average. A higher value means that more high-energy content survived.

On both photos and at every step the order matches paper Fig. 3: per-row pixel removal > seam carving > column removal > best crop. At −50 % the mean rises by 71 / 68 % for pixel removal, 42 / 41 % for seam carving, 16 / 15 % for column removal and 12 / 6 % for best crop (`manzara` / `mimari`). Pixel removal is highest by construction, because each row loses its own lowest-energy pixels. The −30 % images show the price: neighbouring rows lose different pixels and the facade dissolves into horizontal smears. Seam carving keeps the second-highest energy while each seam removes one connected path, so the image stays coherent. On `mimari_01_ahsen` the best crop first falls below the original mean (−2.2 % at 15 %): the energy is spread over the whole width, so every window loses high-energy edges at its sides.

The measure counts kept energy, not visual quality. The faces in D2 show that seam carving can keep much energy and still distort a shape.

![Mean original energy of the kept pixels against the removed width, four strategies (D3)](FIG/d3_energy_preservation_mimari_01_ahsen.png)

![Width −30 %: best crop, column removal, seam carving and per-row pixel removal (D3, cf. paper Fig. 2)](FIG/d3_strategies_mimari_01_ahsen.png)
