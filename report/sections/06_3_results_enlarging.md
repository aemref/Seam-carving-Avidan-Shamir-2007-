## 6.3 Enlarging (D4)

**Question.** Why does enlarging need the k-seam method, and how does it compare to scaling? The experiment reproduces Fig. 8 of the paper on `manzara_01_furkan` and `mimari_01_furkan` (both 800×600). Each photo is widened by 50% with the naive method (find the best seam, duplicate it, repeat), with `enlarge_width` (k-seam, one step) and with bicubic scaling.

![Original, naive repeated best seam +50%, k-seam +50%, scaling +50%](FIG/d4_enlarge_manzara_01_furkan.png)

**Naive vs. k-seam.** The naive method fails exactly as the paper predicts: it inserts the same low-energy path again and again, so all 400 new columns form one wide band of smeared pixels across the buildings and the water. The k-seam result has no such band; the 400 seams are spread over the whole frame. The wooden poles are widened too, the left group somewhat less than by scaling (89 → 111 px with k-seam, 133 px with scaling, measured on one row).

![Chosen seams (red), inserted pixels of the k-seam and naive runs (green)](FIG/d4_seams_manzara_01_furkan.png)

**Where it fails.** On `mimari_01_furkan` the k-seam result bends the monument's straight diagonal edge into a curve, because many seams cross the monument; scaling keeps the line straight. Widening by 75% in one step (`max_step = 1.0`) and in two steps (`max_step = 0.5`) gave visually similar results on both photos; in both cases the diagonal edge was bent. `amplify_content` (scale 1.3) enlarged the monument, but cut its top wedge and left the diagonal edge jagged (`d4_amplify_mimari_01_furkan.png`).

![+75%: one step, two steps, scaling](FIG/d4_staged_mimari_01_furkan.png)

**Run time** (`results/tables/d4_enlarge.csv`; cloud VM, Intel Xeon 2.1 GHz, NumPy 2.5.3):

| Method | manzara (s) | mimari (s) |
|---|---|---|
| Naive +50% | 18.6 | 18.3 |
| k-seam +50% | 10.4 | 10.3 |
| k-seam +75%, one step | 13.9 | 13.4 |
| k-seam +75%, two steps | 19.6 | 19.4 |
| `amplify_content` 1.3 | 19.3 | 19.7 |
| Scaling | < 0.01 | < 0.01 |
