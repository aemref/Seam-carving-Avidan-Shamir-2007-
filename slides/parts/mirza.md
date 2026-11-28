## Shrinking: One Seam at a Time

- Each step: energy → minimum seam (DP) → remove it, width −1
- The energy is recomputed after every seam: new neighbours, new gradients
- An index map remembers each pixel's original column, so we know which step removed it
- Height: transpose, shrink the width, transpose back

![The 200 seams removed at −25 %, drawn on the original (D2)](FIG/d2_seams_manzara_01_mirza.png)

Notes: To make an image k pixels narrower we remove k seams, one after another. After each removal, pixels that were two columns apart become neighbours, so we compute the energy again. Removed pixels lose their position, so we cut an index map together with the image; at the end we know for every original pixel which seam removed it. That is how we draw all seams on the original. Height uses the same code on the transposed image.

## Against Scaling and Cropping

- 13 photos, width −10, −25, −50 %, four methods
- Landscapes: seam carving clearly best; it removes sky and water and keeps the windmills
- People: smooth skin has low energy, so seams cross faces → a protection mask is needed
- Buildings: fine up to −25 %, straight lines break at −50 %

![Width −50 %: original, seam carving, scaling, centre crop, best crop (D2)](FIG/d2_shrink_manzara_01_mirza_50.png)

Notes: Scaling squeezes everything evenly, and cropping throws whole sides away. Seam carving removes what is least visible. On landscapes it wins clearly: all windmills keep their shape. On portraits it fails, because a face is smoother than the background, so the face gets narrower first. Buildings survive moderate shrinking but bend at half width. Seam carving is not always best; it depends on where the energy is.

## How Much Content Survives?

- Measure: mean ORIGINAL energy of the kept pixels (paper Fig. 3)
- Random removal keeps it unchanged; content-aware removal raises it
- At −50 %: per-row pixels +68 %, seams +41 %, columns +15 %, best crop +6 %
- Pixel removal wins the number but destroys the image

![Mean energy of kept pixels against the removed width (D3)](FIG/d3_energy_preservation_mimari_01_ahsen.png)

Notes: We measure how much energy stays in the image, using the energy before any removal. On both test photos the order is the same as in the paper. Removing each row's weakest pixels scores highest, but rows no longer fit together and the picture smears. Seam carving is second while keeping the image coherent, which is the paper's point. A high score does not guarantee a good image.
