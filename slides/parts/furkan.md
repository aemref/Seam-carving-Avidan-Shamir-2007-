## Enlarging: Insert the Seams You Would Remove

- Naive: find the best seam, duplicate it, repeat: the same seam wins every time
- Result: all 400 new columns pile up in one smeared band
- k-seam: take the first k seams shrinking would remove, duplicate them all at once
- New pixel = mean of the seam pixel and its right neighbour; originals untouched

![`manzara_01_furkan` +50 %: original, naive, k-seam, scaling](FIG/d4_enlarge_manzara_01_furkan.png)

Notes: Enlarging is seam removal run backwards. If we duplicate only the best seam, the copy is smooth and gets chosen again, so one band is stretched, top right. Instead we ask the shrinking code which 400 seams it would remove, already in original coordinates, and insert a new pixel next to each. The new columns spread over the whole frame; the poles still get wider, a little less than with scaling.

## Where Enlarging Bends Lines

- k-seam +50 % on the monument: the straight diagonal edge becomes a curve
- Scaling keeps it straight, because it stretches everything evenly
- +75 % in one step or in two steps (max 50 % each) looked similar here
- k-seam +50 %: 10.4 s; naive: 18.6 s; scaling: under 0.01 s

![`mimari_01_furkan` +75 %: one step, two steps, scaling](FIG/d4_staged_mimari_01_furkan.png)

Notes: Seams avoid high energy, but a long diagonal edge cannot be avoided, so many seams cross it and every crossing shifts the line by a pixel. That is why the edge bends. The paper splits enlargements above 50 % into several steps; on our photos the two-step result looked like the one-step result, so we report that honestly. Run times come from our D4 table.

## Limits: People, Lines and the Protect Mask

- Without a mask, 30–52 % of the protected person pixels were removed
- Protect mask: large positive energy, so seams go around the person: 0 pixels removed
- The cost moves to the background: the city is squeezed, the tower bends
- Dense photos at −30 % looked close to scaling: no clear failure

![`insan_01_emre` −30 %: original, seam carving, scaling, with protect mask](FIG/d7_limits_insan_01_emre.png)

Notes: The paper names two limits: content that is too dense and a layout that blocks the seams. Without help, the seams ran straight through the person and her face became narrow. With the protect mask not a single protected pixel was removed, but the seams had to go somewhere: the city behind her is squeezed and the tower on the right bends. A mask protects only what you paint.
