## Removing an Object with a Mask

- Paint a removal mask (green) and, optionally, a protection mask (red)
- Before every seam: removal pixels get $-R$, protected pixels get $+P$; the DP is unchanged
- $R > 2Hm$ and $P > H(R + m)$ with $m = \max|e|$: the weights always dominate the energy
- Remove seams until the mask is empty, then insert as many to restore the size

![Input, masks, object removed, size restored (D5)](FIG/d5_object_removal_nesne_01_berkay.png)

Notes: Object removal reuses everything before it. We only change the energy: pixels to remove become very cheap, pixels to keep very expensive. The weights are not fixed numbers; they come from the energy map, so one more removal pixel always wins and one protected pixel always loses. We pick the direction needing fewer seams, carve until no green pixel is left, then insert the same number of seams.

## Protection and Dilation

- The fountain next to the obelisk is painted as protected
- One protected pixel costs more than the whole removal mask gains, so seams avoid it
- A painted mask misses shadows and soft edges; `dilate` grows it first (max filter, Week 6)
- Every seam crosses the whole image: this is rearrangement, not local inpainting

![Removal without and with the protection mask (D5)](FIG/d5_protect_nesne_01_berkay.png)

Notes: The fountain sits right next to the obelisk, so we painted it red. A seam through one red pixel costs more than the whole green mask can save, so seams go around it; the figure compares the result without and with this mask. Painted masks are never perfect, so we can grow the removal mask by a few pixels with a max filter from Week 6. Keep in mind that each seam runs through the whole picture, so the rest of the scene shifts slightly too.
