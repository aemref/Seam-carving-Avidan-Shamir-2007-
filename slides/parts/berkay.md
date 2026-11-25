## Removing an Object with a Mask

- Paint a removal mask (green) and, optionally, a protection mask (red)
- Before every seam: removal pixels get $-R$, protected pixels get $+P$; the DP is unchanged
- $R > 2Hm$ and $P > H(R + m)$ with $m = \max|e|$: the weights always dominate the energy
- Obelisk: 61 vertical seams in 1.2 s, then 61 seams inserted to restore the size

![Input, masks, object removed, size restored (D5)](FIG/d5_object_removal_nesne_01_berkay.png)

Notes: Object removal reuses everything before it. We only change the energy: pixels to remove become very cheap, pixels to keep very expensive. The weights are not fixed numbers; they come from the energy map, so one more removal pixel always wins and one protected pixel always loses. We pick the direction needing fewer seams, carve until no green pixel is left, then insert the same number of seams.

## Protection Wins

- The fountain next to the obelisk is painted as protected
- 13 base pixels touch the pool rim: no seam reaches them without crossing it
- First version: 708 seams, the image nearly gone, 278 protected pixels cut
- Fix: stop when a seam removes no green pixel; keep those 13 pixels

![Removal without and with the protection mask (D5)](FIG/d5_protect_nesne_01_berkay.png)

Notes: Protection costs more than all the removal can gain, so seams refuse to touch the fountain. Our first version did not notice when the last green pixels were trapped behind the red rim: it kept carving 708 seams until the image was a narrow strip and even cut the fountain. Now, when the cheapest seam removes no green pixel, we stop and keep those few pixels. Protection wins, as the user asked.

## Where Traces Remain

- Five scenes, 23 to 95 seams, under 2 s each; gravel and stone hide the seams
- A mask that misses the outline leaves it: the earring's dark edge becomes a crease
- Growing the mask by 5 pixels (max filter, Week 6) removes most of it: 114 seams
- Seams bend structures they cross; every seam runs through the whole image

![Removal mask as painted and grown by 5 pixels (D5)](FIG/d5_dilate_nesne_01_ahsen.png)

Notes: On textured backgrounds the result is clean. Problems come from the mask and from structure. If the dark outline of the earring is not painted, it stays behind as a crease; dilating the mask by five pixels fixes most of it at the cost of 19 more seams. A curved table edge crossed by the seams gets a kink. This is not inpainting: the whole picture is rearranged.
