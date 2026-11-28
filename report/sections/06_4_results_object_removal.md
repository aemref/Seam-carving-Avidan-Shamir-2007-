### 6.4 Object Removal (D5)

We removed the painted object from all five `nesne` scenes, one per team member, with the default Sobel energy (`python -m experiments.d5_object_removal`). Only `nesne_01_berkay` has a protection mask, painted on the fountain next to the obelisk.

| Scene | Mask area | Direction | Seams | Removal time | With size restored |
|---|---|---|---|---|---|
| ahsen | 5.53 % | vertical | 95 | 1.40 s | 2.50 s |
| berkay | 0.45 % | vertical | 61 | 1.24 s | 2.26 s |
| emre | 1.29 % | vertical | 94 | 1.88 s | 3.46 s |
| furkan | 0.63 % | horizontal | 36 | 0.87 s | 1.57 s |
| mirza | 0.16 % | horizontal | 23 | 0.55 s | 1.01 s |

Table: D5 removal runs (`results/tables/d5_object_removal.csv`); one measurement each, on 800-pixel photos.

**Clean results.** The statue head on gravel (emre) and the small object on the stone sarcophagus (mirza) disappear without a visible trace: the textured background absorbs the removed seams.

![Statue head removed from a gravel courtyard: input, mask, 94 vertical seams removed, size restored](FIG/d5_object_removal_nesne_01_emre.png)

**Traces.** The painted mask on the earring (ahsen) misses its dark outline, which remains as a crease on the leather; growing the mask by 5 pixels removes most of it at the cost of 114 instead of 95 seams. On the obelisk (berkay), a one- to two-pixel line of its edge outside the mask also stays. When a structure crosses the path of the seams, it is bent: the horizontal seams that remove the lighter (furkan) leave a kink in the curved table edge. Restoring the size stretches low-energy regions such as lawn and bushes.

![Removal mask as painted and grown by 5 pixels (ahsen)](FIG/d5_dilate_nesne_01_ahsen.png)

**Protection.** The fountain survives with and without its mask (61 and 63 seams). The mask did matter for termination: 13 base pixels next to the pool rim cannot be reached without crossing it. Before the stopping rule of Section 2.5, the loop carved 708 seams there and finally cut 278 protected pixels.

![Removal without and with the protection mask on the fountain](FIG/d5_protect_nesne_01_berkay.png)
