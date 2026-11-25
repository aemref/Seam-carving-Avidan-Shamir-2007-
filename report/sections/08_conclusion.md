<!-- DRAFT (wave 2b). Final version in wave 3, once sections 6.1-6.3 and 7 are merged.
     Every TODO below names the section whose numbers it waits for; nothing is claimed
     for an experiment that is not yet in results/. Owner: Berkay. -->

# 8. Conclusion

We implemented seam carving [1] from scratch in NumPy: an energy map built on our own correlation, the minimum seam with dynamic programming, shrinking, enlarging by seam insertion, and object removal and protection with masks. All parts share one convention, `energy_fn(img, carry)`, so the energy operator and the masks can be swapped without touching the carving loop.

**What worked.** The vectorised dynamic programming makes the method practical at our 800-pixel working size: one seam takes 20.9 ms, and the time grows linearly with the number of pixels, as the paper states (Section 4.2). Object removal cleared all five painted objects in 23 to 95 seams and under two seconds each; on textured backgrounds, such as gravel and stone, it leaves no visible trace (Section 6.4).

TODO (Ahsen, Section 6.1): which energy operator and colour mode we chose and why (D1); robustness to noise (D8).

TODO (Mirza, Section 6.2): how seam carving compared with scaling and cropping when shrinking (D2), and the energy preservation measure (D3).

TODO (Furkan, Section 6.3): enlarging with k seams against the naive repeated seam (D4).

**What did not work.** Seam carving has no notion of structure. A seam that crosses a straight or curved edge bends it, as the table edge in D5 shows. A painted mask must cover the whole object, including its outline and shadow; dilating the mask helps but costs more seams. Restoring the size after removal stretches the lowest-energy regions.

TODO (Furkan, Section 7): the limits found in D7 (dense scenes, faces, straight lines).

**Future work.** Forward energy (Rubinstein, Shamir and Avidan, 2008), which was outside our scope, measures the energy a seam inserts rather than the energy it removes and targets exactly these bent edges. The paper's optimal order of horizontal and vertical seams (its Section 4.2) and a precomputed seam index for real-time resizing (its Section 5) remain open. The mask painter could show the user which removal pixels the protection mask blocks before carving starts.
