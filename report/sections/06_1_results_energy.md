## 6.1 Energy operators and noise (D1, D8)

### D1: which derivative operator?

We reduced the width of three photos (`manzara`, `mimari`, `insan`; 800 × 600 and 800 × 557) by 25 % with each of the five operators, and with three Sobel variants: per-channel sum, Gaussian pre-blur (σ = 1.5) and L2 norm. As a numerical measure we use the mean original Sobel energy of the pixels that are kept [1, §3.1]: the higher it is compared with the mean over the whole image, the more content survived. The numbers are in `results/tables/d1_energy.csv`.

Sobel and Scharr keep the most energy on all three photos (+15.8 % to +17.7 % over the image mean), forward difference and Roberts follow (+11.2 % to +15.0 %), and the Laplacian keeps the least (+5.8 % to +10.7 %). The per-channel sum and the L2 norm change Sobel's result by at most 0.3 percentage points; the pre-blur lowers it to +10.4 % to +13.7 %. This measure is computed with Sobel energy, so it favours Sobel and Scharr by construction; the gap to the Laplacian is large enough not to depend on that.

Visually the five results are close (Figure 1). Every operator keeps the house and the willow in `manzara` and the person in `insan`, and every operator bends the straight right edge of the library façade in `mimari`. On our laptop one Sobel energy map of an 800 × 600 photo took 24–43 ms; the per-channel sum took 65–84 ms and the pre-blur 77–119 ms. These timings were noisy between runs.

We keep **Sobel, grey, L1, no blur** as the default: it is tied for the best score with Scharr, it is the operator from the lecture, and the variants cost two to four times more time without a better result.

![Width −25 % on `mimari_01_ahsen` with each operator](FIG/d1_carved_mimari_01_ahsen.png)

### D8: noise and pre-filters

We added Gaussian noise (σ = 15) and salt-and-pepper noise (5 % of the pixels) to `manzara_01_ahsen`, removed 160 seams (−20 %) from the noisy image and compared the removed pixels with those of the clean image (intersection over union, `results/tables/d8_noise.csv`). The pre-filters are applied only to the energy; the carved image stays noisy.

Noise changes the seams a lot: the overlap with the clean seams drops to 0.25 (Gaussian) and 0.23 (salt-and-pepper). With salt-and-pepper noise every noise pixel becomes a high-energy dot, and the seams spread evenly over the sky instead of using the smooth regions (Figure 2). A 3 × 3 median filter removes the dots almost completely and doubles the overlap to 0.50; Gaussian pre-blur does not help here (0.20). For Gaussian noise the order is reversed but the effect is small: pre-blur gives 0.29 and the median filter 0.26. This matches the lecture: the median filter suits impulse noise and averaging suits Gaussian noise. Even the best case stays far from 1, because in a flat sky many different seams are almost equally cheap and the noise decides between them.

![Salt-and-pepper noise: energy maps (top) and removed seams (bottom) for clean, noisy, noisy + Gaussian pre-blur and noisy + median](FIG/d8_noise_saltpepper.png)
