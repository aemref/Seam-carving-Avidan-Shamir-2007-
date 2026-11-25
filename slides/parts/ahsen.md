## Resizing Without Losing Content

- Scaling squeezes everything; cropping throws the edges away
- Seam carving removes a connected path of low-energy pixels, one per row
- Repeat k times: width shrinks by k, important content stays
- We implemented it from scratch in NumPy: energy, DP, shrink, enlarge, object removal

![Width −25 % on `manzara_01_ahsen` with five energy operators: the house and the willow stay](FIG/d1_carved_manzara_01_ahsen.png)

Notes: An image often has to fit a different aspect ratio. Scaling makes people thin, cropping cuts them off. Seam carving removes a path from top to bottom that bends around important content and goes through flat areas like sky or water. Here the photo lost a quarter of its width with five different energies, and the house and the willow keep their shape. Everything you will see is our own NumPy code.

## The Energy Map Is a Gradient

- $e = |\partial I/\partial r| + |\partial I/\partial c|$: the paper's Eq. 1 is $M \approx |g_x| + |g_y|$ from Week 5, slide 56
- Derivatives with our own `correlate2d`: sum of products, loop over the 9 kernel weights only
- Kernels sum to 0, so flat regions get zero energy and seams pass there
- Float64 first (uint8 wraps), reflect padding (zero padding makes a fake edge)

![Energy maps of `mimari_01_ahsen`: forward difference, Roberts, Sobel, Scharr, Laplacian](FIG/d1_energy_maps_mimari_01_ahsen.png)

Notes: The energy tells us which pixels are cheap to remove. The paper's formula is exactly the gradient magnitude approximation from Week 5. We wrote the filtering ourselves and checked it against OpenCV to nine decimals. Because derivative kernels sum to zero, the sky is black here and the façade is bright, so seams avoid the building. Two traps: uint8 subtraction wraps around, and zero padding turns the image border into an edge.

## Which Operator, and What About Noise?

- Sobel and Scharr keep the most energy (+16 % to +18 %), Laplacian the least (+6 % to +11 %)
- Default: Sobel, grey, L1; colour sum and blur cost 2–4× more time for no gain
- Salt-and-pepper noise scatters the seams; a 3 × 3 median filter doubles the overlap with the clean seams (0.23 → 0.50)
- Gaussian noise: pre-blur helps a little (0.25 → 0.29)

![Salt-and-pepper noise: energy and seams for clean, noisy, Gaussian pre-blur and median pre-filter](FIG/d8_noise_saltpepper.png)

Notes: D1 compared five operators on three photos. Sobel and Scharr keep the most content; the Laplacian, a second derivative, keeps the least, so we use Sobel. D8 adds noise from Week 6. Every salt-and-pepper dot looks like an edge and the seams scatter across the sky. A median filter before the energy cleans the map and brings the seams back; averaging does not. That is the lecture's rule: median for impulse noise.
