# Lab 1 — Delivery Plan

Source of truth: `Lab1_manual.pdf` (SC4061/CE4003/CZ4003). Four graded
deliverables are named in §3 of the manual:

| # | Deliverable | Manual ref | Target path |
|---|---|---|---|
| D1 | Professional report: all questions answered, results embedded, key source code embedded, personal commentary | §3a | `report/Lab1_Report.pdf` |
| D2 | Complete source code in **ONE** `.m` file | §3b | `matlab/Lab1.m` |
| D3 | The §2.8 creativity program, immediately executable, no install / no extra libraries | §3c | `creativity/<name>.html` |
| D4 | (supporting) Reproducible figure set | — | `results/` |

## Decisions taken

| Question | Decision |
|---|---|
| Primary graded code artefact | **`matlab/Lab1.m`** — one `.m` file, matching §3b literally. Verified by real execution under Octave here. Python is a figure-rendering mirror, not a submitted file. |
| §2.8 topic | **Interactive frequency-domain notch filter** — directly extends §2.5. |
| Report format | **Markdown → PDF**: `report/report.md` is the working copy, `report/Lab1_Report.pdf` the submission. |

## Strategy

The container has no MATLAB licence, so the work is done on two legs:

* **GNU Octave 8.4 + `octave-image` is installed and verified here**, and it
  supplies every toolbox function the manual calls for — `rgb2gray`, `imhist`,
  `histeq`, `conv2`, `medfilt2`, `fft2`/`ifft2`, `maketform`, `imtransform`.
  This means `matlab/Lab1.m` is **actually executed and its numbers verified**
  before submission, rather than written blind. Figure rendering falls back to
  the gnuplot toolkit (Qt is unavailable headless), which is adequate for
  verification but not for report plates.
* **Python (`python/lab1.py`, numpy + scipy + PIL + matplotlib)** mirrors the
  same operations and renders the publication-quality figures that go into the
  report.

Every numeric claim in the report is taken from an actual run of both legs,
never asserted. Any place the two disagree is investigated, not averaged.

Known Octave/MATLAB divergences to guard in `Lab1.m`:

* Octave's `histeq` returns **double**; MATLAB's returns the input class. The
  script casts explicitly so both behave identically.
* Octave's gnuplot toolkit ignores some figure styling; styling is kept minimal
  and portable.

Semantic mapping used to keep the two implementations in agreement:

| Manual / MATLAB | Python mirror |
|---|---|
| `rgb2gray` | ITU-R BT.601 luma, `0.299R + 0.587G + 0.114B`, rounded, uint8 |
| `imhist(P,n)` | uniform binning over `[0,255]` |
| `histeq(P,255)` | CDF mapping over 256 levels, MATLAB's `histeq` grey-level assignment |
| `conv2(...,'same')` | `scipy.signal.convolve2d(..., mode='same', boundary='fill')` |
| `medfilt2(P,[n n])` | `scipy.ndimage.median_filter(..., mode='constant')` |
| `fft2` / `fftshift` / `ifft2` | `numpy.fft` equivalents |
| `A\v`, `maketform`/`imtransform` | `numpy.linalg.lstsq` on the manual's eq. (*) + explicit inverse-warp resampling |

## Section-by-section work plan

### §2.1 Contrast stretching — `mrt-train.jpg`
Read, `whos` check (320×443×3 uint8), `rgb2gray`, report min/max, linear stretch
to [0,255] in two lines, verify new min/max are exactly 0 and 255.
*Figures:* original grey, stretched, before/after histograms.
*Questions:* effect on appearance; why two lines suffice.

### §2.2 Histogram equalisation — `mrt-train.jpg`
`imhist` at 10 and 256 bins; `histeq(P,255)`; re-plot at both bin counts;
re-run equalisation on `P3` and compare.
*Questions:* (a) difference between 10- and 256-bin views; (b) are the
histograms actually equalised, similarities/differences; (c) why the second
pass changes (almost) nothing — the discrete monotone CDF mapping is idempotent
up to rounding; prove it numerically by reporting `max|P4 - P3|`.

### §2.3 Linear spatial filtering — `lib-gn.jpg`, `lib-sp.jpg`
Build two 5×5 Gaussian kernels (σ = 1.0 and σ = 2.0) from the manual's formula,
normalise to sum 1, view as 3-D `mesh`. Filter both noisy images with `conv2`.
*Questions:* effectiveness on Gaussian noise; blur-vs-noise trade-off;
Gaussian-noise vs speckle-noise performance.
*Quantitative backing:* per-image noise σ estimate and edge-sharpness metric so
the trade-off claim is measured rather than eyeballed.

### §2.4 Median filtering — same two images
`medfilt2` at 3×3 and 5×5, repeating §2.3 (b)–(e).
*Questions:* Gaussian vs median filtering per noise type, and the trade-offs.
Expected and to be verified: median wins decisively on speckle/salt-and-pepper,
Gaussian is competitive on additive Gaussian noise, median preserves edges but
erases fine texture at 5×5.

### §2.5 Frequency-domain interference suppression — `pck-int.jpg`, `primate-caged.jpg`
`fft2` → power spectrum → `imagesc(fftshift(S.^0.1))`; locate the two symmetric
peaks *without* `fftshift` and record their exact (row, col); zero a 5×5
neighbourhood around each in **F**; re-display the spectrum; `ifft2` and compare.
Then an improvement pass (wider//soft-edged notch, and zeroing the residual
ridge rather than just the two peaks) with the improvement quantified.
Repeat the whole idea on `primate-caged.jpg`: the fence is a quasi-periodic
structure, so several harmonic pairs must be notched; the honest expected
outcome is partial removal with ringing, which the report will show and explain.
*Note:* `ginput` is interactive, so peak coordinates are found programmatically
(argmax outside a DC exclusion disc) and hard-coded, making the run repeatable.

### §2.6 Undoing perspective distortion — `book.jpg`
Four book corners → target A4 rectangle (210 × 297, 1 px = 1 mm), set up the
8×8 system of eq. (*), solve, reshape to `U`, verify `U·[X;Y;1]` reproduces the
target corners, then warp.
*Questions:* is the result as expected; comment on quality and causes
(resampling/interpolation, corner-click precision, lens distortion, the book not
being perfectly planar, loss of resolution where the original was foreshortened).
*(f)* Locate the pink rectangular "computer screen" region between "Nanyang" and
"2001" — plan: colour-threshold in HSV, morphological clean-up, largest
connected component, report its bounding box in millimetres.
*Note:* corner coordinates likewise hard-coded after being picked once, with the
`ginput` call retained but commented, so the script is non-interactive.

### §2.7 Two perceptrons
Implement Algorithm 1 and Algorithm 2 **exactly as defined in the lecture
slides**, run both on the slides' worked example, compare convergence
(iteration count, final weights, decision boundary) and comment.
*Blocked:* needs the slide pages — see "Inputs still needed" below.

### §2.8 Creativity component
Single self-contained HTML file, opens in any browser, no install, no build
step — the same shape as the reference `gaussian low pass filter pipeline.html`
(three.js pulled from a CDN, everything else inline). Written up as motivation →
problem statement → how it was developed → results, per (a)–(d).
**Topic chosen: an interactive frequency-domain notch filter.** The program
loads an image (or its own built-in test pattern carrying a synthetic
diagonal interference, mirroring `pck-int.jpg`), shows the live `fftshift`ed
power spectrum scaled by `S.^0.1` exactly as §2.5(b) prescribes, lets the user
click spectral peaks to notch them — auto-mirroring each notch to its
conjugate-symmetric partner — and re-renders the inverse transform live, with
a PSNR readout against the clean source.

Constraint honoured: **zero external dependencies**. Pure vanilla JavaScript
and Canvas, including a hand-written radix-2 Cooley-Tukey FFT, so the file
satisfies §2.8(d)'s "no installation and other libraries and plugins" more
strictly than the reference example does (that one pulls three.js from a CDN
and therefore needs a network connection).

## Inputs still needed

1. **The course image files.** Required by name: `mrt-train.jpg`, `lib-gn.jpg`,
   `lib-sp.jpg`, `pck-int.jpg`, `primate-caged.jpg`, `book.jpg`. The image
   pasted into chat is 443×320 RGB, which matches `mrt-train.jpg`'s stated
   320×443×3 — but a chat paste is re-encoded, and §2.1 turns on the *exact*
   min/max intensities of the original JPEG, so the original file is needed.
   Simplest: upload the whole folder into `images/`.
2. **Lecture-slide pages defining perceptron Algorithm 1 and Algorithm 2**, plus
   the worked example the manual says to reproduce. Without these, §2.7 cannot
   be done to spec — there are several textbook variants and the grader will be
   comparing against the slides' exact formulation.
3. **Name and matriculation number**, for the report title page and for the
   `"xxx (your name)'s comment"` annotations the manual asks for, and which
   course code applies (SC4061 / CE4003 / CZ4003).
4. **MATLAB access**, confirm yes/no. If yes, `matlab/Lab1.m` gets a real run
   before submission; if no, it ships validated against the Python mirror and,
   where the toolbox overlap allows, GNU Octave.

## Status

- [x] Manual parsed, all requirements enumerated
- [x] Toolchain verified: numpy 2.4, scipy 1.17, matplotlib 3.11, pillow 12
- [x] GNU Octave 8.4 + image package installed; full end-to-end probe passed
  (contrast stretch, `histeq`, `conv2`, `medfilt2`, `fft2`, `maketform`/`imtransform`, headless figure export)
- [x] Repository scaffolded
- [ ] §2.1  · [ ] §2.2 · [ ] §2.3 · [ ] §2.4 · [ ] §2.5 · [ ] §2.6 · [ ] §2.7 · [ ] §2.8
- [ ] D2 `matlab/Lab1.m` consolidated
- [ ] D1 report written and exported
