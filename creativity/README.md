# §2.8 — Using Your Creativity

**Program:** `notch_filter_studio.html`
**To run:** double-click the file, or open it in any browser. Nothing to
install, no libraries, no plugins, no network connection required.

This file answers §2.8 (a)–(d). It is the source for the corresponding section
of the report.

---

## (a) Motivation and problem definition

§2.5 of this lab removes an interference pattern by transforming an image to the
frequency domain, zeroing a 5×5 neighbourhood at two measured peak locations,
and transforming back. Carried out as a script, the interesting part is
invisible: you read two coordinates off an axis, hard-code them, and look at a
final image. The questions that actually decide whether the method works are
never confronted —

* How do you know *which* peaks are the interference and which are the image?
* Why must peaks always be removed in pairs?
* How big should the notch be, and what does making it bigger cost?
* Why can a fence not be removed as cleanly as a set of diagonal stripes?

**Problem:** build a tool that makes the forward transform, the surgical edit
and the inverse transform directly manipulable, so that each of those questions
can be answered by experiment and measured, rather than asserted.

## (b) How the program was developed

Written as a single HTML file with **zero external dependencies** — deliberately
stricter than §2.8(d) requires, and stricter than the reference example, which
pulls three.js from a CDN and therefore fails to run without a network. Every
numerical routine is implemented from scratch:

* **A radix-2 Cooley–Tukey FFT**, in place, with a bit-reversal permutation;
  applied separably (all rows, then all columns) to give the 2-D transform.
* **Spectrum display** following §2.5(b) exactly: `S = |F|²`, shown as
  `fftshift(S.^0.1)`. Min–max normalisation proved to wash the display out, so
  the range is clipped to the 2nd–99.8th percentile.
* **Notching** with click-to-place. The display is `fftshift`ed, so a click is
  mapped back by `u = (cx + N/2) mod N`. Because the input is real, `F` is
  conjugate-symmetric, so each notch is automatically mirrored to `(N−u, N−v)`.
* **Automatic peak detection**, the non-interactive equivalent of `ginput`.

The peak detector went through three iterations, and the failures were the
instructive part:

1. **Fixed count (take the 6 strongest peaks).** Wrong by construction. A
   measured sweep showed the single-sinusoid case peaks at *1* pair (39.3 dB)
   and degrades monotonically to 24.4 dB by 8 pairs, while the harmonic fence
   peaks at *3*. The right number is a property of the image, not a constant.
2. **Threshold at a multiple of the global median magnitude.** Also wrong: an
   image spectrum spans several orders of magnitude, so ordinary structure
   clears any global cut. It notched a *clean* image down to 24.3 dB.
3. **Local prominence + energy share** — what the program does now. A candidate
   must be a 5×5 local maximum, must exceed its own surrounding annulus
   (radius 6–12) by a set factor, and must carry ≥ 0.1 % of the non-DC energy.
   The prominence test rejects image structure, which sits on smooth ridges;
   the energy test rejects the Nyquist corner and the "most prominent peak" that
   exists in every image even when there is no interference at all.

One further bug is worth recording because it is a modelling error rather than a
coding one: the first test pattern contained seven fine lines evenly spaced 4 px
apart. That is a periodic signal, so it planted genuine spikes at `v ≈ 256/4 = 64`
in the *clean* image, which no detector could distinguish from injected
interference. The detail was replaced with a deterministic pseudo-random texture,
which is broadband.

## (c) Results

Measured by the program itself, with automatic detection at default settings:

| Source | Pairs found | PSNR before | PSNR after |
|---|---|---|---|
| Clean test pattern | **0** | ∞ | ∞ (untouched) |
| Diagonal sinusoid (as `pck-int.jpg`) | **1** | 18.29 dB | **51.32 dB** |
| Periodic fence (as `primate-caged.jpg`) | **3** | 19.01 dB | **32.95 dB** |

The detector recovers exactly the right number of components in each case: none
when there is no interference, one pair for a single sinusoid, and three for the
fence — its fundamental plus the 3rd and 5th harmonics, which is precisely what
a near-square-wave occluder contains.

**Ideal vs Gaussian notch**, on the diagonal case, sweeping the radius:

| Radius | Ideal (hard zero) | Gaussian (soft) |
|---|---|---|
| 1 | 41.80 dB | **53.49 dB** |
| 2 | 38.48 dB | **44.08 dB** |
| 3 | 35.38 dB | **40.41 dB** |
| 5 | 30.39 dB | **35.95 dB** |
| 8 | 25.74 dB | **31.46 dB** |

The soft-edged notch wins at every radius, by 5–12 dB. This is the §2.5(e)
"attempt to further improve the result" instruction, quantified: a hard-zero
notch is a rectangle in the frequency domain, whose inverse transform is a sinc,
so it rings. Both profiles also degrade steadily as the notch widens, which is
the direct measurement of the cost of removing more spectrum than necessary.

A third result emerged from tuning the demo. Moving the interference to a higher
frequency, further from the image's own spectral content, raised the achievable
PSNR from 38.5 dB to 51.3 dB with no change to the algorithm. How cleanly
interference can be removed depends on how well separated it is from the signal
— which is also the honest explanation for why the fence case tops out around
33 dB and `primate-caged.jpg` cannot be fully cleaned: its harmonics lie within
the band the image itself occupies.

## (d) The program

`creativity/notch_filter_studio.html` — self-contained, immediately executable.
