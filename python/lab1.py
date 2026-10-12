#!/usr/bin/env python3
"""
SC4061 / CE4003 / CZ4003 Computer Vision -- Lab 1
Executable Python mirror of matlab/Lab1.m.

Each section function mirrors, operation for operation, the MATLAB calls
prescribed by the lab manual, and writes its figures into results/secNN/.

Usage:
    python3 python/lab1.py --all
    python3 python/lab1.py --section 2.3
"""

import argparse
import pathlib
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
IMAGES = ROOT / "images"
RESULTS = ROOT / "results"


# --------------------------------------------------------------------------
# MATLAB-equivalent primitives
# --------------------------------------------------------------------------

def imread_gray(name):
    """imread + rgb2gray, matching MATLAB's ITU-R BT.601 luma and rounding."""
    from PIL import Image

    path = IMAGES / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Place the course images from NTULearn "
            f"(Course Contents/Laboratory Material/Images) into {IMAGES}/."
        )
    im = np.asarray(Image.open(path))
    if im.ndim == 3:
        # MATLAB rgb2gray: 0.2989 R + 0.5870 G + 0.1140 B, rounded to uint8.
        w = np.array([0.2989, 0.5870, 0.1140])
        im = np.round(im[:, :, :3] @ w).clip(0, 255).astype(np.uint8)
    return im


def gaussian_kernel(dim, sigma):
    """The manual's h(x,y) = 1/(2*pi*sigma^2) * exp(-(x^2+y^2)/(2*sigma^2)),
    sampled on a dim x dim grid centred at zero, then normalised to sum 1."""
    r = (dim - 1) / 2.0
    x, y = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
    h = np.exp(-(x**2 + y**2) / (2.0 * sigma**2)) / (2.0 * np.pi * sigma**2)
    return h / h.sum()


def savefig(fig, section, name):
    out = RESULTS / f"sec{section.replace('.', '')}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


# --------------------------------------------------------------------------
# Section 2.1 -- Contrast stretching
# --------------------------------------------------------------------------

def section_2_1(show=True):
    print("2.1 Contrast stretching")
    from PIL import Image
    raw = np.asarray(Image.open(IMAGES / "mrt-train.jpg"))
    print(f"  whos Pc: {raw.shape} {raw.dtype}  "
          f"({'colour' if raw.ndim == 3 else 'greyscale'} as read by imread)")
    if raw.ndim == 3:
        ch_equal = (raw[:, :, 0] == raw[:, :, 1]).all() and \
                   (raw[:, :, 1] == raw[:, :, 2]).all()
        print(f"  the three channels are "
              f"{'identical, so rgb2gray is exact here' if ch_equal else 'genuinely different'}")

    P = imread_gray("mrt-train.jpg")
    lo, hi = int(P.min()), int(P.max())
    print(f"  min(P(:)) = {lo}, max(P(:)) = {hi}")

    # The two lines of the manual's step (d): subtract, then scale.
    # Done in double and cast back, which is the alternative the manual allows.
    P2 = (P.astype(np.float64) - lo) * (255.0 / (hi - lo))
    P2 = np.round(P2).clip(0, 255).astype(np.uint8)
    print(f"  after stretching: min = {P2.min()}, max = {P2.max()}  "
          f"(expected 0 and 255)")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes[0, 0].imshow(P, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title(f"Original greyscale P\nmin = {lo}, max = {hi}")
    axes[0, 1].imshow(P2, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"Contrast stretched P2\nmin = {P2.min()}, max = {P2.max()}")
    for ax in (axes[0, 0], axes[0, 1]):
        ax.axis("off")
    axes[1, 0].hist(P.ravel(), bins=256, range=(0, 255), color="#475569")
    axes[1, 0].set_title("Histogram of P (256 bins)")
    axes[1, 1].hist(P2.ravel(), bins=256, range=(0, 255), color="#2563eb")
    axes[1, 1].set_title("Histogram of P2 (256 bins)")
    for ax in (axes[1, 0], axes[1, 1]):
        ax.set_xlim(0, 255); ax.set_xlabel("Intensity"); ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, "2.1", "contrast_stretch")
    return P, P2


# --------------------------------------------------------------------------
# Section 2.2 -- Histogram equalisation
# --------------------------------------------------------------------------

def matlab_histeq(I, n=255):
    """MATLAB's histeq(I, n) for a uint8 greyscale image.

    Octave's histeq is NOT MATLAB-compatible -- its own source carries a FIXME
    saying so -- so this reimplements the toolbox algorithm directly: build the
    cumulative histogram of the image and of the flat n-bin target, then for
    each of the 256 input levels choose the output level that minimises the
    tolerance-adjusted difference between the two.
    """
    m = 256
    nn = np.bincount(I.ravel(), minlength=m).astype(np.float64)
    hgram = np.ones(n) * (I.size / n)
    cumd = np.cumsum(nn)
    cum = np.cumsum(hgram)
    tol = np.ones((n, 1)) * np.minimum(
        np.concatenate([nn[:m - 1], [0]]),
        np.concatenate([[0], nn[1:]])) / 2.0
    err = (cum[:, None] * np.ones((1, m)) - np.ones((n, 1)) * cumd[None, :]) + tol
    err[err < -I.size * np.sqrt(np.finfo(float).eps)] = I.size
    T = np.argmin(err, axis=0) / (n - 1)
    return np.round(T[I] * 255).astype(np.uint8)


def section_2_2():
    print("2.2 Histogram equalisation")
    P = imread_gray("mrt-train.jpg")

    P3 = matlab_histeq(P, 255)          # first equalisation
    P4 = matlab_histeq(P3, 255)         # (c) re-run on the result

    def spread(x, bins):
        h, _ = np.histogram(x, bins=bins, range=(0, 255))
        return h

    for label, img in (("P  (original)", P), ("P3 (equalised)", P3),
                       ("P4 (equalised twice)", P4)):
        h10 = spread(img, 10)
        print(f"  {label:22s} mean = {img.mean():7.3f}  "
              f"10-bin counts min/max = {h10.min():6d}/{h10.max():6d}  "
              f"ratio = {h10.max() / max(h10.min(), 1):6.2f}")

    diff = np.abs(P4.astype(int) - P3.astype(int))
    print(f"  max|P4 - P3| = {diff.max()}   mean|P4 - P3| = {diff.mean():.4f}")
    print(f"  pixels changed by the second pass: {100 * (diff > 0).mean():.2f} %")

    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for row, (img, name) in enumerate([(P, "P  original"),
                                       (P3, "P3  histeq once"),
                                       (P4, "P4  histeq twice")]):
        axes[row, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title(name); axes[row, 0].axis("off")
        axes[row, 1].hist(img.ravel(), bins=10, range=(0, 255),
                          color="#475569", edgecolor="k")
        axes[row, 1].set_title(f"{name} -- 10 bins")
        axes[row, 2].hist(img.ravel(), bins=256, range=(0, 255), color="#2563eb")
        axes[row, 2].set_title(f"{name} -- 256 bins")
        for c in (1, 2):
            axes[row, c].set_xlim(0, 255); axes[row, c].grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, "2.2", "histogram_equalisation")
    return P3, P4


# --------------------------------------------------------------------------
# Section 2.3 (a) -- Gaussian averaging filters
# Needs no input image, so it runs before the course images arrive.
# --------------------------------------------------------------------------

def section_2_3a():
    print("2.3(a) Gaussian averaging filters")
    specs = [(5, 1.0), (5, 2.0)]
    kernels = {}
    for dim, sigma in specs:
        h = gaussian_kernel(dim, sigma)
        kernels[sigma] = h
        print(f"  dim={dim} sigma={sigma}: sum={h.sum():.6f} "
              f"min={h.min():.6f} max={h.max():.6f}")
        print("   " + np.array2string(h, precision=5, suppress_small=True)
              .replace("\n", "\n   "))

        r = (dim - 1) / 2.0
        x, y = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
        fig = plt.figure(figsize=(5.5, 4.2))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, h, cmap="viridis", edgecolor="k", linewidth=0.3)
        ax.set_title(f"Gaussian filter, {dim}x{dim}, $\\sigma$ = {sigma}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("h(x,y)")
        savefig(fig, "2.3", f"gaussian_mesh_sigma{sigma:g}")
    return kernels



# --------------------------------------------------------------------------
# Image-quality measures used to back the §2.3 / §2.4 trade-off discussion
# with numbers instead of impressions. Neither noisy image has a clean
# reference, so PSNR is unavailable and these are reference-free measures.
# --------------------------------------------------------------------------

def estimate_noise_sigma(img):
    """Immerkaer's reference-free estimate of additive noise sigma.

    Convolves with a kernel that annihilates locally-linear image content, so
    what survives is dominated by noise.
    """
    from scipy.signal import convolve2d
    M = np.array([[1., -2., 1.], [-2., 4., -2.], [1., -2., 1.]])
    H, W = img.shape
    s = np.sum(np.abs(convolve2d(img.astype(np.float64), M, mode="valid")))
    return s * np.sqrt(0.5 * np.pi) / (6.0 * (W - 2) * (H - 2))


def mean_gradient(img):
    """Mean Sobel gradient magnitude -- a proxy for retained edge sharpness."""
    from scipy.ndimage import sobel
    f = img.astype(np.float64)
    return float(np.hypot(sobel(f, axis=0), sobel(f, axis=1)).mean())


def impulse_fraction(img):
    """Percentage of pixels pinned at 0 or 255 -- the salt-and-pepper signature."""
    return 100.0 * ((img == 0) | (img == 255)).mean()


def _metrics_row(label, img):
    return (label, estimate_noise_sigma(img), mean_gradient(img),
            impulse_fraction(img))


def _print_metrics(rows):
    # Only the unfiltered row's sigma is meaningful: Immerkaer's estimator
    # assumes uncorrelated noise, which a smoothed output violates. Mean
    # gradient is comparable across all rows; impulse % is comparable only
    # within the median column, since convolution spreads impulses rather
    # than removing them.
    print(f"    {'variant':<26s} {'noise sigma':>11s} {'mean grad':>10s} "
          f"{'impulse %':>10s}")
    for i, (label, sig, grad, imp) in enumerate(rows):
        sig_s = f"{sig:11.3f}" if i == 0 else f"{'(n/a)':>11s}"
        print(f"    {label:<26s} {sig_s} {grad:10.2f} {imp:10.3f}")


# --------------------------------------------------------------------------
# Section 2.3 (b)-(e) -- Linear spatial filtering with the Gaussian kernels
# --------------------------------------------------------------------------

def _filter_and_plot(section, name, title, filters, fname):
    """Apply a set of named filters to one image and emit a comparison plate."""
    img = imread_gray(name)
    rows = [_metrics_row("unfiltered", img)]
    outs = [("unfiltered", img)]
    for flabel, fn in filters:
        out = fn(img)
        outs.append((flabel, out))
        rows.append(_metrics_row(flabel, out))

    print(f"  {title}")
    _print_metrics(rows)

    n = len(outs)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4))
    for ax, (flabel, im) in zip(axes, outs):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(flabel, fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    savefig(fig, section, fname)
    return rows


def _conv_filter(h):
    from scipy.signal import convolve2d

    def fn(img):
        # conv2(double(img), h, 'same'); cast back to uint8 as imshow expects
        out = convolve2d(img.astype(np.float64), h, mode="same", boundary="fill")
        return np.round(out).clip(0, 255).astype(np.uint8)
    return fn


def _median_filter(size):
    from scipy.ndimage import median_filter

    def fn(img):
        return median_filter(img, size=size, mode="constant", cval=0.0)
    return fn


def section_2_3():
    print("2.3 Linear spatial filtering")
    kernels = section_2_3a()                       # (a) build + plot the filters
    filters = [(f"Gaussian 5x5, sigma = {s}", _conv_filter(h))
               for s, h in sorted(kernels.items())]

    print("\n  (b),(c) additive Gaussian noise -- lib-gn.jpg")
    gn = _filter_and_plot("2.3", "lib-gn.jpg",
                          "lib-gn.jpg: Gaussian filtering of Gaussian noise",
                          filters, "gaussian_on_gaussian_noise")

    print("\n  (d),(e) speckle noise -- lib-sp.jpg")
    sp = _filter_and_plot("2.3", "lib-sp.jpg",
                          "lib-sp.jpg: Gaussian filtering of speckle noise",
                          filters, "gaussian_on_speckle_noise")
    return gn, sp


# --------------------------------------------------------------------------
# Section 2.4 -- Median filtering, repeating 2.3 (b)-(e)
# --------------------------------------------------------------------------

def section_2_4():
    print("2.4 Median filtering")
    filters = [("median 3x3", _median_filter(3)), ("median 5x5", _median_filter(5))]

    print("\n  additive Gaussian noise -- lib-gn.jpg")
    gn = _filter_and_plot("2.4", "lib-gn.jpg",
                          "lib-gn.jpg: median filtering of Gaussian noise",
                          filters, "median_on_gaussian_noise")

    print("\n  speckle noise -- lib-sp.jpg")
    sp = _filter_and_plot("2.4", "lib-sp.jpg",
                          "lib-sp.jpg: median filtering of speckle noise",
                          filters, "median_on_speckle_noise")
    return gn, sp


# --------------------------------------------------------------------------
# Controlled filter comparison (supports the §2.3(e) / §2.4 questions)
#
# Neither lib-gn.jpg nor lib-sp.jpg comes with a clean original, so no
# reference-based measure can be computed on them directly, and the obvious
# reference-free substitutes are both misleading here:
#
#   * Immerkaer's sigma estimator assumes spatially uncorrelated noise. A
#     Gaussian low-pass leaves correlated residual noise, so the estimator
#     reads far too low and would wrongly suggest Gaussian filtering beats
#     median filtering by an order of magnitude.
#   * Counting pixels still pinned at 0/255 flatters Gaussian filtering too:
#     convolution does not remove an impulse, it spreads it over the kernel
#     footprint, so the pixel leaves the 0/255 count while the damage remains.
#
# So the quantitative claim is made on a controlled experiment instead: take a
# clean image, add noise of each type with known parameters, and measure PSNR
# against the known original. The lab's own images still get the qualitative
# treatment the manual asks for.
# --------------------------------------------------------------------------

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def add_gaussian_noise(img, sigma, rng):
    out = img.astype(np.float64) + rng.normal(0, sigma, img.shape)
    return np.round(out).clip(0, 255).astype(np.uint8)


def add_salt_pepper(img, frac, rng):
    out = img.copy()
    r = rng.random(img.shape)
    out[r < frac / 2] = 0
    out[r > 1 - frac / 2] = 255
    return out


def section_controlled():
    print("Controlled filter comparison (clean reference available)")
    clean = imread_gray("mrt-train.jpg")
    rng = np.random.default_rng(20260924)

    kernels = {s: gaussian_kernel(5, s) for s in (1.0, 2.0)}
    filters = [(f"Gaussian 5x5 s={s}", _conv_filter(h)) for s, h in sorted(kernels.items())]
    filters += [("median 3x3", _median_filter(3)), ("median 5x5", _median_filter(5))]

    # sigma = 20 and 4 % impulses, chosen to match the lab images' severity
    cases = [("additive Gaussian, sigma = 20", add_gaussian_noise(clean, 20, rng)),
             ("salt-and-pepper, 4 %",          add_salt_pepper(clean, 0.04, rng))]

    table = {}
    for case_name, noisy in cases:
        print(f"\n  {case_name}   (noisy input PSNR = {psnr(clean, noisy):.2f} dB)")
        row = {"unfiltered": psnr(clean, noisy)}
        for flabel, fn in filters:
            row[flabel] = psnr(clean, fn(noisy))
            print(f"    {flabel:<22s} PSNR = {row[flabel]:6.2f} dB  "
                  f"({row[flabel] - row['unfiltered']:+6.2f} dB)")
        best = max((k for k in row if k != "unfiltered"), key=lambda k: row[k])
        print(f"    -> best: {best}")
        table[case_name] = row

        fig, axes = plt.subplots(1, len(filters) + 2,
                                 figsize=(3.6 * (len(filters) + 2), 3.9))
        axes[0].imshow(clean, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("clean reference", fontsize=9)
        axes[1].imshow(noisy, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title(f"noisy\n{row['unfiltered']:.2f} dB", fontsize=9)
        for ax, (flabel, fn) in zip(axes[2:], filters):
            ax.imshow(fn(noisy), cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{flabel}\n{row[flabel]:.2f} dB", fontsize=9)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"Controlled comparison -- {case_name}", fontsize=12)
        fig.tight_layout()
        savefig(fig, "2.4", "controlled_" +
                ("gaussian" if "Gaussian" in case_name else "saltpepper"))
    return table


# --------------------------------------------------------------------------
# Section 2.5 -- Suppressing noise interference patterns
# --------------------------------------------------------------------------

def power_spectrum(F):
    """S = |F|^2, real by construction."""
    return np.abs(F) ** 2


def find_peaks(F, dc_guard=10, n_pairs=2, min_share=0.001):
    """Locate isolated interference peaks without ginput.

    Returns one member of each conjugate pair, strongest first, as (v, u)
    indices into the UNSHIFTED transform -- the coordinates the manual asks
    to be read off the un-fftshifted spectrum in step (c).
    """
    mag = np.abs(F)
    N0, N1 = mag.shape
    total = float(np.sum(mag[1:, 1:] ** 2))
    cand = []
    for v in range(N0):
        for u in range(N1):
            dv, du = min(v, N0 - v), min(u, N1 - u)
            if np.hypot(du, dv) < dc_guard:
                continue
            m = mag[v, u]
            nb = mag[max(0, v - 2):v + 3, max(0, u - 2):u + 3]
            if m >= nb.max() and (m * m) / total >= min_share:
                cand.append((m, v, u))
    cand.sort(reverse=True)

    picked = []
    for m, v, u in cand:
        if len(picked) >= n_pairs:
            break
        twin = ((N0 - v) % N0, (N1 - u) % N1)
        if any((abs(v - pv) < 4 and abs(u - pu) < 4) or
               (abs(twin[0] - pv) < 4 and abs(twin[1] - pu) < 4)
               for _, pv, pu in picked):
            continue
        picked.append((m, v, u))
    return [(v, u) for _, v, u in picked]


def notch(F, peaks, half=2, profile="ideal"):
    """Zero (or softly attenuate) a (2*half+1)^2 neighbourhood at each peak and
    at its conjugate twin. `half=2` gives the manual's 5x5 neighbourhood."""
    G = F.copy()
    N0, N1 = F.shape
    sigma = max(half / 1.8, 0.5)
    for (v, u) in peaks:
        for (tv, tu) in ((v, u), ((N0 - v) % N0, (N1 - u) % N1)):
            for dv in range(-half, half + 1):
                for du in range(-half, half + 1):
                    vv, uu = (tv + dv) % N0, (tu + du) % N1
                    if profile == "ideal":
                        G[vv, uu] = 0
                    else:
                        d2 = dv * dv + du * du
                        G[vv, uu] *= 1 - np.exp(-d2 / (2 * sigma * sigma))
    return G


def peak_prominence(F, peaks):
    """Ratio of each peak's magnitude to the median of its surrounding annulus.
    ~1 means the component is gone; large means it is still standing."""
    mag = np.abs(F)
    N0, N1 = mag.shape
    out = []
    for (v, u) in peaks:
        ring = []
        for dv in range(-12, 13):
            for du in range(-12, 13):
                d = np.hypot(dv, du)
                if 6 <= d <= 12:
                    ring.append(mag[(v + dv) % N0, (u + du) % N1])
        out.append(float(mag[v, u] / (np.median(ring) + 1e-9)))
    return out


def _spectrum_image(S):
    """fftshift(S.^0.1), percentile-clipped for display."""
    d = np.fft.fftshift(S ** 0.1)
    lo, hi = np.percentile(d, 2), np.percentile(d, 99.8)
    return np.clip((d - lo) / (hi - lo + 1e-12), 0, 1)


def strongest_remaining(F, dc_guard=10):
    """Prominence of the most interference-like component still standing.
    Near 1 means nothing isolated is left; large means more notching is due."""
    mag = np.abs(F)
    N0, N1 = mag.shape
    best = 0.0
    for v in range(N0):
        for u in range(N1):
            dv, du = min(v, N0 - v), min(u, N1 - u)
            if np.hypot(du, dv) < dc_guard:
                continue
            m = mag[v, u]
            nb = mag[max(0, v - 2):v + 3, max(0, u - 2):u + 3]
            if m < nb.max():
                continue
            best = max(best, peak_prominence(F, [(v, u)])[0])
    return best


def _run_notch_case(name, imgname, n_pairs, variants, fname):
    img = imread_gray(imgname)
    F = np.fft.fft2(img.astype(np.float64))
    S = power_spectrum(F)
    total = float(np.sum(np.abs(F)[1:, 1:] ** 2))

    peaks = find_peaks(F, n_pairs=n_pairs)
    print(f"  {name}: peak locations in the unshifted transform")
    for (v, u) in peaks:
        share = 100 * abs(F[v, u]) ** 2 / total
        print(f"    (row {v:3d}, col {u:3d})  0-based   =  "
              f"({v+1:3d}, {u+1:3d}) in MATLAB 1-based indexing   "
              f"energy share {share:6.3f} %")
    print(f"    prominence before notching: "
          f"{[f'{p:.1f}' for p in peak_prominence(F, peaks)]}")

    print(f"    strongest component before notching: "
          f"prominence {strongest_remaining(F):.1f}")

    results = [("original", img, None)]
    for label, half, profile, npair in variants:
        pk = peaks if npair is None else find_peaks(F, n_pairs=npair)
        G = notch(F, pk, half=half, profile=profile)
        out = np.real(np.fft.ifft2(G))
        out_u8 = np.round(out).clip(0, 255).astype(np.uint8)
        removed = 100 * (1 - float(np.sum(np.abs(G)[1:, 1:] ** 2)) / total)
        print(f"    {label:<34s} {len(pk)} pair(s)  energy removed {removed:6.3f} %   "
              f"strongest left: prominence {strongest_remaining(G):5.1f}")
        results.append((label, out_u8, G))

    ncol = len(results) + 1
    fig, axes = plt.subplots(2, ncol, figsize=(3.5 * ncol, 7.2))
    axes[0, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("original", fontsize=9)
    axes[1, 0].imshow(_spectrum_image(S), cmap="viridis")
    axes[1, 0].set_title("$fftshift(S^{0.1})$", fontsize=9)
    for col, (label, out, G) in enumerate(results[1:], start=1):
        axes[0, col].imshow(out, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(label, fontsize=9)
        axes[1, col].imshow(_spectrum_image(power_spectrum(G)), cmap="viridis")
        axes[1, col].set_title("spectrum after notching", fontsize=9)
    # final column: zoomed spectrum around a peak, before notching
    v, u = peaks[0]
    sh = np.fft.fftshift(S ** 0.1)
    cv, cu = (v + S.shape[0] // 2) % S.shape[0], (u + S.shape[1] // 2) % S.shape[1]
    axes[0, -1].imshow(sh[max(0, cv-20):cv+20, max(0, cu-20):cu+20], cmap="viridis")
    axes[0, -1].set_title("peak, zoomed", fontsize=9)
    axes[1, -1].axis("off")
    for ax in axes.ravel():
        if ax.images:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(name, fontsize=12)
    fig.tight_layout()
    savefig(fig, "2.5", fname)
    return peaks


def section_2_5():
    print("2.5 Suppressing noise interference patterns")

    # (a)-(e) the television-style diagonal interference
    variants = [("(d) ideal 5x5 notch", 2, "ideal", None),
                ("improved: ideal 9x9 notch", 4, "ideal", None),
                ("improved: Gaussian 9x9 notch", 4, "gauss", None)]
    _run_notch_case("pck-int.jpg -- diagonal interference",
                    "pck-int.jpg", 2, variants, "pck_int_notch")

    # (f) the caged primate: a fence is a harmonic series, not one sinusoid
    print()
    variants_f = [("1 pair, ideal 5x5", 2, "ideal", 1),
                  ("6 pairs, ideal 5x5", 2, "ideal", 6),
                  ("6 pairs, Gaussian 9x9", 4, "gauss", 6)]
    # the pair count differs per variant, so run them separately
    img = imread_gray("primate-caged.jpg")
    F = np.fft.fft2(img.astype(np.float64))
    total = float(np.sum(np.abs(F)[1:, 1:] ** 2))
    print("  primate-caged.jpg -- fence harmonics")
    for npair in (2, 4, 6, 8):
        pk = find_peaks(F, n_pairs=npair)
        G = notch(F, pk, half=4, profile="gauss")
        removed = 100 * (1 - float(np.sum(np.abs(G)[1:, 1:] ** 2)) / total)
        print(f"    {npair} conjugate pairs notched -> energy removed {removed:6.3f} %")
    _run_notch_case("primate-caged.jpg -- fence removal",
                    "primate-caged.jpg", 6, variants_f, "primate_notch")


# --------------------------------------------------------------------------
# Section 2.6 -- Undoing perspective distortion of a planar surface
# --------------------------------------------------------------------------

# Corners of the book cover in book.jpg, clockwise from the top-left, in the
# same order as the desired corners below. The manual obtains these with
# ginput, which is interactive; to keep the script reproducible they were
# measured once by segmenting the bright cover, fitting a straight line to
# each of its four edges (sub-pixel residuals, 0.31-0.94 px) and intersecting
# adjacent lines. The equivalent ginput call is kept below, commented out.
#
#     >> [X Y] = ginput(4)
BOOK_CORNERS = np.array([[142.55,  18.65],    # top-left,     near "School of..."
                         [307.74,  46.71],    # top-right,    near the logo
                         [255.37, 217.37],    # bottom-right
                         [  4.25, 158.64]])   # bottom-left

# The book is A4: 210 mm wide x 297 mm tall, and the warped image is to be
# scaled 1 pixel = 1 mm.
A4_CORNERS = np.array([[  0.0,   0.0],
                       [210.0,   0.0],
                       [210.0, 297.0],
                       [  0.0, 297.0]])


def projective_matrix(src, dst):
    """Solve for the 3x3 projective transform using the manual's equation (*).

    Builds the 8x8 system A u = v in which u holds
    [m11 m12 m13 m21 m22 m23 m31 m32]' and each correspondence contributes two
    rows, then returns U = reshape([u;1], 3, 3)'.
    """
    A, v = [], []
    for (Xw, Yw), (xim, yim) in zip(src, dst):
        A.append([Xw, Yw, 1,  0,  0, 0, -xim * Xw, -xim * Yw])
        A.append([0,  0,  0, Xw, Yw, 1, -yim * Xw, -yim * Yw])
        v += [xim, yim]
    A = np.array(A); v = np.array(v)
    # square for 4 points -> solve; over-determined for more -> least squares
    u = np.linalg.solve(A, v) if A.shape[0] == 8 else np.linalg.lstsq(A, v, rcond=None)[0]
    return np.append(u, 1.0).reshape(3, 3)


def warp_projective(img, U, width, height):
    """Inverse-map every output pixel through U^-1 and bilinearly resample,
    which is what imtransform does for 'XData',[0 width],'YData',[0 height]."""
    Uinv = np.linalg.inv(U)
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    P = np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5, np.ones(xx.size)])
    Q = Uinv @ P
    sx, sy = Q[0] / Q[2], Q[1] / Q[2]

    H, W = img.shape[:2]
    x0 = np.floor(sx).astype(int); y0 = np.floor(sy).astype(int)
    fx = sx - x0; fy = sy - y0
    valid = (x0 >= 0) & (x0 < W - 1) & (y0 >= 0) & (y0 < H - 1)
    x0c = np.clip(x0, 0, W - 2); y0c = np.clip(y0, 0, H - 2)

    def sample(ch):
        a = ch[y0c, x0c];     b = ch[y0c, x0c + 1]
        c = ch[y0c + 1, x0c]; d = ch[y0c + 1, x0c + 1]
        out = (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) +
               c * (1 - fx) * fy + d * fx * fy)
        return np.where(valid, out, 0.0)

    if img.ndim == 3:
        chans = [sample(img[:, :, k].astype(np.float64)) for k in range(img.shape[2])]
        out = np.stack(chans, axis=-1).reshape(height, width, img.shape[2])
    else:
        out = sample(img.astype(np.float64)).reshape(height, width)
    return np.round(out).clip(0, 255).astype(np.uint8)


def section_2_6():
    print("2.6 Undoing perspective distortion")
    from PIL import Image
    P = np.asarray(Image.open(IMAGES / "book.jpg"))
    print(f"  book.jpg: {P.shape} {P.dtype}")

    X, Y = BOOK_CORNERS[:, 0], BOOK_CORNERS[:, 1]
    x, y = A4_CORNERS[:, 0], A4_CORNERS[:, 1]
    print("  corner correspondences (image -> desired):")
    for (Xi, Yi), (xi, yi) in zip(BOOK_CORNERS, A4_CORNERS):
        print(f"    ({Xi:7.2f}, {Yi:7.2f})  ->  ({xi:6.1f}, {yi:6.1f})")

    U = projective_matrix(BOOK_CORNERS, A4_CORNERS)
    print("\n  projective transformation matrix U:")
    for row in U:
        print("    [" + "  ".join(f"{val: .6e}" for val in row) + "]")

    # (c) verification: transform the original corners and check they land on
    #     the desired ones.  w = U*[X'; Y'; ones(1,4)];  w = w ./ (ones(3,1)*w(3,:))
    w = U @ np.vstack([X, Y, np.ones(4)])
    w = w / w[2]
    print("\n  verification -- U applied to the original corners:")
    err = 0.0
    for i in range(4):
        e = np.hypot(w[0, i] - x[i], w[1, i] - y[i]); err = max(err, e)
        print(f"    ({w[0, i]:10.6f}, {w[1, i]:10.6f})   target "
              f"({x[i]:6.1f}, {y[i]:6.1f})   error {e:.3e}")
    print(f"  max corner error = {err:.3e} mm  "
          f"(exact to machine precision, as 4 points determine the 8 unknowns)")

    # (d),(e) warp to a 210 x 297 image, 1 pixel = 1 mm
    P2 = warp_projective(P, U, 210, 297)
    print(f"\n  warped image: {P2.shape}  (1 pixel = 1 mm)")

    # (f) locate the pink "computer screen" region
    rgb = P2.astype(np.float64)
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    mx = rgb.max(axis=2); mn = rgb.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)
    # pink/salmon: red clearly dominant, reasonably bright, decently saturated
    pink = (R > G + 25) & (R > B + 25) & (R > 90) & (sat > 0.18)
    from scipy import ndimage
    pink = ndimage.binary_opening(pink, np.ones((3, 3)))
    pink = ndimage.binary_closing(pink, np.ones((5, 5)))
    lab, n = ndimage.label(pink)
    print(f"  (f) pink-region search: {n} candidate components")
    if n:
        sizes = ndimage.sum(pink, lab, range(1, n + 1))
        k = int(np.argmax(sizes)) + 1
        ys, xs = np.nonzero(lab == k)
        print(f"      largest component: {int(sizes.max())} px "
              f"({100*sizes.max()/pink.size:.2f} % of the page)")
        print(f"      bounding box: x {xs.min()}-{xs.max()} mm, "
              f"y {ys.min()}-{ys.max()} mm  "
              f"-> {xs.max()-xs.min()+1} x {ys.max()-ys.min()+1} mm")
        bbox = (xs.min(), ys.min(), xs.max(), ys.max())
    else:
        bbox = None

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.4))
    axes[0].imshow(P)
    cl = np.vstack([BOOK_CORNERS, BOOK_CORNERS[:1]])
    axes[0].plot(cl[:, 0], cl[:, 1], "-o", c="lime", lw=2, ms=6)
    axes[0].set_title("book.jpg with the four measured corners")
    axes[1].imshow(P2)
    axes[1].set_title("P2: warped to a frontal A4 view\n210 x 297, 1 px = 1 mm")
    axes[2].imshow(P2)
    if bbox:
        x0, y0, x1, y1 = bbox
        axes[2].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                        fill=False, ec="#00ffcc", lw=2.2))
        axes[2].set_title(f"(f) pink screen region\n{x1-x0+1} x {y1-y0+1} mm "
                          f"at ({x0}, {y0})")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    savefig(fig, "2.6", "perspective_warp")
    return U, P2


# --------------------------------------------------------------------------
# Section 2.7 -- Two perceptrons
#
# Algorithm 1 (sequential sample update) and Algorithm 2 (batch delta / LMS
# rule), as defined in the lecture slides, run on the slides' own example:
#
#     x(1) = [3, 3, 1]^T  in c1       x(2) = [1, 1, 1]^T  in c2
#     alpha = 1,  w(1) = [0, 0, 0]^T
#
# The bias is folded into w by the constant 1 appended to each x, so the
# decision rule is simply sign(w' x).
# --------------------------------------------------------------------------

PERCEPTRON_X = np.array([[3.0, 3.0, 1.0],
                         [1.0, 1.0, 1.0]])
PERCEPTRON_CLASS = [1, 2]              # x(1) in c1, x(2) in c2
PERCEPTRON_R = np.array([1.0, -1.0])   # target labels r for Algorithm 2


def perceptron_alg1(X, cls, alpha=1.0, w0=None, max_epochs=100):
    """Algorithm 1: sequential update, applied the moment a sample is misclassified.

        x(k) in c1 and w'x <= 0  ->  w(k+1) = w(k) + alpha * x(k)
        x(k) in c2 and w'x >= 0  ->  w(k+1) = w(k) - alpha * x(k)
        otherwise                ->  w(k+1) = w(k)
    """
    w = np.zeros(X.shape[1]) if w0 is None else np.array(w0, dtype=float)
    trace, k, epochs = [], 0, 0
    for epoch in range(max_epochs):
        epochs = epoch + 1
        changed = False
        for i in range(len(X)):
            k += 1
            a = float(w @ X[i])
            if cls[i] == 1 and a <= 0:
                w = w + alpha * X[i]; rule, changed = 1, True
            elif cls[i] == 2 and a >= 0:
                w = w - alpha * X[i]; rule, changed = 2, True
            else:
                rule = 3
            trace.append({"k": k, "sample": i + 1, "a": a,
                          "rule": rule, "w": w.copy()})
        if not changed:
            break
    return w, trace, epochs


def perceptron_alg2(X, r, alpha=0.05, w0=None, max_epochs=5000, tol=1e-12):
    """Algorithm 2: batch delta (LMS) rule, one update per epoch.

        E(w)      = 1/2 sum_i (r_i - w'x_i)^2
        dE/dw     = -sum_i (r_i - w'x_i) x_i
        w(k+1)    = w(k) + alpha * sum_i [r_i - w'(k) x_i] x_i
    """
    w = np.zeros(X.shape[1]) if w0 is None else np.array(w0, dtype=float)
    costs = []
    for _ in range(max_epochs):
        e = r - X @ w
        cost = 0.5 * float(np.sum(e**2))
        costs.append(cost)
        if not np.isfinite(cost) or cost > 1e12:
            return w, costs, False               # diverged
        if cost < tol:
            break
        grad = -(e[:, None] * X).sum(axis=0)     # accumulate over the epoch
        w = w - alpha * grad
    return w, costs, True


def _boundary_stats(w, X):
    """Signed perpendicular distance from each sample to the line w'x = 0."""
    n = np.linalg.norm(w[:2])
    return [(float(w @ x) / n) for x in X]


def section_2_7():
    print("2.7 Two perceptrons")
    X, cls, r = PERCEPTRON_X, PERCEPTRON_CLASS, PERCEPTRON_R

    # ---- Algorithm 1 ------------------------------------------------------
    w1, trace, ep1 = perceptron_alg1(X, cls, alpha=1.0)
    print(f"  Algorithm 1 (alpha = 1): converged at k = {trace[-1]['k']}, "
          f"{ep1} epochs, {sum(t['rule'] != 3 for t in trace)} updates")
    print(f"    w = {w1}  ->  boundary  {w1[0]:g}*x1 + {w1[1]:g}*x2 + {w1[2]:g} = 0")
    print("    step trace:")
    for t in trace:
        print(f"      k={t['k']:2d}  x({t['sample']})  w'x={t['a']:6.1f}  "
              f"rule {t['rule']}  ->  w = {t['w']}")

    # ---- Algorithm 2: stability bound -------------------------------------
    R = sum(np.outer(x, x) for x in X)
    lam = float(np.linalg.eigvalsh(R).max())
    print(f"\n  Algorithm 2 stability: lambda_max(sum x x') = {lam:.4f}, "
          f"so convergence needs alpha < 2/lambda_max = {2/lam:.4f}")
    w_div, _, ok_div = perceptron_alg2(X, r, alpha=1.0)
    print(f"    alpha = 1.0 (as used by Algorithm 1): "
          f"{'converged' if ok_div else 'DIVERGES, as predicted'}")

    alpha2 = 0.05
    w2, costs, ok = perceptron_alg2(X, r, alpha=alpha2)
    print(f"    alpha = {alpha2}: converged in {len(costs)} epochs, "
          f"final cost {costs[-1]:.3e}")
    print(f"    w = {np.round(w2, 4)}  ->  boundary  "
          f"{w2[0]:g}*x1 + {w2[1]:g}*x2 + {w2[2]:g} = 0")

    # ---- Comparison -------------------------------------------------------
    d1, d2 = _boundary_stats(w1, X), _boundary_stats(w2, X)
    print("\n  Comparison of the two boundaries:")
    print(f"    Algorithm 1:  x1 + x2 = {-w1[2]/w1[0]:.2f}   "
          f"distances {d1[0]:+.4f} / {d1[1]:+.4f}   "
          f"margin ratio {min(map(abs,d1))/max(map(abs,d1)):.3f}")
    print(f"    Algorithm 2:  x1 + x2 = {-w2[2]/w2[0]:.2f}   "
          f"distances {d2[0]:+.4f} / {d2[1]:+.4f}   "
          f"margin ratio {min(map(abs,d2))/max(map(abs,d2)):.3f}")

    # ---- Figures ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    ax.scatter([3], [3], s=140, c="#2563eb", marker="o",
               label="$x(1)=[3,3]^T \\in c_1$", zorder=3, edgecolors="k")
    ax.scatter([1], [1], s=140, c="#dc2626", marker="s",
               label="$x(2)=[1,1]^T \\in c_2$", zorder=3, edgecolors="k")
    xs = np.linspace(-0.5, 4.5, 100)
    ax.plot(xs, (-w1[2] - w1[0]*xs) / w1[1], "-",  c="#2563eb", lw=2,
            label=f"Alg. 1:  $x_1+x_2={-w1[2]/w1[0]:.0f}$")
    ax.plot(xs, (-w2[2] - w2[0]*xs) / w2[1], "--", c="#059669", lw=2,
            label=f"Alg. 2:  $x_1+x_2={-w2[2]/w2[0]:.0f}$")
    ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.5, 4.5); ax.set_aspect("equal")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title("Decision boundaries")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.semilogy(costs, lw=2, c="#059669")
    ax.set_xlabel("Epoch"); ax.set_ylabel("$E(w) = \\frac{1}{2}\\sum (r - w^Tx)^2$")
    ax.set_title(f"Algorithm 2: cost vs iterations ($\\alpha$ = {alpha2})")
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    savefig(fig, "2.7", "perceptron_comparison")
    return w1, w2


# --------------------------------------------------------------------------
# Sections pending their input images.
# --------------------------------------------------------------------------

def _pending(section, what):
    def run():
        print(f"{section}: NOT YET IMPLEMENTED -- blocked on {what}")
    return run



def dominant_ridge_angle(F, r_min=8, r_max=120):
    """Angle (radians) of the line through DC carrying the most spectral energy.

    A wavy occluder like the fence is not a single sinusoid: its energy spreads
    along a continuous ridge through the origin rather than collapsing into
    isolated peaks, which is why point-notching alone cannot remove it.
    """
    mag = np.abs(F)
    N0, N1 = mag.shape
    best, best_a = -1.0, 0.0
    for deg in np.arange(0, 180, 0.5):
        a = np.deg2rad(deg)
        rr = np.arange(r_min, r_max)
        vv = np.round(rr * np.sin(a)).astype(int) % N0
        uu = np.round(rr * np.cos(a)).astype(int) % N1
        s = float(np.sum(mag[vv, uu]))
        if s > best:
            best, best_a = s, a
    return best_a


def ridge_notch(F, angle, width=2.0, r_min=8, profile="gauss"):
    """Attenuate a narrow band along the ridge through DC, sparing low
    frequencies so overall image structure survives."""
    N0, N1 = F.shape
    v = np.fft.fftfreq(N0) * N0
    u = np.fft.fftfreq(N1) * N1
    U, V = np.meshgrid(u, v)
    # perpendicular distance from each coefficient to the ridge line
    d = np.abs(-np.sin(angle) * U + np.cos(angle) * V)
    radius = np.hypot(U, V)
    if profile == "ideal":
        w = np.where((d <= width) & (radius >= r_min), 0.0, 1.0)
    else:
        w = 1 - np.exp(-(d ** 2) / (2 * width ** 2))
        w = np.where(radius >= r_min, w, 1.0)
    return F * w


def section_2_5f_improved():
    print("2.5(f) further improvement: ridge notching on primate-caged.jpg")
    img = imread_gray("primate-caged.jpg")
    F = np.fft.fft2(img.astype(np.float64))
    total = float(np.sum(np.abs(F)[1:, 1:] ** 2))

    ang = dominant_ridge_angle(F)
    print(f"  dominant spectral ridge at {np.rad2deg(ang):.1f} deg "
          f"(fence stripes run perpendicular to this)")

    peaks = find_peaks(F, n_pairs=6)
    base = notch(F, peaks, half=4, profile="gauss")

    outs = [("original", img)]
    print(f"  {'variant':<40s} {'removed':>9s} {'strongest left':>15s}")
    print(f"  {'original':<40s} {'--':>9s} {strongest_remaining(F):15.1f}")
    for label, G in [
            ("6 point notches only", base),
            ("ridge notch, width 2", ridge_notch(F, ang, width=2.0)),
            ("ridge notch, width 4", ridge_notch(F, ang, width=4.0)),
            ("6 point notches + ridge width 3",
             ridge_notch(base, ang, width=3.0))]:
        out = np.round(np.real(np.fft.ifft2(G))).clip(0, 255).astype(np.uint8)
        removed = 100 * (1 - float(np.sum(np.abs(G)[1:, 1:] ** 2)) / total)
        print(f"  {label:<40s} {removed:8.3f}% {strongest_remaining(G):15.1f}")
        outs.append((label, out))

    fig, axes = plt.subplots(1, len(outs), figsize=(3.6 * len(outs), 4.0))
    for ax, (label, im) in zip(axes, outs):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(label, fontsize=9); ax.axis("off")
    fig.suptitle("2.5(f) freeing the primate: point notches vs ridge notching",
                 fontsize=12)
    fig.tight_layout()
    savefig(fig, "2.5", "primate_ridge_notch")


SECTIONS = {
    "2.1": section_2_1,
    "2.2": section_2_2,
    "2.3": section_2_3,
    "2.4": section_2_4,
    "ctrl": section_controlled,
    "2.5f": section_2_5f_improved,
    "2.5": section_2_5,
    "2.6": section_2_6,
    "2.7": section_2_7,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--section", help="run one section, e.g. 2.3")
    ap.add_argument("--all", action="store_true", help="run every section")
    args = ap.parse_args(argv)

    if args.section:
        if args.section not in SECTIONS:
            ap.error(f"unknown section {args.section}; "
                     f"choose from {', '.join(sorted(SECTIONS))}")
        SECTIONS[args.section]()
    elif args.all:
        for key in sorted(SECTIONS):
            SECTIONS[key]()
    else:
        ap.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
