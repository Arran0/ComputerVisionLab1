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


SECTIONS = {
    "2.1": section_2_1,
    "2.2": section_2_2,
    "2.3": section_2_3,
    "2.4": section_2_4,
    "ctrl": section_controlled,
    "2.5": _pending("2.5", "images/pck-int.jpg, images/primate-caged.jpg"),
    "2.6": _pending("2.6", "images/book.jpg"),
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
