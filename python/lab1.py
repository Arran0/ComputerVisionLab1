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
    "2.1": _pending("2.1", "images/mrt-train.jpg"),
    "2.2": _pending("2.2", "images/mrt-train.jpg"),
    "2.3": section_2_3a,
    "2.4": _pending("2.4", "images/lib-gn.jpg, images/lib-sp.jpg"),
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
