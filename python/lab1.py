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
# Sections pending their input images / lecture-slide definitions.
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
    "2.7": _pending("2.7", "lecture-slide definitions of Algorithm 1 / 2"),
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
