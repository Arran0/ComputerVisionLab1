#!/usr/bin/env python3
"""Recompute every headline number quoted in report/report.md and check it.

Each CLAIM below is the value as the report states it. The script recomputes
the value from the images and the lab code, then compares. Any mismatch is
printed and the script exits non-zero, so a stale number in the report cannot
pass silently.

Usage:  python3 python/verify_report.py
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import lab1  # noqa: E402

PASS, FAIL = [], []


def check(label, claimed, actual, tol=0.0):
    ok = (abs(float(actual) - float(claimed)) <= tol
          if isinstance(claimed, (int, float)) and not isinstance(claimed, bool)
          else actual == claimed)
    (PASS if ok else FAIL).append((label, claimed, actual))
    print(("  ok   " if ok else "  FAIL ") +
          f"{label:<52s} report={claimed!s:<22s} actual={actual!s}")


print("2.1 Contrast stretching")
from PIL import Image
raw = np.asarray(Image.open(lab1.IMAGES / "mrt-train.jpg"))
check("shape of Pc", (320, 443, 3), raw.shape)
check("max|R-G| (channels identical)", 0, int(np.abs(raw[:, :, 0].astype(int) - raw[:, :, 1].astype(int)).max()))
check("max|G-B| (channels identical)", 0, int(np.abs(raw[:, :, 1].astype(int) - raw[:, :, 2].astype(int)).max()))
P = lab1.imread_gray("mrt-train.jpg")
check("min(P)", 13, int(P.min()))
check("max(P)", 204, int(P.max()))
check("levels used (204-13+1)", 192, int(P.max()) - int(P.min()) + 1)
lo, hi = int(P.min()), int(P.max())
P2 = np.round((P.astype(float) - lo) * (255.0 / (hi - lo))).clip(0, 255).astype(np.uint8)
check("min(P2)", 0, int(P2.min()))
check("max(P2)", 255, int(P2.max()))
check("stretch factor 255/191", 1.335, 255 / 191, 0.001)

print("\n2.2 Histogram equalisation")
P3 = lab1.matlab_histeq(P, 255)
P4 = lab1.matlab_histeq(P3, 255)
check("mean(P)", 73.989, P.mean(), 0.001)
check("mean(P3)", 127.502, P3.mean(), 0.001)
check("mean(P4)", 127.506, P4.mean(), 0.001)
# imhist places n bin CENTRES at linspace(0,255,n), so the end bins are half
# width. These counts come from MATLAB's own imhist.
h10p, h10_3, h10_4 = (lab1.matlab_imhist(x, 10) for x in (P, P3, P4))
check("P 10-bin counts match MATLAB", True,
      list(h10p) == [3, 50391, 39642, 13218, 12633, 9236, 15414, 1223, 0, 0])
check("P3 inner-eight minimum", 15037, int(h10_3[1:-1].min()))
check("P3 inner-eight maximum", 16251, int(h10_3[1:-1].max()))
check("P3 end bins", [7776, 8742], [int(h10_3[0]), int(h10_3[-1])])
check("P distinct levels", 192, len(np.unique(P)))
check("P3 distinct levels", 136, len(np.unique(P3)))
check("P4 distinct levels", 135, len(np.unique(P4)))
check("pixels changed by second pass", 615, int((P4 != P3).sum()))
check("P3/P4 10-bin counts identical", True, bool((h10_3 == h10_4).all()))
d = np.abs(P4.astype(int) - P3.astype(int))
check("max|P4-P3|", 1, int(d.max()))
check("mean|P4-P3|", 0.0043, d.mean(), 0.0001)
check("% pixels changed", 0.43, 100 * (d > 0).mean(), 0.005)

print("\n2.3 Gaussian filters and noise")
k1, k2 = lab1.gaussian_kernel(5, 1.0), lab1.gaussian_kernel(5, 2.0)
check("sum of sigma=1 kernel", 1.0, k1.sum(), 1e-12)
check("sum of sigma=2 kernel", 1.0, k2.sum(), 1e-12)
check("sigma=1 centre", 0.162103, k1[2, 2], 1e-6)
check("sigma=1 corner", 0.002969, k1[0, 0], 1e-6)
check("sigma=2 centre", 0.063191, k2[2, 2], 1e-6)
check("sigma=2 corner", 0.023247, k2[0, 0], 1e-6)
check("sigma=1 centre:corner", 54.6, k1[2, 2] / k1[0, 0], 0.05)
check("sigma=2 centre:corner", 2.7, k2[2, 2] / k2[0, 0], 0.05)

gn, sp = lab1.imread_gray("lib-gn.jpg"), lab1.imread_gray("lib-sp.jpg")
check("lib-sp %% at exactly 0 or 255", 3.80, lab1.impulse_fraction(sp), 0.01)
check("lib-sp %% <= 10 or >= 245", 10.96,
      100 * ((sp <= 10) | (sp >= 245)).mean(), 0.01)
# Separate true impulses from genuinely saturated scene content by the size
# of the connected component each extreme pixel belongs to.
from scipy.ndimage import label as _label
_m = (sp <= 10) | (sp >= 245)
_lab, _n = _label(_m)
_sizes = np.bincount(_lab.ravel())[1:]
check("lib-sp %% in blobs of <= 4 px (impulses)", 2.20,
      100 * _sizes[_sizes <= 4].sum() / sp.size, 0.01)
check("lib-sp %% in blobs of >= 21 px (real scene)", 8.42,
      100 * _sizes[_sizes >= 21].sum() / sp.size, 0.01)
from scipy.ndimage import median_filter as _mf
_r_gn = gn.astype(int) - _mf(gn, size=3, mode="reflect").astype(int)
_r_sp = sp.astype(int) - _mf(sp, size=3, mode="reflect").astype(int)
check("lib-gn residual std, impulses excluded", 7.69,
      _r_gn[np.abs(_r_gn) <= 20].std(), 0.01)
check("lib-sp residual std, impulses excluded", 3.98,
      _r_sp[np.abs(_r_sp) <= 20].std(), 0.01)

g1, g2 = lab1._conv_filter(k1), lab1._conv_filter(k2)
check("lib-gn mean gradient", 79.90, lab1.mean_gradient(gn), 0.01)
check("lib-gn grad sigma=1", 54.94, lab1.mean_gradient(g1(gn)), 0.01)
check("lib-gn grad sigma=2", 49.30, lab1.mean_gradient(g2(gn)), 0.01)
check("lib-sp mean gradient", 103.32, lab1.mean_gradient(sp), 0.01)
check("lib-sp grad sigma=1", 70.83, lab1.mean_gradient(g1(sp)), 0.01)
check("lib-sp grad sigma=2", 62.06, lab1.mean_gradient(g2(sp)), 0.01)

print("\n2.4 Controlled PSNR comparison")
rng = np.random.default_rng(20260924)
noisy_g = lab1.add_gaussian_noise(P, 20, rng)
noisy_s = lab1.add_salt_pepper(P, 0.04, rng)
m3, m5 = lab1._median_filter(3), lab1._median_filter(5)
check("Gaussian noise, unfiltered PSNR", 22.31, lab1.psnr(P, noisy_g), 0.01)
check("Gaussian noise, Gauss s=1", 25.20, lab1.psnr(P, g1(noisy_g)), 0.01)
check("Gaussian noise, Gauss s=2", 23.66, lab1.psnr(P, g2(noisy_g)), 0.01)
check("Gaussian noise, median 3x3", 24.55, lab1.psnr(P, m3(noisy_g)), 0.01)
check("Gaussian noise, median 5x5", 23.66, lab1.psnr(P, m5(noisy_g)), 0.01)
check("S&P, unfiltered PSNR", 18.67, lab1.psnr(P, noisy_s), 0.01)
check("S&P, Gauss s=1", 24.12, lab1.psnr(P, g1(noisy_s)), 0.01)
check("S&P, Gauss s=2", 23.15, lab1.psnr(P, g2(noisy_s)), 0.01)
check("S&P, median 3x3", 26.18, lab1.psnr(P, m3(noisy_s)), 0.01)
check("S&P, median 5x5", 24.27, lab1.psnr(P, m5(noisy_s)), 0.01)
check("S&P median-vs-Gaussian margin", 2.06,
      lab1.psnr(P, m3(noisy_s)) - lab1.psnr(P, g1(noisy_s)), 0.01)

print("\n2.5 Frequency-domain notching")
pck = lab1.imread_gray("pck-int.jpg")
F = np.fft.fft2(pck.astype(float))
total = float(np.sum(np.abs(F)[1:, 1:] ** 2))
peaks = lab1.find_peaks(F, n_pairs=2)
v, u = peaks[0]
# find_peaks returns one member of the conjugate pair; which one depends on
# scan order, and the report tabulates both, so accept either.
pair = {(16, 248), (240, 8)}
check("peak is a member of the reported pair", True, (v, u) in pair)
check("its conjugate twin is the other member", True,
      ((256 - v) % 256, (256 - u) % 256) in pair)
check("peak energy share %%", 20.751, 100 * abs(F[v, u]) ** 2 / total, 0.001)
# The report ranks local maxima, not raw coefficients: a bright sample on the
# shoulder of the main peak is not itself a competing component.
mag = np.abs(F)
maxima = []
for a in range(256):
    for b in range(256):
        if np.hypot(min(a, 256 - a), min(b, 256 - b)) < 12:
            continue
        nb = mag[max(0, a - 2):a + 3, max(0, b - 2):b + 3]
        if mag[a, b] >= nb.max():
            maxima.append(mag[a, b])
maxima.sort(reverse=True)
nxt = maxima[2]                      # [0] and [1] are the conjugate pair
check("next local maximum share %%", 0.0740, 100 * nxt ** 2 / total, 0.0005)
check("isolation factor", 280, (abs(F[v, u]) ** 2) / (nxt ** 2), 1.0)
check("prominence before notch", 89.5, lab1.peak_prominence(F, peaks)[0], 0.1)
for lbl, half, prof, claim_e, claim_p in [
        ("ideal 5x5", 2, "ideal", 57.12, 10.1),
        ("ideal 9x9", 4, "ideal", 58.99, 9.1),
        ("Gaussian 9x9", 4, "gauss", 57.54, 7.7)]:
    G = lab1.notch(F, peaks, half=half, profile=prof)
    e = 100 * (1 - float(np.sum(np.abs(G)[1:, 1:] ** 2)) / total)
    check(f"{lbl}: energy removed %%", claim_e, e, 0.01)
    check(f"{lbl}: strongest left", claim_p, lab1.strongest_remaining(G), 0.1)
check("coefficients zeroed by 5x5 pair", 50, 2 * 25)
check("as %% of 256x256 transform", 0.08, 100 * 50 / (256 * 256), 0.005)

prim = lab1.imread_gray("primate-caged.jpg")
FP = np.fft.fft2(prim.astype(float))
totP = float(np.sum(np.abs(FP)[1:, 1:] ** 2))
pk1 = lab1.find_peaks(FP, n_pairs=1)
pk6 = lab1.find_peaks(FP, n_pairs=6)
check("primate fundamental share %%", 5.002, 100 * abs(FP[pk1[0]]) ** 2 / totP, 0.005)
check("primate pairs found at n=6", 5, len(pk6))
check("primate prominence before", 17.6, lab1.strongest_remaining(FP), 0.1)
G1 = lab1.notch(FP, pk1, half=2, profile="ideal")
G5 = lab1.notch(FP, pk6, half=2, profile="ideal")
check("primate 1 pair: removed %%", 29.46, 100 * (1 - float(np.sum(np.abs(G1)[1:, 1:] ** 2)) / totP), 0.01)
check("primate 1 pair: left", 13.2, lab1.strongest_remaining(G1), 0.1)
check("primate 5 pairs: removed %%", 41.448, 100 * (1 - float(np.sum(np.abs(G5)[1:, 1:] ** 2)) / totP), 0.01)
check("primate 5 pairs: left", 6.8, lab1.strongest_remaining(G5), 0.1)
ang = lab1.dominant_ridge_angle(FP)
check("ridge angle (deg)", 155.0, np.rad2deg(ang), 0.5)
Gb = lab1.notch(FP, pk6, half=4, profile="gauss")
check("primate point notches: removed %%", 43.569,
      100 * (1 - float(np.sum(np.abs(Gb)[1:, 1:] ** 2)) / totP), 0.01)
Gr = lab1.ridge_notch(FP, ang, width=2.0)
check("ridge width 2: removed %%", 44.588,
      100 * (1 - float(np.sum(np.abs(Gr)[1:, 1:] ** 2)) / totP), 0.01)
Gc = lab1.ridge_notch(Gb, ang, width=3.0)
check("combined: removed %%", 51.567,
      100 * (1 - float(np.sum(np.abs(Gc)[1:, 1:] ** 2)) / totP), 0.01)

print("\n2.6 Perspective correction")
U = lab1.projective_matrix(lab1.BOOK_CORNERS, lab1.A4_CORNERS)
claimed_U = np.array([[1.464099, 1.560748, -249.7471],
                      [-0.4226511, 3.694565, -37.21058],
                      [1.653224e-4, 5.347072e-3, 1.0]])
check("U matches report to 6 s.f.", True,
      bool(np.allclose(U, claimed_U, rtol=1e-5, atol=1e-8)))
X, Y = lab1.BOOK_CORNERS[:, 0], lab1.BOOK_CORNERS[:, 1]
w = U @ np.vstack([X, Y, np.ones(4)])
w = w / w[2]
err = float(np.max(np.hypot(w[0] - lab1.A4_CORNERS[:, 0], w[1] - lab1.A4_CORNERS[:, 1])))
check("max corner error < 1e-13 mm", True, bool(err < 1e-13))
top = np.hypot(*(lab1.BOOK_CORNERS[1] - lab1.BOOK_CORNERS[0]))
bot = np.hypot(*(lab1.BOOK_CORNERS[2] - lab1.BOOK_CORNERS[3]))
check("near edge px for 210 mm", 261, bot, 1.0)
check("far edge px for 210 mm", 166, top, 1.0)
check("near edge px/mm", 1.24, bot / 210, 0.01)
check("far edge px/mm", 0.79, top / 210, 0.01)

print("\n2.7 Perceptrons")
w1, trace, ep1 = lab1.perceptron_alg1(lab1.PERCEPTRON_X, lab1.PERCEPTRON_CLASS, 1.0)
check("Alg1 final w", [1.0, 1.0, -3.0], list(w1))
check("Alg1 converges at k", 12, trace[-1]["k"])
check("Alg1 epochs", 6, ep1)
check("Alg1 updates", 7, sum(t["rule"] != 3 for t in trace))
R = sum(np.outer(x, x) for x in lab1.PERCEPTRON_X)
lam = float(np.linalg.eigvalsh(R).max())
check("lambda_max(R)", 21.63, lam, 0.01)
check("stability bound 2/lambda_max", 0.0925, 2 / lam, 0.0001)
_, _, ok1 = lab1.perceptron_alg2(lab1.PERCEPTRON_X, lab1.PERCEPTRON_R, alpha=1.0)
check("Alg2 diverges at alpha=1", False, ok1)
w2, costs, ok2 = lab1.perceptron_alg2(lab1.PERCEPTRON_X, lab1.PERCEPTRON_R, alpha=0.05)
check("Alg2 converges at alpha=0.05", True, ok2)
check("Alg2 epochs", 737, len(costs))
check("Alg2 w, analytical limit to 4 dp", [0.5, 0.5, -2.0],
      [round(float(c), 4) for c in w2])
check("Alg2 iterate as reported", [0.5, 0.5, -1.999998],
      [round(float(c), 6) for c in w2])
d1 = lab1._boundary_stats(w1, lab1.PERCEPTRON_X)
d2 = lab1._boundary_stats(w2, lab1.PERCEPTRON_X)
check("Alg1 distance to x(1)", 2.1213, d1[0], 0.0001)
check("Alg1 distance to x(2)", -0.7071, d1[1], 0.0001)
check("Alg2 distance to x(1)", 1.4142, d2[0], 0.0001)
check("Alg2 distance to x(2)", -1.4142, d2[1], 0.0001)
check("Alg1 margin ratio", 0.333, min(map(abs, d1)) / max(map(abs, d1)), 0.001)
check("Alg2 margin ratio of the iterate", 0.9999992,
      min(map(abs, d2)) / max(map(abs, d2)), 1e-6)
d2_exact = lab1._boundary_stats(np.array([0.5, 0.5, -2.0]), lab1.PERCEPTRON_X)
check("Alg2 margin ratio, analytical", 1.0,
      min(map(abs, d2_exact)) / max(map(abs, d2_exact)), 1e-12)

print("\n" + "=" * 72)
print(f"  {len(PASS)} checks passed, {len(FAIL)} failed")
if FAIL:
    print("\n  MISMATCHES:")
    for label, claimed, actual in FAIL:
        print(f"    {label}: report says {claimed}, actual {actual}")
print("=" * 72)
sys.exit(1 if FAIL else 0)
