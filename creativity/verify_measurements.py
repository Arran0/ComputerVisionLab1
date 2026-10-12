#!/usr/bin/env python3
"""Re-measure every number report/report.md quotes for section 2.8.

Those figures come from running notch_filter_studio.html in a browser, so they
cannot be checked by the Python mirror. This drives the real page with
Playwright and compares. It exists because an earlier edit to the test
pattern's interference frequency silently invalidated a whole table.

Usage:  python3 creativity/verify_measurements.py
Needs:  pip install playwright
"""
import glob
import os
import pathlib
import sys

from playwright.sync_api import sync_playwright

PAGE = pathlib.Path(__file__).resolve().parent / "notch_filter_studio.html"

# Values exactly as the report states them.
SOURCES = {            # source      pairs  before    after
    "clean":        (0, None,  None),
    "interference": (1, 18.29, 51.32),
    "fence":        (3, 19.01, 32.95),
}
RADIUS_SWEEP = {       # radius: (ideal, gaussian)
    1: (55.50, 68.87), 2: (51.32, 57.56), 3: (47.47, 53.40),
    5: (42.45, 48.28), 8: (38.12, 43.76),
}
FIXED_COUNT = {1: 51.32, 2: 31.61, 3: 28.53, 8: 28.53}
CLEAN_FORCED = {1: 37.66, 8: 25.59}

TOL = 0.02
fails = []


def psnr(pg):
    return float(pg.inner_text("#psnrStat").split()[0])


def check(label, claimed, actual, tol=TOL):
    ok = abs(actual - claimed) <= tol
    print(("  ok   " if ok else "  FAIL ") +
          f"{label:<46s} report={claimed:<9} actual={actual}")
    if not ok:
        fails.append(label)


exe = os.environ.get("CHROMIUM_PATH") or next(
    iter(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome")), None)

with sync_playwright() as pw:
    browser = pw.chromium.launch(executable_path=exe, args=["--no-sandbox"])
    pg = browser.new_page()
    pg.goto("file://" + str(PAGE))
    pg.wait_for_timeout(2500)

    print("Detector on each source")
    for src, (npair, before, after) in SOURCES.items():
        pg.select_option("#source", src)
        pg.wait_for_timeout(1800)
        pg.evaluate("notches = []; applyNotchesAndInvert();")
        pg.wait_for_timeout(800)
        pre = pg.inner_text("#psnrStat")
        pg.evaluate("runDetect()")
        pg.wait_for_timeout(1800)
        check(f"{src}: pairs found", npair, pg.evaluate("notches.length"), 0)
        if before is not None:
            check(f"{src}: PSNR before", before, float(pre.split()[0]))
            check(f"{src}: PSNR after", after, psnr(pg))
        else:
            ok = "∞" in pre and "∞" in pg.inner_text("#psnrStat")
            print(("  ok   " if ok else "  FAIL ") +
                  f"{'clean: left untouched':<46s} report=inf      actual={pg.inner_text('#psnrStat')}")
            if not ok:
                fails.append("clean untouched")

    print("\nIdeal versus Gaussian notch, by radius")
    pg.select_option("#source", "interference")
    pg.wait_for_timeout(1500)
    for r, (ideal, gauss) in RADIUS_SWEEP.items():
        for prof, claimed in (("ideal", ideal), ("gauss", gauss)):
            pg.select_option("#profile", prof)
            pg.evaluate(f"document.getElementById('radius').value={r}; runDetect();")
            pg.wait_for_timeout(900)
            check(f"radius {r}, {prof}", claimed, psnr(pg))

    print("\nForcing a fixed pair count (the rejected design)")
    pg.select_option("#profile", "ideal")
    pg.evaluate("document.getElementById('radius').value=2")
    for k, claimed in FIXED_COUNT.items():
        pg.evaluate(f"notches = detectPeaks(0.0001, {k}); applyNotchesAndInvert();")
        pg.wait_for_timeout(900)
        check(f"{k} pair(s) forced", claimed, psnr(pg))

    print("\nNo prominence test, on a clean image")
    pg.select_option("#source", "clean")
    pg.wait_for_timeout(1600)
    for k, claimed in CLEAN_FORCED.items():
        pg.evaluate(f"notches = detectPeaks(0.0001, {k}); applyNotchesAndInvert();")
        pg.wait_for_timeout(900)
        check(f"{k} pair(s) forced on clean", claimed, psnr(pg))

    browser.close()

print("\n" + "=" * 72)
print(f"  {len(fails)} failed" if fails else "  all section 2.8 numbers verified")
print("=" * 72)
sys.exit(1 if fails else 0)
