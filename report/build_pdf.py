"""Render report/report.md to report/Lab1_Report.pdf.

Layout follows a conventional academic specification: a serif body face for
reading endurance, sans-serif headings as visual anchors, 1 inch margins,
three-line (booktabs) tables with no vertical rules, and numbered figure
captions set tight beneath each figure.

Usage:  python3 report/build_pdf.py
Needs:  pip install markdown pillow playwright
"""
import base64
import glob
import io
import os
import pathlib
import re

import markdown
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parent.parent
TITLE = "SC4061 Computer Vision: Laboratory 1"
AUTHOR = "AMARAVEL BABU ARRAN (U2423193C)"

md = (ROOT / "report" / "report.md").read_text()
html_body = markdown.markdown(md, extensions=["tables", "fenced_code"])


def inline_image(m):
    """Embed each figure as a compressed JPEG so the PDF is self-contained.

    The source PNGs are 150 dpi matplotlib output and inline to roughly 17 MB
    of base64, which is awkward to upload; re-encoding keeps them legible at
    A4 width while cutting the file by about a factor of eight.
    """
    src = m.group(1)
    path = (ROOT / "report" / src).resolve()
    if not path.exists():
        print("MISSING IMAGE:", src)
        return m.group(0)
    im = Image.open(path).convert("RGB")
    max_w = 1700
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82, optimize=True, progressive=True)
    return 'src="data:image/jpeg;base64,{}"'.format(base64.b64encode(buf.getvalue()).decode())


html_body, n_img = re.subn(r'src="([^"]+\.png)"', inline_image, html_body)

# Wrap every figure in <figure> and number its caption from the alt text.
counter = {"n": 0}


def caption(m):
    counter["n"] += 1
    tag, alt = m.group(0), m.group(1)
    return ('<figure>{}<figcaption>Figure {}. {}</figcaption></figure>'
            .format(tag, counter["n"], alt))


html_body = re.sub(r'<img alt="([^"]*)"[^>]*>', caption, html_body)
print("inlined {} figures, captioned {}".format(n_img, counter["n"]))

CSS = """
@page { size: A4; margin: 25.4mm; }

body { font-family: "Liberation Serif", "Times New Roman", "Nimbus Roman", serif;
       font-size: 10.5pt; line-height: 1.32; color: #111111; background: #FFFFFF;
       text-align: justify; hyphens: auto; }

h1, h2, h3, h4 { font-family: "Liberation Sans", Helvetica, Arial, sans-serif;
                 color: #111111; text-align: left; }

/* Title block */
h1 { font-size: 22pt; font-weight: bold; text-align: center;
     margin: 0 0 6pt 0; line-height: 1.22; }
h1 + h3 { font-size: 12pt; font-weight: normal; text-align: center;
          color: #1A365D; margin: 0 0 14pt 0; letter-spacing: .2px; }
h1 + h3 + p { text-align: center; font-size: 10.5pt; line-height: 1.5;
              margin-bottom: 4pt; }

h2 { font-size: 14pt; font-weight: bold; margin: 20pt 0 7pt 0;
     padding-bottom: 3pt; border-bottom: 0.8pt solid #1A365D;
     page-break-after: avoid; }
h3 { font-size: 12pt; font-weight: bold; font-style: italic;
     margin: 13pt 0 5pt 0; page-break-after: avoid; }

p { margin: 0 0 7pt 0; }
ul, ol { margin: 0 0 7pt 0; padding-left: 17pt; }
li { margin-bottom: 3pt; }

/* Three-line tables: horizontal rules only, never vertical. */
table { border-collapse: collapse; margin: 9pt auto 4pt auto; font-size: 9pt;
        font-family: "Liberation Sans", Helvetica, Arial, sans-serif;
        border-top: 1.1pt solid #111111; border-bottom: 1.1pt solid #111111;
        page-break-inside: avoid; }
thead th { border-bottom: 0.6pt solid #111111; font-weight: bold;
           padding: 4pt 10pt; text-align: left; background: none; }
td { padding: 3.2pt 10pt; border: none; text-align: left; }
tbody tr { background: none; }

figure { margin: 10pt 0 11pt 0; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; height: auto; display: block; margin: 0 auto; }
figcaption { font-size: 8.5pt; line-height: 1.35; color: #111111;
             font-family: "Liberation Sans", Helvetica, Arial, sans-serif;
             margin-top: 3.5pt; text-align: center; letter-spacing: .1px; }

code { font-family: "Liberation Mono", "DejaVu Sans Mono", monospace;
       font-size: 9pt; }
pre { font-size: 8.4pt; line-height: 1.38; background: #FAFAFA;
      border: 0.5pt solid #D8DEE6; border-left: 2pt solid #1A365D;
      padding: 7pt 10pt; margin: 8pt 0; overflow-x: auto;
      page-break-inside: avoid; text-align: left; }
pre code { font-size: 8.4pt; }

/* Personal commentary: set apart but kept sober. */
blockquote { margin: 9pt 0; padding: 7pt 12pt; border-left: 2pt solid #1A365D;
             background: #F7F9FC; page-break-inside: avoid; font-size: 10pt; }
blockquote p { margin: 4pt 0; }

hr { border: none; border-top: 0.6pt solid #1A365D; margin: 14pt 0; }
strong { font-weight: bold; }
a { color: #1A365D; text-decoration: none; }
"""

html = ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>{}</style></head><body>{}</body></html>".format(CSS, html_body))
out_html = ROOT / "report" / "_report.html"
out_html.write_text(html)

FOOT = ("<div style=\"width:100%;font-size:8pt;color:#777777;text-align:center;"
        "font-family:'Liberation Sans',Helvetica,Arial,sans-serif\">"
        "<span class='pageNumber'></span></div>")
HEAD = ("<div style=\"width:100%;font-size:7.5pt;color:#999999;padding:0 25.4mm;"
        "font-family:'Liberation Sans',Helvetica,Arial,sans-serif;"
        "display:flex;justify-content:space-between\">"
        "<span>{}</span><span>{}</span></div>".format(TITLE, AUTHOR))

pdf = ROOT / "report" / "Lab1_Report.pdf"
exe = os.environ.get("CHROMIUM_PATH") or next(
    iter(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome")), None)
with sync_playwright() as pw:
    browser = pw.chromium.launch(executable_path=exe, args=["--no-sandbox"])
    page = browser.new_page()
    page.goto("file://" + str(out_html))
    page.wait_for_timeout(2500)
    page.pdf(path=str(pdf), format="A4", print_background=True,
             margin={"top": "22mm", "bottom": "20mm",
                     "left": "25.4mm", "right": "25.4mm"},
             display_header_footer=True,
             header_template=HEAD, footer_template=FOOT)
    browser.close()
print("PDF:", pdf, "{:.0f} KB".format(pdf.stat().st_size / 1024))
