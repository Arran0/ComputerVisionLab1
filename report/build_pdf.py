"""Render report/report.md to report/Lab1_Report.pdf.

Usage:  python3 report/build_pdf.py
Needs:  pip install markdown pillow playwright
"""
import pathlib, markdown, base64, re
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parent.parent
md = (ROOT / 'report' / 'report.md').read_text()
html_body = markdown.markdown(md, extensions=['tables', 'fenced_code', 'toc'])

# inline every figure so the PDF is self-contained
def inline(m):
    # Re-encode as JPEG at a sane print width. The source PNGs are 150 dpi
    # matplotlib output and inline to ~17 MB of base64, which is too heavy to
    # upload; this keeps them legible at A4 while cutting the file by ~10x.
    import io
    from PIL import Image
    src = m.group(1)
    path = (ROOT / 'report' / src).resolve()
    if not path.exists():
        print('MISSING IMAGE:', src); return m.group(0)
    im = Image.open(path).convert('RGB')
    maxw = 1700
    if im.width > maxw:
        im = im.resize((maxw, round(im.height * maxw / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=82, optimize=True, progressive=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'src="data:image/jpeg;base64,{b64}"'

html_body, n = re.subn(r'src="([^"]+\.png)"', inline, html_body)
print(f'inlined {n} figures')

CSS = """
@page { size: A4; margin: 17mm 15mm; }
body { font-family: "DejaVu Sans", Arial, sans-serif; font-size: 10.2pt;
       line-height: 1.5; color: #15202b; max-width: 100%; }
h1 { font-size: 20pt; border-bottom: 2.5px solid #1f4e79; padding-bottom: 7px;
     color: #1f4e79; margin-top: 0; }
h2 { font-size: 14pt; color: #1f4e79; margin-top: 22px;
     border-bottom: 1px solid #c8d4e0; padding-bottom: 4px;
     page-break-after: avoid; }
h3 { font-size: 11.5pt; color: #2d3f52; margin-top: 15px; page-break-after: avoid; }
table { border-collapse: collapse; margin: 11px 0; font-size: 9.1pt; width: auto; }
th, td { border: 1px solid #b9c6d4; padding: 4px 9px; text-align: left; }
th { background: #eaf0f6; font-weight: 600; }
tr:nth-child(even) td { background: #f7f9fc; }
code { background: #f0f3f7; padding: 1px 4px; border-radius: 3px;
       font-family: "DejaVu Sans Mono", Consolas, monospace; font-size: 8.9pt; }
pre { background: #f7f9fc; border: 1px solid #d6e0ea; border-left: 3.5px solid #1f4e79;
      padding: 9px 12px; border-radius: 4px; overflow-x: auto;
      page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.6pt; line-height: 1.42; }
blockquote { border-left: 3.5px solid #d97706; background: #fffaf2;
             margin: 11px 0; padding: 9px 14px; border-radius: 0 4px 4px 0;
             page-break-inside: avoid; }
blockquote p { margin: 5px 0; font-size: 9.7pt; }
img { max-width: 100%; height: auto; display: block; margin: 11px auto;
      border: 1px solid #d6e0ea; border-radius: 3px; page-break-inside: avoid; }
hr { border: none; border-top: 1px solid #c8d4e0; margin: 22px 0; }
strong { color: #10263a; }
"""

html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_body}</body></html>"
out_html = ROOT / 'report' / '_report.html'
out_html.write_text(html)

pdf = ROOT / 'report' / 'Lab1_Report.pdf'
with sync_playwright() as pw:
    # Chromium is used purely as a print engine. executable_path is only
    # needed where the bundled browser version does not match playwright's.
    import os, glob
    exe = os.environ.get('CHROMIUM_PATH') or next(
        iter(glob.glob('/opt/pw-browsers/chromium-*/chrome-linux/chrome')), None)
    b = pw.chromium.launch(executable_path=exe, args=['--no-sandbox'])
    pg = b.new_page()
    pg.goto('file://' + str(out_html))
    pg.wait_for_timeout(2500)
    pg.pdf(path=str(pdf), format='A4', print_background=True,
           margin={'top':'17mm','bottom':'17mm','left':'15mm','right':'15mm'},
           display_header_footer=True,
           header_template='<div></div>',
           footer_template='<div style="width:100%;font-size:8pt;color:#8899aa;'
                           'text-align:center;font-family:Arial">'
                           'SC4061 Lab 1 &mdash; AMARAVEL BABU ARRAN (U2423193C) '
                           '&mdash; page <span class="pageNumber"></span> of '
                           '<span class="totalPages"></span></div>')
    b.close()
print('PDF:', pdf, f'{pdf.stat().st_size/1024:.0f} KB')
