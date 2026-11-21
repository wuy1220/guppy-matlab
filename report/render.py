# -*- coding: utf-8 -*-
"""Rebuild numbers, rebuild the docx, export a PDF with Word, render page PNGs."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCX = HERE / "out" / "MATLAB大作业_零工具箱Transformer.docx"
PDF = HERE / "out" / "preview.pdf"

PS = r"""
$src = '{src}'
$pdf = '{pdf}'
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$w.DisplayAlerts = 0
$doc = $w.Documents.Open($src, $false, $true)
$doc.ExportAsFixedFormat($pdf, 17)
"PAGES " + $doc.ComputeStatistics(2)
$doc.Close(0)
$w.Quit()
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($w) | Out-Null
""".format(src=str(DOCX), pdf=str(PDF))


def run(cmd, enc="gbk", **kw):
    # The console host emits the OEM code page, not UTF-8: decoding as utf-8
    # throws inside subprocess's reader thread and leaves stdout empty, which
    # looks like a silent failure of a call that actually worked.
    r = subprocess.run(cmd, capture_output=True, **kw)
    out = r.stdout.decode(enc, errors="replace")
    err = r.stderr.decode(enc, errors="replace")
    if r.returncode != 0:
        print("FAILED:", " ".join(cmd))
        print(out[-3000:])
        print(err[-3000:])
        sys.exit(1)
    return out


if "--nocollect" not in sys.argv:
    print(run([sys.executable, str(HERE / "collect_numbers.py")], enc="utf-8").strip())
print(run([sys.executable, str(HERE / "build_report.py")], enc="utf-8").strip())

ps1 = HERE / "out" / "_export.ps1"
# utf-8-sig: Windows PowerShell 5.1 reads a BOM-less .ps1 as ANSI, so the
# Chinese characters in the docx path would arrive mangled and Open() would
# fail while still returning exit code 0.
ps1.write_text(PS, encoding="utf-8-sig")
PDF.unlink(missing_ok=True)
out = run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1)])
print(out.strip())
if not PDF.exists():
    print("EXPORT PRODUCED NO PDF")
    sys.exit(1)

import pymupdf  # noqa: E402

d = pymupdf.open(str(PDF))
print("pdf pages:", d.page_count)
for old in (HERE / "out").glob("p*.png"):
    old.unlink()
for i, pg in enumerate(d):
    pg.get_pixmap(dpi=110).save(str(HERE / "out" / f"p{i+1}.png"))
print("rendered", d.page_count, "pages")
