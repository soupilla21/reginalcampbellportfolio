"""Prints the rendered HTML (see render.ts) to the leave-behind PDFs.

    python3 scripts/first-90-days-pdfs/print.py /tmp/first-90-days-pdf
"""
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/first-90-days-pdf")
DEST = Path(__file__).resolve().parents[2] / "advisory-packages" / "first-90-days" / "files"
JOBS = [("summary.html", "first-decision-by-day-90-summary.pdf"), ("kit.html", "governance-starter-kit.pdf")]


async def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        for src, dest in JOBS:
            await page.goto((SRC / src).as_uri(), wait_until="load")
            await page.pdf(path=str(DEST / dest), prefer_css_page_size=True, print_background=True)
            print("wrote", DEST / dest)
        await browser.close()


asyncio.run(main())
