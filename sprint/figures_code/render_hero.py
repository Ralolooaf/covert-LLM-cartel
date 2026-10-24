"""Render hero.html to hero.png (3200x1800, i.e. 1600x900 at 2x) with headless Chromium (Playwright).
Usage: python render_hero.py [hero.html] [hero.png]
Needs the Inter and Noto Color Emoji fonts installed."""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

src = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("hero.html")).resolve()
out = Path(sys.argv[2] if len(sys.argv) > 2 else src.with_suffix(".png"))
with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1600, "height": 900}, device_scale_factor=2)
    page.goto(src.as_uri())
    page.wait_for_timeout(800)
    bottom = page.evaluate("() => document.getElementById('wrap').lastElementChild.getBoundingClientRect().bottom")
    if bottom > 900:
        print(f"warning: content ends at {bottom:.0f}px, below the 900px canvas")
    page.screenshot(path=str(out))
    browser.close()
print("wrote", out)
