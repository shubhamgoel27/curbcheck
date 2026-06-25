"""Load the Space, run an inference, wait as long as needed, screenshot the result."""
import os
from playwright.sync_api import sync_playwright

URL = "https://build-small-hackathon-curbcheck.hf.space"
IMG = os.path.abspath("hf_space/examples/dpw_0b6a0cec08.jpg")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_context(viewport={"width": 1280, "height": 800},
                               device_scale_factor=2).new_page()
    page.goto(URL, wait_until="load", timeout=90000)
    page.wait_for_timeout(4000)
    page.locator("input[type=file]").first.set_input_files(IMG)
    page.wait_for_timeout(3000)
    page.get_by_role("button", name="Can I park here?").click()
    print("clicked, waiting up to 240s for result...")
    found = False
    try:
        page.get_by_text("What the model read", exact=False).wait_for(timeout=240000)
        found = True
        print("RESULT FOUND")
    except Exception as e:
        print("NO RESULT in 240s:", str(e)[:80])
    page.evaluate("window.scrollTo(0,0)")
    page.wait_for_timeout(1500)
    page.screenshot(path="/tmp/result_top.png")
    print("screenshot saved, found =", found)
    browser.close()
