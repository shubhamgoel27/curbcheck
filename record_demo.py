"""Drive the live curbcheck Space headlessly and record a short demo video."""
import os
import time
from playwright.sync_api import sync_playwright

URL = "https://build-small-hackathon-curbcheck.hf.space"
IMG = os.path.abspath("hf_space/examples/dpw_0b6a0cec08.jpg")
os.makedirs("video", exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch()
    ctx = browser.new_context(
        viewport={"width": 1280, "height": 800},
        record_video_dir="video",
        record_video_size={"width": 1280, "height": 800},
        device_scale_factor=2,
    )
    page = ctx.new_page()
    print("loading...")
    page.goto(URL, wait_until="load", timeout=90000)
    page.wait_for_timeout(4000)  # let gradio hydrate
    page.mouse.wheel(0, 0)
    page.wait_for_timeout(2500)  # show the title / intro

    # upload the example photo
    print("uploading photo...")
    page.locator("input[type=file]").first.set_input_files(IMG)
    page.wait_for_timeout(4500)  # image preview

    # click the predict button
    print("clicking predict...")
    page.get_by_role("button", name="Can I park here?").click()
    page.wait_for_timeout(800)

    # wait for the verdict / signs output to appear
    print("waiting for inference...")
    try:
        page.get_by_text("What the model read", exact=False).wait_for(timeout=120000)
        print("got result")
    except Exception as e:
        print("wait fallback:", str(e)[:100])
        page.wait_for_timeout(20000)

    # scroll to top so the photo (left) and the verdict (right) are both in frame
    page.evaluate("window.scrollTo({top:0,behavior:'smooth'})")
    page.wait_for_timeout(8000)  # hold on the result
    path = page.video.path()
    ctx.close()
    browser.close()
    print("RAW VIDEO:", path)
