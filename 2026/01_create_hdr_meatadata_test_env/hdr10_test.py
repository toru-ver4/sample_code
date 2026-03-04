# Required packages:
#   pip install playwright
#   python -m playwright install chromium
#
# Usage:
#   python hdr10_test.py

from __future__ import annotations

import ctypes
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright
from screeninfo import get_monitors

BASE_URL = "https://toru-ver4.github.io/pages_test/MDCV_CLLI_Test/index.html"
LINK_TEXT_LIST = [
    "./metadata_img/av1_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-100.mp4",
    "./metadata_img/av1_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-10000.mp4",
    "./metadata_img/av1_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-None.mp4",
    "./metadata_img/av1_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-100.mp4",
    "./metadata_img/av1_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-10000.mp4",
    "./metadata_img/av1_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-None.mp4",
    "./metadata_img/av1_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-100.mp4",
    "./metadata_img/av1_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-10000.mp4",
    "./metadata_img/av1_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-None.mp4",
    "./metadata_img/av1_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-100.mp4",
    "./metadata_img/av1_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.mp4",
    "./metadata_img/av1_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-None.mp4",
    "./metadata_img/av1_mdcv-p-None_mdcv-l-None_clli-100.mp4",
    "./metadata_img/av1_mdcv-p-None_mdcv-l-None_clli-10000.mp4",
    "./metadata_img/av1_mdcv-p-None_mdcv-l-None_clli-None.mp4",
    "./metadata_img/hevc_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-100.mp4",
    "./metadata_img/hevc_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-10000.mp4",
    "./metadata_img/hevc_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-None.mp4",
    "./metadata_img/hevc_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-100.mp4",
    "./metadata_img/hevc_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-10000.mp4",
    "./metadata_img/hevc_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-None.mp4",
    "./metadata_img/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-100.mp4",
    "./metadata_img/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-10000.mp4",
    "./metadata_img/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-None.mp4",
    "./metadata_img/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-100.mp4",
    "./metadata_img/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.mp4",
    "./metadata_img/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-None.mp4",
    "./metadata_img/hevc_mdcv-p-None_mdcv-l-None_clli-100.mp4",
    "./metadata_img/hevc_mdcv-p-None_mdcv-l-None_clli-10000.mp4",
    "./metadata_img/hevc_mdcv-p-None_mdcv-l-None_clli-None.mp4",
    "./metadata_img/avif_mdcv-p-None_mdcv-l-None_clli-100.avif",
    "./metadata_img/avif_mdcv-p-None_mdcv-l-None_clli-10000.avif",
    "./metadata_img/avif_mdcv-p-None_mdcv-l-None_clli-None.avif",
    "./metadata_img/png_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-100.png",
    "./metadata_img/png_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-10000.png",
    "./metadata_img/png_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-None.png",
    "./metadata_img/png_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-100.png",
    "./metadata_img/png_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-10000.png",
    "./metadata_img/png_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-None.png",
    "./metadata_img/png_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-100.png",
    "./metadata_img/png_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-10000.png",
    "./metadata_img/png_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-None.png",
    "./metadata_img/png_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-100.png",
    "./metadata_img/png_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.png",
    "./metadata_img/png_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-None.png",
    "./metadata_img/png_mdcv-p-None_mdcv-l-None_clli-100.png",
    "./metadata_img/png_mdcv-p-None_mdcv-l-None_clli-10000.png",
    "./metadata_img/png_mdcv-p-None_mdcv-l-None_clli-None.png",
]

SCRIPT_DIR = Path(__file__).resolve().parent
CAPTURE_EXE = SCRIPT_DIR / "capture_scRGB" / "build" / "my_capture_app.exe"
CAPTURE_OUTPUT_DIR = SCRIPT_DIR / "capture_img"

user32 = ctypes.windll.user32


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_dpi_awareness() -> None:
    logging.info("Setting DPI awareness")
    try:
        per_monitor_v2 = ctypes.c_void_p(-4)
        if user32.SetProcessDpiAwarenessContext(per_monitor_v2):
            return
    except Exception:
        pass

    try:
        shcore = ctypes.windll.shcore
        shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass

    try:
        user32.SetProcessDPIAware()
    except Exception:
        logging.warning("Failed to set DPI awareness")


def get_display2_geometry() -> tuple[int, int, int, int]:
    logging.info("Detecting Display No.2 geometry")
    raw_monitors = get_monitors()
    if not raw_monitors:
        raise RuntimeError("No monitors found")

    monitors: list[dict[str, Any]] = []
    for monitor in raw_monitors:
        device = str(getattr(monitor, "name", "") or "")
        monitors.append(
            {
                "device": device,
                "left": int(monitor.x),
                "top": int(monitor.y),
                "width": int(monitor.width),
                "height": int(monitor.height),
            }
        )

    for monitor in monitors:
        logging.info(
            "Monitor found: device=%s rect=(%d,%d %dx%d)",
            monitor["device"],
            monitor["left"],
            monitor["top"],
            monitor["width"],
            monitor["height"],
        )

    for monitor in monitors:
        if monitor["device"].upper().endswith("DISPLAY2"):
            logging.info("Using monitor device=%s", monitor["device"])
            return monitor["left"], monitor["top"], monitor["width"], monitor["height"]

    if len(monitors) >= 2:
        monitor = monitors[1]
        logging.warning("DISPLAY2 not found explicitly; fallback to second monitor: %s", monitor["device"])
        return monitor["left"], monitor["top"], monitor["width"], monitor["height"]

    raise RuntimeError("Display No.2 was not found")


def launch_browser(playwright: Playwright, left: int, top: int, width: int, height: int) -> Browser:
    args = [
        "--start-fullscreen",
        "--start-maximized",
        f"--window-position={left},{top}",
        f"--window-size={width},{height}",
    ]
    launch_trials: list[tuple[str | None, str]] = [
        ("msedge", "Microsoft Edge"),
        ("chrome", "Google Chrome"),
        (None, "Bundled Chromium"),
    ]
    last_error: Exception | None = None
    for channel, name in launch_trials:
        try:
            logging.info("Launching browser: %s", name)
            kwargs: dict[str, Any] = {"headless": False, "args": args}
            if channel is not None:
                kwargs["channel"] = channel
            return playwright.chromium.launch(**kwargs)
        except Exception as exc:
            last_error = exc
            logging.warning("Failed to launch %s: %s", name, exc)
    raise RuntimeError(f"Failed to launch any Chromium browser: {last_error}")


def enforce_window_on_display2(page: Page, left: int, top: int, width: int, height: int) -> None:
    logging.info("Enforcing browser window bounds on Display No.2")
    session = page.context.new_cdp_session(page)
    window_id = session.send("Browser.getWindowForTarget")["windowId"]
    session.send(
        "Browser.setWindowBounds",
        {
            "windowId": window_id,
            "bounds": {
                "windowState": "normal",
                "left": left,
                "top": top,
                "width": width,
                "height": height,
            },
        },
    )
    page.set_viewport_size({"width": width, "height": height})


def make_capture_output_path(href: str) -> Path:
    out_name = f"{Path(Path(href).name).stem}.jxr"
    return CAPTURE_OUTPUT_DIR / out_name


def run_capture_exe(output_path: Path) -> None:
    cmd = [str(CAPTURE_EXE), "2", str(output_path)]
    logging.info("Running capture command: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(SCRIPT_DIR), check=True)


def process_one_link(context: BrowserContext, page_a: Page, href: str) -> None:
    logging.info("Processing link: %s", href)
    page_a.bring_to_front()
    page_a.goto(BASE_URL, wait_until="domcontentloaded", timeout=60000)
    selector = f"a[href='{href}']"
    if page_a.locator(selector).count() == 0:
        logging.warning("Link with exact href not found on page A, continuing via direct URL: %s", href)

    page_b = context.new_page()
    try:
        target_url = urljoin(BASE_URL, href)
        logging.info("Opening page B: %s", target_url)
        page_b.bring_to_front()
        page_b.goto(target_url, wait_until="load", timeout=60000)
        logging.info("Waiting 5 seconds")
        page_b.wait_for_timeout(5000)
        output_path = make_capture_output_path(href)
        logging.info("Capture output path: %s", output_path)
        run_capture_exe(output_path)
    finally:
        logging.info("Closing page B")
        page_b.close()
        page_a.bring_to_front()


def main() -> int:
    setup_logger()
    set_dpi_awareness()

    if not CAPTURE_EXE.exists():
        logging.error("Capture EXE not found: %s", CAPTURE_EXE)
        return 1

    CAPTURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        left, top, width, height = get_display2_geometry()
    except Exception:
        logging.exception("Failed to get Display No.2 geometry")
        return 1

    with sync_playwright() as playwright:
        browser: Browser | None = None
        try:
            browser = launch_browser(playwright, left, top, width, height)
            context = browser.new_context(viewport={"width": width, "height": height})
            page_a = context.new_page()
            page_a.goto(BASE_URL, wait_until="domcontentloaded", timeout=60000)
            enforce_window_on_display2(page_a, left, top, width, height)

            for idx, href in enumerate(LINK_TEXT_LIST, start=1):
                logging.info("[%d/%d] Start", idx, len(LINK_TEXT_LIST))
                try:
                    process_one_link(context, page_a, href)
                    logging.info("[%d/%d] Success", idx, len(LINK_TEXT_LIST))
                except Exception:
                    logging.exception("[%d/%d] Failed for href=%s", idx, len(LINK_TEXT_LIST), href)
                    continue

            logging.info("All links processed")
            context.close()
            return 0
        except Exception:
            logging.exception("Fatal error")
            return 1
        finally:
            if browser is not None:
                browser.close()


if __name__ == "__main__":
    sys.exit(main())
