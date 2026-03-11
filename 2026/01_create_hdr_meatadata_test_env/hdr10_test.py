# Required packages:
#   pip install playwright
#   python -m playwright install chromium
#
# Usage:
#   python hdr10_test.py

from __future__ import annotations

import ctypes
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright

BASE_URL = "https://toru-ver4.github.io/pages_test/MDCV_CLLI_Test/index.html"
LINK_TEXT_LIST = [
    # "./metadata_img/av1_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-100.mp4",
    # "./metadata_img/av1_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-10000.mp4",
    # "./metadata_img/av1_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-None.mp4",
    # "./metadata_img/av1_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-100.mp4",
    # "./metadata_img/av1_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-10000.mp4",
    # "./metadata_img/av1_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-None.mp4",
    # "./metadata_img/av1_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-100.mp4",
    # "./metadata_img/av1_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-10000.mp4",
    # "./metadata_img/av1_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-None.mp4",
    # "./metadata_img/av1_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-100.mp4",
    # "./metadata_img/av1_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.mp4",
    # "./metadata_img/av1_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-None.mp4",
    # "./metadata_img/av1_mdcv-p-None_mdcv-l-None_clli-100.mp4",
    # "./metadata_img/av1_mdcv-p-None_mdcv-l-None_clli-10000.mp4",
    # "./metadata_img/av1_mdcv-p-None_mdcv-l-None_clli-None.mp4",
    # "./metadata_img/hevc_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-100.mp4",
    # "./metadata_img/hevc_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-10000.mp4",
    # "./metadata_img/hevc_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-None.mp4",
    # "./metadata_img/hevc_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-100.mp4",
    # "./metadata_img/hevc_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-10000.mp4",
    # "./metadata_img/hevc_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-None.mp4",
    # "./metadata_img/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-100.mp4",
    # "./metadata_img/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-10000.mp4",
    # "./metadata_img/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-None.mp4",
    # "./metadata_img/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-100.mp4",
    # "./metadata_img/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.mp4",
    # "./metadata_img/hevc_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-None.mp4",
    # "./metadata_img/hevc_mdcv-p-None_mdcv-l-None_clli-100.mp4",
    # "./metadata_img/hevc_mdcv-p-None_mdcv-l-None_clli-10000.mp4",
    # "./metadata_img/hevc_mdcv-p-None_mdcv-l-None_clli-None.mp4",
    # "./metadata_img/avif_mdcv-p-None_mdcv-l-None_clli-100.avif",
    # "./metadata_img/avif_mdcv-p-None_mdcv-l-None_clli-10000.avif",
    # "./metadata_img/avif_mdcv-p-None_mdcv-l-None_clli-None.avif",
    # "./metadata_img/png_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-100.png",
    # "./metadata_img/png_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-10000.png",
    # "./metadata_img/png_mdcv-p-ITU-R BT.709_mdcv-l-100_clli-None.png",
    # "./metadata_img/png_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-100.png",
    # "./metadata_img/png_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-10000.png",
    # "./metadata_img/png_mdcv-p-ITU-R BT.709_mdcv-l-10000_clli-None.png",
    # "./metadata_img/png_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-100.png",
    # "./metadata_img/png_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-10000.png",
    # "./metadata_img/png_mdcv-p-ITU-R BT.2020_mdcv-l-100_clli-None.png",
    # "./metadata_img/png_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-100.png",
    # "./metadata_img/png_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-10000.png",
    # "./metadata_img/png_mdcv-p-ITU-R BT.2020_mdcv-l-10000_clli-None.png",
    "./metadata_img/png_mdcv-p-None_mdcv-l-None_clli-100.png",
    "./metadata_img/png_mdcv-p-None_mdcv-l-None_clli-10000.png",
    "./metadata_img/png_mdcv-p-None_mdcv-l-None_clli-None.png",
]

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
PLAYWRIGHT_ENVIRONMENT_PATH = DATA_DIR / "playwright_environment.json"
CAPTURE_EXE = SCRIPT_DIR / "capture_scRGB" / "build" / "my_capture_app.exe"
CAPTURE_OUTPUT_DIR = SCRIPT_DIR / "capture_img"
CAPTURE_TARGET_DISPLAY_NUMBER = 1
DISPLAY_GEOMETRY = (-1920, 0, 1920, 1080)
CAPTURE_WAIT_SECONDS = 2
# DISPLAY_GEOMETRY = (0, 0, 1920, 1080)
# CAPTURE_WAIT_SECONDS = 2

user32 = ctypes.windll.user32

JS_DUMP = r"""
() => {
  const mm = (q) => {
    try {
      return matchMedia(q).matches;
    } catch {
      return null;
    }
  };

  const safe = (fn) => {
    try {
      return fn();
    } catch (e) {
      return { error: String(e) };
    }
  };

  const canvas2dInfo = safe(() => {
    const canvas = document.createElement("canvas");

    const ctxDefault = canvas.getContext("2d");
    const ctxP3 = canvas.getContext("2d", { colorSpace: "display-p3" });
    const ctxFloat16 = canvas.getContext("2d", { colorType: "float16" });
    const ctxP3Float16 = canvas.getContext("2d", {
      colorSpace: "display-p3",
      colorType: "float16",
    });

    return {
      defaultContext: !!ctxDefault,
      displayP3Context: !!ctxP3,
      float16Context: !!ctxFloat16,
      displayP3Float16Context: !!ctxP3Float16,
    };
  });

  const webglInfo = safe(() => {
    const canvas = document.createElement("canvas");
    const gl =
      canvas.getContext("webgl2") ||
      canvas.getContext("webgl") ||
      canvas.getContext("experimental-webgl");

    if (!gl) {
      return { supported: false };
    }

    return {
      supported: true,
      drawingBufferColorSpace:
        "drawingBufferColorSpace" in gl ? gl.drawingBufferColorSpace : null,
      unpackColorSpace:
        "unpackColorSpace" in gl ? gl.unpackColorSpace : null,
    };
  });

  return {
    userAgent: navigator.userAgent,

    mediaQueries: {
      dynamicRangeStandard: mm("(dynamic-range: standard)"),
      dynamicRangeHigh: mm("(dynamic-range: high)"),

      colorGamutSrgb: mm("(color-gamut: srgb)"),
      colorGamutP3: mm("(color-gamut: p3)"),
      colorGamutRec2020: mm("(color-gamut: rec2020)"),

      videoDynamicRangeStandard: mm("(video-dynamic-range: standard)"),
      videoDynamicRangeHigh: mm("(video-dynamic-range: high)"),

      videoColorGamutSrgb: mm("(video-color-gamut: srgb)"),
      videoColorGamutP3: mm("(video-color-gamut: p3)"),
      videoColorGamutRec2020: mm("(video-color-gamut: rec2020)"),

      forcedColorsNone: mm("(forced-colors: none)"),
      forcedColorsActive: mm("(forced-colors: active)"),

      prefersContrastNoPreference: mm("(prefers-contrast: no-preference)"),
      prefersContrastMore: mm("(prefers-contrast: more)"),

      prefersColorSchemeLight: mm("(prefers-color-scheme: light)"),
      prefersColorSchemeDark: mm("(prefers-color-scheme: dark)"),
    },

    screen: {
      width: screen.width,
      height: screen.height,
      availWidth: screen.availWidth,
      availHeight: screen.availHeight,
      colorDepth: screen.colorDepth,
      pixelDepth: screen.pixelDepth,
      devicePixelRatio: window.devicePixelRatio,
      orientationType: screen.orientation?.type ?? null,
      orientationAngle: screen.orientation?.angle ?? null,
      isExtended: "isExtended" in screen ? screen.isExtended : null,
    },

    canvas2d: canvas2dInfo,
    webgl: webglInfo,
  };
}
"""


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
            kwargs: dict[str, Any] = {
                "headless": False,
                "args": args,
                "ignore_default_args": ["--force-color-profile=srgb"],
            }
            if channel is not None:
                kwargs["channel"] = channel
            return playwright.chromium.launch(**kwargs)
        except Exception as exc:
            last_error = exc
            logging.warning("Failed to launch %s: %s", name, exc)
    raise RuntimeError(f"Failed to launch any Chromium browser: {last_error}")


def make_capture_output_path(href: str) -> Path:
    out_name = f"{Path(Path(href).name).stem}.jxr"
    return CAPTURE_OUTPUT_DIR / out_name


def run_capture_exe(output_path: Path) -> None:
    cmd = [str(CAPTURE_EXE), str(CAPTURE_TARGET_DISPLAY_NUMBER), str(output_path)]
    logging.info("Running capture command: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(SCRIPT_DIR), check=True)


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


def apply_cdp_fullscreen_on_display(page: Page, left: int, top: int, width: int, height: int) -> None:
    logging.info("Applying fullscreen via CDP on target display")
    page.bring_to_front()
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
    session.send(
        "Browser.setWindowBounds",
        {
            "windowId": window_id,
            "bounds": {"windowState": "fullscreen"},
        },
    )
    page.wait_for_timeout(500)


def dump_playwright_environment(page: Page) -> None:
    logging.info("Dumping Playwright browser environment: %s", PLAYWRIGHT_ENVIRONMENT_PATH)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    page.goto("about:blank", wait_until="load", timeout=60000)
    result = page.evaluate(JS_DUMP)
    PLAYWRIGHT_ENVIRONMENT_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def process_one_link(
    context: BrowserContext,
    page_a: Page,
    href: str,
    left: int,
    top: int,
    width: int,
    height: int,
) -> None:
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
        apply_cdp_fullscreen_on_display(page_b, left, top, width, height)
        logging.info("Waiting %.1f seconds", CAPTURE_WAIT_SECONDS)
        page_b.wait_for_timeout(int(CAPTURE_WAIT_SECONDS * 1000))
        output_path = make_capture_output_path(href)
        logging.info("Capture output path: %s", output_path)
        run_capture_exe(output_path)
    finally:
        logging.info("Closing page B")
        page_b.close()
        page_a.bring_to_front()


def main() -> int:
    setup_logger()
    # set_dpi_awareness()

    if not CAPTURE_EXE.exists():
        logging.error("Capture EXE not found: %s", CAPTURE_EXE)
        return 1

    CAPTURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    left, top, width, height = DISPLAY_GEOMETRY
    logging.info(
        "Using configured geometry: left=%d top=%d width=%d height=%d",
        left,
        top,
        width,
        height,
    )
    with sync_playwright() as playwright:
        browser: Browser | None = None
        try:
            browser = launch_browser(playwright, left, top, width, height)
            context = browser.new_context(viewport={"width": width, "height": height})
            page_a = context.new_page()
            dump_playwright_environment(page_a)
            page_a.goto(BASE_URL, wait_until="domcontentloaded", timeout=60000)
            enforce_window_on_display2(page_a, left, top, width, height)

            for idx, href in enumerate(LINK_TEXT_LIST, start=1):
                logging.info("[%d/%d] Start", idx, len(LINK_TEXT_LIST))
                try:
                    process_one_link(context, page_a, href, left, top, width, height)
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
