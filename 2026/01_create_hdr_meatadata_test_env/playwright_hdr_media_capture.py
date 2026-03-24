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
DATA_DIR = SCRIPT_DIR / "data"
CAPTURE_EXE = SCRIPT_DIR / "capture_scRGB" / "build" / "my_capture_app.exe"
CAPTURE_OUTPUT_ROOT_DIR = SCRIPT_DIR / "capture_img"
CAPTURE_TARGET_DISPLAY_NUMBER = 1
DISPLAY_GEOMETRY = (-1920, 0, 1920, 1080)
CAPTURE_WAIT_SECONDS = 2
BROWSER_TARGETS = [
    {"name": "Edge", "channel": "msedge"},
    {"name": "Chrome", "channel": "chrome"},
]
# DISPLAY_GEOMETRY = (0, 0, 1920, 1080)
# CAPTURE_WAIT_SECONDS = 2

user32 = ctypes.windll.user32

JS_DUMP = r"""
async () => {
  const mm = (q) => {
    try {
      return window.matchMedia(q).matches;
    } catch (e) {
      return null;
    }
  };

  const safe = async (fn) => {
    try {
      return await fn();
    } catch (e) {
      return { error: String(e) };
    }
  };

  const detectColorBitsPerComponent = () => {
    for (let bits = 48; bits >= 1; bits--) {
      if (mm(`(color: ${bits})`) || mm(`(min-color: ${bits})`)) {
        return bits;
      }
    }
    return 0;
  };

  const getCanvas2dInfo = () => {
    const canvas = document.createElement("canvas");

    const test2d = (options, expected = {}) => {
      try {
        const c = document.createElement("canvas");
        const ctx =
          options === undefined
            ? c.getContext("2d")
            : c.getContext("2d", options);

        if (!ctx) return false;
        if (typeof ctx.getContextAttributes !== "function") return null;

        const actual = ctx.getContextAttributes();
        return Object.entries(expected).every(([k, v]) => actual?.[k] === v);
      } catch (e) {
        return { error: String(e) };
      }
    };

    return {
      supported: !!canvas.getContext("2d"),
      displayP3: test2d(
        { colorSpace: "display-p3" },
        { colorSpace: "display-p3" }
      ),
      float16: test2d(
        { colorType: "float16" },
        { colorType: "float16" }
      ),
      "displayP3 + float16": test2d(
        { colorSpace: "display-p3", colorType: "float16" },
        { colorSpace: "display-p3", colorType: "float16" }
      ),
    };
  };

  const getWebglInfo = () => {
    try {
      const canvas = document.createElement("canvas");

      const gl =
        canvas.getContext("webgl2") ||
        canvas.getContext("webgl") ||
        canvas.getContext("experimental-webgl");

      if (!gl) {
        return { supported: false };
      }

      const info = {
        supported: true,
        context: gl instanceof WebGL2RenderingContext ? "webgl2" : "webgl",
      };

      try {
        info.drawingBufferColorSpace = gl.drawingBufferColorSpace ?? null;
      } catch (e) {
        info.drawingBufferColorSpace = { error: String(e) };
      }

      try {
        info.unpackColorSpace = gl.unpackColorSpace ?? null;
      } catch (e) {
        info.unpackColorSpace = { error: String(e) };
      }

      try {
        const r = gl.getParameter(gl.RED_BITS);
        const g = gl.getParameter(gl.GREEN_BITS);
        const b = gl.getParameter(gl.BLUE_BITS);
        const a = gl.getParameter(gl.ALPHA_BITS);
        info.drawingBufferFormat = `R${r}G${g}B${b}A${a}`;
      } catch (e) {
        info.drawingBufferFormat = { error: String(e) };
      }

      return info;
    } catch (e) {
      return { supported: false, error: String(e) };
    }
  };

  const getMediaCapabilitiesInfo = async () => {
    if (!("mediaCapabilities" in navigator)) {
      return { supported: false };
    }

    const probes = {
      h264_sdr: {
        type: "file",
        video: {
          contentType: 'video/mp4; codecs="avc1.640028"',
          width: 1920,
          height: 1080,
          bitrate: 8000000,
          framerate: 30
        }
      },
      hevc_hdr_pq: {
        type: "file",
        video: {
          contentType: 'video/mp4; codecs="hvc1.2.4.L153.B0"',
          width: 3840,
          height: 2160,
          bitrate: 20000000,
          framerate: 60,
          colorGamut: "rec2020",
          transferFunction: "pq",
          hdrMetadataType: "smpteSt2086"
        }
      },
      av1_hdr_pq: {
        type: "file",
        video: {
          contentType: 'video/mp4; codecs="av01.0.10M.10.0.110.09.16.09.0"',
          width: 3840,
          height: 2160,
          bitrate: 20000000,
          framerate: 60,
          colorGamut: "rec2020",
          transferFunction: "pq",
          hdrMetadataType: "smpteSt2086"
        }
      },
      vp9_hdr_pq: {
        type: "file",
        video: {
          contentType: 'video/webm; codecs="vp09.02.10.10.01.09.16.09.01"',
          width: 3840,
          height: 2160,
          bitrate: 20000000,
          framerate: 60,
          colorGamut: "rec2020",
          transferFunction: "pq"
        }
      }
    };

    const results = {};
    for (const [name, config] of Object.entries(probes)) {
      try {
        const r = await navigator.mediaCapabilities.decodingInfo(config);
        results[name] = !!r?.supported;
      } catch (e) {
        results[name] = false;
      }
    }

    return {
      supported: true,
      ...results,
    };
  };

  return await safe(async () => ({
    timestamp: new Date().toISOString(),
    location: {
      href: location.href,
      origin: location.origin,
      pathname: location.pathname,
    },
    userAgent: navigator.userAgent,
    mediaQueries: {
      "dynamic-range: high": mm("(dynamic-range: high)"),
      "color-gamut: srgb": mm("(color-gamut: srgb)"),
      "color-gamut: p3": mm("(color-gamut: p3)"),
      "color-gamut: rec2020": mm("(color-gamut: rec2020)"),
      "video-dynamic-range: high": mm("(video-dynamic-range: high)"),
      "video-color-gamut: srgb": mm("(video-color-gamut: srgb)"),
      "video-color-gamut: p3": mm("(video-color-gamut: p3)"),
      "video-color-gamut: rec2020": mm("(video-color-gamut: rec2020)"),
      "prefers-color-scheme: light": mm("(prefers-color-scheme: light)"),
      "prefers-color-scheme: dark": mm("(prefers-color-scheme: dark)"),
      colorBitsPerComponent: detectColorBitsPerComponent(),
    },
    mediaCapabilities: await getMediaCapabilitiesInfo(),
    screen: {
      width: screen.width,
      height: screen.height,
      availWidth: screen.availWidth,
      availHeight: screen.availHeight,
      colorDepth: screen.colorDepth,
      pixelDepth: screen.pixelDepth,
      devicePixelRatio: window.devicePixelRatio,
    },
    canvas2d: getCanvas2dInfo(),
    webgl: getWebglInfo(),
  }));
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


def make_playwright_environment_path(browser_name: str) -> Path:
    return DATA_DIR / f"playwright_environment_{browser_name.lower()}.json"


def launch_browser(
    playwright: Playwright,
    browser_name: str,
    channel: str,
    left: int,
    top: int,
    width: int,
    height: int,
) -> Browser:
    args = [
        "--start-fullscreen",
        "--start-maximized",
        f"--window-position={left},{top}",
        f"--window-size={width},{height}",
    ]
    try:
        logging.info("Launching browser: %s", browser_name)
        kwargs: dict[str, Any] = {
            "headless": False,
            "args": args,
            "ignore_default_args": ["--force-color-profile=srgb"],
            "channel": channel,
        }
        return playwright.chromium.launch(**kwargs)
    except Exception as exc:
        raise RuntimeError(f"Failed to launch {browser_name}: {exc}") from exc


def make_capture_output_path(browser_name: str, href: str) -> Path:
    out_name = f"{Path(Path(href).name).stem}.jxr"
    return CAPTURE_OUTPUT_ROOT_DIR / browser_name / out_name


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


def dump_playwright_environment(page: Page, browser_name: str) -> None:
    output_path = make_playwright_environment_path(browser_name)
    logging.info("Dumping %s Playwright browser environment: %s", browser_name, output_path)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    page.goto("about:blank", wait_until="load", timeout=60000)
    result = page.evaluate(JS_DUMP)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def process_one_link(
    browser_name: str,
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
        output_path = make_capture_output_path(browser_name, href)
        logging.info("Capture output path: %s", output_path)
        run_capture_exe(output_path)
    finally:
        logging.info("Closing page B")
        page_b.close()
        page_a.bring_to_front()


def run_browser_sequence(
    playwright: Playwright,
    browser_name: str,
    channel: str,
    left: int,
    top: int,
    width: int,
    height: int,
) -> None:
    browser: Browser | None = None
    context: BrowserContext | None = None
    capture_output_dir = CAPTURE_OUTPUT_ROOT_DIR / browser_name
    capture_output_dir.mkdir(parents=True, exist_ok=True)
    try:
        browser = launch_browser(playwright, browser_name, channel, left, top, width, height)
        context = browser.new_context(viewport={"width": width, "height": height})
        page_a = context.new_page()
        page_a.goto(BASE_URL, wait_until="domcontentloaded", timeout=60000)
        enforce_window_on_display2(page_a, left, top, width, height)
        dump_playwright_environment(page_a, browser_name)

        for idx, href in enumerate(LINK_TEXT_LIST, start=1):
            logging.info("[%s %d/%d] Start", browser_name, idx, len(LINK_TEXT_LIST))
            try:
                process_one_link(browser_name, context, page_a, href, left, top, width, height)
                logging.info("[%s %d/%d] Success", browser_name, idx, len(LINK_TEXT_LIST))
            except Exception:
                logging.exception(
                    "[%s %d/%d] Failed for href=%s",
                    browser_name,
                    idx,
                    len(LINK_TEXT_LIST),
                    href,
                )
                continue

        logging.info("%s: all links processed", browser_name)
    finally:
        if context is not None:
            context.close()
        if browser is not None:
            browser.close()


def main() -> int:
    setup_logger()
    # set_dpi_awareness()

    if not CAPTURE_EXE.exists():
        logging.error("Capture EXE not found: %s", CAPTURE_EXE)
        return 1

    CAPTURE_OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    left, top, width, height = DISPLAY_GEOMETRY
    logging.info(
        "Using configured geometry: left=%d top=%d width=%d height=%d",
        left,
        top,
        width,
        height,
    )
    with sync_playwright() as playwright:
        try:
            for browser_target in BROWSER_TARGETS:
                browser_name = browser_target["name"]
                channel = browser_target["channel"]
                logging.info("Starting browser sequence: %s", browser_name)
                run_browser_sequence(playwright, browser_name, channel, left, top, width, height)
                logging.info("Finished browser sequence: %s", browser_name)

            logging.info("All browser sequences processed")
            return 0
        except Exception:
            logging.exception("Fatal error")
            return 1


if __name__ == "__main__":
    sys.exit(main())
