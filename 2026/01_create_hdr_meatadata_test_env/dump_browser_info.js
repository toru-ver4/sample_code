(() => {
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

  const detectBrowser = () => {
    const ua = navigator.userAgent;

    const rules = [
      { name: "Edge", regex: /Edg\/(\d+)/ },
      { name: "Chrome", regex: /Chrome\/(\d+)/ },
      { name: "Firefox", regex: /Firefox\/(\d+)/ },
      { name: "Safari", regex: /Version\/(\d+).+Safari/ }
    ];

    for (const r of rules) {
      const m = ua.match(r.regex);
      if (m) {
        return {
          name: r.name,
          version: m[1]
        };
      }
    }

    return {
      name: "Unknown",
      version: "0"
    };
  };

  const canvas2dInfo = safe(() => {
    const test2d = (options, expected = {}) => {
      const canvas = document.createElement("canvas");
      const ctx =
        options === undefined
          ? canvas.getContext("2d")
          : canvas.getContext("2d", options);

      if (!ctx) return false;

      if (typeof ctx.getContextAttributes !== "function") {
        return null;
      }

      const actual = ctx.getContextAttributes();
      return Object.entries(expected).every(([k, v]) => actual?.[k] === v);
    };

    const base = (() => {
      const canvas = document.createElement("canvas");
      return !!canvas.getContext("2d");
    })();

    const displayP3 = test2d(
      { colorSpace: "display-p3" },
      { colorSpace: "display-p3" }
    );

    const float16 = test2d(
      { colorType: "float16" },
      { colorType: "float16" }
    );

    const displayP3Float16 = test2d(
      { colorSpace: "display-p3", colorType: "float16" },
      { colorSpace: "display-p3", colorType: "float16" }
    );

    return {
      "2d": base,
      "display-p3": displayP3,
      "float16": float16,
      "display-p3 + float16": displayP3Float16
    };
  });

  const webglInfo = safe(() => {
    const canvas = document.createElement("canvas");

    const gl =
      canvas.getContext("webgl2") ||
      canvas.getContext("webgl") ||
      canvas.getContext("experimental-webgl");

    if (!gl) {
      return {
        supported: false,
        drawingBufferColorSpace: null,
        unpackColorSpace: null,
      };
    }

    return {
      supported: true,
      drawingBufferColorSpace:
        "drawingBufferColorSpace" in gl ? gl.drawingBufferColorSpace : null,
      unpackColorSpace:
        "unpackColorSpace" in gl ? gl.unpackColorSpace : null,
    };
  });

  const browser = detectBrowser();

  const result = {
    timestamp: new Date().toISOString(),
    url: location.href,
    title: document.title,
    userAgent: navigator.userAgent,

    mediaQueries: {
      "dynamic-range: standard": mm("(dynamic-range: standard)"),
      "dynamic-range: high": mm("(dynamic-range: high)"),

      "color-gamut: srgb": mm("(color-gamut: srgb)"),
      "color-gamut: p3": mm("(color-gamut: p3)"),
      "color-gamut: rec2020": mm("(color-gamut: rec2020)"),

      "video-dynamic-range: standard": mm("(video-dynamic-range: standard)"),
      "video-dynamic-range: high": mm("(video-dynamic-range: high)"),

      "video-color-gamut: srgb": mm("(video-color-gamut: srgb)"),
      "video-color-gamut: p3": mm("(video-color-gamut: p3)"),
      "video-color-gamut: rec2020": mm("(video-color-gamut: rec2020)"),

      "forced-colors: none": mm("(forced-colors: none)"),
      "forced-colors: active": mm("(forced-colors: active)"),

      "prefers-contrast: no-preference": mm("(prefers-contrast: no-preference)"),
      "prefers-contrast: more": mm("(prefers-contrast: more)"),

      "prefers-color-scheme: light": mm("(prefers-color-scheme: light)"),
      "prefers-color-scheme: dark": mm("(prefers-color-scheme: dark)"),
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

  const json = JSON.stringify(result, null, 2);

  const date = new Date().toISOString().slice(0, 10);

  const filename =
    `dump-browser-info_${date}_${browser.name}-${browser.version}.json`;

  const blob = new Blob([json], { type: "application/json" });
  const blobUrl = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = blobUrl;
  a.download = filename;

  document.body.appendChild(a);
  a.click();
  a.remove();

  setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);

  return result;
})();
