(async () => {
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

  const data = {
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
  };

  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);

  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const filename = `browser-capability-dump-${timestamp}.json`;

  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();

  setTimeout(() => URL.revokeObjectURL(url), 1000);

  console.log("Downloaded:", filename);
  console.log(data);
})();
