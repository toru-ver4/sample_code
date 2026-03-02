// my_capture_app.exe output.jxr
//
// Win11 + HDR環境を想定。
// Windows.Graphics.Capture (WGC) でモニタ全体を FP16 (R16G16B16A16Float) で1フレーム取得し、WIC(JPEG XR)で保存。
// 依存: Windows SDK, MSVC Build Tools
//
// 参考:
// - Screen capture (Windows.Graphics.Capture) 概要: https://learn.microsoft.com/windows/uwp/audio-video-camera/screen-capture :contentReference[oaicite:8]{index=8}
// - Direct3D11CaptureFramePool::CreateFreeThreaded: https://learn.microsoft.com/uwp/api/windows.graphics.capture.direct3d11captureframepool.createfreethreaded :contentReference[oaicite:9]{index=9}
// - JPEG XR Codec Overview (floating-point 対応): https://learn.microsoft.com/windows/win32/wic/jpeg-xr-codec :contentReference[oaicite:10]{index=10}

#include <windows.h>
#include <wrl/client.h>

#include <d3d11.h>
#include <dxgi1_2.h>

#include <wincodec.h> // WIC
#include <algorithm>
#include <vector>

// C++/WinRT
#include <winrt/base.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>

// Interop: WinRT surface -> ID3D11Texture2D
#include <windows.graphics.directx.direct3d11.interop.h>

// GraphicsCaptureItem を HWND/HMONITOR から作るための interop
#include <windows.graphics.capture.interop.h>

using Microsoft::WRL::ComPtr;

static void ThrowIfFailed(HRESULT hr, const char* msg)
{
    if (FAILED(hr)) {
        winrt::throw_hresult(hr);
    }
}

static ComPtr<ID3D11Device> CreateD3D11Device(ComPtr<ID3D11DeviceContext>& ctx)
{
    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#if defined(_DEBUG)
    flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    D3D_FEATURE_LEVEL levels[] = {
        D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0
    };
    ComPtr<ID3D11Device> dev;
    D3D_FEATURE_LEVEL fl{};
    HRESULT hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        flags,
        levels,
        _countof(levels),
        D3D11_SDK_VERSION,
        dev.GetAddressOf(),
        &fl,
        ctx.GetAddressOf()
    );
    ThrowIfFailed(hr, "D3D11CreateDevice failed");
    return dev;
}

static winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice CreateWinRTD3DDevice(ID3D11Device* d3d11)
{
    ComPtr<IDXGIDevice> dxgi;
    ThrowIfFailed(d3d11->QueryInterface(IID_PPV_ARGS(dxgi.GetAddressOf())), "QI IDXGIDevice failed");

    winrt::com_ptr<IInspectable> insp;
    ThrowIfFailed(CreateDirect3D11DeviceFromDXGIDevice(dxgi.Get(), insp.put()), "CreateDirect3D11DeviceFromDXGIDevice failed");
    return insp.as<winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice>();
}

static winrt::Windows::Graphics::Capture::GraphicsCaptureItem CreateItemForMonitor(HMONITOR mon)
{
    auto interopFactory = winrt::get_activation_factory<
        winrt::Windows::Graphics::Capture::GraphicsCaptureItem,
        IGraphicsCaptureItemInterop>();

    winrt::Windows::Graphics::Capture::GraphicsCaptureItem item{ nullptr };
    HRESULT hr = interopFactory->CreateForMonitor(
        mon,
        winrt::guid_of<winrt::Windows::Graphics::Capture::GraphicsCaptureItem>(),
        winrt::put_abi(item)
    );
    ThrowIfFailed(hr, "CreateForMonitor failed");
    return item;
}

static ComPtr<ID3D11Texture2D> GetFrameTexture(winrt::Windows::Graphics::Capture::Direct3D11CaptureFrame const& frame)
{
    auto surface = frame.Surface();
    //auto access = surface.as<winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
    auto access = surface.as<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();

    ComPtr<ID3D11Texture2D> tex;
    HRESULT hr = access->GetInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void**>(tex.GetAddressOf()));
    ThrowIfFailed(hr, "GetInterface(ID3D11Texture2D) failed");
    return tex;
}

static void SaveAsJXR_Float16RGBA(
    const wchar_t* path,
    const uint16_t* rgbaHalf, // RGBA half-float interleaved
    UINT width,
    UINT height,
    UINT strideBytes)
{
    // WIC init
    ComPtr<IWICImagingFactory> factory;
    ThrowIfFailed(CoCreateInstance(
        CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(factory.GetAddressOf())), "CoCreateInstance(WIC) failed");

    ComPtr<IWICStream> stream;
    ThrowIfFailed(factory->CreateStream(stream.GetAddressOf()), "CreateStream failed");
    ThrowIfFailed(stream->InitializeFromFilename(path, GENERIC_WRITE), "InitializeFromFilename failed");

    ComPtr<IWICBitmapEncoder> encoder;
    // JPEG XR container
    ThrowIfFailed(factory->CreateEncoder(GUID_ContainerFormatWmp, nullptr, encoder.GetAddressOf()), "CreateEncoder(JXR) failed");
    ThrowIfFailed(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache), "Encoder Initialize failed");

    ComPtr<IWICBitmapFrameEncode> frame;
    ComPtr<IPropertyBag2> props;
    ThrowIfFailed(encoder->CreateNewFrame(frame.GetAddressOf(), props.GetAddressOf()), "CreateNewFrame failed");

    // Request lossless JPEG XR encoding.
    PROPBAG2 option{};
    option.pstrName = const_cast<LPOLESTR>(L"Lossless");
    VARIANT value{};
    value.vt = VT_BOOL;
    value.boolVal = VARIANT_TRUE;
    ThrowIfFailed(props->Write(1, &option, &value), "Set Lossless property failed");

    ThrowIfFailed(frame->Initialize(props.Get()), "Frame Initialize failed");

    ThrowIfFailed(frame->SetSize(width, height), "SetSize failed");

    // 64bpp RGBA Half (FP16x4)
    ThrowIfFailed(frame->SetPixelFormat(const_cast<WICPixelFormatGUID*>(&GUID_WICPixelFormat64bppRGBAHalf)),
                  "SetPixelFormat(64bppRGBAHalf) failed");

    // 書き込み
    ThrowIfFailed(frame->WritePixels(height, strideBytes, strideBytes * height, (BYTE*)rgbaHalf), "WritePixels failed");

    ThrowIfFailed(frame->Commit(), "Frame Commit failed");
    ThrowIfFailed(encoder->Commit(), "Encoder Commit failed");
}

static BOOL CALLBACK EnumMonitorsProc(HMONITOR monitor, HDC, LPRECT, LPARAM userData)
{
    auto monitors = reinterpret_cast<std::vector<HMONITOR>*>(userData);
    monitors->push_back(monitor);
    return TRUE;
}

static std::vector<HMONITOR> EnumerateMonitorsOrdered()
{
    std::vector<HMONITOR> monitors;
    if (!EnumDisplayMonitors(nullptr, nullptr, EnumMonitorsProc, reinterpret_cast<LPARAM>(&monitors))) {
        ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()), "EnumDisplayMonitors failed");
    }

    std::sort(monitors.begin(), monitors.end(), [](HMONITOR a, HMONITOR b) {
        MONITORINFOEXW ia{};
        MONITORINFOEXW ib{};
        ia.cbSize = sizeof(ia);
        ib.cbSize = sizeof(ib);
        GetMonitorInfoW(a, &ia);
        GetMonitorInfoW(b, &ib);
        if (ia.rcMonitor.top != ib.rcMonitor.top) {
            return ia.rcMonitor.top < ib.rcMonitor.top;
        }
        return ia.rcMonitor.left < ib.rcMonitor.left;
    });

    return monitors;
}

int wmain(int argc, wchar_t** argv)
{
    if (argc < 3) {
        wprintf(L"Usage: %s <display-number> <output.jxr>\n", argv[0]);
        return 2;
    }
    wchar_t* end = nullptr;
    long displayNumber = wcstol(argv[1], &end, 10);
    if (end == argv[1] || *end != L'\0' || displayNumber <= 0) {
        wprintf(L"Invalid display-number: %s\n", argv[1]);
        return 2;
    }
    const wchar_t* outputPath = argv[2];

    // COM + WinRT init
    ThrowIfFailed(CoInitializeEx(nullptr, COINIT_MULTITHREADED), "CoInitializeEx failed");
    winrt::init_apartment(winrt::apartment_type::multi_threaded);

    // D3D11 device
    ComPtr<ID3D11DeviceContext> ctx;
    auto d3d = CreateD3D11Device(ctx);
    auto winrtDev = CreateWinRTD3DDevice(d3d.Get());

    auto monitors = EnumerateMonitorsOrdered();
    if (displayNumber > static_cast<long>(monitors.size())) {
        wprintf(L"Display-number out of range: %ld (available: 1..%zu)\n", displayNumber, monitors.size());
        for (size_t i = 0; i < monitors.size(); ++i) {
            MONITORINFOEXW info{};
            info.cbSize = sizeof(info);
            if (GetMonitorInfoW(monitors[i], &info)) {
                wprintf(L"  %zu: %s rect=(%ld,%ld)-(%ld,%ld)\n",
                        i + 1,
                        info.szDevice,
                        info.rcMonitor.left,
                        info.rcMonitor.top,
                        info.rcMonitor.right,
                        info.rcMonitor.bottom);
            }
        }
        return 2;
    }

    HMONITOR mon = monitors[displayNumber - 1];
    {
        MONITORINFOEXW info{};
        info.cbSize = sizeof(info);
        if (GetMonitorInfoW(mon, &info)) {
            wprintf(L"Capturing display %ld: %s\n", displayNumber, info.szDevice);
        } else {
            wprintf(L"Capturing display %ld\n", displayNumber);
        }
    }

    auto item = CreateItemForMonitor(mon);
    auto size = item.Size();

    using namespace winrt::Windows::Graphics::Capture;
    using namespace winrt::Windows::Graphics::DirectX;

    // FP16でフレームプール作成
    auto framePool = Direct3D11CaptureFramePool::CreateFreeThreaded(
        winrtDev,
        DirectXPixelFormat::R16G16B16A16Float,
        1,
        size);

    auto session = framePool.CreateCaptureSession(item);

    // 1フレーム到着を待つ
    HANDLE ev = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    winrt::event_token token = framePool.FrameArrived([&](auto const& sender, auto const&) {
        SetEvent(ev);
    });

    session.StartCapture();

    DWORD wait = WaitForSingleObject(ev, 5000);
    framePool.FrameArrived(token);
    CloseHandle(ev);

    if (wait != WAIT_OBJECT_0) {
        wprintf(L"Timed out waiting for frame.\n");
        return 1;
    }

    auto frame = framePool.TryGetNextFrame();
    if (!frame) {
        wprintf(L"TryGetNextFrame returned null.\n");
        return 1;
    }

    auto tex = GetFrameTexture(frame);

    // テクスチャをCPUで読める staging にコピーして map
    D3D11_TEXTURE2D_DESC desc{};
    tex->GetDesc(&desc);

    // 期待：DXGI_FORMAT_R16G16B16A16_FLOAT
    if (desc.Format != DXGI_FORMAT_R16G16B16A16_FLOAT) {
        // ここに来た場合、HDR/FP16として取れていない可能性が高い
        wprintf(L"Warning: captured format is not R16G16B16A16_FLOAT. Format=%u\n", (unsigned)desc.Format);
    }

    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0;

    ComPtr<ID3D11Texture2D> staging;
    ThrowIfFailed(d3d->CreateTexture2D(&desc, nullptr, staging.GetAddressOf()), "CreateTexture2D(staging) failed");

    ctx->CopyResource(staging.Get(), tex.Get());

    D3D11_MAPPED_SUBRESOURCE mapped{};
    ThrowIfFailed(ctx->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped), "Map failed");

    // mapped.pData は RGBA half-float (interleaved) として扱える
    // stride は mapped.RowPitch
    SaveAsJXR_Float16RGBA(outputPath,
                         reinterpret_cast<const uint16_t*>(mapped.pData),
                         desc.Width,
                         desc.Height,
                         mapped.RowPitch);

    ctx->Unmap(staging.Get(), 0);

    wprintf(L"Saved: %s\n", outputPath);

    CoUninitialize();
    return 0;
}
