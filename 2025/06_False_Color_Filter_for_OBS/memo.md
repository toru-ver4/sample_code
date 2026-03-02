# OBS の User-defined shader に関するメモ

## プラグインの URL

https://obsproject.com/forum/resources/obs-shaderfilter.1736/

## 内部空間のデバッグ

### `current space` が `GS_CS_SRGB` の場合

* Image で SDR画像を開くと 0.0-1.0 の Non-Linear に見える
* Window Capture で SDRアプリケーションをキャプチャすると Non-Linear に見える

### `current space` が `GS_CS_709_SCRGB` の場合

* Image で SDR画像を開くと 0.0-1.0 の Non-Linear に見える
* Window Capture で SDRアプリケーションをキャプチャすると Non-Linear に見える

## obs-shaderfilter のカスタムビルドを試す

```
winget install --id=Kitware.CMake
winget install --id=GnuWin32.DiffUtils
winget install --id=GnuWin32.Patch

winget install --id=Microsoft.VisualStudio.2022.BuildTools --exact --override "--add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.ATLMFC --includeRecommended --quiet --wait --norestart"

```

launch "x64 Native Tools Command Prompt for VS 2022"

```
# build OBS Studio
cd C:\home\build_tools
git clone --branch 31.0.3 --recursive https://github.com/obsproject/obs-studio.git

cd C:\home\build_tools\obs-studio
rmdir /s /q .git
rmdir /s /q .github
cd plugins
git clone --branch 2.4.3 https://github.com/exeldro/obs-shaderfilter.git

ここで plugins\CMakeLists.txt にpっちを当てる

cd ..
cmake --preset windows-x64
cmake --build --preset windows-x64

```

とりあえず、obs-shaderfileter.c の `get_input_source` と `draw_output` の `gs_get_format_from_space` の戻り値を確認したい

確認したら、Canvas は GS_CS_709_EXTENDED っぽい。
filter->context が映像ソースっぽい？

## 1.0 を超える値の維持

以下のようにして gs_texrender_create に GS_RGBA16F を食わせる

```
gs_texrender_t *create_or_reset_texrender(gs_texrender_t *render)
{
	if (!render) {
		// render = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
		render = gs_texrender_create(GS_RGBA16F, GS_ZS_NONE);
	} else {
		gs_texrender_reset(render);
	}
	return render;
}
```

obs_source_process_filter_tech_end の最終行の gs_set_linear_srgb(previous); は previous が true になってしまってる。たぶん、これで linear to srgb 変換が適用されてる気がする

たぶんだけど、obs-shaderfilter.c の shader_filter_render がメインでレンダリングしてるところ

obs-source.c としては obs_source_render_filters がメインっぽいかな


obs-scence.c の render_item の linear_srgb を強制 false にしてみる？


OBS のキャプチャのコールバックについて

* Display Capture: duplicator_capture_render
  * これは Windows の scRGB を GS_CS_709_EXTENDED に変換している
* Window Capture: wc_render
* Game Capture: game_capture_render
  * GS_R10G10B10A2 と GS_RGBA16F もちゃんと棲み分けしてる


# OBS の描画について
* 大元は obs-video.c の obs_graphics_thread_loop と思われる
  * output_frames でソースを GS_CS_709_EXTENDED に変換したり、フィルタを適用してる（きっと）
    * render_displays で GS_CS_709_EXTENDED を scRGB に変換して Windows にお任せしてる
* output_frames は最終的に source_render をコールしている
* source_render で色変換が行われている
* render_displays からはディスプレイ描画用のコールバックとして window-basic-main.cpp の OBSBasic::RenderMain がコールされている
  * その中の obs_render_main_texture_internal で scRGB への変換が行われているっぽいぞ！


# HDRアプリの Swap Chain について

* FF7RB: GS_R10G10B10A2
* Resident Evil 4: GS_R10G10B10A2
* Monster Hunter Wilds: GS_R10G10B10A2
* 自作アプリ: GS_RGBA16F
* Sky: Children of the Light: GS_R10G10B10A2
